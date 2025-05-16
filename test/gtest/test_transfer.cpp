/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "common.h"
#include "gtest/gtest.h"

#include "nixl.h"
#include "nixl_types.h"

#include <gtest/gtest.h>
#include <absl/strings/str_format.h>
#include <absl/time/clock.h>
#include <memory>
#include <string>
#include <vector>
#include <random>
#include <set>

namespace gtest {

template<nixl_mem_t MemType>
class MemBuffer;

template<>
class MemBuffer<DRAM_SEG> {
public:
    MemBuffer(size_t size) :
        buffer_(size)
    {
    }

    MemBuffer(std::vector<uint8_t>&& data) :
        buffer_(std::move(data))
    {
    }

    uintptr_t data() const
    {
        return reinterpret_cast<uintptr_t>(buffer_.data());
    }

    size_t size() const
    {
        return buffer_.size();
    }

    bool operator==(const MemBuffer<DRAM_SEG>& other) const
    {
        return buffer_ == other.buffer_;
    }

private:
    std::vector<uint8_t> buffer_;
};

template<nixl_mem_t MemType>
std::vector<uint8_t> createRandomData(size_t size)
{
    std::vector<uint8_t> data(size);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(0, 255);

    for (auto& byte : data) {
        byte = static_cast<uint8_t>(distrib(gen));
    }
    return data;
}

class TestTransfer : public testing::TestWithParam<std::string> {
protected:
    static nixlAgentConfig getConfig()
    {
        return nixlAgentConfig(true, false, 0,
                               nixl_thread_sync_t::NIXL_THREAD_SYNC_STRICT, 0,
                               100000);
    }

    void SetUp() override
    {
        // Create two agents
        for (size_t i = 0; i < 2; i++) {
            agents.emplace_back(std::make_unique<nixlAgent>(getAgentName(i), getConfig()));
            nixlBackendH *backend_handle = nullptr;
            nixl_status_t status = agents.back()->createBackend(getBackendName(), {},
                                                                backend_handle);
            ASSERT_EQ(status, NIXL_SUCCESS);
            EXPECT_NE(backend_handle, nullptr);
        }
    }

    void TearDown() override
    {
        agents.clear();
    }

    std::string getBackendName() const
    {
        return GetParam();
    }

    template<typename Desc, typename Iter>
    nixlDescList<Desc>
    makeDescList(Iter begin, Iter end, nixl_mem_t mem_type)
    {
        nixlDescList<Desc> desc_list(mem_type);
        for (auto it = begin; it != end; ++it) {
            desc_list.addDesc(Desc(it->data(), it->size(), DEV_ID));
        }
        return desc_list;
    }

    void registerMem(nixlAgent &agent, const std::vector<MemBuffer<DRAM_SEG>> &buffers,
                     nixl_mem_t mem_type)
    {
        auto reg_list = makeDescList<nixlBlobDesc>(buffers.begin(), buffers.end(), mem_type);
        agent.registerMem(reg_list);
    }

    void exchangeMD()
    {
        // Connect the existing agents and exchange metadata
        for (size_t i = 0; i < agents.size(); i++) {
            nixl_blob_t md;
            nixl_status_t status = agents[i]->getLocalMD(md);
            ASSERT_EQ(status, NIXL_SUCCESS);

            for (size_t j = 0; j < agents.size(); j++) {
                if (i == j)
                    continue;
                std::string remote_agent_name;
                status = agents[j]->loadRemoteMD(md, remote_agent_name);
                ASSERT_EQ(status, NIXL_SUCCESS);
                EXPECT_EQ(remote_agent_name, getAgentName(i));
            }
        }
    }

    void invalidateMD()
    {
        // Disconnect the agents and invalidate remote metadata
        for (size_t i = 0; i < agents.size(); i++) {
            for (size_t j = 0; j < agents.size(); j++) {
                if (i == j)
                    continue;
                nixl_status_t status = agents[j]->invalidateRemoteMD(
                        getAgentName(i));
                ASSERT_EQ(status, NIXL_SUCCESS);
            }
        }
    }

    void waitForXfer(nixlAgent &from, const std::string &from_name,
                     nixlAgent &to, nixlXferReqH *xfer_req)
    {
        bool xfer_done;
        do {
            // progress on "from" agent while waiting for completion
            nixl_status_t status = from.getXferStatus(xfer_req);
            EXPECT_TRUE((status == NIXL_SUCCESS) || (status == NIXL_IN_PROG));
            xfer_done = (status == NIXL_SUCCESS);
        } while (!xfer_done);
    }

    void doTransfer(nixlAgent &from, const std::string &from_name,
                    nixlAgent &to, const std::string &to_name, size_t size,
                    size_t count, size_t batch_size,
                    nixl_mem_t src_mem_type, nixl_mem_t dst_mem_type,
                    nixl_xfer_op_t mode,
                    const std::vector<std::string>& expected_notifs)
    {
        std::vector<MemBuffer<DRAM_SEG>> src_buffers, dst_buffers;
        for (size_t i = 0; i < count; i++) {
            if (mode == NIXL_WRITE) {
                src_buffers.emplace_back(createRandomData<DRAM_SEG>(size));
                dst_buffers.emplace_back(size);
            } else {
                src_buffers.emplace_back(size);
                dst_buffers.emplace_back(createRandomData<DRAM_SEG>(size));
            }
        }

        registerMem(from, src_buffers, src_mem_type);
        registerMem(to, dst_buffers, dst_mem_type);
        exchangeMD();

        auto start_time = absl::Now();
        size_t total_transferred = 0;
        size_t notif_idx = 0;

        for (size_t batch_start = 0; batch_start < count; batch_start += batch_size) {
            size_t batch_end = std::min(batch_start + batch_size, count);

            nixl_opt_args_t extra_params;
            extra_params.hasNotif = true;
            extra_params.notifMsg = expected_notifs[notif_idx++];

            nixlXferReqH *xfer_req = nullptr;
            nixl_status_t status = from.createXferReq(
                mode,
                makeDescList<nixlBasicDesc>(
                    src_buffers.begin() + batch_start,
                    src_buffers.begin() + batch_end,
                    src_mem_type),
                makeDescList<nixlBasicDesc>(
                    dst_buffers.begin() + batch_start,
                    dst_buffers.begin() + batch_end,
                    dst_mem_type),
                to_name,
                xfer_req,
                &extra_params);
            ASSERT_EQ(status, NIXL_SUCCESS);
            EXPECT_NE(xfer_req, nullptr);

            status = from.postXferReq(xfer_req);
            ASSERT_GE(status, NIXL_SUCCESS);

            waitForXfer(from, from_name, to, xfer_req);

            status = from.getXferStatus(xfer_req);
            EXPECT_EQ(status, NIXL_SUCCESS);

            // Verify transfer was successful for this batch
            for (size_t j = batch_start; j < batch_end; j++) {
                EXPECT_EQ(src_buffers[j], dst_buffers[j])
                    << "Transfer validation failed for buffer " << j;
            }

            status = from.releaseXferReq(xfer_req);
            EXPECT_EQ(status, NIXL_SUCCESS);

            total_transferred += (batch_end - batch_start) * size;
        }

        auto total_time = absl::ToDoubleSeconds(absl::Now() - start_time);
        auto bandwidth = total_transferred / total_time / (1024 * 1024 * 1024);
        Logger() << (mode == NIXL_WRITE ? "Write" : "Read") << " transfer: "
                 << size << "x" << count << "=" << total_transferred
                 << " bytes in " << total_time << " seconds "
                 << "(" << bandwidth << " GB/s)";

        invalidateMD();
    }

    void doTransfers(nixlAgent &from, const std::string &from_name,
                     nixlAgent &to, const std::string &to_name, size_t size,
                     size_t count, size_t batch_size,
                     nixl_mem_t src_mem_type, nixl_mem_t dst_mem_type,
                     nixl_xfer_op_t mode)
    {
        std::vector<std::string> expected_notifs;
        for (size_t batch_start = 0; batch_start < count; batch_start += batch_size) {
            size_t batch_idx = batch_start / batch_size;
            expected_notifs.push_back(absl::StrFormat("notification_%zu", batch_idx));
        }

        doTransfer(from, from_name, to, to_name, size, count, batch_size,
                  src_mem_type, dst_mem_type, mode, expected_notifs);

        nixl_notifs_t notif_map;
        nixl_status_t status = to.getNotifs(notif_map);
        ASSERT_EQ(status, NIXL_SUCCESS);

        auto& notif_list = notif_map[from_name];
        EXPECT_EQ(notif_list.size(), expected_notifs.size())
            << "Expected " << expected_notifs.size() << " notifications, got " << notif_list.size();

        std::set<std::string> expected_msgs(expected_notifs.begin(), expected_notifs.end());

        for (const auto& msg : notif_list) {
            EXPECT_TRUE(expected_msgs.find(msg) != expected_msgs.end())
                << "Unexpected notification message: " << msg;
        }
    }

    nixlAgent &getAgent(size_t idx)
    {
        return *agents[idx];
    }

    std::string getAgentName(size_t idx)
    {
        return absl::StrFormat("agent_%d", idx);
    }

private:
    static constexpr uint64_t DEV_ID = 0;

    std::vector<std::unique_ptr<nixlAgent>> agents;
};

TEST_P(TestTransfer, RandomSizes)
{
    // Tuple fields are: size, count, batch_size
    constexpr std::array<std::tuple<size_t, size_t, size_t>, 3> test_cases = {
        {{4096, 128, 32},
         {32768, 32, 4},
         {1000000, 8, 1}}
    };

    for (const auto &[size, count, batch_size] : test_cases) {
        doTransfers(getAgent(0), getAgentName(0), getAgent(1), getAgentName(1),
                    size, count, batch_size, DRAM_SEG, DRAM_SEG, NIXL_WRITE);
        doTransfers(getAgent(0), getAgentName(0), getAgent(1), getAgentName(1),
                    size, count, batch_size, DRAM_SEG, DRAM_SEG, NIXL_READ);
    }
}

INSTANTIATE_TEST_SUITE_P(ucx, TestTransfer, testing::Values("UCX"));

} // namespace gtest
