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
#include <map>
#include <thread>
#include <mutex>

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
    auto start_time = absl::Now();
    std::vector<uint8_t> data(size);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(0, 255);

    for (auto& byte : data) {
        byte = static_cast<uint8_t>(distrib(gen));
    }
    auto total_time = absl::ToDoubleSeconds(absl::Now() - start_time);
    Logger() << "createRandomData: " << size << " bytes in " << total_time << " seconds";
    return data;
}

class TestTransfer : public testing::TestWithParam<std::string> {
protected:
    static nixlAgentConfig getConfig()
    {
        return nixlAgentConfig(true, false, 0,
                               nixl_thread_sync_t::NIXL_THREAD_SYNC_RW, 0,
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

    std::pair<std::vector<std::vector<MemBuffer<DRAM_SEG>>>, std::vector<std::vector<MemBuffer<DRAM_SEG>>>>
    initThreadBuffers(nixlAgent &from, nixlAgent &to, size_t num_threads, size_t count, size_t size,
                      nixl_mem_t src_mem_type, nixl_mem_t dst_mem_type, nixl_xfer_op_t mode)
    {
        auto start_time = absl::Now();
        std::vector<std::vector<MemBuffer<DRAM_SEG>>> thread_src_buffers(num_threads);
        std::vector<std::vector<MemBuffer<DRAM_SEG>>> thread_dst_buffers(num_threads);
        for (size_t thread_id = 0; thread_id < num_threads; ++thread_id) {
            for (size_t i = 0; i < count; i++) {
                if (mode == NIXL_WRITE) {
                    thread_src_buffers[thread_id].emplace_back(createRandomData<DRAM_SEG>(size));
                    thread_dst_buffers[thread_id].emplace_back(size);
                } else {
                    thread_src_buffers[thread_id].emplace_back(size);
                    thread_dst_buffers[thread_id].emplace_back(createRandomData<DRAM_SEG>(size));
                }
            }
            registerMem(from, thread_src_buffers[thread_id], src_mem_type);
            registerMem(to, thread_dst_buffers[thread_id], dst_mem_type);
        }
        auto total_time = absl::ToDoubleSeconds(absl::Now() - start_time);
        Logger() << "initThreadBuffers: " << num_threads << " threads, " << count << " buffers of " 
                 << size << " bytes in " << total_time << " seconds";
        return {std::move(thread_src_buffers), std::move(thread_dst_buffers)};
    }

    void validateThreadBuffers(const std::vector<std::vector<MemBuffer<DRAM_SEG>>>& thread_src_buffers,
                               const std::vector<std::vector<MemBuffer<DRAM_SEG>>>& thread_dst_buffers,
                               size_t num_threads, size_t count)
    {
        auto start_time = absl::Now();
        for (size_t thread_id = 0; thread_id < num_threads; ++thread_id) {
            for (size_t i = 0; i < count; i++) {
                EXPECT_EQ(thread_src_buffers[thread_id][i], thread_dst_buffers[thread_id][i])
                    << "Transfer validation failed for thread " << thread_id << " buffer " << i;
            }
        }
        auto total_time = absl::ToDoubleSeconds(absl::Now() - start_time);
        Logger() << "validateThreadBuffers: " << num_threads << " threads, " << count 
                 << " buffers in " << total_time << " seconds";
    }

    std::pair<std::vector<std::vector<std::string>>, std::set<std::string>>
    initThreadNotifications(size_t num_threads, size_t count, size_t batch_size)
    {
        auto start_time = absl::Now();
        std::vector<std::vector<std::string>> thread_notifs(num_threads);
        std::set<std::string> expected_msgs;
        for (size_t thread_id = 0; thread_id < num_threads; ++thread_id) {
            for (size_t batch_start = 0; batch_start < count; batch_start += batch_size) {
                size_t batch_idx = batch_start / batch_size;
                std::string notif = absl::StrFormat("notification_thread_%zu_batch_%zu", thread_id, batch_idx);
                thread_notifs[thread_id].push_back(notif);
                expected_msgs.insert(notif);
            }
        }
        auto total_time = absl::ToDoubleSeconds(absl::Now() - start_time);
        Logger() << "initThreadNotifications: " << num_threads << " threads, " 
                 << count << " total notifications in " << total_time << " seconds";
        return {std::move(thread_notifs), std::move(expected_msgs)};
    }

    void validateNotifications(const nixl_notifs_t& notif_map, const std::string& from_name,
                               const std::set<std::string>& expected_msgs)
    {
        auto start_time = absl::Now();
        auto& notif_list = notif_map.at(from_name);
        EXPECT_EQ(notif_list.size(), expected_msgs.size())
            << "Expected " << expected_msgs.size() << " notifications, got " << notif_list.size();

        std::set<std::string> remaining_msgs = expected_msgs;
        for (const auto& msg : notif_list) {
            EXPECT_TRUE(remaining_msgs.erase(msg) > 0)
                << "Unexpected or duplicate notification: " << msg;
        }
        EXPECT_TRUE(remaining_msgs.empty())
            << "Missing " << remaining_msgs.size() << " notifications";
        auto total_time = absl::ToDoubleSeconds(absl::Now() - start_time);
        Logger() << "validateNotifications: " << expected_msgs.size() 
                 << " notifications in " << total_time << " seconds";
    }

    void doTransfer(nixlAgent &from, const std::string &from_name,
                    nixlAgent &to, const std::string &to_name, size_t size,
                    size_t count, size_t batch_size,
                    nixl_mem_t src_mem_type, nixl_mem_t dst_mem_type,
                    nixl_xfer_op_t mode,
                    const std::vector<std::string>& expected_notifs,
                    const std::vector<MemBuffer<DRAM_SEG>>& src_buffers,
                    const std::vector<MemBuffer<DRAM_SEG>>& dst_buffers)
    {
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
    }

    void doTransfers(nixlAgent &from, const std::string &from_name,
                    nixlAgent &to, const std::string &to_name, size_t size,
                    size_t count, size_t batch_size,
                    nixl_mem_t src_mem_type, nixl_mem_t dst_mem_type,
                    nixl_xfer_op_t mode, size_t num_threads = 1)
    {
        auto [thread_src_buffers, thread_dst_buffers] = initThreadBuffers(
            from, to, num_threads, count, size, src_mem_type, dst_mem_type, mode);
        exchangeMD();

        auto [thread_notifs, expected_msgs] = initThreadNotifications(num_threads, count, batch_size);

        std::vector<std::thread> threads;
        for (size_t thread_id = 0; thread_id < num_threads; ++thread_id) {
            threads.emplace_back([&, thread_id]() {
                doTransfer(from, from_name, to, to_name, size, count, batch_size,
                          src_mem_type, dst_mem_type, mode, thread_notifs[thread_id],
                          thread_src_buffers[thread_id], thread_dst_buffers[thread_id]);
            });
        }

        for (auto& thread : threads) {
            thread.join();
        }

        validateThreadBuffers(thread_src_buffers, thread_dst_buffers, num_threads, count);

        invalidateMD();

        nixl_notifs_t notif_map;
        nixl_status_t status = to.getNotifs(notif_map);
        ASSERT_EQ(status, NIXL_SUCCESS);

        validateNotifications(notif_map, from_name, expected_msgs);
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
    // Tuple fields are: size, count, batch_size, num_threads
    constexpr std::array<std::tuple<size_t, size_t, size_t, size_t>, 3> test_cases = {
        {{4096, 1024, 32, 8},
         {32768, 64, 4, 8},
         {1000000, 8, 1, 8}}
    };

    for (const auto &[size, count, batch_size, num_threads] : test_cases) {
        doTransfers(getAgent(0), getAgentName(0), getAgent(1), getAgentName(1),
                    size, count, batch_size, DRAM_SEG, DRAM_SEG, NIXL_WRITE, num_threads);
        doTransfers(getAgent(0), getAgentName(0), getAgent(1), getAgentName(1),
                    size, count, batch_size, DRAM_SEG, DRAM_SEG, NIXL_READ, num_threads);
    }
}

INSTANTIATE_TEST_SUITE_P(ucx, TestTransfer, testing::Values("UCX"));

} // namespace gtest
