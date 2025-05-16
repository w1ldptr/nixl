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

    void zero()
    {
        std::fill(buffer_.begin(), buffer_.end(), 0);
    }

    bool operator==(const MemBuffer<DRAM_SEG>& other) const
    {
        return buffer_ == other.buffer_;
    }

private:
    std::vector<uint8_t> buffer_;
};

class TestTransfer : public testing::TestWithParam<std::string> {
protected:
    TestTransfer() : rd(), gen(rd()), distrib() {}

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

    template<nixl_mem_t MemType>
    void registerMem(nixlAgent &agent, const std::vector<MemBuffer<MemType>> &buffers)
    {
        auto reg_list = makeDescList<nixlBlobDesc>(buffers.begin(), buffers.end(), MemType);
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

    template<nixl_mem_t SrcMemType, nixl_mem_t DstMemType>
    std::pair<std::vector<std::vector<MemBuffer<SrcMemType>>>, std::vector<std::vector<MemBuffer<DstMemType>>>>
    initThreadBuffers(nixlAgent &from, nixlAgent &to, size_t num_threads, size_t count, size_t size,
                      nixl_xfer_op_t mode)
    {
        std::vector<std::vector<MemBuffer<SrcMemType>>> thread_src_buffers(num_threads);
        std::vector<std::vector<MemBuffer<DstMemType>>> thread_dst_buffers(num_threads);
        for (size_t thread_id = 0; thread_id < num_threads; ++thread_id) {
            for (size_t i = 0; i < count; i++) {
                if (mode == NIXL_WRITE) {
                    thread_src_buffers[thread_id].emplace_back(createRandomData<SrcMemType>(size));
                    thread_dst_buffers[thread_id].emplace_back(size);
                } else {
                    thread_src_buffers[thread_id].emplace_back(size);
                    thread_dst_buffers[thread_id].emplace_back(createRandomData<DstMemType>(size));
                }
            }
            registerMem(from, thread_src_buffers[thread_id]);
            registerMem(to, thread_dst_buffers[thread_id]);
        }
        return {std::move(thread_src_buffers), std::move(thread_dst_buffers)};
    }

    template<nixl_mem_t SrcMemType, nixl_mem_t DstMemType>
    void validateThreadBuffers(const std::vector<std::vector<MemBuffer<SrcMemType>>>& thread_src_buffers,
                              const std::vector<std::vector<MemBuffer<DstMemType>>>& thread_dst_buffers,
                              size_t num_threads, size_t count)
    {
        for (size_t thread_id = 0; thread_id < num_threads; ++thread_id) {
            for (size_t i = 0; i < count; i++) {
                EXPECT_EQ(thread_src_buffers[thread_id][i], thread_dst_buffers[thread_id][i])
                    << "Transfer validation failed for thread " << thread_id << " buffer " << i;
            }
        }
    }

    std::vector<std::vector<std::string>>
    initThreadNotifications(size_t num_threads, size_t count, size_t batch_size)
    {
        std::vector<std::vector<std::string>> thread_notifs(num_threads);
        for (size_t thread_id = 0; thread_id < num_threads; ++thread_id) {
            for (size_t batch_start = 0; batch_start < count; batch_start += batch_size) {
                size_t batch_idx = batch_start / batch_size;
                std::string notif = absl::StrFormat("notification_thread_%zu_batch_%zu", thread_id, batch_idx);
                thread_notifs[thread_id].push_back(notif);
            }
        }
        return thread_notifs;
    }

    void validateNotifications(const nixl_notifs_t& notif_map, const std::string& from_name,
                               const std::vector<std::vector<std::string>>& thread_notifs)
    {
        std::set<std::string> expected_msgs;
        for (const auto& thread_notif_list : thread_notifs) {
            expected_msgs.insert(thread_notif_list.begin(), thread_notif_list.end());
        }

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
    }

    template<nixl_mem_t SrcMemType, nixl_mem_t DstMemType>
    void zeroBuffers(std::vector<std::vector<MemBuffer<SrcMemType>>>& thread_src_buffers,
                    std::vector<std::vector<MemBuffer<DstMemType>>>& thread_dst_buffers,
                    size_t num_threads, size_t count, nixl_xfer_op_t mode)
    {
        for (size_t thread_id = 0; thread_id < num_threads; ++thread_id) {
            for (size_t i = 0; i < count; i++) {
                if (mode == NIXL_WRITE) {
                    thread_dst_buffers[thread_id][i].zero();
                } else {
                    thread_src_buffers[thread_id][i].zero();
                }
            }
        }
    }

    template<nixl_mem_t SrcMemType, nixl_mem_t DstMemType>
    void doTransfer(nixlAgent &from, const std::string &from_name,
                   nixlAgent &to, const std::string &to_name, size_t size,
                   size_t count, size_t batch_size,
                   nixl_xfer_op_t mode,
                   const std::vector<std::string>& expected_notifs,
                   const std::vector<MemBuffer<SrcMemType>>& src_buffers,
                   const std::vector<MemBuffer<DstMemType>>& dst_buffers)
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
                    SrcMemType),
                makeDescList<nixlBasicDesc>(
                    dst_buffers.begin() + batch_start,
                    dst_buffers.begin() + batch_end,
                    DstMemType),
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

    template<nixl_mem_t SrcMemType, nixl_mem_t DstMemType>
    void doTransfers(nixlAgent &from, const std::string &from_name,
                    nixlAgent &to, const std::string &to_name, size_t size,
                    size_t count, size_t batch_size, size_t repeat,
                    nixl_xfer_op_t mode, size_t num_threads = 1)
    {
        auto [thread_src_buffers, thread_dst_buffers] = initThreadBuffers<SrcMemType, DstMemType>(
            from, to, num_threads, count, size, mode);
        exchangeMD();

        auto base_thread_notifs = initThreadNotifications(num_threads, count, batch_size);

        for (size_t repeat_idx = 0; repeat_idx < repeat; ++repeat_idx) {
            zeroBuffers<SrcMemType, DstMemType>(thread_src_buffers, thread_dst_buffers, num_threads, count, mode);

            std::vector<std::thread> threads;
            for (size_t thread_id = 0; thread_id < num_threads; ++thread_id) {
                threads.emplace_back([&, thread_id]() {
                    doTransfer<SrcMemType, DstMemType>(from, from_name, to, to_name, size, count, batch_size,
                              mode, base_thread_notifs[thread_id],
                              thread_src_buffers[thread_id], thread_dst_buffers[thread_id]);
                });
            }

            for (auto& thread : threads) {
                thread.join();
            }

            validateThreadBuffers<SrcMemType, DstMemType>(thread_src_buffers, thread_dst_buffers, num_threads, count);

            nixl_notifs_t notif_map;
            nixl_status_t status = to.getNotifs(notif_map);
            ASSERT_EQ(status, NIXL_SUCCESS);

            validateNotifications(notif_map, from_name, base_thread_notifs);
        }

        invalidateMD();
    }

    nixlAgent &getAgent(size_t idx)
    {
        return *agents[idx];
    }

    std::string getAgentName(size_t idx)
    {
        return absl::StrFormat("agent_%d", idx);
    }

    template<nixl_mem_t MemType>
    std::vector<uint8_t> createRandomData(size_t size)
    {
        size_t aligned_size = (size + 7) & ~7;
        std::vector<uint8_t> data(aligned_size);

        for (size_t i = 0; i < aligned_size; i += 8) {
            uint64_t rand_val = distrib(gen);
            for (size_t j = 0; j < 8; ++j) {
                data[i + j] = static_cast<uint8_t>(rand_val >> (j * 8));
            }
        }

        data.resize(size);
        return data;
    }

private:
    static constexpr uint64_t DEV_ID = 0;

    std::vector<std::unique_ptr<nixlAgent>> agents;
    std::random_device rd;
    std::mt19937_64 gen;
    std::uniform_int_distribution<uint64_t> distrib;
};

TEST_P(TestTransfer, RandomSizes)
{
    // Tuple fields are: size, count, batch_size, repeat, num_threads
    constexpr std::array<std::tuple<size_t, size_t, size_t, size_t, size_t>, 3> test_cases = {
        {{4096, 32, 4, 3, 4},
         {32768, 16, 2, 2, 4},
         {1000000, 8, 1, 2, 4}}
    };

    for (const auto &[size, count, batch_size, repeat, num_threads] : test_cases) {
        doTransfers<DRAM_SEG, DRAM_SEG>(getAgent(0), getAgentName(0), getAgent(1), getAgentName(1),
                    size, count, batch_size, repeat, NIXL_WRITE, num_threads);
        doTransfers<DRAM_SEG, DRAM_SEG>(getAgent(0), getAgentName(0), getAgent(1), getAgentName(1),
                    size, count, batch_size, repeat, NIXL_READ, num_threads);
    }
}

INSTANTIATE_TEST_SUITE_P(ucx, TestTransfer, testing::Values("UCX"));

} // namespace gtest
