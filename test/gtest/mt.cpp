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
#include <gtest/gtest.h>
#include "nixl.h"
#include "plugin_manager.h"
#include <thread>

namespace gtest {
namespace mt {

class Mt : public testing::Test {
protected:
    // Helper method to create an agent with default config
    static nixlAgent createAgent() {
        nixlAgentConfig cfg(false);
        return nixlAgent("test_agent", cfg);
    }

    // Helper method to create a mock DRAM backend
    static nixlBackendH* createMockDramBackend(nixlAgent& agent) {
        nixlBackendH* backend_handle = nullptr;
        nixl_b_params_t params;
        nixl_status_t status = agent.createBackend("MOCK_DRAM", params, backend_handle);
        EXPECT_EQ(status, NIXL_SUCCESS);
        EXPECT_NE(backend_handle, nullptr);
        return backend_handle;
    }

    // Helper method to set up extra params with a backend
    static nixl_opt_args_t createExtraParams(nixlBackendH* backend) {
        nixl_opt_args_t extra_params;
        extra_params.backends = {backend};
        return extra_params;
    }

    // Helper method to create and register memory
    static void registerMemory(nixlAgent& agent, const nixl_opt_args_t& extra_params) {
        nixlBlobDesc blob(0, 1024, 0, "");
        nixlDescList<nixlBlobDesc> desc_list(DRAM_SEG);
        desc_list.addDesc(blob);

        nixl_status_t status = agent.registerMem(desc_list, &extra_params);
        EXPECT_EQ(status, NIXL_SUCCESS);
    }

    // Helper method to perform a transfer operation
    static void performTransfer(nixlAgent& agent, const nixl_opt_args_t& extra_params) {
        nixlXferReqH* xfer_req = nullptr;
        nixl_xfer_op_t op = NIXL_WRITE;
        nixlDescList<nixlBasicDesc> src_list(DRAM_SEG);
        nixlDescList<nixlBasicDesc> dst_list(DRAM_SEG);

        // Create basic descriptor for transfer
        nixlBasicDesc basic_desc(0, 1024, 0);
        src_list.addDesc(basic_desc);
        dst_list.addDesc(basic_desc);

        nixl_status_t status = agent.createXferReq(op, src_list, dst_list, "test_agent", xfer_req, &extra_params);
        EXPECT_EQ(status, NIXL_SUCCESS);
        EXPECT_NE(xfer_req, nullptr);

        status = agent.postXferReq(xfer_req);
        EXPECT_EQ(status, NIXL_SUCCESS);

        status = agent.getXferStatus(xfer_req);
        EXPECT_EQ(status, NIXL_SUCCESS);

        status = agent.releaseXferReq(xfer_req);
        EXPECT_EQ(status, NIXL_SUCCESS);
    }
};

TEST_F(Mt, ConcurrentTransfersWithPerThreadAgent) {
    // Function to execute transfer sequence with per-thread agent and memory registration
    auto transfer_sequence = []() {
        nixlAgent agent = createAgent();
        nixlBackendH* backend = createMockDramBackend(agent);
        nixl_opt_args_t extra_params = createExtraParams(backend);

        registerMemory(agent, extra_params);
        performTransfer(agent, extra_params);
    };

    // Create and run two threads executing the transfer sequence
    std::thread t1(transfer_sequence);
    std::thread t2(transfer_sequence);

    // Wait for both threads to complete
    t1.join();
    t2.join();
}

TEST_F(Mt, ConcurrentTransfersWithPerThreadMemory) {
    nixlAgent agent = createAgent();
    nixlBackendH* backend = createMockDramBackend(agent);
    nixl_opt_args_t extra_params = createExtraParams(backend);

    // Function to execute transfer sequence with per-thread memory registration
    auto transfer_sequence = [&agent, &extra_params]() {
        registerMemory(agent, extra_params);
        performTransfer(agent, extra_params);
    };

    // Create and run two threads executing the transfer sequence
    std::thread t1(transfer_sequence);
    std::thread t2(transfer_sequence);

    // Wait for both threads to complete
    t1.join();
    t2.join();
}

TEST_F(Mt, ConcurrentTransfers) {
    nixlAgent agent = createAgent();
    nixlBackendH* backend = createMockDramBackend(agent);
    nixl_opt_args_t extra_params = createExtraParams(backend);

    registerMemory(agent, extra_params);

    // Function to execute transfer sequence
    auto transfer_sequence = [&agent, &extra_params]() {
        performTransfer(agent, extra_params);
    };

    // Create and run two threads executing the transfer sequence
    std::thread t1(transfer_sequence);
    std::thread t2(transfer_sequence);

    // Wait for both threads to complete
    t1.join();
    t2.join();
}

TEST_F(Mt, RegisterMemWithMockDram) {
    nixlAgent agent = createAgent();
    nixlBackendH* backend = createMockDramBackend(agent);
    nixl_opt_args_t extra_params = createExtraParams(backend);

    registerMemory(agent, extra_params);
    performTransfer(agent, extra_params);
}

} // namespace mt
} // namespace gtest
