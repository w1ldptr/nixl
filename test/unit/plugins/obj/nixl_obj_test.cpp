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
#include <iostream>
#include <string>
#include <getopt.h>
#include "nixl.h"
#include "nixl_params.h"
#include "nixl_descriptors.h"
#include "nixl_types.h"
#include <stdexcept>
#include <thread>
#include <cctype>

int main(int argc, char *argv[])
{
    std::cout << "NIXL Object Storage Plugin Test" << std::endl;

    std::string access_key;
    std::string secret_key;
    std::string token;
    int opt;

    while ((opt = getopt(argc, argv, "a:s:t:h")) != -1) {
        switch (opt) {
            case 'a':
                access_key = optarg;
                break;
            case 's':
                secret_key = optarg;
                break;
            case 't':
                token = optarg;
                break;
            case 'h':
            default:
                std::cout << "Usage: " << argv[0] << " [-a access_key] [-s secret_key] [-t token] [-h]" << std::endl;
                std::cout << "  -a access_key    AWS access key (required)" << std::endl;
                std::cout << "  -s secret_key    AWS secret key (required)" << std::endl;
                std::cout << "  -t token         AWS session token (optional)" << std::endl;
                std::cout << "  -h               Show this help message" << std::endl;
                return (opt == 'h') ? 0 : 1;
        }
    }

    if (access_key.empty() || secret_key.empty()) {
        std::cerr << "Error: Access key and secret key are required" << std::endl;
        return 1;
    }

    try {
        // Initialize NIXL agent
        nixlAgent agent("ObjTester", nixlAgentConfig(true));

        // Set up backend parameters
        nixl_b_params_t params;
        params["access_key"] = access_key;
        params["secret_key"] = secret_key;
        if (!token.empty()) {
            params["token"] = token;
        }
        params["bucket"] = "test-bucket";
        params["scheme"] = "http";
        params["endpoint_override"] = "http://localstack:4566";
        params["use_virtual_addressing"] = "false";

        // Create object storage backend
        nixlBackendH* obj = nullptr;
        nixl_status_t status = agent.createBackend("OBJ", params, obj);
        if (status != NIXL_SUCCESS) {
            std::cerr << "Error creating object storage backend: " << nixlEnumStrings::statusStr(status) << std::endl;
            return 1;
        }

        nixlBlobDesc dram_buf;
        std::string test_data = "test data";
        dram_buf.addr = (uintptr_t)(test_data.c_str());
        dram_buf.len = test_data.size();
        dram_buf.devId = 0;

        nixl_reg_dlist_t dram_reg(DRAM_SEG);
        dram_reg.addDesc (dram_buf);
        nixl_xfer_dlist_t dram_xfer(DRAM_SEG);
        dram_xfer.addDesc (dram_buf);

        nixl_status_t ret = agent.registerMem(dram_reg);
        if (ret != NIXL_SUCCESS) {
            std::cerr << "Failed to register DRAM memory with NIXL" << std::endl;
            return 1;
        }

        nixlBlobDesc obj_buf;
        obj_buf.addr = 0;
        obj_buf.len = test_data.size();
        obj_buf.devId = 0;
        obj_buf.metaInfo = "test-key";

        nixl_reg_dlist_t obj_reg(OBJ_SEG);
        obj_reg.addDesc (obj_buf);
        nixl_xfer_dlist_t obj_xfer(OBJ_SEG);
        obj_xfer.addDesc (obj_buf);

        ret = agent.registerMem(obj_reg);
        if (ret != NIXL_SUCCESS) {
            std::cerr << "Failed to register OBJ memory with NIXL" << std::endl;
            return 1;
        }

        nixlXferReqH* treq = nullptr;
        status = agent.createXferReq(NIXL_WRITE, dram_xfer, obj_xfer,
                                     "ObjTester", treq);
        if (status != NIXL_SUCCESS || !treq) {
            std::cerr << "Failed to create write transfer request - status: " << nixlEnumStrings::statusStr(status) << std::endl;
            return 1;
        }

        status = agent.postXferReq(treq);
        if (status < 0) {
            std::cerr << "Failed to post write transfer request - status: " << nixlEnumStrings::statusStr(status) << std::endl;
            agent.releaseXferReq(treq);
            return 1;
        }

        do {
            status = agent.getXferStatus(treq);
            if (status < 0) {
                std::cerr << "Error during write transfer - status: " << nixlEnumStrings::statusStr(status) << std::endl;
                agent.releaseXferReq(treq);
                return 1;
            }
            std::cout << "Transfer status: " << nixlEnumStrings::statusStr(status) << std::endl;
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        } while (status == NIXL_IN_PROG);

        for (char &c : test_data) {
            c = std::toupper(c);
        }
        std::cout << "Test data: " << test_data << std::endl;

        treq = nullptr;
        status = agent.createXferReq(NIXL_READ, dram_xfer, obj_xfer,
                                     "ObjTester", treq);
        if (status != NIXL_SUCCESS || !treq) {
            std::cerr << "Failed to create read transfer request - status: " << nixlEnumStrings::statusStr(status) << std::endl;
            return 1;
        }

        status = agent.postXferReq(treq);
        if (status < 0) {
            std::cerr << "Failed to post read transfer request - status: " << nixlEnumStrings::statusStr(status) << std::endl;
            agent.releaseXferReq(treq);
            return 1;
        }

        do {
            status = agent.getXferStatus(treq);
            if (status < 0) {
                std::cerr << "Error during read transfer - status: " << nixlEnumStrings::statusStr(status) << std::endl;
                agent.releaseXferReq(treq);
                return 1;
            }
            std::cout << "Transfer status: " << nixlEnumStrings::statusStr(status) << std::endl;
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        } while (status == NIXL_IN_PROG);

        std::cout << "Test data: " << test_data << std::endl;
        if (test_data != "test data") {
            std::cerr << "Test data mismatch" << std::endl;
            return 1;
        }
        std::cout << "Test passed" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Exception during initialization: " << e.what() << std::endl;
        return 1;
    }
}
