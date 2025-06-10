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

#ifndef OBJ_BACKEND_H
#define OBJ_BACKEND_H

#include "obj_executor.h"
#include <string>
#include <memory>
#include <aws/core/Aws.h>
#include <aws/core/client/ClientConfiguration.h>
#include <aws/core/auth/AWSCredentials.h>
#include <aws/core/auth/AWSCredentialsProvider.h>
#include <aws/s3/S3Client.h>
#include <aws/s3/S3ClientConfiguration.h>
#include "backend/backend_engine.h"

class nixlObjEngine : public nixlBackendEngine {
public:
    nixlObjEngine(const nixlBackendInitParams* init_params);
    virtual ~nixlObjEngine();

    bool supportsRemote() const override {
        return false;
    }

    bool supportsLocal() const override {
        return true;
    }

    bool supportsNotif() const override {
        return false;
    }

    bool supportsProgTh() const override {
        return false;
    }

    nixl_mem_list_t getSupportedMems() const override {
        return {OBJ_SEG, DRAM_SEG};
    }

    nixl_status_t registerMem(const nixlBlobDesc &mem,
                              const nixl_mem_t &nixl_mem,
                              nixlBackendMD* &out) override;

    nixl_status_t deregisterMem(nixlBackendMD* meta) override;

    nixl_status_t connect(const std::string &remote_agent) override {
        return NIXL_SUCCESS;
    }

    nixl_status_t disconnect(const std::string &remote_agent) override {
        return NIXL_SUCCESS;
    }

    nixl_status_t unloadMD(nixlBackendMD* input) override {
        return NIXL_SUCCESS;
    }

    nixl_status_t prepXfer(const nixl_xfer_op_t &operation,
                           const nixl_meta_dlist_t &local,
                           const nixl_meta_dlist_t &remote,
                           const std::string &remote_agent,
                           nixlBackendReqH* &handle,
                           const nixl_opt_b_args_t* opt_args=nullptr) const override;

    nixl_status_t postXfer(const nixl_xfer_op_t &operation,
                           const nixl_meta_dlist_t &local,
                           const nixl_meta_dlist_t &remote,
                           const std::string &remote_agent,
                           nixlBackendReqH* &handle,
                           const nixl_opt_b_args_t* opt_args=nullptr) const override;

    nixl_status_t checkXfer(nixlBackendReqH* handle) const override;
    nixl_status_t releaseReqH(nixlBackendReqH* handle) const override;

    nixl_status_t loadLocalMD(nixlBackendMD* input, nixlBackendMD* &output) override {
        output = input;
        return NIXL_SUCCESS;
    }

private:
    std::shared_ptr<AsioThreadPoolExecutor> executor_;
    // This member must be defined before other AWS SDK members so that it
    // is destroyed last.
    std::unique_ptr<Aws::SDKOptions, void(*)(Aws::SDKOptions*)> aws_options_;
    std::unique_ptr<Aws::S3::S3Client> s3_client_;
    std::string bucket_name_;
};

#endif // OBJ_BACKEND_H
