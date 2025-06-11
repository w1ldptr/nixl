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

#include "obj_backend.h"
#include "common/nixl_log.h"
#include "nixl_types.h"
#include <memory>

class nixlObjBackendReqH : public nixlBackendReqH {
public:
    std::atomic_bool completed_ = false;
    std::atomic<nixl_status_t> status_;
};

class nixlObjMetadata : public nixlBackendMD {
public:
    nixl_mem_t nixl_mem;
    uint64_t dev_id;
    std::string obj_key;

    nixlObjMetadata() : nixlBackendMD (true) {}
    ~nixlObjMetadata() {}
};

// -----------------------------------------------------------------------------
// Obj Engine Implementation
// -----------------------------------------------------------------------------

nixlObjEngine::nixlObjEngine (const nixlBackendInitParams *init_params) :
        nixlBackendEngine (init_params),
        executor_ (std::make_shared<AsioThreadPoolExecutor> (std::thread::hardware_concurrency())) {
    auto *custom_params = init_params->customParams;

    s3_client_ = std::make_shared<AwsS3Client>(custom_params, executor_);

    NIXL_INFO << "Object storage backend initialized with S3 client wrapper";
}

nixlObjEngine::nixlObjEngine (const nixlBackendInitParams *init_params, std::shared_ptr<IS3Client> s3_client) :
        nixlBackendEngine (init_params),
        executor_ (std::make_shared<AsioThreadPoolExecutor> (std::thread::hardware_concurrency())),
        s3_client_ (s3_client) {
    s3_client_->setExecutor(executor_);

    NIXL_INFO << "Object storage backend initialized with injected S3 client";
}

nixlObjEngine::~nixlObjEngine() {
    executor_->WaitUntilStopped();
}

nixl_status_t
nixlObjEngine::registerMem (const nixlBlobDesc &mem,
                            const nixl_mem_t &nixl_mem,
                            nixlBackendMD *&out) {
    if (nixl_mem != OBJ_SEG) {
        return NIXL_SUCCESS;
    }

    std::unique_ptr<nixlObjMetadata> obj_md = std::make_unique<nixlObjMetadata>();
    obj_md->nixl_mem = nixl_mem;
    obj_md->dev_id = mem.devId;

    if (mem.metaInfo.empty()) {
        obj_md->obj_key = std::to_string (mem.devId);
    } else {
        obj_md->obj_key = mem.metaInfo;
    }
    dev_id_to_obj_key[mem.devId] = obj_md->obj_key;

    out = obj_md.release();
    return NIXL_SUCCESS;
}

nixl_status_t
nixlObjEngine::deregisterMem (nixlBackendMD *meta) {
    nixlObjMetadata *obj_md = static_cast<nixlObjMetadata *> (meta);
    if (obj_md && obj_md->nixl_mem == OBJ_SEG) {
        std::unique_ptr<nixlObjMetadata> obj_md_ptr = std::unique_ptr<nixlObjMetadata> (obj_md);
        dev_id_to_obj_key.erase (obj_md->dev_id);
    }

    return NIXL_SUCCESS;
}

nixl_status_t
nixlObjEngine::prepXfer (const nixl_xfer_op_t &operation,
                         const nixl_meta_dlist_t &local,
                         const nixl_meta_dlist_t &remote,
                         const std::string &remote_agent,
                         nixlBackendReqH *&handle,
                         const nixl_opt_b_args_t *opt_args) const {
    nixlObjBackendReqH *req_h = new nixlObjBackendReqH();
    handle = req_h;
    return NIXL_SUCCESS;
}

nixl_status_t
nixlObjEngine::postXfer (const nixl_xfer_op_t &operation,
                         const nixl_meta_dlist_t &local,
                         const nixl_meta_dlist_t &remote,
                         const std::string &remote_agent,
                         nixlBackendReqH *&handle,
                         const nixl_opt_b_args_t *opt_args) const {
    nixlObjBackendReqH *req_h = static_cast<nixlObjBackendReqH *> (handle);

    nixlObjMetadata* obj_md = static_cast<nixlObjMetadata*> (remote.begin()->metadataP);
    if (!obj_md) {
        NIXL_ERROR << "No metadata found for remote descriptor";
        return NIXL_ERR_INVALID_PARAM;
    }

    std::string obj_key = obj_md->obj_key;
    if (obj_key.empty()) {
        NIXL_ERROR << "No object key found for device ID: " << obj_md->dev_id;
        return NIXL_ERR_INVALID_PARAM;
    }

    uintptr_t data_ptr = local.begin()->addr;
    size_t data_len = local.begin()->len;
    size_t offset = remote.begin()->addr;

    req_h->completed_ = false;
    if (operation == NIXL_WRITE) {
        s3_client_->PutObjectAsync(
            obj_key,
            data_ptr,
            data_len,
            offset,
            [req_h](bool success) {
                req_h->status_ = success ? NIXL_SUCCESS : NIXL_ERR_BACKEND;
                req_h->completed_ = true;
            }
        );
    } else if (operation == NIXL_READ) {
        s3_client_->GetObjectAsync(
            obj_key,
            data_ptr,
            data_len,
            offset,
            [req_h](bool success) {
                req_h->status_ = success ? NIXL_SUCCESS : NIXL_ERR_BACKEND;
                req_h->completed_ = true;
            }
        );
    } else {
        return NIXL_ERR_INVALID_PARAM;
    }

    return NIXL_IN_PROG;
}

nixl_status_t
nixlObjEngine::checkXfer (nixlBackendReqH *handle) const {
    nixlObjBackendReqH *req_h = static_cast<nixlObjBackendReqH *> (handle);
    if (req_h->completed_) {
        return req_h->status_;
    }
    return NIXL_IN_PROG;
}

nixl_status_t
nixlObjEngine::releaseReqH (nixlBackendReqH *handle) const {
    nixlObjBackendReqH *req_h = static_cast<nixlObjBackendReqH *> (handle);
    delete req_h;
    return NIXL_SUCCESS;
}
