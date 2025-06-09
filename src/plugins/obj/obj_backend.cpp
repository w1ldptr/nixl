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
#include <aws/core/auth/AWSCredentials.h>
#include <aws/s3/S3Client.h>
#include <aws/s3/model/PutObjectRequest.h>
#include <aws/s3/model/GetObjectRequest.h>
#include <memory>
#include <stdexcept>
#include <optional>
#include <cstdlib>

namespace {
Aws::Client::ClientConfiguration
createClientConfiguration (nixl_b_params_t *custom_params) {
    Aws::Client::ClientConfiguration config;

    if (!custom_params) {
        return config;
    }

    auto endpoint_override_it = custom_params->find ("endpoint_override");
    if (endpoint_override_it != custom_params->end()) {
        config.endpointOverride = endpoint_override_it->second;
    }

    auto scheme_it = custom_params->find ("scheme");
    if (scheme_it != custom_params->end()) {
        if (scheme_it->second == "http") {
            config.scheme = Aws::Http::Scheme::HTTP;
        } else if (scheme_it->second == "https") {
            config.scheme = Aws::Http::Scheme::HTTPS;
        } else {
            throw std::runtime_error ("Invalid scheme: " + scheme_it->second);
        }
    }

    auto region_it = custom_params->find ("region");
    if (region_it != custom_params->end()) {
        config.region = region_it->second;
    }

    return config;
}

std::optional<Aws::Auth::AWSCredentials>
createAWSCredentials (nixl_b_params_t *custom_params) {
    if (!custom_params) {
        return std::nullopt;
    }

    std::string access_key, secret_key, session_token;

    auto access_key_it = custom_params->find ("access_key");
    if (access_key_it != custom_params->end()) {
        access_key = access_key_it->second;
    }

    auto secret_key_it = custom_params->find ("secret_key");
    if (secret_key_it != custom_params->end()) {
        secret_key = secret_key_it->second;
    }

    auto session_token_it = custom_params->find ("session_token");
    if (session_token_it != custom_params->end()) {
        session_token = session_token_it->second;
    }

    if (access_key.empty() || secret_key.empty()) {
        return std::nullopt;
    }

    if (session_token.empty()) {
        return Aws::Auth::AWSCredentials (access_key, secret_key);
    }
    return Aws::Auth::AWSCredentials (access_key, secret_key, session_token);
}

bool
getUserVirtualAddressing (nixl_b_params_t *custom_params) {
    if (!custom_params) {
        return false;
    }

    auto virtual_addressing_it = custom_params->find ("use_virtual_addressing");
    if (virtual_addressing_it != custom_params->end()) {
        const std::string &value = virtual_addressing_it->second;
        if (value == "true") {
            return true;
        } else if (value == "false") {
            return false;
        } else {
            throw std::runtime_error ("Invalid value for use_virtual_addressing: '" + value +
                                      "'. Must be 'true' or 'false'");
        }
    }

    return false;
}

std::string
getBucketName (nixl_b_params_t *custom_params) {
    // First try custom_params
    if (custom_params) {
        auto bucket_it = custom_params->find ("bucket");
        if (bucket_it != custom_params->end() && !bucket_it->second.empty()) {
            return bucket_it->second;
        }
    }

    // Then try environment variable
    const char *env_bucket = std::getenv ("AWS_DEFAULT_BUCKET");
    if (env_bucket && env_bucket[0] != '\0') {
        return std::string (env_bucket);
    }

    // Both methods failed
    throw std::runtime_error ("Bucket name not found. Please provide 'bucket' in custom_params or "
                              "set AWS_DEFAULT_BUCKET environment variable");
}
} // namespace

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
        executor_ (std::make_shared<AsioThreadPoolExecutor> (std::thread::hardware_concurrency())),
        aws_options_ (
                []() {
                    auto *opts = new Aws::SDKOptions();
                    Aws::InitAPI (*opts);
                    return opts;
                }(),
                [] (Aws::SDKOptions *opts) {
                    Aws::ShutdownAPI (*opts);
                    delete opts;
                }) {
    auto *custom_params = init_params->customParams;
    auto config = createClientConfiguration (custom_params);
    config.executor = executor_;
    auto credentials_opt = createAWSCredentials (custom_params);
    bool use_virtual_addressing = getUserVirtualAddressing (custom_params);
    bucket_name_ = getBucketName (custom_params);

    if (credentials_opt.has_value()) {
        s3_client_ = std::make_unique<Aws::S3::S3Client> (
                credentials_opt.value(),
                config,
                Aws::Client::AWSAuthV4Signer::PayloadSigningPolicy::RequestDependent,
                use_virtual_addressing);
    } else {
        s3_client_ = std::make_unique<Aws::S3::S3Client> (
                config,
                Aws::Client::AWSAuthV4Signer::PayloadSigningPolicy::RequestDependent,
                use_virtual_addressing);
    }

    NIXL_INFO << "Object storage backend initialized with AWS S3 client";
}

nixlObjEngine::~nixlObjEngine() {
    executor_->WaitUntilStopped();
}

nixl_status_t
nixlObjEngine::registerMem (const nixlBlobDesc &mem,
                            const nixl_mem_t &nixl_mem,
                            nixlBackendMD *&out) {
    nixlObjMetadata *obj_md = new nixlObjMetadata();
    obj_md->nixl_mem = nixl_mem;
    obj_md->dev_id = mem.devId;

    if (nixl_mem == OBJ_SEG) {
        if (mem.metaInfo.empty()) {
            obj_md->obj_key = std::to_string (mem.devId);
        } else {
            obj_md->obj_key = mem.metaInfo;
        }
        dev_id_to_obj_key[mem.devId] = obj_md->obj_key;
    }

    out = (nixlBackendMD *)obj_md;
    return NIXL_SUCCESS;
}

nixl_status_t
nixlObjEngine::deregisterMem (nixlBackendMD *meta) {
    nixlObjMetadata *obj_md = static_cast<nixlObjMetadata *> (meta);
    if (obj_md->nixl_mem == OBJ_SEG) {
        dev_id_to_obj_key.erase (obj_md->dev_id);
    }
    delete obj_md;

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

    req_h->completed_ = false;
    if (operation == NIXL_WRITE) {
        Aws::S3::Model::PutObjectRequest object_request;
        object_request.WithBucket (bucket_name_).WithKey (obj_key);
        auto data = Aws::MakeShared<Aws::StringStream> (
                "PutObjectInputStream",
                std::stringstream::in | std::stringstream::out | std::stringstream::binary);
        data->write (reinterpret_cast<char *> (local.begin()->addr),
                     local.begin()->len); // TODO: fix this to avoid copying data
        object_request.SetBody (data);


        s3_client_->PutObjectAsync (
                object_request,
                [req_h] (const Aws::S3::S3Client *client,
                         const Aws::S3::Model::PutObjectRequest &request,
                         const Aws::S3::Model::PutObjectOutcome &outcome,
                         const std::shared_ptr<const Aws::Client::AsyncCallerContext> &context) {
                    if (outcome.IsSuccess()) {
                        req_h->status_ = NIXL_SUCCESS;
                    } else {
                        req_h->status_ = NIXL_ERR_BACKEND;
                    }
                    req_h->completed_ = true;
                },
                nullptr);
    } else if (operation == NIXL_READ) {
        Aws::S3::Model::GetObjectRequest object_request;
        object_request.WithBucket (bucket_name_).WithKey (obj_key);
        void *addr = reinterpret_cast<void *> (local.begin()->addr);
        size_t len = local.begin()->len;
        s3_client_->GetObjectAsync (
                object_request,
                [req_h, addr, len] (
                        const Aws::S3::S3Client *client,
                        const Aws::S3::Model::GetObjectRequest &request,
                        const Aws::S3::Model::GetObjectOutcome &outcome,
                        const std::shared_ptr<const Aws::Client::AsyncCallerContext> &context) {
                    if (outcome.IsSuccess()) {
                        auto &stream = outcome.GetResult().GetBody();
                        stream.seekg (0, std::ios::beg);
                        stream.read (static_cast<char *> (addr), len);
                        req_h->status_ = NIXL_SUCCESS;
                        req_h->completed_ = true;
                    } else {
                        req_h->status_ = NIXL_ERR_BACKEND;
                        req_h->completed_ = true;
                    }
                });
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
