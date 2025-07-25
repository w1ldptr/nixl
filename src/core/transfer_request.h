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
#ifndef __TRANSFER_REQUEST_H_
#define __TRANSFER_REQUEST_H_

constexpr auto min_chrono_time = std::chrono::time_point<std::chrono::high_resolution_clock>::min();
using chrono_point_t = std::chrono::high_resolution_clock::time_point;

// Contains pointers to corresponding backend engine and its handler, and populated
// and verified DescLists, and other state and metadata needed for a NIXL transfer
class nixlXferReqH {
    private:
        nixlBackendEngine* engine         = nullptr;
        nixlBackendReqH*   backendHandle  = nullptr;

        nixl_meta_dlist_t* initiatorDescs = nullptr;
        nixl_meta_dlist_t* targetDescs    = nullptr;

        std::string        remoteAgent;
        nixl_blob_t        notifMsg;
        bool               hasNotif       = false;

        nixl_xfer_op_t     backendOp;
        nixl_status_t      status;

        struct {
            chrono_point_t startTime;
            size_t totalBytes;
        } telemetry;

    public:
        inline nixlXferReqH() { }

        inline ~nixlXferReqH() {
            // delete checks for nullptr itself
            delete initiatorDescs;
            delete targetDescs;
            if (backendHandle != nullptr)
                engine->releaseReqH(backendHandle);
        }

        void
        updateRequestStats(const std::string &dbg_msg_type);

        friend class nixlAgent;
};

class nixlDlistH {
    private:
        std::unordered_map<nixlBackendEngine*, nixl_meta_dlist_t*> descs;

        std::string        remoteAgent;
        bool               isLocal;

    public:
        inline nixlDlistH() { }

        inline ~nixlDlistH() {
            for (auto & elm : descs)
                delete elm.second;
        }

    friend class nixlAgent;
};

#endif
