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
#include "ucx_backend.h"
#include "serdes/serdes.h"
#include "common/nixl_log.h"

#ifdef HAVE_CUDA

#include <cuda_runtime.h>
#include <cufile.h>

#endif

#include <poll.h>

static constexpr int const noSyncIters = 32;

/****************************************
 * CUDA related code
 *****************************************/

class nixlUcxCudaCtx {
public:
#ifdef HAVE_CUDA
    CUcontext pthrCudaCtx;
    int myDevId;

    nixlUcxCudaCtx() {
        pthrCudaCtx = NULL;
        myDevId = -1;
    }
#endif
    void cudaResetCtxPtr();
    int cudaUpdateCtxPtr(void *address, int expected_dev, bool &was_updated);
    int cudaSetCtx();
};

#ifdef HAVE_CUDA

static int cudaQueryAddr(void *address, bool &is_dev,
                         CUdevice &dev, CUcontext &ctx)
{
    CUmemorytype mem_type = CU_MEMORYTYPE_HOST;
    uint32_t is_managed = 0;
#define NUM_ATTRS 4
    CUpointer_attribute attr_type[NUM_ATTRS];
    void *attr_data[NUM_ATTRS];
    CUresult result;

    attr_type[0] = CU_POINTER_ATTRIBUTE_MEMORY_TYPE;
    attr_data[0] = &mem_type;
    attr_type[1] = CU_POINTER_ATTRIBUTE_IS_MANAGED;
    attr_data[1] = &is_managed;
    attr_type[2] = CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL;
    attr_data[2] = &dev;
    attr_type[3] = CU_POINTER_ATTRIBUTE_CONTEXT;
    attr_data[3] = &ctx;

    result = cuPointerGetAttributes(4, attr_type, attr_data, (CUdeviceptr)address);

    is_dev = (mem_type == CU_MEMORYTYPE_DEVICE);

    return (CUDA_SUCCESS != result);
}

int nixlUcxCudaCtx::cudaUpdateCtxPtr(void *address, int expected_dev, bool &was_updated)
{
    bool is_dev;
    CUdevice dev;
    CUcontext ctx;
    int ret;

    was_updated = false;

    /* TODO: proper error codes and log outputs through this method */
    if (expected_dev == -1)
        return -1;

    // incorrect dev id from first registration
    if (myDevId != -1 && expected_dev != myDevId)
        return -1;

    ret = cudaQueryAddr(address, is_dev, dev, ctx);
    if (ret) {
        return ret;
    }

    if (!is_dev) {
        return 0;
    }

    if (dev != expected_dev) {
        // User provided address that does not match dev_id
        return -1;
    }

    if (pthrCudaCtx) {
        // Context was already set previously, and does not match new context
        if (pthrCudaCtx != ctx) {
            return -1;
        }
        return 0;
    }

    pthrCudaCtx = ctx;
    was_updated = true;
    myDevId = expected_dev;

    return 0;
}

int nixlUcxCudaCtx::cudaSetCtx()
{
    CUresult result;
    if (NULL == pthrCudaCtx) {
        return 0;
    }

    result = cuCtxSetCurrent(pthrCudaCtx);

    return (CUDA_SUCCESS == result);
}

#else

int nixlUcxCudaCtx::cudaUpdateCtxPtr(void *address, int expected_dev, bool &was_updated)
{
    was_updated = false;
    return 0;
}

int nixlUcxCudaCtx::cudaSetCtx() {
    return 0;
}

#endif


void nixlUcxEngine::vramInitCtx()
{
    cudaCtx = new nixlUcxCudaCtx;
}

int nixlUcxEngine::vramUpdateCtx(void *address, uint64_t  devId, bool &restart_reqd)
{
    int ret;
    bool was_updated;

    restart_reqd = false;

    if(!cuda_addr_wa) {
        // Nothing to do
        return 0;
    }

    ret = cudaCtx->cudaUpdateCtxPtr(address, devId, was_updated);
    if (ret) {
        return ret;
    }

    restart_reqd = was_updated;

    return 0;
}

int nixlUcxEngine::vramApplyCtx()
{
    if(!cuda_addr_wa) {
        // Nothing to do
        return 0;
    }

    return cudaCtx->cudaSetCtx();
}

void nixlUcxEngine::vramFiniCtx()
{
    delete cudaCtx;
}

/****************************************
 * UCX request management
*****************************************/


class nixlUcxIntReq : public nixlLinkElem<nixlUcxIntReq> {
    private:
        int _completed;
    public:
        std::string *amBuffer;

        nixlUcxIntReq() : nixlLinkElem() {
            _completed = 0;
            amBuffer = NULL;
        }

        ~nixlUcxIntReq() {
            _completed = 0;
            if (amBuffer) {
                delete amBuffer;
            }
        }

        bool is_complete() { return _completed; }
        void completed() { _completed = 1; }
};

static void _internalRequestInit(void *request)
{
    /* Initialize request in-place (aka "placement new")*/
    new(request) nixlUcxIntReq;
}

static void _internalRequestFini(void *request)
{
    /* Finalize request */
    nixlUcxIntReq *req = (nixlUcxIntReq*)request;
    req->~nixlUcxIntReq();
}


static void _internalRequestReset(nixlUcxIntReq *req) {
    _internalRequestFini((void *)req);
    _internalRequestInit((void *)req);
}

/****************************************
 * Backend request management
*****************************************/

class nixlUcxBackendH : public nixlBackendReqH {
private:
    nixlUcxIntReq head;
    nixlUcxEngine &eng;
    size_t worker_id;

public:

    nixlUcxBackendH(nixlUcxEngine &eng_, size_t worker_id_): eng(eng_), worker_id(worker_id_) {}

    void append(nixlUcxIntReq *req) {
        head.link(req);
    }

    nixl_status_t release()
    {
        nixlUcxIntReq *req = head.next();

        if (!req) {
            return NIXL_SUCCESS;
        }

        const auto &uw = eng.getWorker(worker_id);
        // TODO: Error log: uncompleted requests found! Cancelling ...
        while(req) {
            nixlUcxIntReq *cur = req;
            bool done = cur->is_complete();
            req = cur->unlink();
            if (!done) {
                // TODO: Need process this properly.
                // it may not be enough to cancel UCX request
                uw->reqCancel((nixlUcxReq)cur);
            }
            _internalRequestReset(cur);
            uw->reqRelease((nixlUcxReq)cur);
        }
        return NIXL_SUCCESS;
    }


    nixl_status_t status()
    {
        nixlUcxIntReq *req = head.next();
        nixl_status_t out_ret = NIXL_SUCCESS;

        if (NULL == req) {
            /* No pending transmissions */
            return NIXL_SUCCESS;
        }

        const auto &uw = eng.getWorker(worker_id);
        /* Go over all request updating their status */
        while(req) {
            nixl_status_t ret;
            if (!req->is_complete()) {
                ret = uw->test((nixlUcxReq)req);
                switch (ret) {
                    case NIXL_SUCCESS:
                        /* Mark as completed */
                        req->completed();
                        break;
                    case NIXL_IN_PROG:
                        out_ret = NIXL_IN_PROG;
                        break;
                    default:
                        /* Any other ret value is ERR and will be returned */
                        return ret;
                }
            }
            req = req->next();
        }

        /* Remove completed requests keeping the first one as
        request representative */
        req = head.unlink();
        while(req) {
            nixlUcxIntReq *next_req = req->unlink();
            if (req->is_complete()) {
                _internalRequestReset(req);
                uw->reqRelease((nixlUcxReq)req);
            } else {
                /* Enqueue back */
                append(req);
            }
            req = next_req;
        }

        return out_ret;
    }

    size_t getWorkerId() const {
        return worker_id;
    }
};

/****************************************
 * Progress thread management
*****************************************/

void nixlUcxEngine::progressFunc()
{
    using namespace nixlTime;

    vramApplyCtx();

    {
        std::unique_lock<std::mutex> lock(pthrActiveLock);
        pthrActive = true;
    }
    pthrActiveCV.notify_one();

    std::vector<pollfd> poll_fds;
    for (auto &uw: uws) {
        int fd;
        if (ucp_worker_get_efd(uw->getWorker(), &fd) == UCS_OK)
            poll_fds.push_back({fd, POLLIN, 0});
        else
            NIXL_ERROR << "Couldn't obtain fd for a worker";
    }

    while (!pthrStop) {
        bool made_progress = false;
        for (auto &uw: uws)
            while (uw->progress())
                made_progress = true;

        if (made_progress) {
            notifProgress();
            continue;
        }

        while (true) {
            bool all_armed = true;
            for (auto &uw: uws) {
                ucs_status_t status = ucp_worker_arm(uw->getWorker());
                if (status != UCS_OK) {
                    if (status != UCS_ERR_BUSY)
                        NIXL_ERROR << "Couldn't arm worker, error " << status << std::endl;
                    all_armed = false;
                    break;
                }
            }
            if (!all_armed || pthrStop)
                break;

            if (poll(poll_fds.data(), poll_fds.size(), pthrDelay) > 0) {
                for (size_t i = 0; i < poll_fds.size(); i++) {
                    if (poll_fds[i].revents & POLLIN) {
                        while (uws[i]->progress());
                        poll_fds[i].revents = 0;
                    }
                }
                notifProgress();
            }
        }
    }
}

void nixlUcxEngine::progressThreadStart()
{
    pthrStop = false;

    if (!pthrOn) {
        // not enabled
        return;
    }

    pthr = std::thread(&nixlUcxEngine::progressFunc, this);

    std::unique_lock<std::mutex> lock(pthrActiveLock);
    pthrActiveCV.wait(lock, [&]{ return pthrActive; });
}

void nixlUcxEngine::progressThreadStop()
{
    if (!pthrOn) {
        // not enabled
        return;
    }

    pthrStop = true;
    pthr.join();
}

void nixlUcxEngine::progressThreadRestart()
{
    progressThreadStop();
    progressThreadStart();
}

/****************************************
 * Constructor/Destructor
*****************************************/

nixlUcxEngine::nixlUcxEngine (const nixlBackendInitParams* init_params)
: nixlBackendEngine (init_params) {
    std::vector<std::string> devs; /* Empty vector */
    nixl_b_params_t* custom_params = init_params->customParams;

    if (!init_params->numWorkers) {
        this->initErr = true;
        return;
    }
    if (init_params->enableProgTh) {
        if (!nixlUcxContext::mtLevelIsSupproted(NIXL_UCX_MT_WORKER)) {
            this->initErr = true;
            return;
        }
    }

    if (custom_params->count("device_list")!=0)
        devs = str_split((*custom_params)["device_list"], ", ");

    uc = std::make_shared<nixlUcxContext>(devs, sizeof(nixlUcxIntReq),
                                          _internalRequestInit, _internalRequestFini, NIXL_UCX_MT_WORKER,
                                          init_params->enableProgTh);
    for (unsigned int i = 0; i < init_params->numWorkers; i++) {
        uint64_t n_addr;
        auto uw = std::make_unique<nixlUcxWorker>(uc);
        uw->epAddr(n_addr, workerSize);
        workerAddr = (void*) n_addr;

        uw->regAmCallback(CONN_CHECK, connectionCheckAmCb, this);
        uw->regAmCallback(DISCONNECT, connectionTermAmCb, this);
        uw->regAmCallback(NOTIF_STR, notifAmCb, this);

        uws.push_back(std::move(uw));
    }

    if (init_params->enableProgTh) {
        pthrOn = true;
        pthrDelay = init_params->pthrDelay;
    } else {
        pthrOn = false;
    }

    // Temp fixup
    if (getenv("NIXL_DISABLE_CUDA_ADDR_WA")) {
        std::cout << "WARNING: disabling CUDA address workaround" << std::endl;
        cuda_addr_wa = false;
    } else {
        cuda_addr_wa = true;
    }
    vramInitCtx();
    progressThreadStart();
}

nixl_mem_list_t nixlUcxEngine::getSupportedMems () const {
    nixl_mem_list_t mems;
    mems.push_back(DRAM_SEG);
    mems.push_back(VRAM_SEG);
    return mems;
}

// Through parent destructor the unregister will be called.
nixlUcxEngine::~nixlUcxEngine () {
    // per registered memory deregisters it, which removes the corresponding metadata too
    // parent destructor takes care of the desc list
    // For remote metadata, they should be removed here
    if (this->initErr) {
        // Nothing to do
        return;
    }

    progressThreadStop();
    vramFiniCtx();
    free(workerAddr);
}

/****************************************
 * Connection management
*****************************************/

nixl_status_t nixlUcxEngine::checkConn(const std::string &remote_agent) {
     if(remoteConnMap.find(remote_agent) == remoteConnMap.end()) {
        return NIXL_ERR_NOT_FOUND;
    }
    return NIXL_SUCCESS;
}

nixl_status_t nixlUcxEngine::endConn(const std::string &remote_agent) {

    auto search = remoteConnMap.find(remote_agent);

    if(search == remoteConnMap.end()) {
        return NIXL_ERR_NOT_FOUND;
    }

    nixlUcxConnection &conn = remoteConnMap[remote_agent];

    bool error = false;
    for (size_t i = 0; i < conn.eps.size(); i++)
        if(getWorker(i)->disconnect_nb(conn.getEp(i)) < 0)
            error = true;
    //thread safety?
    remoteConnMap.erase(remote_agent);

    return error ? NIXL_ERR_BACKEND : NIXL_SUCCESS;
}

nixl_status_t nixlUcxEngine::getConnInfo(std::string &str) const {
    str = nixlSerDes::_bytesToString(workerAddr, workerSize);
    return NIXL_SUCCESS;
}

ucs_status_t
nixlUcxEngine::connectionCheckAmCb(void *arg, const void *header,
                                   size_t header_length, void *data,
                                   size_t length,
                                   const ucp_am_recv_param_t *param)
{
    struct nixl_ucx_am_hdr* hdr = (struct nixl_ucx_am_hdr*) header;

    std::string remote_agent( (char*) data, length);
    nixlUcxEngine* engine = (nixlUcxEngine*) arg;

    if(hdr->op != CONN_CHECK) {
        //is this the best way to ERR?
        return UCS_ERR_INVALID_PARAM;
    }

    //send_am should be forcing EAGER protocol
    if((param->recv_attr & UCP_AM_RECV_ATTR_FLAG_RNDV) != 0) {
        //is this the best way to ERR?
        return UCS_ERR_INVALID_PARAM;
    }

    if(engine->checkConn(remote_agent)) {
        //TODO: received connect AM from agent we don't recognize
        return UCS_ERR_INVALID_PARAM;
    }

    return UCS_OK;
}

ucs_status_t
nixlUcxEngine::connectionTermAmCb (void *arg, const void *header,
                                   size_t header_length, void *data,
                                   size_t length,
                                   const ucp_am_recv_param_t *param)
{
    struct nixl_ucx_am_hdr* hdr = (struct nixl_ucx_am_hdr*) header;

    std::string remote_agent( (char*) data, length);

    if(hdr->op != DISCONNECT) {
        //is this the best way to ERR?
        return UCS_ERR_INVALID_PARAM;
    }

    //send_am should be forcing EAGER protocol
    if((param->recv_attr & UCP_AM_RECV_ATTR_FLAG_RNDV) != 0) {
        //is this the best way to ERR?
        return UCS_ERR_INVALID_PARAM;
    }
/*
    // TODO: research UCX connection logic and fix.
    nixlUcxEngine* engine = (nixlUcxEngine*) arg;
    if(NIXL_SUCCESS != engine->endConn(remote_agent)) {
        //TODO: received connect AM from agent we don't recognize
        return UCS_ERR_INVALID_PARAM;
    }
*/
    return UCS_OK;
}

nixl_status_t nixlUcxEngine::connect(const std::string &remote_agent) {
    struct nixl_ucx_am_hdr hdr;
    uint32_t flags = 0;

    if (remote_agent == localAgent)
        return loadRemoteConnInfo (remote_agent,
                   nixlSerDes::_bytesToString(workerAddr, workerSize));

    auto search = remoteConnMap.find(remote_agent);

    if(search == remoteConnMap.end()) {
        return NIXL_ERR_NOT_FOUND;
    }

    nixlUcxConnection &conn = remoteConnMap[remote_agent];

    hdr.op = CONN_CHECK;
    //agent names should never be long enough to need RNDV
    flags |= UCP_AM_SEND_FLAG_EAGER;

    bool error = false;
    nixl_status_t ret = NIXL_SUCCESS;
    std::vector<nixlUcxReq> reqs;
    for (size_t i = 0; i < uws.size(); i++) {
        reqs.emplace_back();
        ret = getWorker(i)->sendAm(conn.getEp(i), CONN_CHECK,
                                   &hdr, sizeof(struct nixl_ucx_am_hdr),
                                   (void*) localAgent.data(), localAgent.size(),
                                   flags, reqs.back());
        if(ret < 0) {
            error = true;
            break;
        }
    }

    //wait for AM to send
    for (size_t i = 0; i < reqs.size(); i++)
        while(ret == NIXL_IN_PROG)
            ret = getWorker(i)->test(reqs[i]);

    return error ? NIXL_SUCCESS : NIXL_ERR_BACKEND;
}

nixl_status_t nixlUcxEngine::disconnect(const std::string &remote_agent) {

    static struct nixl_ucx_am_hdr hdr;
    uint32_t flags = 0;

    if (remote_agent != localAgent) {
        auto search = remoteConnMap.find(remote_agent);

        if(search == remoteConnMap.end()) {
            return NIXL_ERR_NOT_FOUND;
        }

        nixlUcxConnection &conn = remoteConnMap[remote_agent];

        hdr.op = DISCONNECT;
        //agent names should never be long enough to need RNDV
        flags |= UCP_AM_SEND_FLAG_EAGER;

        nixl_status_t ret = NIXL_SUCCESS;
        for (size_t i = 0; i < uws.size(); i++) {
            nixlUcxReq req;
            ret = getWorker(i)->sendAm(conn.getEp(i), DISCONNECT,
                                 &hdr, sizeof(struct nixl_ucx_am_hdr),
                                 (void*) localAgent.data(), localAgent.size(),
                                 flags, req);
            //don't care
            if (ret == NIXL_IN_PROG)
                getWorker(i)->reqRelease(req);
        }
    }

    endConn(remote_agent);

    return NIXL_SUCCESS;
}

nixl_status_t nixlUcxEngine::loadRemoteConnInfo (const std::string &remote_agent,
                                                 const std::string &remote_conn_info)
{
    size_t size = remote_conn_info.size();
    nixlUcxConnection conn;
    int ret;
    std::vector<char> addr(size);

    if(remoteConnMap.find(remote_agent) != remoteConnMap.end()) {
        return NIXL_ERR_INVALID_PARAM;
    }

    nixlSerDes::_stringToBytes((void*) addr.data(), remote_conn_info, size);
    bool error = false;
    for (auto &uw: uws) {
        nixlUcxEp ep;
        ret = uw->connect(addr.data(), size, ep);
        if (ret) {
            error = true;
            break;
        }
        conn.eps.push_back(ep);
    }

    if (error) {
        for (size_t i = 0; i < conn.eps.size(); i++)
            getWorker(i)->disconnect_nb(conn.getEp(i));
        return NIXL_ERR_BACKEND;
    }

    conn.remoteAgent = remote_agent;
    conn.connected = false;

    remoteConnMap[remote_agent] = conn;

    return NIXL_SUCCESS;
}

/****************************************
 * Memory management
*****************************************/
nixl_status_t nixlUcxEngine::registerMem (const nixlBlobDesc &mem,
                                          const nixl_mem_t &nixl_mem,
                                          nixlBackendMD* &out)
{
    int ret;
    nixlUcxPrivateMetadata *priv = new nixlUcxPrivateMetadata;
    uint64_t rkey_addr;
    size_t rkey_size;

    if (nixl_mem == VRAM_SEG) {
        bool need_restart;
        if (vramUpdateCtx((void*)mem.addr, mem.devId, need_restart)) {
            return NIXL_ERR_NOT_SUPPORTED;
            //TODO Add to logging
        }
        if (need_restart) {
            progressThreadRestart();
        }
    }

    // TODO: Add nixl_mem check?
    ret = nixlUcxWorker::memReg(uc, (void*) mem.addr, mem.len, priv->mem);
    if (ret) {
        return NIXL_ERR_BACKEND;
    }
    ret = nixlUcxWorker::packRkey(uc, priv->mem, rkey_addr, rkey_size);
    if (ret) {
        return NIXL_ERR_BACKEND;
    }
    priv->rkeyStr = nixlSerDes::_bytesToString((void*) rkey_addr, rkey_size);

    out = (nixlBackendMD*) priv; //typecast?

    free((void*)rkey_addr);
    return NIXL_SUCCESS; // Or errors
}

nixl_status_t nixlUcxEngine::deregisterMem (nixlBackendMD* meta)
{
    nixlUcxPrivateMetadata *priv = (nixlUcxPrivateMetadata*) meta;
    nixlUcxWorker::memDereg(uc, priv->mem);
    delete priv;
    return NIXL_SUCCESS;
}

nixl_status_t nixlUcxEngine::getPublicData (const nixlBackendMD* meta,
                                            std::string &str) const {
    const nixlUcxPrivateMetadata *priv = (nixlUcxPrivateMetadata*) meta;
    str = priv->get();
    return NIXL_SUCCESS;
}


// To be cleaned up
nixl_status_t
nixlUcxEngine::internalMDHelper (const nixl_blob_t &blob,
                                 const std::string &agent,
                                 nixlBackendMD* &output) {
    nixlUcxPublicMetadata *md = new nixlUcxPublicMetadata;
    size_t size = blob.size();

    auto search = remoteConnMap.find(agent);

    if(search == remoteConnMap.end()) {
        //TODO: err: remote connection not found
        return NIXL_ERR_NOT_FOUND;
    }

    nixlUcxConnection &conn = search->second;

    //directly copy underlying conn struct
    md->conn = conn;

    std::vector<char> addr(size);
    nixlSerDes::_stringToBytes(addr.data(), blob, size);

    bool error = false;
    for (size_t worker_id = 0; worker_id < uws.size(); worker_id++) {
        nixlUcxRkey rkey;
        error = nixlUcxWorker::rkeyImport(conn.getEp(worker_id), addr.data(), size, rkey);
        if (error)
            // TODO: error out. Should we indicate which desc failed or unroll everything prior
            break;
        md->rkeys.push_back(rkey);
    }
    if (error) {
        for (auto &rkey: md->rkeys)
            nixlUcxWorker::rkeyDestroy(rkey);
        return NIXL_ERR_BACKEND;
    }

    output = (nixlBackendMD*) md;

    return NIXL_SUCCESS;
}

nixl_status_t
nixlUcxEngine::loadLocalMD (nixlBackendMD* input,
                            nixlBackendMD* &output)
{
    nixlUcxPrivateMetadata* input_md = (nixlUcxPrivateMetadata*) input;
    return internalMDHelper(input_md->rkeyStr, localAgent, output);
}

// To be cleaned up
nixl_status_t nixlUcxEngine::loadRemoteMD (const nixlBlobDesc &input,
                                           const nixl_mem_t &nixl_mem,
                                           const std::string &remote_agent,
                                           nixlBackendMD* &output)
{
    return internalMDHelper(input.metaInfo, remote_agent, output);
}

nixl_status_t nixlUcxEngine::unloadMD (nixlBackendMD* input) {

    nixlUcxPublicMetadata *md = (nixlUcxPublicMetadata*) input; //typecast?

    for (auto &rkey: md->rkeys)
        nixlUcxWorker::rkeyDestroy(rkey);
    delete md;

    return NIXL_SUCCESS;
}

/****************************************
 * Data movement
*****************************************/

static nixl_status_t _retHelper(nixl_status_t ret,  nixlUcxBackendH *hndl, nixlUcxReq &req)
{
    /* if transfer wasn't immediately completed */
    switch(ret) {
        case NIXL_IN_PROG:
            hndl->append((nixlUcxIntReq*)req);
        case NIXL_SUCCESS:
            // Nothing to do
            break;
        default:
            // Error. Release all previously initiated ops and exit:
            hndl->release();
            return NIXL_ERR_BACKEND;
    }
    return NIXL_SUCCESS;
}

nixl_status_t nixlUcxEngine::prepXfer (const nixl_xfer_op_t &operation,
                                       const nixl_meta_dlist_t &local,
                                       const nixl_meta_dlist_t &remote,
                                       const std::string &remote_agent,
                                       nixlBackendReqH* &handle,
                                       const nixl_opt_b_args_t* opt_args)
{
    /* TODO: try to get from a pool first */
    nixlUcxBackendH *intHandle = new nixlUcxBackendH(*this, getWorkerId());

    handle = (nixlBackendReqH*)intHandle;
    return NIXL_SUCCESS;
}

nixl_status_t nixlUcxEngine::postXfer (const nixl_xfer_op_t &operation,
                                       const nixl_meta_dlist_t &local,
                                       const nixl_meta_dlist_t &remote,
                                       const std::string &remote_agent,
                                       nixlBackendReqH* &handle,
                                       const nixl_opt_b_args_t* opt_args)
{
    size_t lcnt = local.descCount();
    size_t rcnt = remote.descCount();
    size_t i;
    nixl_status_t ret;
    nixlUcxBackendH *intHandle = (nixlUcxBackendH *)handle;
    nixlUcxPrivateMetadata *lmd;
    nixlUcxPublicMetadata *rmd;
    nixlUcxReq req;
    size_t workerId = intHandle->getWorkerId();
    auto &uw = getWorker(workerId);


    if (lcnt != rcnt) {
        return NIXL_ERR_INVALID_PARAM;
    }

    for(i = 0; i < lcnt; i++) {
        void *laddr = (void*) local[i].addr;
        size_t lsize = local[i].len;
        void *raddr = (void*) remote[i].addr;
        size_t rsize = remote[i].len;

        lmd = (nixlUcxPrivateMetadata*) local[i].metadataP;
        rmd = (nixlUcxPublicMetadata*) remote[i].metadataP;

        if (lsize != rsize) {
            return NIXL_ERR_INVALID_PARAM;
        }

        // TODO: remote_agent and msg should be cached in nixlUCxReq or another way

        switch (operation) {
        case NIXL_READ:
            ret = uw->read(rmd->conn.getEp(workerId), (uint64_t) raddr, rmd->getRkey(workerId), laddr, lmd->mem, lsize, req);
            break;
        case NIXL_WRITE:
            ret = uw->write(rmd->conn.getEp(workerId), laddr, lmd->mem, (uint64_t) raddr, rmd->getRkey(workerId), lsize, req);
            break;
        default:
            return NIXL_ERR_INVALID_PARAM;
        }

        if (_retHelper(ret, intHandle, req)) {
            return ret;
        }
    }

    rmd = (nixlUcxPublicMetadata*) remote[0].metadataP;
    ret = uw->flushEp(rmd->conn.getEp(workerId), req);
    if (_retHelper(ret, intHandle, req)) {
        return ret;
    }

    if(opt_args && opt_args->hasNotif) {
        ret = notifSendPriv(remote_agent, opt_args->notifMsg, req, workerId);
        if (_retHelper(ret, intHandle, req)) {
            return ret;
        }
    }

    return intHandle->status();
}

nixl_status_t nixlUcxEngine::checkXfer (nixlBackendReqH* handle)
{
    nixlUcxBackendH *intHandle = (nixlUcxBackendH *)handle;

    return intHandle->status();
}

nixl_status_t nixlUcxEngine::releaseReqH(nixlBackendReqH* handle)
{
    nixlUcxBackendH *intHandle = (nixlUcxBackendH *)handle;
    nixl_status_t status = intHandle->release();

    /* TODO: return to a pool instead. */
    delete intHandle;

    return status;
}

int nixlUcxEngine::progress() {
    // TODO: add listen for connection handling if necessary
    int ret = 0;
    for (auto &uw: uws)
        ret += uw->progress();
    return ret;
}

/****************************************
 * Notifications
*****************************************/

//agent will provide cached msg
nixl_status_t nixlUcxEngine::notifSendPriv(const std::string &remote_agent,
                                           const std::string &msg,
                                           nixlUcxReq &req,
                                           size_t worker_id)
{
    nixlSerDes ser_des;
    std::string *ser_msg;
    nixlUcxConnection conn;
    // TODO - temp fix, need to have an mpool
    static struct nixl_ucx_am_hdr hdr;
    uint32_t flags = 0;
    nixl_status_t ret;

    auto search = remoteConnMap.find(remote_agent);

    if(search == remoteConnMap.end()) {
        //TODO: err: remote connection not found
        return NIXL_ERR_NOT_FOUND;
    }

    conn = remoteConnMap[remote_agent];

    hdr.op = NOTIF_STR;
    flags |= UCP_AM_SEND_FLAG_EAGER;

    ser_des.addStr("name", localAgent);
    ser_des.addStr("msg", msg);
    // TODO: replace with mpool for performance
    ser_msg = new std::string(ser_des.exportStr());

    ret = getWorker(worker_id)->sendAm(conn.getEp(worker_id), NOTIF_STR,
                                       &hdr, sizeof(struct nixl_ucx_am_hdr),
                                       (void*) ser_msg->data(), ser_msg->size(),
                                       flags, req);

    if (ret == NIXL_IN_PROG) {
        nixlUcxIntReq* nReq = (nixlUcxIntReq*)req;
        nReq->amBuffer = ser_msg;
    } else {
        delete ser_msg;
    }
    return ret;
}

ucs_status_t
nixlUcxEngine::notifAmCb(void *arg, const void *header,
                         size_t header_length, void *data,
                         size_t length,
                         const ucp_am_recv_param_t *param)
{
    struct nixl_ucx_am_hdr* hdr = (struct nixl_ucx_am_hdr*) header;
    nixlSerDes ser_des;

    std::string ser_str( (char*) data, length);
    nixlUcxEngine* engine = (nixlUcxEngine*) arg;
    std::string remote_name, msg;

    if(hdr->op != NOTIF_STR) {
        //is this the best way to ERR?
        return UCS_ERR_INVALID_PARAM;
    }

    //send_am should be forcing EAGER protocol
    if((param->recv_attr & UCP_AM_RECV_ATTR_FLAG_RNDV) != 0) {
        //is this the best way to ERR?
        return UCS_ERR_INVALID_PARAM;
    }

    ser_des.importStr(ser_str);
    remote_name = ser_des.getStr("name");
    msg = ser_des.getStr("msg");

    if (engine->isProgressThread()) {
        /* Append to the private list to allow batching */
        engine->notifPthrPriv.push_back(std::make_pair(remote_name, msg));
    } else {
        engine->notifMainList.push_back(std::make_pair(remote_name, msg));
    }

    return UCS_OK;
}


void nixlUcxEngine::notifCombineHelper(notif_list_t &src, notif_list_t &tgt)
{
    if (!src.size()) {
        // Nothing to do. Exit
        return;
    }

    move(src.begin(), src.end(), back_inserter(tgt));
    src.erase(src.begin(), src.end());
}

void nixlUcxEngine::notifProgressCombineHelper(notif_list_t &src, notif_list_t &tgt)
{
    notifMtx.lock();

    if (src.size()) {
        move(src.begin(), src.end(), back_inserter(tgt));
        src.erase(src.begin(), src.end());
    }

    notifMtx.unlock();
}

void nixlUcxEngine::notifProgress()
{
    notifProgressCombineHelper(notifPthrPriv, notifPthr);
}

nixl_status_t nixlUcxEngine::getNotifs(notif_list_t &notif_list)
{
    if (notif_list.size()!=0)
        return NIXL_ERR_INVALID_PARAM;

    if(!pthrOn) while(progress());

    notifCombineHelper(notifMainList, notif_list);
    notifProgressCombineHelper(notifPthr, notif_list);

    return NIXL_SUCCESS;
}

nixl_status_t nixlUcxEngine::genNotif(const std::string &remote_agent, const std::string &msg)
{
    nixl_status_t ret;
    nixlUcxReq req;
    size_t worker_id = getWorkerId();

    ret = notifSendPriv(remote_agent, msg, req, worker_id);

    switch(ret) {
    case NIXL_IN_PROG:
        /* do not track the request */
        getWorker(worker_id)->reqRelease(req);
    case NIXL_SUCCESS:
        break;
    default:
        /* error case */
        return ret;
    }
    return NIXL_SUCCESS;
}

const std::unique_ptr<nixlUcxWorker> &nixlUcxEngine::getWorker(size_t worker_id) const {
    return uws[worker_id];
}

size_t nixlUcxEngine::getWorkerId() const {
    return std::hash<std::thread::id>{}(std::this_thread::get_id()) % uws.size();
}
