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

#ifndef MEM_BUFFER_H
#define MEM_BUFFER_H

#include "nixl_types.h"
#include <cstdint>

namespace gtest {

template<nixl_mem_t MemType> class MemBuffer;

template<> class MemBuffer<DRAM_SEG> {
public:
    MemBuffer(size_t size) : buffer_(size) {}

    MemBuffer(std::vector<uint8_t> &&data) : buffer_(std::move(data)) {}

    bool
    operator==(const MemBuffer<DRAM_SEG> &other) const {
        return buffer_ == other.buffer_;
    }

    uintptr_t
    data() const {
        return reinterpret_cast<uintptr_t>(buffer_.data());
    }

    size_t
    size() const {
        return buffer_.size();
    }

    void
    zero() {
        std::fill(buffer_.begin(), buffer_.end(), 0);
    }

private:
    std::vector<uint8_t> buffer_;
};

} // namespace gtest

#ifdef HAVE_CUDA

#include <cuda_runtime.h>

namespace gtest {

template<> class MemBuffer<VRAM_SEG> {
public:
    MemBuffer(size_t size) : size_(size) {
        cudaError_t err = cudaMalloc(&buffer_, size);
        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to allocate CUDA memory");
        }
    }

    MemBuffer(std::vector<uint8_t> &&data) : MemBuffer(data.size()) {
        // TODO
    }

    ~MemBuffer() {
        if (buffer_) {
            cudaFree(buffer_);
        }
    }

    MemBuffer(const MemBuffer &) = delete;
    MemBuffer &
    operator=(const MemBuffer &) = delete;

    MemBuffer(MemBuffer &&other) noexcept : buffer_(other.buffer_), size_(other.size_) {
        other.buffer_ = nullptr;
        other.size_ = 0;
    }

    MemBuffer &
    operator=(MemBuffer &&other) noexcept {
        if (this != &other) {
            if (buffer_) {
                cudaFree(buffer_);
            }
            buffer_ = other.buffer_;
            size_ = other.size_;
            other.buffer_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    bool
    operator==(const MemBuffer<VRAM_SEG> &other) const {
        // TODO
        return true;
    }

    uintptr_t
    data() const {
        return reinterpret_cast<uintptr_t>(buffer_);
    }

    size_t
    size() const {
        return size_;
    }

    void
    zero() {
        // TODO
    }

private:
    void *buffer_ = nullptr;
    size_t size_;
};

} // namespace gtest

#endif // HAVE_CUDA


#endif /* MEM_BUFFER_H */
