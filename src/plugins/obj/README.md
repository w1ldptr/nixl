<!--
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# NIXL Object Storage Plugin

This backend provides AWS S3 object storage using aws-sdk-cpp version 1.11.

## Dependencies
This backend requires aws-sdk-cpp version 1.11 to be installed. Example CLI to compile from sources:

```bash
# Ubuntu/Debian
apt-get install -y libcurl4-openssl-dev libssl-dev uuid-dev zlib1g-dev
git clone --recurse-submodules https://github.com/aws/aws-sdk-cpp.git --branch 1.11.581 && mkdir sdk_build && cd sdk_build && cmake ../aws-sdk-cpp/ -DCMAKE_BUILD_TYPE=Release -DBUILD_ONLY="s3" -DENABLE_TESTING=OFF -DCMAKE_INSTALL_PREFIX=/usr/local && make -j && make install
```
