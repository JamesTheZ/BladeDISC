#!/bin/usr/env bash
# Copyright 2022 The BladeDISC Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.



CUTLASS_ROOT=$1
PREPROCESS_FILE=$2
OUTPUT_FILE1=$3
OUTPUT_FILE4=$4

CUDA_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv -i 0 | sed -n '2p' | sed -e 's/\([0-9]\)\.\([0-9]\)/\1\20/g')
CUDA_MAJOR=$(nvcc --version | grep -o 'V[0-9]*\.[0-9]*\.[0-9]*' | sed -e 's/V\([0-9]*\)\.\([0-9]*\)\.\([0-9]*\)/\1/g')
CUDA_MINOR=$(nvcc --version | grep -o 'V[0-9]*\.[0-9]*\.[0-9]*' | sed -e 's/V\([0-9]*\)\.\([0-9]*\)\.\([0-9]*\)/\2/g')
CUDA_BUILD=$(nvcc --version | grep -o 'V[0-9]*\.[0-9]*\.[0-9]*' | sed -e 's/V\([0-9]*\)\.\([0-9]*\)\.\([0-9]*\)/\3/g')
CUDA_HOME=$(dirname $(which nvcc))/..

gcc -std=c++11 -D__CUDA_ARCH__=${CUDA_ARCH} -E -x c++  \
    -DCUDA_DOUBLE_MATH_FUNCTIONS -D__CUDACC__ -D__NVCC__ -fPIC -O3 \
    -I"${CUTLASS_ROOT}/include" -I"${CUTLASS_ROOT}/tools/util/include" "-I${CUDA_HOME}/targets/x86_64-linux/include" \
    -D__CUDACC_VER_MAJOR__=${CUDA_MAJOR} -D__CUDACC_VER_MINOR__=${CUDA_MINOR} -D__CUDACC_VER_BUILD__=${CUDA_BUILD} \
    -D__CUDA_API_VER_MAJOR__=${CUDA_MAJOR} -D__CUDA_API_VER_MINOR__=${CUDA_MINOR} -include "cuda_runtime.h" -m64 "${PREPROCESS_FILE}" \
    -o "${OUTPUT_FILE1}"

gcc -std=c++11 -E -x c++ \
    -D__CUDACC__ -D__NVCC__ -fPIC -O3 \
    -I"${CUTLASS_ROOT}/include" -I"${CUTLASS_ROOT}/tools/util/include" "-I${CUDA_HOME}/targets/x86_64-linux/include" \
    -D__CUDACC_VER_MAJOR__=${CUDA_MAJOR} -D__CUDACC_VER_MINOR__=${CUDA_MINOR} -D__CUDACC_VER_BUILD__=${CUDA_BUILD} \
    -D__CUDA_API_VER_MAJOR__=${CUDA_MAJOR} -D__CUDA_API_VER_MINOR__=${CUDA_MINOR} -include "cuda_runtime.h" -m64 "${PREPROCESS_FILE}" \
    -o "${OUTPUT_FILE4}"