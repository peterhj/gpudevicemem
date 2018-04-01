/*
Copyright 2017 the gpudevicemem authors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#include "lib.h"
#include "common.cuh"
#include <cuda_runtime.h>
#include <math_constants.h>

template <typename T>
class SetConstantFlatMap {
public:
  __forceinline__ __device__ static void ConstantFlatMapInplaceIndex(uint32_t idx, T c, T *y) {
    y[idx] = c;
  }
};

template <typename T, typename FlatMap>
__global__ void gpudevicemem_constant_flat_map_inplace_kernel(
    uint32_t len,
    T c,
    T *y)
{
  for (uint32_t idx = gtindex(); idx < len; idx += gtcount()) {
    FlatMap::ConstantFlatMapInplaceIndex(idx, c, y);
  }
}

extern "C" void gpudevicemem_set_constant_flat_map_f32(
    uint32_t len,
    float c,
    float *y,
    const KernelConfig *cfg,
    cudaStream_t stream)
{
  gpudevicemem_constant_flat_map_inplace_kernel<float, SetConstantFlatMap<float>><<<cfg->flat_grid_dim(len), cfg->flat_block_dim(), 0, stream>>>(
      len, c, y);
}

template <typename T>
class MultConstantFlatMap {
public:
  __forceinline__ __device__ static void ConstantFlatMapIndex(uint32_t idx, T c, const T *x, T *y) {
    y[idx] = c * x[idx];
  }
};

template <typename T, typename FlatMap>
__global__ void gpudevicemem_constant_flat_map_kernel(
    uint32_t len,
    T c,
    const T *x,
    T *y)
{
  for (uint32_t idx = gtindex(); idx < len; idx += gtcount()) {
    FlatMap::ConstantFlatMapIndex(idx, c, x, y);
  }
}

extern "C" void gpudevicemem_mult_constant_flat_map_f32(
    uint32_t len,
    float c,
    const float *x,
    float *y,
    const KernelConfig *cfg,
    cudaStream_t stream)
{
  gpudevicemem_constant_flat_map_kernel<float, MultConstantFlatMap<float>><<<cfg->flat_grid_dim(len), cfg->flat_block_dim(), 0, stream>>>(
      len, c, x, y);
}

/*template <typename T, typename FlatMap>
__global__ void gpudevicemem_generic_flat_map_kernel(
    uint32_t len,
    const T *x,
    T *y)
{
  for (uint32_t idx = gtindex(); idx < len; idx += gtcount()) {
    FlatMap::FlatMapIndex(idx, x, y);
  }
}*/