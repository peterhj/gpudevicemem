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
class SetConstantFlatMapInplace {
public:
  __forceinline__ __device__ static void ConstantFlatMapInplaceIndex(uint32_t idx, T c, T *y) {
    y[idx] = c;
  }
};

template <typename T>
class AddConstantFlatMapInplace {
public:
  __forceinline__ __device__ static void ConstantFlatMapInplaceIndex(uint32_t idx, T c, T *y) {
    y[idx] = y[idx] + c;
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

extern "C" void gpudevicemem_set_constant_flat_map_inplace_f32(
    uint32_t len,
    float c,
    float *y,
    const KernelConfig *cfg,
    cudaStream_t stream)
{
  gpudevicemem_constant_flat_map_inplace_kernel<float, SetConstantFlatMapInplace<float>><<<cfg->flat_grid_dim(len), cfg->flat_block_dim(), 0, stream>>>(
      len, c, y);
}

extern "C" void gpudevicemem_add_constant_flat_map_inplace_f32(
    uint32_t len,
    float c,
    float *y,
    const KernelConfig *cfg,
    cudaStream_t stream)
{
  gpudevicemem_constant_flat_map_inplace_kernel<float, AddConstantFlatMapInplace<float>><<<cfg->flat_grid_dim(len), cfg->flat_block_dim(), 0, stream>>>(
      len, c, y);
}

template <typename T>
class AddConstantFlatMap {
public:
  __forceinline__ __device__ static void ConstantFlatMapIndex(uint32_t idx, T c, const T *x, T *y) {
    y[idx] = x[idx] + c;
  }
};

template <typename T>
class MultConstantFlatMap {
public:
  __forceinline__ __device__ static void ConstantFlatMapIndex(uint32_t idx, T c, const T *x, T *y) {
    y[idx] = x[idx] * c;
  }
};

template <typename T>
class RDivConstantFlatMap {
public:
  __forceinline__ __device__ static void ConstantFlatMapIndex(uint32_t idx, T c, const T *x, T *y) {
    y[idx] = x[idx] / c;
  }
};

template <typename T>
class LDivConstantFlatMap {
public:
  __forceinline__ __device__ static void ConstantFlatMapIndex(uint32_t idx, T c, const T *x, T *y) {
    y[idx] = c / x[idx];
  }
};

template <typename T>
class OnlineAddFlatMapAccumulate {
public:
  __forceinline__ __device__ static void ConstantFlatMapIndex(uint32_t idx, T c, const T *x, T *y) {
    T y_i = y[idx];
    y[idx] = y_i + c * x[idx];
  }
};

template <typename T>
class OnlineDiscountFlatMapAccumulate {
public:
  __forceinline__ __device__ static void ConstantFlatMapIndex(uint32_t idx, T c, const T *x, T *y) {
    T y_i = y[idx];
    y[idx] = c * y_i + x[idx];
  }
};

template <typename T>
class OnlineAverageFlatMapAccumulate {
public:
  __forceinline__ __device__ static void ConstantFlatMapIndex(uint32_t idx, T c, const T *x, T *y) {
    T y_i = y[idx];
    y[idx] = y_i + c * (x[idx] - y_i);
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

extern "C" void gpudevicemem_add_constant_flat_map_f32(
    uint32_t len,
    float c,
    const float *x,
    float *y,
    const KernelConfig *cfg,
    cudaStream_t stream)
{
  gpudevicemem_constant_flat_map_kernel<float, AddConstantFlatMap<float>><<<cfg->flat_grid_dim(len), cfg->flat_block_dim(), 0, stream>>>(
      len, c, x, y);
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

extern "C" void gpudevicemem_rdiv_constant_flat_map_f32(
    uint32_t len,
    float c,
    const float *x,
    float *y,
    const KernelConfig *cfg,
    cudaStream_t stream)
{
  gpudevicemem_constant_flat_map_kernel<float, RDivConstantFlatMap<float>><<<cfg->flat_grid_dim(len), cfg->flat_block_dim(), 0, stream>>>(
      len, c, x, y);
}

extern "C" void gpudevicemem_ldiv_constant_flat_map_f32(
    uint32_t len,
    float c,
    const float *x,
    float *y,
    const KernelConfig *cfg,
    cudaStream_t stream)
{
  gpudevicemem_constant_flat_map_kernel<float, LDivConstantFlatMap<float>><<<cfg->flat_grid_dim(len), cfg->flat_block_dim(), 0, stream>>>(
      len, c, x, y);
}

extern "C" void gpudevicemem_online_add_flat_map_accum_f32(
    uint32_t len,
    float c,
    const float *x,
    float *y,
    const KernelConfig *cfg,
    cudaStream_t stream)
{
  gpudevicemem_constant_flat_map_kernel<float, OnlineAddFlatMapAccumulate<float>><<<cfg->flat_grid_dim(len), cfg->flat_block_dim(), 0, stream>>>(
      len, c, x, y);
}

extern "C" void gpudevicemem_online_discount_flat_map_accum_f32(
    uint32_t len,
    float c,
    const float *x,
    float *y,
    const KernelConfig *cfg,
    cudaStream_t stream)
{
  gpudevicemem_constant_flat_map_kernel<float, OnlineDiscountFlatMapAccumulate<float>><<<cfg->flat_grid_dim(len), cfg->flat_block_dim(), 0, stream>>>(
      len, c, x, y);
}

extern "C" void gpudevicemem_online_average_flat_map_accum_f32(
    uint32_t len,
    float c,
    const float *x,
    float *y,
    const KernelConfig *cfg,
    cudaStream_t stream)
{
  gpudevicemem_constant_flat_map_kernel<float, OnlineAverageFlatMapAccumulate<float>><<<cfg->flat_grid_dim(len), cfg->flat_block_dim(), 0, stream>>>(
      len, c, x, y);
}

template <typename T>
class IsNonzeroFlatMap {
public:
  __forceinline__ __device__ static void FlatMapIndex(uint32_t idx, const T *x, T *y);
};

template <>
class IsNonzeroFlatMap<float> {
public:
  __forceinline__ __device__ static void FlatMapIndex(uint32_t idx, const float *x, float *y) {
    float x_i = x[idx];
    y[idx] = (float)(0.0f != x_i);
  }
};

template <typename T>
class IsZeroFlatMap {
public:
  __forceinline__ __device__ static void FlatMapIndex(uint32_t idx, const T *x, T *y);
};

template <>
class IsZeroFlatMap<float> {
public:
  __forceinline__ __device__ static void FlatMapIndex(uint32_t idx, const float *x, float *y) {
    float x_i = x[idx];
    y[idx] = (float)(0.0f == x_i);
  }
};

template <typename T, typename FlatMap>
__global__ void gpudevicemem_flat_map_kernel(
    uint32_t len,
    const T *x,
    T *y)
{
  for (uint32_t idx = gtindex(); idx < len; idx += gtcount()) {
    FlatMap::FlatMapIndex(idx, x, y);
  }
}

extern "C" void gpudevicemem_is_nonzero_flat_map_f32(
    uint32_t len,
    const float *x,
    float *y,
    const KernelConfig *cfg,
    cudaStream_t stream)
{
  gpudevicemem_flat_map_kernel<float, IsNonzeroFlatMap<float>><<<cfg->flat_grid_dim(len), cfg->flat_block_dim(), 0, stream>>>(
      len, x, y);
}

extern "C" void gpudevicemem_is_zero_flat_map_f32(
    uint32_t len,
    const float *x,
    float *y,
    const KernelConfig *cfg,
    cudaStream_t stream)
{
  gpudevicemem_flat_map_kernel<float, IsZeroFlatMap<float>><<<cfg->flat_grid_dim(len), cfg->flat_block_dim(), 0, stream>>>(
      len, x, y);
}
