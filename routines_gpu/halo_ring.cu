/*
Copyright 2017-2018 Peter Jin

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

struct InteriorRegion {
  __forceinline__ __device__ static uint32_t CalculateArrayIndex(uint32_t ridx, uint32_t halo_radius, uint32_t arr_dim) {
    return ridx + halo_radius;
  }
};

struct LEdgeRegion {
  __forceinline__ __device__ static uint32_t CalculateArrayIndex(uint32_t ridx, uint32_t halo_radius, uint32_t arr_dim) {
    return ridx + halo_radius;
  }
};

struct LGhostRegion {
  __forceinline__ __device__ static uint32_t CalculateArrayIndex(uint32_t ridx, uint32_t halo_radius, uint32_t arr_dim) {
    return ridx;
  }
};

struct REdgeRegion {
  __forceinline__ __device__ static uint32_t CalculateArrayIndex(uint32_t ridx, uint32_t halo_radius, uint32_t arr_dim) {
    return ridx + arr_dim - halo_radius * 2;
  }
};

struct RGhostRegion {
  __forceinline__ __device__ static uint32_t CalculateArrayIndex(uint32_t ridx, uint32_t halo_radius, uint32_t arr_dim) {
    return ridx + arr_dim - halo_radius;
  }
};

struct IgnoreBufDir {};
struct ToBufDir {};
struct FromBufDir {};

template <typename T, typename Dir>
struct ZeroOp {
  __forceinline__ __device__ static void Apply(T *arr_elem, T *region_buf_elem);
};

template <>
struct ZeroOp<float, IgnoreBufDir> {
  __forceinline__ __device__ static void Apply(float *arr_elem, float *_region_buf_elem) {
    *arr_elem = 0.0f;
  }
};

template <typename T, typename Dir>
struct CopyOp {
  __forceinline__ __device__ static void Apply(T *arr_elem, T *region_buf_elem);
};

template <typename T>
struct CopyOp<T, ToBufDir> {
  __forceinline__ __device__ static void Apply(T *arr_elem, T *region_buf_elem) {
    *region_buf_elem = *arr_elem;
  }
};

template <typename T>
struct CopyOp<T, FromBufDir> {
  __forceinline__ __device__ static void Apply(T *arr_elem, T *region_buf_elem) {
    *arr_elem = *region_buf_elem;
  }
};

template <typename T, typename Dir>
struct AccumulateOp {
  __forceinline__ __device__ static void Apply(T *arr_elem, T *region_buf_elem);
};

template <typename T>
struct AccumulateOp<T, FromBufDir> {
  __forceinline__ __device__ static void Apply(T *arr_elem, T *region_buf_elem) {
    *arr_elem = *arr_elem + *region_buf_elem;
  }
};

template <typename Region, typename T, typename Op>
__global__ void gpudevicemem_halo_ring_3d1_generic_packed_kernel(
    uint32_t region_len,
    uint32_t region_dim1,
    uint32_t halo_radius,
    uint32_t dim0,
    uint32_t arr_dim1,
    uint32_t dim2,
    T *arr,
    T *region_buf)
{
  for (uint32_t idx = gtindex(); idx < region_len; idx += gtcount()) {
    uint32_t i0, r1, i2;
    Index3::Unpack(idx,
        &i0, dim0,
        &r1, region_dim1,
        &i2);
    uint32_t arr_i1 = Region::CalculateArrayIndex(r1, halo_radius, arr_dim1);
    uint32_t arr_idx = Index3::Pack(
        i0, dim0,
        arr_i1, arr_dim1,
        i2);
    Op::Apply(arr + arr_idx, region_buf + idx);
  }
}

extern "C" void gpudevicemem_halo_ring_3d1_fill_f32(
    uint32_t halo_radius,
    uint32_t dim0,
    uint32_t dim1,
    uint32_t dim2,
    // XXX: NOTE: The array param order is flipped here vs the kernel order.
    float *src_arr,
    float *dst_arr,
    const KernelConfig *cfg,
    cudaStream_t stream)
{
  uint32_t region_dim1 = dim1 - halo_radius * 2;
  uint32_t region_len = dim0 * region_dim1 * dim2;
  gpudevicemem_halo_ring_3d1_generic_packed_kernel<
      InteriorRegion,
      float,
      CopyOp<float, FromBufDir>
  ><<<cfg->flat_grid_dim(region_len), cfg->flat_block_dim(), 0, stream>>>(
      region_len,
      region_dim1,
      halo_radius,
      dim0,
      dim1,
      dim2,
      dst_arr,
      src_arr);
}

extern "C" void gpudevicemem_halo_ring_3d1_unfill_f32(
    uint32_t halo_radius,
    uint32_t dim0,
    uint32_t dim1,
    uint32_t dim2,
    float *src_arr,
    float *dst_arr,
    const KernelConfig *cfg,
    cudaStream_t stream)
{
  uint32_t region_dim1 = dim1 - halo_radius * 2;
  uint32_t region_len = dim0 * region_dim1 * dim2;
  gpudevicemem_halo_ring_3d1_generic_packed_kernel<
      InteriorRegion,
      float,
      CopyOp<float, ToBufDir>
  ><<<cfg->flat_grid_dim(region_len), cfg->flat_block_dim(), 0, stream>>>(
      region_len,
      region_dim1,
      halo_radius,
      dim0,
      dim1,
      dim2,
      src_arr,
      dst_arr);
}

extern "C" void gpudevicemem_halo_ring_3d1_zero_lghost_f32(
    uint32_t halo_radius,
    uint32_t dim0,
    uint32_t dim1,
    uint32_t dim2,
    float *arr,
    const KernelConfig *cfg,
    cudaStream_t stream)
{
  uint32_t region_len = dim0 * halo_radius * dim2;
  gpudevicemem_halo_ring_3d1_generic_packed_kernel<
      LGhostRegion,
      float,
      ZeroOp<float, IgnoreBufDir>
  ><<<cfg->flat_grid_dim(region_len), cfg->flat_block_dim(), 0, stream>>>(
      region_len,
      halo_radius,
      halo_radius,
      dim0,
      dim1,
      dim2,
      arr,
      NULL);
}

extern "C" void gpudevicemem_halo_ring_3d1_zero_rghost_f32(
    uint32_t halo_radius,
    uint32_t dim0,
    uint32_t dim1,
    uint32_t dim2,
    float *arr,
    const KernelConfig *cfg,
    cudaStream_t stream)
{
  uint32_t region_len = dim0 * halo_radius * dim2;
  gpudevicemem_halo_ring_3d1_generic_packed_kernel<
      RGhostRegion,
      float,
      ZeroOp<float, IgnoreBufDir>
  ><<<cfg->flat_grid_dim(region_len), cfg->flat_block_dim(), 0, stream>>>(
      region_len,
      halo_radius,
      halo_radius,
      dim0,
      dim1,
      dim2,
      arr,
      NULL);
}

extern "C" void gpudevicemem_halo_ring_3d1_copy_ledge_to_buf_f32(
    uint32_t halo_radius,
    uint32_t dim0,
    uint32_t dim1,
    uint32_t dim2,
    float *arr,
    float *region_buf,
    const KernelConfig *cfg,
    cudaStream_t stream)
{
  uint32_t region_len = dim0 * halo_radius * dim2;
  gpudevicemem_halo_ring_3d1_generic_packed_kernel<
      LEdgeRegion,
      float,
      CopyOp<float, ToBufDir>
  ><<<cfg->flat_grid_dim(region_len), cfg->flat_block_dim(), 0, stream>>>(
      region_len,
      halo_radius,
      halo_radius,
      dim0,
      dim1,
      dim2,
      arr,
      region_buf);
}

extern "C" void gpudevicemem_halo_ring_3d1_copy_redge_to_buf_f32(
    uint32_t halo_radius,
    uint32_t dim0,
    uint32_t dim1,
    uint32_t dim2,
    float *arr,
    float *region_buf,
    const KernelConfig *cfg,
    cudaStream_t stream)
{
  uint32_t region_len = dim0 * halo_radius * dim2;
  gpudevicemem_halo_ring_3d1_generic_packed_kernel<
      REdgeRegion,
      float,
      CopyOp<float, ToBufDir>
  ><<<cfg->flat_grid_dim(region_len), cfg->flat_block_dim(), 0, stream>>>(
      region_len,
      halo_radius,
      halo_radius,
      dim0,
      dim1,
      dim2,
      arr,
      region_buf);
}

extern "C" void gpudevicemem_halo_ring_3d1_copy_buf_to_lghost_f32(
    uint32_t halo_radius,
    uint32_t dim0,
    uint32_t dim1,
    uint32_t dim2,
    float *arr,
    float *region_buf,
    const KernelConfig *cfg,
    cudaStream_t stream)
{
  uint32_t region_len = dim0 * halo_radius * dim2;
  gpudevicemem_halo_ring_3d1_generic_packed_kernel<
      LGhostRegion,
      float,
      CopyOp<float, FromBufDir>
  ><<<cfg->flat_grid_dim(region_len), cfg->flat_block_dim(), 0, stream>>>(
      region_len,
      halo_radius,
      halo_radius,
      dim0,
      dim1,
      dim2,
      arr,
      region_buf);
}

extern "C" void gpudevicemem_halo_ring_3d1_copy_buf_to_rghost_f32(
    uint32_t halo_radius,
    uint32_t dim0,
    uint32_t dim1,
    uint32_t dim2,
    float *arr,
    float *region_buf,
    const KernelConfig *cfg,
    cudaStream_t stream)
{
  uint32_t region_len = dim0 * halo_radius * dim2;
  gpudevicemem_halo_ring_3d1_generic_packed_kernel<
      RGhostRegion,
      float,
      CopyOp<float, FromBufDir>
  ><<<cfg->flat_grid_dim(region_len), cfg->flat_block_dim(), 0, stream>>>(
      region_len,
      halo_radius,
      halo_radius,
      dim0,
      dim1,
      dim2,
      arr,
      region_buf);
}

extern "C" void gpudevicemem_halo_ring_3d1_copy_lghost_to_buf_f32(
    uint32_t halo_radius,
    uint32_t dim0,
    uint32_t dim1,
    uint32_t dim2,
    float *arr,
    float *region_buf,
    const KernelConfig *cfg,
    cudaStream_t stream)
{
  uint32_t region_len = dim0 * halo_radius * dim2;
  gpudevicemem_halo_ring_3d1_generic_packed_kernel<
      LGhostRegion,
      float,
      CopyOp<float, ToBufDir>
  ><<<cfg->flat_grid_dim(region_len), cfg->flat_block_dim(), 0, stream>>>(
      region_len,
      halo_radius,
      halo_radius,
      dim0,
      dim1,
      dim2,
      arr,
      region_buf);
}

extern "C" void gpudevicemem_halo_ring_3d1_copy_rghost_to_buf_f32(
    uint32_t halo_radius,
    uint32_t dim0,
    uint32_t dim1,
    uint32_t dim2,
    float *arr,
    float *region_buf,
    const KernelConfig *cfg,
    cudaStream_t stream)
{
  uint32_t region_len = dim0 * halo_radius * dim2;
  gpudevicemem_halo_ring_3d1_generic_packed_kernel<
      RGhostRegion,
      float,
      CopyOp<float, ToBufDir>
  ><<<cfg->flat_grid_dim(region_len), cfg->flat_block_dim(), 0, stream>>>(
      region_len,
      halo_radius,
      halo_radius,
      dim0,
      dim1,
      dim2,
      arr,
      region_buf);
}

extern "C" void gpudevicemem_halo_ring_3d1_accumulate_buf_to_ledge_f32(
    uint32_t halo_radius,
    uint32_t dim0,
    uint32_t dim1,
    uint32_t dim2,
    float *arr,
    float *region_buf,
    const KernelConfig *cfg,
    cudaStream_t stream)
{
  uint32_t region_len = dim0 * halo_radius * dim2;
  gpudevicemem_halo_ring_3d1_generic_packed_kernel<
      LEdgeRegion,
      float,
      AccumulateOp<float, FromBufDir>
  ><<<cfg->flat_grid_dim(region_len), cfg->flat_block_dim(), 0, stream>>>(
      region_len,
      halo_radius,
      halo_radius,
      dim0,
      dim1,
      dim2,
      arr,
      region_buf);
}

extern "C" void gpudevicemem_halo_ring_3d1_accumulate_buf_to_redge_f32(
    uint32_t halo_radius,
    uint32_t dim0,
    uint32_t dim1,
    uint32_t dim2,
    float *arr,
    float *region_buf,
    const KernelConfig *cfg,
    cudaStream_t stream)
{
  uint32_t region_len = dim0 * halo_radius * dim2;
  gpudevicemem_halo_ring_3d1_generic_packed_kernel<
      REdgeRegion,
      float,
      AccumulateOp<float, FromBufDir>
  ><<<cfg->flat_grid_dim(region_len), cfg->flat_block_dim(), 0, stream>>>(
      region_len,
      halo_radius,
      halo_radius,
      dim0,
      dim1,
      dim2,
      arr,
      region_buf);
}
