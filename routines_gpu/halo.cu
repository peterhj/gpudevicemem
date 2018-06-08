/*
Copyright 2017-2018 the gpudevicemem authors

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

template <typename T>
__global__ void gpudevicemem_halo_expand_packed3d1_kernel(
    uint32_t len,
    uint32_t ax0_size,
    uint32_t ax1_size,
    uint32_t ax2_size,
    uint32_t halo_size,
    const T *x,
    T *y)
{
  for (uint32_t idx = gtindex(); idx < len; idx += gtcount()) {
    uint32_t i0, i1, i2;
    Index3::Unpack(
        idx,
        &i0, ax0_size,
        &i1, ax1_size + halo_size * 2,
        &i2
    );
    if (i0 < ax0_size && i1 < (ax1_size + halo_size * 2) && i2 < ax2_size) {
      T x_i;
      if (i1 < halo_size) {
        x_i = static_cast<T>(0);
      } else if (i1 >= ax1_size + halo_size) {
        x_i = static_cast<T>(0);
      } else {
        uint32_t src_idx = Index3::Pack(
            i0, ax0_size,
            i1 - halo_size, ax1_size,
            i2
        );
        x_i = x[src_idx];
      }
      y[idx] = x_i;
    }
  }
}

template <typename T>
__global__ void gpudevicemem_halo_project_packed3d1_kernel(
    uint32_t len,
    uint32_t ax0_size,
    uint32_t ax1_size,
    uint32_t ax2_size,
    uint32_t halo_size,
    const T *x,
    T *y)
{
  for (uint32_t idx = gtindex(); idx < len; idx += gtcount()) {
    uint32_t i0, i1, i2;
    Index3::Unpack(
        idx,
        &i0, ax0_size,
        &i1, ax1_size,
        &i2
    );
    if (i0 < ax0_size && i1 < ax1_size && i2 < ax2_size) {
      uint32_t src_idx = Index3::Pack(
          i0, ax0_size,
          i1 + halo_size, ax1_size + halo_size * 2,
          i2
      );
      y[idx] = x[src_idx];
    }
  }
}

template <typename T>
__global__ void gpudevicemem_halo_set_constant_3d1_kernel(
    uint32_t len,
    uint32_t ax0_size,
    uint32_t ax0_offset,
    uint32_t ax0_stride,
    uint32_t ax1_size,
    uint32_t ax1_offset,
    uint32_t ax1_stride,
    uint32_t ax2_size,
    uint32_t ax2_offset,
    uint32_t ax2_stride,
    uint32_t halo_size,
    T c,
    T *y)
{
  for (uint32_t idx = gtindex(); idx < len; idx += gtcount()) {
    uint32_t i0, i1, i2;
    Index3::Unpack(
        idx,
        &i0, ax0_size,
        &i1, halo_size * 2,
        &i2
    );
    if (i0 < ax0_size && i1 < (halo_size * 2) && i2 < ax2_size) {
      uint32_t dst_i1;
      if (i1 < halo_size) {
        dst_i1 = i1;
      } else {
        dst_i1 = i1 + ax1_size - (halo_size * 2);
      }
      uint32_t dst_idx = Index3::Pack(
          ax0_offset + i0, ax0_stride,
          ax1_offset + dst_i1, ax1_stride,
          ax2_offset + i2
      );
      y[dst_idx] = c;
    }
  }
}

template <typename T>
__global__ void gpudevicemem_halo_pack_3d1_kernel(
    uint32_t len,
    uint32_t ax0_size,
    uint32_t ax0_offset,
    uint32_t ax0_stride,
    uint32_t ax1_size,
    uint32_t ax1_offset,
    uint32_t ax1_stride,
    uint32_t ax2_size,
    uint32_t ax2_offset,
    uint32_t ax2_stride,
    uint32_t halo_size,
    const T *x,
    T *y)
{
  for (uint32_t idx = gtindex(); idx < len; idx += gtcount()) {
    uint32_t i0, i1, i2;
    Index3::Unpack(
        idx,
        &i0, ax0_size,
        &i1, halo_size * 2,
        &i2
    );
    if (i0 < ax0_size && i1 < (halo_size * 2) && i2 < ax2_size) {
      uint32_t src_i1;
      if (i1 < halo_size) {
        src_i1 = i1;
      } else {
        src_i1 = i1 + ax1_size - (halo_size * 2);
      }
      uint32_t src_idx = Index3::Pack(
          ax0_offset + i0, ax0_stride,
          ax1_offset + src_i1, ax1_stride,
          ax2_offset + i2
      );
      y[idx] = x[src_idx];
    }
  }
}

template <typename T>
__global__ void gpudevicemem_halo_pack_lo_3d1_kernel(
    uint32_t len,
    uint32_t ax0_size,
    uint32_t ax0_offset,
    uint32_t ax0_stride,
    uint32_t ax1_size,
    uint32_t ax1_offset,
    uint32_t ax1_stride,
    uint32_t ax2_size,
    uint32_t ax2_offset,
    uint32_t ax2_stride,
    uint32_t halo_size,
    const T *x,
    T *y)
{
  for (uint32_t idx = gtindex(); idx < len; idx += gtcount()) {
    uint32_t i0, i1, i2;
    Index3::Unpack(
        idx,
        &i0, ax0_size,
        &i1, halo_size,
        &i2
    );
    if (i0 < ax0_size && i1 < halo_size && i2 < ax2_size) {
      uint32_t src_i1 = i1;
      uint32_t src_idx = Index3::Pack(
          ax0_offset + i0, ax0_stride,
          ax1_offset + src_i1, ax1_stride,
          ax2_offset + i2
      );
      y[idx] = x[src_idx];
    }
  }
}

template <typename T>
__global__ void gpudevicemem_halo_pack_hi_3d1_kernel(
    uint32_t len,
    uint32_t ax0_size,
    uint32_t ax0_offset,
    uint32_t ax0_stride,
    uint32_t ax1_size,
    uint32_t ax1_offset,
    uint32_t ax1_stride,
    uint32_t ax2_size,
    uint32_t ax2_offset,
    uint32_t ax2_stride,
    uint32_t halo_size,
    const T *x,
    T *y)
{
  for (uint32_t idx = gtindex(); idx < len; idx += gtcount()) {
    uint32_t i0, i1, i2;
    Index3::Unpack(
        idx,
        &i0, ax0_size,
        &i1, halo_size,
        &i2
    );
    if (i0 < ax0_size && i1 < halo_size && i2 < ax2_size) {
      uint32_t src_i1 = i1 + ax1_size - halo_size;
      uint32_t src_idx = Index3::Pack(
          ax0_offset + i0, ax0_stride,
          ax1_offset + src_i1, ax1_stride,
          ax2_offset + i2
      );
      y[idx] = x[src_idx];
    }
  }
}

template <typename T>
__global__ void gpudevicemem_halo_unpack_3d1_kernel(
    uint32_t len,
    uint32_t ax0_size,
    uint32_t ax0_offset,
    uint32_t ax0_stride,
    uint32_t ax1_size,
    uint32_t ax1_offset,
    uint32_t ax1_stride,
    uint32_t ax2_size,
    uint32_t ax2_offset,
    uint32_t ax2_stride,
    uint32_t halo_size,
    const T *x,
    T *y)
{
  for (uint32_t idx = gtindex(); idx < len; idx += gtcount()) {
    uint32_t i0, i1, i2;
    Index3::Unpack(
        idx,
        &i0, ax0_size,
        &i1, halo_size * 2,
        &i2
    );
    if (i0 < ax0_size && i1 < (halo_size * 2) && i2 < ax2_size) {
      int32_t dst_i1;
      if (i1 < halo_size) {
        dst_i1 = static_cast<int32_t>(i1) - halo_size;
      } else {
        dst_i1 = static_cast<int32_t>(i1) + ax1_size - halo_size;
      }
      uint32_t dst_idx = Index3::Pack(
          ax0_offset + i0, ax0_stride,
          ax1_offset + dst_i1, ax1_stride,
          ax2_offset + i2
      );
      y[dst_idx] = x[idx];
    }
  }
}

template <typename T>
__global__ void gpudevicemem_halo_unpack_lo_3d1_kernel(
    uint32_t len,
    uint32_t ax0_size,
    uint32_t ax0_offset,
    uint32_t ax0_stride,
    uint32_t ax1_size,
    uint32_t ax1_offset,
    uint32_t ax1_stride,
    uint32_t ax2_size,
    uint32_t ax2_offset,
    uint32_t ax2_stride,
    uint32_t halo_size,
    const T *x,
    T *y)
{
  for (uint32_t idx = gtindex(); idx < len; idx += gtcount()) {
    uint32_t i0, i1, i2;
    Index3::Unpack(
        idx,
        &i0, ax0_size,
        &i1, halo_size,
        &i2
    );
    if (i0 < ax0_size && i1 < halo_size && i2 < ax2_size) {
      int32_t dst_i1 = static_cast<int32_t>(i1) - halo_size;
      uint32_t dst_idx = Index3::Pack(
          ax0_offset + i0, ax0_stride,
          ax1_offset + dst_i1, ax1_stride,
          ax2_offset + i2
      );
      y[dst_idx] = x[idx];
    }
  }
}

template <typename T>
__global__ void gpudevicemem_halo_unpack_hi_3d1_kernel(
    uint32_t len,
    uint32_t ax0_size,
    uint32_t ax0_offset,
    uint32_t ax0_stride,
    uint32_t ax1_size,
    uint32_t ax1_offset,
    uint32_t ax1_stride,
    uint32_t ax2_size,
    uint32_t ax2_offset,
    uint32_t ax2_stride,
    uint32_t halo_size,
    const T *x,
    T *y)
{
  for (uint32_t idx = gtindex(); idx < len; idx += gtcount()) {
    uint32_t i0, i1, i2;
    Index3::Unpack(
        idx,
        &i0, ax0_size,
        &i1, halo_size,
        &i2
    );
    if (i0 < ax0_size && i1 < halo_size && i2 < ax2_size) {
      uint32_t dst_i1 = i1 + ax1_size - halo_size;
      uint32_t dst_idx = Index3::Pack(
          ax0_offset + i0, ax0_stride,
          ax1_offset + dst_i1, ax1_stride,
          ax2_offset + i2
      );
      y[dst_idx] = x[idx];
    }
  }
}

extern "C" void gpudevicemem_halo_expand_packed3d1_f32(
    uint32_t ax0_size,
    uint32_t ax1_size,
    uint32_t ax2_size,
    uint32_t halo_size,
    const float *x,
    float *y,
    const struct KernelConfig *cfg,
    struct CUstream_st *stream)
{
  // TODO
  uint32_t len = ax0_size * (ax1_size + halo_size * 2) * ax2_size;
  gpudevicemem_halo_expand_packed3d1_kernel<float><<<cfg->flat_grid_dim(len), cfg->flat_block_dim(), 0, stream>>>(
      len,
      ax0_size,
      ax1_size,
      ax2_size,
      halo_size,
      x,
      y);
}

extern "C" void gpudevicemem_halo_project_packed3d1_f32(
    uint32_t ax0_size,
    uint32_t ax1_size,
    uint32_t ax2_size,
    uint32_t halo_size,
    const float *x,
    float *y,
    const struct KernelConfig *cfg,
    struct CUstream_st *stream)
{
  // TODO
  uint32_t len = ax0_size * ax1_size * ax2_size;
  gpudevicemem_halo_project_packed3d1_kernel<float><<<cfg->flat_grid_dim(len), cfg->flat_block_dim(), 0, stream>>>(
      len,
      ax0_size,
      ax1_size,
      ax2_size,
      halo_size,
      x,
      y);
}

extern "C" void gpudevicemem_halo_set_constant_3d1_f32(
    uint32_t ax0_size,
    uint32_t ax0_offset,
    uint32_t ax0_stride,
    uint32_t ax1_size,
    uint32_t ax1_offset,
    uint32_t ax1_stride,
    uint32_t ax2_size,
    uint32_t ax2_offset,
    uint32_t ax2_stride,
    uint32_t halo_size,
    float c,
    float *y,
    const struct KernelConfig *cfg,
    struct CUstream_st *stream)
{
  // TODO
  uint32_t len = ax0_size * (halo_size * 2) * ax2_size;
  gpudevicemem_halo_set_constant_3d1_kernel<float><<<cfg->flat_grid_dim(len), cfg->flat_block_dim(), 0, stream>>>(
      len,
      ax0_size, ax0_offset, ax0_stride,
      ax1_size, ax1_offset, ax1_stride,
      ax2_size, ax2_offset, ax2_stride,
      halo_size,
      c,
      y);
}

extern "C" void gpudevicemem_halo_pack_3d1_f32(
    uint32_t ax0_size,
    uint32_t ax0_offset,
    uint32_t ax0_stride,
    uint32_t ax1_size,
    uint32_t ax1_offset,
    uint32_t ax1_stride,
    uint32_t ax2_size,
    uint32_t ax2_offset,
    uint32_t ax2_stride,
    uint32_t halo_size,
    const float *x,
    float *y,
    const struct KernelConfig *cfg,
    struct CUstream_st *stream)
{
  // TODO
  uint32_t len = ax0_size * (halo_size * 2) * ax2_size;
  gpudevicemem_halo_pack_3d1_kernel<float><<<cfg->flat_grid_dim(len), cfg->flat_block_dim(), 0, stream>>>(
      len,
      ax0_size, ax0_offset, ax0_stride,
      ax1_size, ax1_offset, ax1_stride,
      ax2_size, ax2_offset, ax2_stride,
      halo_size,
      x,
      y);
}

extern "C" void gpudevicemem_halo_pack_lo_3d1_f32(
    uint32_t ax0_size,
    uint32_t ax0_offset,
    uint32_t ax0_stride,
    uint32_t ax1_size,
    uint32_t ax1_offset,
    uint32_t ax1_stride,
    uint32_t ax2_size,
    uint32_t ax2_offset,
    uint32_t ax2_stride,
    uint32_t halo_size,
    const float *x,
    float *y,
    const struct KernelConfig *cfg,
    struct CUstream_st *stream)
{
  // TODO
  uint32_t len = ax0_size * halo_size * ax2_size;
  gpudevicemem_halo_pack_lo_3d1_kernel<float><<<cfg->flat_grid_dim(len), cfg->flat_block_dim(), 0, stream>>>(
      len,
      ax0_size, ax0_offset, ax0_stride,
      ax1_size, ax1_offset, ax1_stride,
      ax2_size, ax2_offset, ax2_stride,
      halo_size,
      x,
      y);
}

extern "C" void gpudevicemem_halo_pack_hi_3d1_f32(
    uint32_t ax0_size,
    uint32_t ax0_offset,
    uint32_t ax0_stride,
    uint32_t ax1_size,
    uint32_t ax1_offset,
    uint32_t ax1_stride,
    uint32_t ax2_size,
    uint32_t ax2_offset,
    uint32_t ax2_stride,
    uint32_t halo_size,
    const float *x,
    float *y,
    const struct KernelConfig *cfg,
    struct CUstream_st *stream)
{
  // TODO
  uint32_t len = ax0_size * halo_size * ax2_size;
  gpudevicemem_halo_pack_hi_3d1_kernel<float><<<cfg->flat_grid_dim(len), cfg->flat_block_dim(), 0, stream>>>(
      len,
      ax0_size, ax0_offset, ax0_stride,
      ax1_size, ax1_offset, ax1_stride,
      ax2_size, ax2_offset, ax2_stride,
      halo_size,
      x,
      y);
}

extern "C" void gpudevicemem_halo_unpack_3d1_f32(
    uint32_t ax0_size,
    uint32_t ax0_offset,
    uint32_t ax0_stride,
    uint32_t ax1_size,
    uint32_t ax1_offset,
    uint32_t ax1_stride,
    uint32_t ax2_size,
    uint32_t ax2_offset,
    uint32_t ax2_stride,
    uint32_t halo_size,
    const float *x,
    float *y,
    const struct KernelConfig *cfg,
    struct CUstream_st *stream)
{
  // TODO
  uint32_t len = ax0_size * (halo_size * 2) * ax2_size;
  gpudevicemem_halo_unpack_3d1_kernel<float><<<cfg->flat_grid_dim(len), cfg->flat_block_dim(), 0, stream>>>(
      len,
      ax0_size, ax0_offset, ax0_stride,
      ax1_size, ax1_offset, ax1_stride,
      ax2_size, ax2_offset, ax2_stride,
      halo_size,
      x,
      y);
}

extern "C" void gpudevicemem_halo_unpack_lo_3d1_f32(
    uint32_t ax0_size,
    uint32_t ax0_offset,
    uint32_t ax0_stride,
    uint32_t ax1_size,
    uint32_t ax1_offset,
    uint32_t ax1_stride,
    uint32_t ax2_size,
    uint32_t ax2_offset,
    uint32_t ax2_stride,
    uint32_t halo_size,
    const float *x,
    float *y,
    const struct KernelConfig *cfg,
    struct CUstream_st *stream)
{
  // TODO
  uint32_t len = ax0_size * halo_size * ax2_size;
  gpudevicemem_halo_unpack_lo_3d1_kernel<float><<<cfg->flat_grid_dim(len), cfg->flat_block_dim(), 0, stream>>>(
      len,
      ax0_size, ax0_offset, ax0_stride,
      ax1_size, ax1_offset, ax1_stride,
      ax2_size, ax2_offset, ax2_stride,
      halo_size,
      x,
      y);
}

extern "C" void gpudevicemem_halo_unpack_hi_3d1_f32(
    uint32_t ax0_size,
    uint32_t ax0_offset,
    uint32_t ax0_stride,
    uint32_t ax1_size,
    uint32_t ax1_offset,
    uint32_t ax1_stride,
    uint32_t ax2_size,
    uint32_t ax2_offset,
    uint32_t ax2_stride,
    uint32_t halo_size,
    const float *x,
    float *y,
    const struct KernelConfig *cfg,
    struct CUstream_st *stream)
{
  // TODO
  uint32_t len = ax0_size * halo_size * ax2_size;
  gpudevicemem_halo_unpack_hi_3d1_kernel<float><<<cfg->flat_grid_dim(len), cfg->flat_block_dim(), 0, stream>>>(
      len,
      ax0_size, ax0_offset, ax0_stride,
      ax1_size, ax1_offset, ax1_stride,
      ax2_size, ax2_offset, ax2_stride,
      halo_size,
      x,
      y);
}
