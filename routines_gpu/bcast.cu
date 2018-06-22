/*
Copyright 2018 the gpudevicemem authors

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

template <typename T, typename Write>
__global__ void gpudevicemem_bcast_packed_kernel(
    uint32_t bcast_dim,
    const T *x,
    T *y)
{
  float x0 = x[0];
  for (uint32_t idx = gtindex(); idx < bcast_dim; idx += gtcount()) {
    Write::Write(&y[idx], x0);
  }
}

extern "C" void gpudevicemem_bcast_packed_f32(
    uint32_t bcast_dim,
    const float *x,
    float *y,
    const KernelConfig *cfg,
    cudaStream_t stream)
{
  uint32_t len = bcast_dim;
  gpudevicemem_bcast_packed_kernel<float, AssignWrite<float>><<<cfg->flat_grid_dim(len), cfg->flat_block_dim(), 0, stream>>>(
      bcast_dim, x, y);
}

extern "C" void gpudevicemem_bcast_packed_accumulate_f32(
    uint32_t bcast_dim,
    const float *x,
    float *y,
    const KernelConfig *cfg,
    cudaStream_t stream)
{
  uint32_t len = bcast_dim;
  gpudevicemem_bcast_packed_kernel<float, AccumulateWrite<float>><<<cfg->flat_grid_dim(len), cfg->flat_block_dim(), 0, stream>>>(
      bcast_dim, x, y);
}
