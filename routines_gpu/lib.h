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

#ifndef __DEVICEMEM_ROUTINES_GPU_LIB_H__
#define __DEVICEMEM_ROUTINES_GPU_LIB_H__

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>

struct KernelConfig;
struct CUstream_st;

// "bcast_flat_linear.cu"

void gpudevicemem_bcast_flat_add_I1a_I2ab_Oab_packed_f32(
    uint32_t bcast_dim,
    uint32_t outer_dim,
    const float *lx,
    const float *rx,
    float *y,
    const struct KernelConfig *cfg,
    struct CUstream_st *stream);
void gpudevicemem_bcast_flat_add_I1a_IO2ab_inplace_packed_f32(
    uint32_t bcast_dim,
    uint32_t outer_dim,
    const float *lx,
    float *rx,
    const struct KernelConfig *cfg,
    struct CUstream_st *stream);
void gpudevicemem_bcast_flat_add_I1b_I2abc_Oabc_packed_f32(
    uint32_t inner_dim,
    uint32_t bcast_dim,
    uint32_t outer_dim,
    const float *lx,
    const float *rx,
    float *y,
    const struct KernelConfig *cfg,
    struct CUstream_st *stream);
void gpudevicemem_bcast_flat_add_I1b_IO2abc_inplace_packed_f32(
    uint32_t inner_dim,
    uint32_t bcast_dim,
    uint32_t outer_dim,
    const float *lx,
    float *rx,
    const struct KernelConfig *cfg,
    struct CUstream_st *stream);
void gpudevicemem_bcast_flat_mult_I1b_I2ab_Oab_packed_f32(
    uint32_t inner_dim,
    uint32_t bcast_dim,
    const float *lx,
    const float *rx,
    float *y,
    const struct KernelConfig *cfg,
    struct CUstream_st *stream);
void gpudevicemem_bcast_flat_mult_add_I1b_I2ab_Oab_packed_f32(
    uint32_t inner_dim,
    uint32_t bcast_dim,
    const float *lx,
    const float *rx,
    const float *shift,
    float *y,
    const struct KernelConfig *cfg,
    struct CUstream_st *stream);
void gpudevicemem_bcast_flat_mult_I1b_I2abc_Oabc_packed_f32(
    uint32_t inner_dim,
    uint32_t bcast_dim,
    uint32_t outer_dim,
    const float *lx,
    const float *rx,
    float *y,
    const struct KernelConfig *cfg,
    struct CUstream_st *stream);
void gpudevicemem_bcast_flat_mult_add_I1b_I2abc_Oabc_packed_f32(
    uint32_t inner_dim,
    uint32_t bcast_dim,
    uint32_t outer_dim,
    const float *lx,
    const float *rx,
    const float *shift,
    float *y,
    const struct KernelConfig *cfg,
    struct CUstream_st *stream);

// "flat_linear.cu"
void gpudevicemem_flat_mult_f32(
    uint32_t len,
    const float *lx,
    const float *rx,
    float *y,
    const struct KernelConfig *cfg,
    struct CUstream_st *stream);
void gpudevicemem_flat_mult_add_f32(
    uint32_t len,
    const float *lx,
    const float *rx,
    const float *shift,
    float *y,
    const struct KernelConfig *cfg,
    struct CUstream_st *stream);

// "flat_map.cu"
void gpudevicemem_set_constant_flat_map_f32(
    uint32_t len,
    float c,
    float *y,
    const struct KernelConfig *cfg,
    struct CUstream_st *stream);
void gpudevicemem_add_constant_flat_map_f32(
    uint32_t len,
    float c,
    const float *x,
    float *y,
    const struct KernelConfig *cfg,
    struct CUstream_st *stream);
void gpudevicemem_mult_constant_flat_map_f32(
    uint32_t len,
    float c,
    const float *x,
    float *y,
    const struct KernelConfig *cfg,
    struct CUstream_st *stream);
/*void gpudevicemem_copy_flat_map_f32(
    uint32_t len,
    const float *x,
    float *y,
    const struct KernelConfig *cfg,
    struct CUstream_st *stream);
void gpudevicemem_modulus_flat_map_f32(
    uint32_t len,
    const float *x,
    float *y,
    const struct KernelConfig *cfg,
    struct CUstream_st *stream);
void gpudevicemem_square_flat_map_f32(
    uint32_t len,
    const float *x,
    float *y,
    const struct KernelConfig *cfg,
    struct CUstream_st *stream);
void gpudevicemem_positive_clip_flat_map_f32(
    uint32_t len,
    const float *x,
    float *y,
    const struct KernelConfig *cfg,
    struct CUstream_st *stream);
void gpudevicemem_unit_step_flat_map_f32(
    uint32_t len,
    const float *x,
    float *y,
    const struct KernelConfig *cfg,
    struct CUstream_st *stream);
void gpudevicemem_normal_cdf_flat_map_f32(
    uint32_t len,
    const float *x,
    float *y,
    const struct KernelConfig *cfg,
    struct CUstream_st *stream);
void gpudevicemem_tanh_flat_map_f32(
    uint32_t len,
    const float *x,
    float *y,
    const struct KernelConfig *cfg,
    struct CUstream_st *stream);
void gpudevicemem_rcosh2_flat_map_f32(
    uint32_t len,
    const float *x,
    float *y,
    const struct KernelConfig *cfg,
    struct CUstream_st *stream);*/

// "halo.cu"

void gpudevicemem_halo_expand_packed3d1_f32(
    uint32_t ax0_size,
    uint32_t ax1_size,
    uint32_t ax2_size,
    uint32_t halo_size,
    const float *x,
    float *y,
    const struct KernelConfig *cfg,
    struct CUstream_st *stream);
void gpudevicemem_halo_project_packed3d1_f32(
    uint32_t ax0_size,
    uint32_t ax1_size,
    uint32_t ax2_size,
    uint32_t halo_size,
    const float *x,
    float *y,
    const struct KernelConfig *cfg,
    struct CUstream_st *stream);
void gpudevicemem_halo_pack_lo_packed3d1_f32(
    uint32_t ax0_size,
    uint32_t ax1_size,
    uint32_t ax2_size,
    uint32_t halo_size,
    const float *x,
    float *y,
    const struct KernelConfig *cfg,
    struct CUstream_st *stream);
void gpudevicemem_halo_pack_hi_packed3d1_f32(
    uint32_t ax0_size,
    uint32_t ax1_size,
    uint32_t ax2_size,
    uint32_t halo_size,
    const float *x,
    float *y,
    const struct KernelConfig *cfg,
    struct CUstream_st *stream);
void gpudevicemem_halo_unpack_lo_packed3d1_f32(
    uint32_t ax0_size,
    uint32_t ax1_size,
    uint32_t ax2_size,
    uint32_t halo_size,
    const float *x,
    float *y,
    const struct KernelConfig *cfg,
    struct CUstream_st *stream);
void gpudevicemem_halo_unpack_hi_packed3d1_f32(
    uint32_t ax0_size,
    uint32_t ax1_size,
    uint32_t ax2_size,
    uint32_t halo_size,
    const float *x,
    float *y,
    const struct KernelConfig *cfg,
    struct CUstream_st *stream);
void gpudevicemem_halo_ghost_pack_lo_packed3d1_f32(
    uint32_t ax0_size,
    uint32_t ax1_size,
    uint32_t ax2_size,
    uint32_t halo_size,
    const float *x,
    float *y,
    const struct KernelConfig *cfg,
    struct CUstream_st *stream);
void gpudevicemem_halo_ghost_pack_hi_packed3d1_f32(
    uint32_t ax0_size,
    uint32_t ax1_size,
    uint32_t ax2_size,
    uint32_t halo_size,
    const float *x,
    float *y,
    const struct KernelConfig *cfg,
    struct CUstream_st *stream);
void gpudevicemem_halo_edge_reduce_lo_packed3d1_f32(
    uint32_t ax0_size,
    uint32_t ax1_size,
    uint32_t ax2_size,
    uint32_t halo_size,
    const float *x,
    float *y,
    const struct KernelConfig *cfg,
    struct CUstream_st *stream);
void gpudevicemem_halo_edge_reduce_hi_packed3d1_f32(
    uint32_t ax0_size,
    uint32_t ax1_size,
    uint32_t ax2_size,
    uint32_t halo_size,
    const float *x,
    float *y,
    const struct KernelConfig *cfg,
    struct CUstream_st *stream);
void gpudevicemem_halo_set_constant_3d1_f32(
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
    struct CUstream_st *stream);
void gpudevicemem_halo_pack_3d1_f32(
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
    struct CUstream_st *stream);
void gpudevicemem_halo_pack_lo_3d1_f32(
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
    struct CUstream_st *stream);
void gpudevicemem_halo_pack_hi_3d1_f32(
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
    struct CUstream_st *stream);
void gpudevicemem_halo_unpack_3d1_f32(
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
    struct CUstream_st *stream);
void gpudevicemem_halo_unpack_lo_3d1_f32(
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
    struct CUstream_st *stream);
void gpudevicemem_halo_unpack_hi_3d1_f32(
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
    struct CUstream_st *stream);

// "reduce.cu"

void gpudevicemem_sum_I1ab_Oa_packed_deterministic_f32(
    uint32_t inner_dim,
    uint32_t reduce_dim,
    const float *x,
    float *y,
    const struct KernelConfig *cfg,
    struct CUstream_st *stream);
void gpudevicemem_sum_I1ab_Ob_packed_deterministic_f32(
    uint32_t reduce_dim,
    uint32_t outer_dim,
    const float *x,
    float *y,
    const struct KernelConfig *cfg,
    struct CUstream_st *stream);
void gpudevicemem_sum_I1abc_Ob_packed_deterministic_f32(
    uint32_t reduce_inner_dim,
    uint32_t mid_dim,
    uint32_t reduce_outer_dim,
    const float *x,
    float *y,
    const struct KernelConfig *cfg,
    struct CUstream_st *stream);
void gpudevicemem_square_sum_I1abc_Ob_packed_deterministic_f32(
    uint32_t reduce_inner_dim,
    uint32_t mid_dim,
    uint32_t reduce_outer_dim,
    const float *x,
    float *y,
    const struct KernelConfig *cfg,
    struct CUstream_st *stream);

#ifdef __cplusplus
} // extern "C"
#endif

#endif
