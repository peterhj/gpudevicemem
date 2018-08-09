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

// "bcast.cu"

void gpudevicemem_bcast_packed_f32(
    uint32_t bcast_dim,
    const float *x,
    float *y,
    const struct KernelConfig *cfg,
    struct CUstream_st *stream);
void gpudevicemem_bcast_packed_accumulate_f32(
    uint32_t bcast_dim,
    const float *x,
    float *y,
    const struct KernelConfig *cfg,
    struct CUstream_st *stream);
void gpudevicemem_bcast_Ib_Oab_packed_f32(
    uint32_t inner_dim,
    uint32_t outer_dim,
    const float *x,
    float *y,
    const struct KernelConfig *cfg,
    struct CUstream_st *stream);
void gpudevicemem_bcast_Ib_Oab_packed_accumulate_f32(
    uint32_t inner_dim,
    uint32_t outer_dim,
    const float *x,
    float *y,
    const struct KernelConfig *cfg,
    struct CUstream_st *stream);

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
void gpudevicemem_bcast_flat_mult_add_I1b_I2abc_I3b_Oabc_packed_f32(
    uint32_t inner_dim,
    uint32_t bcast_dim,
    uint32_t outer_dim,
    const float *lx,
    const float *rx,
    const float *shift,
    float *y,
    const struct KernelConfig *cfg,
    struct CUstream_st *stream);
void gpudevicemem_flat_bcast_rdiv_I1ab_I2b_Oab_packed_f32(
    uint32_t inner_dim,
    uint32_t outer_dim,
    const float *lx,
    const float *rx,
    float *y,
    const struct KernelConfig *cfg,
    struct CUstream_st *stream);

// "flat_linear.cu"
void gpudevicemem_flat_add_inplace_f32(
    uint32_t len,
    const float *x,
    float *y,
    const struct KernelConfig *cfg,
    struct CUstream_st *stream);
void gpudevicemem_flat_mult_inplace_f32(
    uint32_t len,
    const float *x,
    float *y,
    const struct KernelConfig *cfg,
    struct CUstream_st *stream);
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
void gpudevicemem_flat_rdiv_inplace_f32(
    uint32_t len,
    const float *x,
    float *y,
    const struct KernelConfig *cfg,
    struct CUstream_st *stream);

// "flat_map.cu"
void gpudevicemem_set_constant_flat_map_inplace_f32(
    uint32_t len,
    float c,
    float *y,
    const struct KernelConfig *cfg,
    struct CUstream_st *stream);
void gpudevicemem_add_constant_flat_map_inplace_f32(
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
void gpudevicemem_rdiv_constant_flat_map_f32(
    uint32_t len,
    float c,
    const float *x,
    float *y,
    const struct KernelConfig *cfg,
    struct CUstream_st *stream);
void gpudevicemem_ldiv_constant_flat_map_f32(
    uint32_t len,
    float c,
    const float *x,
    float *y,
    const struct KernelConfig *cfg,
    struct CUstream_st *stream);
void gpudevicemem_online_add_flat_map_accum_f32(
    uint32_t len,
    float c,
    const float *x,
    float *y,
    const struct KernelConfig *cfg,
    struct CUstream_st *stream);
void gpudevicemem_online_discount_flat_map_accum_f32(
    uint32_t len,
    float c,
    const float *x,
    float *y,
    const struct KernelConfig *cfg,
    struct CUstream_st *stream);
void gpudevicemem_online_average_flat_map_accum_f32(
    uint32_t len,
    float c,
    const float *x,
    float *y,
    const struct KernelConfig *cfg,
    struct CUstream_st *stream);
void gpudevicemem_is_nonzero_flat_map_f32(
    uint32_t len,
    const float *x,
    float *y,
    const struct KernelConfig *cfg,
    struct CUstream_st *stream);
void gpudevicemem_is_zero_flat_map_f32(
    uint32_t len,
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

// "halo_ring.cu"

void gpudevicemem_halo_ring_3d1_fill_f32(
    uint32_t halo_radius,
    uint32_t dim0,
    uint32_t dim1,
    uint32_t dim2,
    float *src_arr,
    float *dst_arr,
    const struct KernelConfig *cfg,
    struct CUstream_st *stream);
void gpudevicemem_halo_ring_3d1_zero_lghost_f32(
    uint32_t halo_radius,
    uint32_t dim0,
    uint32_t dim1,
    uint32_t dim2,
    float *arr,
    const struct KernelConfig *cfg,
    struct CUstream_st *stream);
void gpudevicemem_halo_ring_3d1_zero_rghost_f32(
    uint32_t halo_radius,
    uint32_t dim0,
    uint32_t dim1,
    uint32_t dim2,
    float *arr,
    const struct KernelConfig *cfg,
    struct CUstream_st *stream);
void gpudevicemem_halo_ring_3d1_copy_ledge_to_buf_f32(
    uint32_t halo_radius,
    uint32_t dim0,
    uint32_t dim1,
    uint32_t dim2,
    float *arr,
    float *region_buf,
    const struct KernelConfig *cfg,
    struct CUstream_st *stream);
void gpudevicemem_halo_ring_3d1_copy_redge_to_buf_f32(
    uint32_t halo_radius,
    uint32_t dim0,
    uint32_t dim1,
    uint32_t dim2,
    float *arr,
    float *region_buf,
    const struct KernelConfig *cfg,
    struct CUstream_st *stream);
void gpudevicemem_halo_ring_3d1_copy_buf_to_lghost_f32(
    uint32_t halo_radius,
    uint32_t dim0,
    uint32_t dim1,
    uint32_t dim2,
    float *arr,
    float *region_buf,
    const struct KernelConfig *cfg,
    struct CUstream_st *stream);
void gpudevicemem_halo_ring_3d1_copy_buf_to_rghost_f32(
    uint32_t halo_radius,
    uint32_t dim0,
    uint32_t dim1,
    uint32_t dim2,
    float *arr,
    float *region_buf,
    const struct KernelConfig *cfg,
    struct CUstream_st *stream);
void gpudevicemem_halo_ring_3d1_copy_lghost_to_buf_f32(
    uint32_t halo_radius,
    uint32_t dim0,
    uint32_t dim1,
    uint32_t dim2,
    float *arr,
    float *region_buf,
    const struct KernelConfig *cfg,
    struct CUstream_st *stream);
void gpudevicemem_halo_ring_3d1_copy_rghost_to_buf_f32(
    uint32_t halo_radius,
    uint32_t dim0,
    uint32_t dim1,
    uint32_t dim2,
    float *arr,
    float *region_buf,
    const struct KernelConfig *cfg,
    struct CUstream_st *stream);
void gpudevicemem_halo_ring_3d1_accumulate_buf_to_ledge_f32(
    uint32_t halo_radius,
    uint32_t dim0,
    uint32_t dim1,
    uint32_t dim2,
    float *arr,
    float *region_buf,
    const struct KernelConfig *cfg,
    struct CUstream_st *stream);
void gpudevicemem_halo_ring_3d1_accumulate_buf_to_redge_f32(
    uint32_t halo_radius,
    uint32_t dim0,
    uint32_t dim1,
    uint32_t dim2,
    float *arr,
    float *region_buf,
    const struct KernelConfig *cfg,
    struct CUstream_st *stream);

// "reduce.cu"

void gpudevicemem_sum_packed_deterministic_f32(
    uint32_t reduce_dim,
    const float *x,
    float *y,
    const struct KernelConfig *cfg,
    struct CUstream_st *stream);
void gpudevicemem_sum_packed_accumulate_deterministic_f32(
    uint32_t reduce_dim,
    const float *x,
    float *y,
    const struct KernelConfig *cfg,
    struct CUstream_st *stream);
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
void gpudevicemem_mult_then_sum_I1abc_I2abc_Ob_packed_deterministic_f32(
    uint32_t reduce_inner_dim,
    uint32_t mid_dim,
    uint32_t reduce_outer_dim,
    const float *x1,
    const float *x2,
    float *y,
    const struct KernelConfig *cfg,
    struct CUstream_st *stream);

#ifdef __cplusplus
} // extern "C"
#endif

#endif
