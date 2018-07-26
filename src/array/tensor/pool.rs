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

use ::{GPUDeviceId, GPUDeviceConn};
use ::array::*;
use ::array::tensor::conv::*;

//use arithmetic::{PseudoField, PseudoRing};
use cuda_dnn::*;
use cuda_dnn::ffi::*;
use float::stub::*;
use num_traits::identities::*;

use std::collections::{HashMap};
use std::mem::{uninitialized};
//use std::ptr::{null, null_mut};
use std::sync::{Arc, Mutex};

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum XPoolOp {
  Average,
  Max,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum XPoolFullShape {
  Pool1d(Pool1dFullShape),
  Pool2d(Pool2dFullShape),
  Pool3d(Pool3dFullShape),
}

impl XPoolFullShape {
  pub fn preset_conv2d_nchw(n: usize, c: usize, h: usize, w: usize) -> Self {
    // TODO
    unimplemented!();
  }

  pub fn preset_conv2d_nwhc(n: usize, h: usize, w: usize, c: usize) -> Self {
    // TODO
    unimplemented!();
  }
}

pub type Pool1dFullShape = PoolFullShape<isize, usize, [usize; 3]>;
pub type Pool2dFullShape = PoolFullShape<[isize; 2], [usize; 2], [usize; 4]>;
pub type Pool3dFullShape = PoolFullShape<[isize; 3], [usize; 3], [usize; 5]>;

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct PoolFullShape<Axes, WIdx, XIdx> {
  pub src_space_axes:   Axes,
  pub src_feature_axis: isize,
  pub src_batch_axis:   isize,
  pub src_size: XIdx,
  pub dst_space_axes:   Axes,
  pub dst_feature_axis: isize,
  pub dst_batch_axis:   isize,
  pub dst_size: XIdx,
  pub ker_size: WIdx,
  pub stride:   WIdx,
  pub zero_pad: WIdx,
}

impl Pool2dFullShape {
  pub fn is_default_nchw(&self) -> bool {
        self.src_space_axes[0] == 0
    &&  self.src_space_axes[1] == 1
    &&  self.src_feature_axis == 2
    &&  self.src_batch_axis == 3
    &&  self.dst_space_axes[0] == 0
    &&  self.dst_space_axes[1] == 1
    &&  self.dst_feature_axis == 2
    &&  self.dst_batch_axis == 3
  }

  pub fn is_default_nhwc(&self) -> bool {
        self.src_space_axes[0] == 1
    &&  self.src_space_axes[1] == 2
    &&  self.src_feature_axis == 0
    &&  self.src_batch_axis == 3
    &&  self.dst_space_axes[0] == 1
    &&  self.dst_space_axes[1] == 2
    &&  self.dst_feature_axis == 0
    &&  self.dst_batch_axis == 3
  }
}

impl Pool3dFullShape {
  pub fn is_default_ncdhw(&self) -> bool {
        self.src_space_axes[0] == 0
    &&  self.src_space_axes[1] == 1
    &&  self.src_space_axes[2] == 2
    &&  self.src_feature_axis == 3
    &&  self.src_batch_axis == 4
    &&  self.dst_space_axes[0] == 0
    &&  self.dst_space_axes[1] == 1
    &&  self.dst_space_axes[2] == 2
    &&  self.dst_feature_axis == 3
    &&  self.dst_batch_axis == 4
  }
}

pub enum XGPUPoolState<T> {
  Cudnn(CudnnGPUPoolState<T>),
}

pub struct CudnnGPUPoolState<T> {
  src_desc:     CudnnTensorDesc<T>,
  src2_desc:    CudnnTensorDesc<T>,
  dst_desc:     CudnnTensorDesc<T>,
  dst2_desc:    CudnnTensorDesc<T>,
  pool_desc:    CudnnPoolDesc,
}

fn sz2int(sz: usize) -> i32 {
  assert!(sz <= i32::max_value() as usize);
  sz as i32
}

fn encode_index3(size: [usize; 3]) -> [i32; 3] {
  [
    sz2int(size[2]),
    sz2int(size[1]),
    sz2int(size[0]),
  ]
}

fn encode_index5(size: [usize; 5]) -> [i32; 5] {
  [
    sz2int(size[4]),
    sz2int(size[3]),
    sz2int(size[2]),
    sz2int(size[1]),
    sz2int(size[0]),
  ]
}

pub fn query_gpu_pool_state<T>(
    dev: GPUDeviceId,
    pool_op: XPoolOp,
    pool_shape: XPoolFullShape,
    conn: GPUDeviceConn)
-> Option<XGPUPoolState<T>>
where T: GPUDataTyped + CudnnDataTypeExt,
{
  let mut src_desc = CudnnTensorDesc::<T>::create().unwrap();
  let mut src2_desc = CudnnTensorDesc::<T>::create().unwrap();
  let mut dst_desc = CudnnTensorDesc::<T>::create().unwrap();
  let mut dst2_desc = CudnnTensorDesc::<T>::create().unwrap();
  let mut pool_desc = CudnnPoolDesc::create().unwrap();
  match pool_shape {
    XPoolFullShape::Pool2d(shape) => {
      // TODO: configure tensor layout.
      if shape.is_default_nchw() {
        assert!(src_desc.set_4d_nchw(
            sz2int(shape.src_size[3]),
            sz2int(shape.src_size[2]),
            sz2int(shape.src_size[1]),
            sz2int(shape.src_size[0]),
        ).is_ok());
        assert!(src2_desc.set_4d_nchw(
            sz2int(shape.src_size[3]),
            sz2int(shape.src_size[2]),
            sz2int(shape.src_size[1]),
            sz2int(shape.src_size[0]),
        ).is_ok());
        assert!(dst_desc.set_4d_nchw(
            sz2int(shape.dst_size[3]),
            sz2int(shape.dst_size[2]),
            sz2int(shape.dst_size[1]),
            sz2int(shape.dst_size[0]),
        ).is_ok());
        assert!(dst2_desc.set_4d_nchw(
            sz2int(shape.dst_size[3]),
            sz2int(shape.dst_size[2]),
            sz2int(shape.dst_size[1]),
            sz2int(shape.dst_size[0]),
        ).is_ok());
      } else if shape.is_default_nhwc() {
        unimplemented!();
      } else {
        unimplemented!("only nchw layout is currently supported");
      }
      assert!(pool_desc.set_2d(
          sz2int(shape.ker_size[1]),  sz2int(shape.ker_size[0]),
          sz2int(shape.zero_pad[1]),  sz2int(shape.zero_pad[0]),
          sz2int(shape.stride[1]),    sz2int(shape.stride[0]),
          match pool_op {
            XPoolOp::Average    => cudnnPoolingMode_t_CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING,
            XPoolOp::Max        => cudnnPoolingMode_t_CUDNN_POOLING_MAX_DETERMINISTIC,
          },
          // TODO: configure cudnn NaN propagation.
          cudnnNanPropagation_t_CUDNN_NOT_PROPAGATE_NAN,
      ).is_ok());
    }
    XPoolFullShape::Pool3d(shape) => {
      // TODO: configure tensor layout.
      if shape.is_default_ncdhw() {
        let src_stride = shape.src_size.to_packed_stride();
        assert!(src_desc.set_nd(
            &encode_index5(shape.src_size),
            &encode_index5(src_stride),
        ).is_ok());
        assert!(src2_desc.set_nd(
            &encode_index5(shape.src_size),
            &encode_index5(src_stride),
        ).is_ok());
        let dst_stride = shape.dst_size.to_packed_stride();
        assert!(dst_desc.set_nd(
            &encode_index5(shape.dst_size),
            &encode_index5(dst_stride),
        ).is_ok());
        assert!(dst2_desc.set_nd(
            &encode_index5(shape.dst_size),
            &encode_index5(dst_stride),
        ).is_ok());
      } else {
        unimplemented!("only nchw layout is currently supported");
      }
      assert!(pool_desc.set_nd(
          &encode_index3(shape.ker_size),
          &encode_index3(shape.zero_pad),
          &encode_index3(shape.stride),
          match pool_op {
            XPoolOp::Average    => cudnnPoolingMode_t_CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING,
            XPoolOp::Max        => cudnnPoolingMode_t_CUDNN_POOLING_MAX_DETERMINISTIC,
          },
          // TODO: configure cudnn NaN propagation.
          cudnnNanPropagation_t_CUDNN_NOT_PROPAGATE_NAN,
      ).is_ok());
    }
    _ => unimplemented!(),
  }

  return Some(XGPUPoolState::Cudnn(CudnnGPUPoolState{
    src_desc, src2_desc, dst_desc, dst2_desc, pool_desc,
  }));
}

pub trait GPUBatchPoolOps<T: Copy>
where CudnnHandle: CudnnPoolExt<T>,
      <CudnnHandle as CudnnPoolExt<T>>::HostScalar: Zero + One,
{
  fn batch_pool2d(&mut self,
      state: &mut XGPUPoolState<T>,
      x: GPUDeviceArrayView4d<T>,
      conn: GPUDeviceConn);
  fn batch_pool2d_bwd(&mut self,
      state: &mut XGPUPoolState<T>,
      y: GPUDeviceArrayView4d<T>,
      dy: GPUDeviceArrayView4d<T>,
      x: GPUDeviceArrayView4d<T>,
      conn: GPUDeviceConn);
}

pub trait GPUBatchPool3dOps<T: Copy>
where CudnnHandle: CudnnPoolExt<T>,
      <CudnnHandle as CudnnPoolExt<T>>::HostScalar: Zero + One,
{
  fn batch_pool3d(&mut self,
      state: &mut XGPUPoolState<T>,
      alpha: <CudnnHandle as CudnnPoolExt<T>>::HostScalar,
      x: GPUDeviceArrayView5d<T>,
      beta: <CudnnHandle as CudnnPoolExt<T>>::HostScalar,
      conn: GPUDeviceConn);
  fn batch_pool3d_bwd(&mut self,
      state: &mut XGPUPoolState<T>,
      alpha: <CudnnHandle as CudnnPoolExt<T>>::HostScalar,
      y: GPUDeviceArrayView5d<T>,
      dy: GPUDeviceArrayView5d<T>,
      x: GPUDeviceArrayView5d<T>,
      beta: <CudnnHandle as CudnnPoolExt<T>>::HostScalar,
      conn: GPUDeviceConn);
}

impl<T: Copy> GPUBatchPoolOps<T> for GPUDeviceArrayViewMut4d<T>
where CudnnHandle: CudnnPoolExt<T>,
      <CudnnHandle as CudnnPoolExt<T>>::HostScalar: Zero + One,
{
  fn batch_pool2d(&mut self,
      state: &mut XGPUPoolState<T>,
      x: GPUDeviceArrayView4d<T>,
      conn: GPUDeviceConn)
  {
    match state {
      &mut XGPUPoolState::Cudnn(ref mut state) => {
        let mut stream = conn.cuda_stream();
        let mut cudnn_h = conn.cudnn();
        assert!(cudnn_h.set_stream(&mut stream).is_ok());
        let alpha: <CudnnHandle as CudnnPoolExt<T>>::HostScalar = one();
        let beta: <CudnnHandle as CudnnPoolExt<T>>::HostScalar = zero();
        let status = unsafe { cudnn_h.pool_fwd(
            &mut state.pool_desc,
            alpha,
            &mut state.src_desc,
            x.raw_dptr(),
            beta,
            &mut state.dst_desc,
            self.raw_mut_dptr(),
        ) };
        assert!(status.is_ok());
      }
      //_ => unimplemented!(),
    }
  }

  fn batch_pool2d_bwd(&mut self,
      state: &mut XGPUPoolState<T>,
      y: GPUDeviceArrayView4d<T>,
      dy: GPUDeviceArrayView4d<T>,
      x: GPUDeviceArrayView4d<T>,
      conn: GPUDeviceConn)
  {
    match state {
      &mut XGPUPoolState::Cudnn(ref mut state) => {
        let mut stream = conn.cuda_stream();
        let mut cudnn_h = conn.cudnn();
        assert!(cudnn_h.set_stream(&mut stream).is_ok());
        let alpha: <CudnnHandle as CudnnPoolExt<T>>::HostScalar = one();
        let beta: <CudnnHandle as CudnnPoolExt<T>>::HostScalar = zero();
        let status = unsafe { cudnn_h.pool_bwd(
            &mut state.pool_desc,
            alpha,
            &mut state.dst_desc,
            y.raw_dptr(),
            &mut state.dst2_desc,
            dy.raw_dptr(),
            &mut state.src_desc,
            x.raw_dptr(),
            beta,
            &mut state.src2_desc,
            self.raw_mut_dptr(),
        ) };
        assert!(status.is_ok());
      }
      //_ => unimplemented!(),
    }
  }
}

impl<T: Copy> GPUBatchPool3dOps<T> for GPUDeviceArrayViewMut5d<T>
where CudnnHandle: CudnnPoolExt<T>,
      <CudnnHandle as CudnnPoolExt<T>>::HostScalar: Zero + One,
{
  fn batch_pool3d(&mut self,
      state: &mut XGPUPoolState<T>,
      alpha: <CudnnHandle as CudnnPoolExt<T>>::HostScalar,
      x: GPUDeviceArrayView5d<T>,
      beta: <CudnnHandle as CudnnPoolExt<T>>::HostScalar,
      conn: GPUDeviceConn)
  {
    match state {
      &mut XGPUPoolState::Cudnn(ref mut state) => {
        let mut stream = conn.cuda_stream();
        let mut cudnn_h = conn.cudnn();
        assert!(cudnn_h.set_stream(&mut stream).is_ok());
        let status = unsafe { cudnn_h.pool_fwd(
            &mut state.pool_desc,
            alpha,
            &mut state.src_desc,
            x.raw_dptr(),
            beta,
            &mut state.dst_desc,
            self.raw_mut_dptr(),
        ) };
        assert!(status.is_ok());
      }
      //_ => unimplemented!(),
    }
  }

  fn batch_pool3d_bwd(&mut self,
      state: &mut XGPUPoolState<T>,
      alpha: <CudnnHandle as CudnnPoolExt<T>>::HostScalar,
      y: GPUDeviceArrayView5d<T>,
      dy: GPUDeviceArrayView5d<T>,
      x: GPUDeviceArrayView5d<T>,
      beta: <CudnnHandle as CudnnPoolExt<T>>::HostScalar,
      conn: GPUDeviceConn)
  {
    match state {
      &mut XGPUPoolState::Cudnn(ref mut state) => {
        let mut stream = conn.cuda_stream();
        let mut cudnn_h = conn.cudnn();
        assert!(cudnn_h.set_stream(&mut stream).is_ok());
        let status = unsafe { cudnn_h.pool_bwd(
            &mut state.pool_desc,
            alpha,
            &mut state.dst_desc,
            y.raw_dptr(),
            &mut state.dst2_desc,
            dy.raw_dptr(),
            &mut state.src_desc,
            x.raw_dptr(),
            beta,
            &mut state.src2_desc,
            self.raw_mut_dptr(),
        ) };
        assert!(status.is_ok());
      }
      //_ => unimplemented!(),
    }
  }
}
