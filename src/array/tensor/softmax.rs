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
pub enum XSoftmaxFullShape {
  Softmax0d(Softmax0dFullShape),
  //Softmax1d(Softmax1dFullShape),
  //Softmax2d(Softmax2dFullShape),
}

pub type Softmax0dFullShape = SoftmaxFullShape<[usize; 2]>;
//pub type Softmax1dFullShape = SoftmaxFullShape<[usize; 3]>;
//pub type Softmax2dFullShape = SoftmaxFullShape<[usize; 4]>;

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct SoftmaxFullShape<XIdx> {
  pub src_feature_axis: isize,
  pub src_batch_axis:   isize,
  pub src_size: XIdx,
  pub dst_feature_axis: isize,
  pub dst_batch_axis:   isize,
  pub dst_size: XIdx,
}

impl Softmax0dFullShape {
  pub fn is_default_nchw(&self) -> bool {
        self.src_feature_axis == 0
    &&  self.src_batch_axis == 1
    &&  self.dst_feature_axis == 0
    &&  self.dst_batch_axis == 1
  }

  pub fn is_default_nhwc(&self) -> bool {
        self.src_feature_axis == 0
    &&  self.src_batch_axis == 1
    &&  self.dst_feature_axis == 0
    &&  self.dst_batch_axis == 1
  }
}

pub enum XGPUSoftmaxState<T> {
  Cudnn(CudnnGPUSoftmaxState<T>),
}

pub struct CudnnGPUSoftmaxState<T> {
  src_desc:     CudnnTensorDesc<T>,
  dst_desc:     CudnnTensorDesc<T>,
  dst2_desc:    CudnnTensorDesc<T>,
}

fn sz2int(sz: usize) -> i32 {
  assert!(sz <= i32::max_value() as usize);
  sz as i32
}

pub fn query_gpu_softmax_state<T>(
    dev: GPUDeviceId,
    softmax_shape: XSoftmaxFullShape,
    conn: GPUDeviceConn)
-> Option<XGPUSoftmaxState<T>>
where T: GPUDataTyped + CudnnDataTypeExt,
{
  let mut src_desc = CudnnTensorDesc::<T>::create().unwrap();
  let mut dst_desc = CudnnTensorDesc::<T>::create().unwrap();
  let mut dst2_desc = CudnnTensorDesc::<T>::create().unwrap();
  match softmax_shape {
    XSoftmaxFullShape::Softmax0d(shape) => {
      // TODO: configure tensor layout.
      if shape.is_default_nchw() {
        assert!(src_desc.set_4d_nchw(
            sz2int(shape.src_size[1]),
            sz2int(shape.src_size[0]),
            1,
            1,
        ).is_ok());
        assert!(dst_desc.set_4d_nchw(
            sz2int(shape.dst_size[1]),
            sz2int(shape.dst_size[0]),
            1,
            1,
        ).is_ok());
        assert!(dst2_desc.set_4d_nchw(
            sz2int(shape.dst_size[1]),
            sz2int(shape.dst_size[0]),
            1,
            1,
        ).is_ok());
      } else if shape.is_default_nhwc() {
        unreachable!();
      } else {
        unimplemented!("only nchw layout is currently supported");
      }
    }
    _ => unimplemented!(),
  }

  return Some(XGPUSoftmaxState::Cudnn(CudnnGPUSoftmaxState{
    src_desc, dst_desc, dst2_desc,
  }));
}

pub trait GPUBatchSoftmaxOps<T: Copy> {
  fn batch_softmax(&mut self,
      state: &mut XGPUSoftmaxState<T>,
      x: GPUDeviceArrayView2d<T>,
      conn: GPUDeviceConn);
  fn batch_softmax_bwd(&mut self,
      state: &mut XGPUSoftmaxState<T>,
      y: GPUDeviceArrayView2d<T>,
      dy: GPUDeviceArrayView2d<T>,
      conn: GPUDeviceConn);
}

impl<T: Copy + 'static> GPUBatchSoftmaxOps<T> for GPUDeviceArrayViewMut2d<T>
where CudnnHandle: CudnnSoftmaxExt<T>,
      <CudnnHandle as CudnnSoftmaxExt<T>>::HostScalar: Zero + One,
{
  fn batch_softmax(&mut self,
      state: &mut XGPUSoftmaxState<T>,
      x: GPUDeviceArrayView2d<T>,
      conn: GPUDeviceConn)
  {
    match state {
      &mut XGPUSoftmaxState::Cudnn(ref mut state) => {
        let x = x.wait(conn.clone());
        let mut y = self.wait_mut(conn.clone());
        let mut stream = conn.cuda_stream();
        let mut cudnn_h = conn.cudnn();
        assert!(cudnn_h.set_stream(&mut stream).is_ok());
        let alpha: <CudnnHandle as CudnnSoftmaxExt<T>>::HostScalar = one();
        let beta: <CudnnHandle as CudnnSoftmaxExt<T>>::HostScalar = zero();
        let status = unsafe { cudnn_h.softmax_fwd(
            cudnnSoftmaxAlgorithm_t_CUDNN_SOFTMAX_ACCURATE,
            cudnnSoftmaxMode_t_CUDNN_SOFTMAX_MODE_INSTANCE,
            alpha,
            &mut state.src_desc,
            x.as_dptr(),
            beta,
            &mut state.dst_desc,
            y.as_mut_dptr(),
        ) };
        assert!(status.is_ok());
      }
      //_ => unimplemented!(),
    }
  }

  fn batch_softmax_bwd(&mut self,
      state: &mut XGPUSoftmaxState<T>,
      y: GPUDeviceArrayView2d<T>,
      dy: GPUDeviceArrayView2d<T>,
      conn: GPUDeviceConn)
  {
    match state {
      &mut XGPUSoftmaxState::Cudnn(ref mut state) => {
        let y = y.wait(conn.clone());
        let dy = dy.wait(conn.clone());
        let mut dx = self.wait_mut(conn.clone());
        let mut stream = conn.cuda_stream();
        let mut cudnn_h = conn.cudnn();
        assert!(cudnn_h.set_stream(&mut stream).is_ok());
        let alpha: <CudnnHandle as CudnnSoftmaxExt<T>>::HostScalar = one();
        let beta: <CudnnHandle as CudnnSoftmaxExt<T>>::HostScalar = zero();
        let status = unsafe { cudnn_h.softmax_bwd(
            cudnnSoftmaxAlgorithm_t_CUDNN_SOFTMAX_ACCURATE,
            cudnnSoftmaxMode_t_CUDNN_SOFTMAX_MODE_INSTANCE,
            alpha,
            &mut state.dst_desc,
            y.as_dptr(),
            &mut state.dst2_desc,
            dy.as_dptr(),
            beta,
            &mut state.src_desc,
            dx.as_mut_dptr(),
        ) };
        assert!(status.is_ok());
      }
      //_ => unimplemented!(),
    }
  }
}
