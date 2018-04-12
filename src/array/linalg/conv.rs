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

use ::{GPUDeviceId, GPUDeviceConn};
use ::array::*;

use arithmetic::{PseudoField, PseudoRing};
use cuda_dnn::*;
use cuda_dnn::ffi::*;
use float::stub::*;

use std::collections::{HashMap};
use std::mem::{uninitialized};
//use std::ptr::{null, null_mut};
use std::sync::{Arc, Mutex};

lazy_static! {
  static ref CACHED_CONV_FWD_ALGOS: Arc<Mutex<HashMap<XGPUConvKey, XGPUConvFwdConfig>>> = {
    Arc::new(Mutex::new(HashMap::new()))
  };
  static ref CACHED_CONV_BWD_W_ALGOS: Arc<Mutex<HashMap<XGPUConvKey, XGPUConvBwdWConfig>>> = {
    Arc::new(Mutex::new(HashMap::new()))
  };
  static ref CACHED_CONV_BWD_X_ALGOS: Arc<Mutex<HashMap<XGPUConvKey, XGPUConvBwdXConfig>>> = {
    Arc::new(Mutex::new(HashMap::new()))
  };
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct XGPUConvKey {
  dev:          GPUDeviceId,
  determinism:  GPUMathDeterminism,
  math_mode:    GPUMathMode,
  conv_shape:   XConvFullShape,
  conv_type:    XConvType,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum GPUMathDeterminism {
  Nondeterministic,
  Deterministic,
}

impl Default for GPUMathDeterminism {
  fn default() -> Self {
    GPUMathDeterminism::Nondeterministic
  }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum GPUMathMode {
  Fp32,
  Fp64,
  Fp16,
  Fp16MMA,
}

impl Default for GPUMathMode {
  fn default() -> Self {
    GPUMathMode::Fp32
  }
}

impl GPUMathMode {
  pub fn gpu_data_ty(&self) -> GPUDataType {
    match *self {
      GPUMathMode::Fp32     => GPUDataType::Fp32,
      GPUMathMode::Fp64     => GPUDataType::Fp64,
      GPUMathMode::Fp16     |
      GPUMathMode::Fp16MMA  => GPUDataType::Fp16,
    }
  }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum XConvFullShape {
  Conv1d(Conv1dFullShape),
  Conv2d(Conv2dFullShape),
  Conv3d(Conv3dFullShape),
}

impl XConvFullShape {
  pub fn preset_conv2d_nchw(n: usize, c: usize, h: usize, w: usize) -> Self {
    // TODO
    unimplemented!();
  }

  pub fn preset_conv2d_nwhc(n: usize, h: usize, w: usize, c: usize) -> Self {
    // TODO
    unimplemented!();
  }
}

pub type Conv1dFullShape = ConvFullShape<usize, [usize; 3]>;
pub type Conv2dFullShape = ConvFullShape<[usize; 2], [usize; 4]>;
pub type Conv3dFullShape = ConvFullShape<[usize; 3], [usize; 5]>;

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct ConvFullShape<WIdx, XIdx> {
  pub ker_axes: (),
  pub ker_size: WIdx,
  pub stride:   WIdx,
  pub zero_pad: WIdx,
  pub dilation: WIdx,
  pub src_axes: (),
  pub src_size: XIdx,
  pub dst_axes: (),
  pub dst_size: XIdx,
  pub groups:   usize,
  pub cross:    bool,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum GPUDataType {
  Fp32,
  Fp64,
  Fp16,
}

pub trait GPUDataTyped {
  fn gpu_data_ty() -> GPUDataType;
}

impl GPUDataTyped for f32       { fn gpu_data_ty() -> GPUDataType { GPUDataType::Fp32 } }
impl GPUDataTyped for f64       { fn gpu_data_ty() -> GPUDataType { GPUDataType::Fp64 } }
impl GPUDataTyped for f16_stub  { fn gpu_data_ty() -> GPUDataType { GPUDataType::Fp16 } }

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct XConvType {
  pub w_ty: GPUDataType,
  pub x_ty: GPUDataType,
  pub y_ty: GPUDataType,
}

#[derive(Clone, Copy)]
pub enum XGPUConvFwdConfig {
  Cudnn(CudnnGPUConvFwdConfig),
}

#[derive(Clone, Copy)]
pub struct CudnnGPUConvFwdConfig {
  pub algo_desc:    cudnnConvolutionFwdAlgo_t,
  pub workspace:    usize,
}

#[derive(Clone, Copy)]
pub enum XGPUConvBwdWConfig {
  Cudnn(CudnnGPUConvBwdWConfig),
}

#[derive(Clone, Copy)]
pub struct CudnnGPUConvBwdWConfig {
  pub algo_desc:    cudnnConvolutionBwdFilterAlgo_t,
  pub workspace:    usize,
}

#[derive(Clone, Copy)]
pub enum XGPUConvBwdXConfig {
  Cudnn(CudnnGPUConvBwdXConfig),
}

#[derive(Clone, Copy)]
pub struct CudnnGPUConvBwdXConfig {
  pub algo_desc:    cudnnConvolutionBwdDataAlgo_t,
  pub workspace:    usize,
}

pub enum XGPUConvState<WTy, XTy, YTy> {
  Cudnn(CudnnGPUConvState<WTy, XTy, YTy>),
}

pub struct CudnnGPUConvState<WTy, XTy, YTy> {
  kernel_desc:  CudnnFilterDesc<WTy>,
  src_desc:     CudnnTensorDesc<XTy>,
  dst_desc:     CudnnTensorDesc<YTy>,
  conv_desc:    CudnnConvDesc,
}

fn sz2int(sz: usize) -> i32 {
  assert!(sz <= i32::max_value() as usize);
  sz as i32
}

fn query_gpu_workspace_limit() -> usize {
  // TODO: configurable.
  3_000_000_000
}

pub fn query_gpu_conv_fwd_algo<WTy, XTy, YTy>(
    dev: GPUDeviceId,
    maybe_determinism: Option<GPUMathDeterminism>,
    maybe_math_mode: Option<GPUMathMode>,
    conv_shape: XConvFullShape,
    conn: GPUDeviceConn)
-> Option<(XGPUConvFwdConfig, XGPUConvState<WTy, XTy, YTy>)>
where WTy: GPUDataTyped + CudnnDataTypeExt,
      XTy: GPUDataTyped + CudnnDataTypeExt,
      YTy: GPUDataTyped + CudnnDataTypeExt,
{
  let determinism = maybe_determinism.unwrap_or(GPUMathDeterminism::default());
  let math_mode = maybe_math_mode.unwrap_or(GPUMathMode::default());
  let conv_type = XConvType{
    w_ty:   WTy::gpu_data_ty(),
    x_ty:   XTy::gpu_data_ty(),
    y_ty:   YTy::gpu_data_ty(),
  };

  let mut kernel_desc = CudnnFilterDesc::<WTy>::create().unwrap();
  let mut src_desc = CudnnTensorDesc::<XTy>::create().unwrap();
  let mut dst_desc = CudnnTensorDesc::<YTy>::create().unwrap();
  let mut conv_desc = CudnnConvDesc::create().unwrap();
  match conv_shape {
    XConvFullShape::Conv2d(shape) => {
      // TODO: configure tensor layout.
      assert!(kernel_desc.set_4d_nchw(
          sz2int(shape.dst_size[2]),
          sz2int(shape.src_size[2]),
          sz2int(shape.ker_size[1]),
          sz2int(shape.ker_size[0]),
      ).is_ok());
      assert!(src_desc.set_4d_nchw(
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
      assert!(conv_desc.set_2d(
          sz2int(shape.zero_pad[1]),  sz2int(shape.zero_pad[0]),
          sz2int(shape.stride[1]),    sz2int(shape.stride[0]),
          sz2int(shape.dilation[1]),  sz2int(shape.dilation[0]),
          match shape.cross {
            false => cudnnConvolutionMode_t_CUDNN_CONVOLUTION,
            true  => cudnnConvolutionMode_t_CUDNN_CROSS_CORRELATION,
          },
          match math_mode {
            GPUMathMode::Fp32 => f32::cudnn_data_ty(),
            GPUMathMode::Fp64 => f64::cudnn_data_ty(),
            GPUMathMode::Fp16 | GPUMathMode::Fp16MMA => f16_stub::cudnn_data_ty(),
          },
      ).is_ok());
      if shape.groups > 1 {
        assert!(conv_desc.set_group_count(sz2int(shape.groups)).is_ok());
      }
    }
    _ => unimplemented!(),
  }
  match math_mode {
    GPUMathMode::Fp16MMA => {
      assert!(CUDNN_MAJOR >= 7);
      assert_eq!(conv_type.w_ty, GPUDataType::Fp16);
      assert_eq!(conv_type.x_ty, GPUDataType::Fp16);
      assert_eq!(conv_type.y_ty, GPUDataType::Fp16);
      assert!(conv_desc.set_math_type(cudnnMathType_t_CUDNN_TENSOR_OP_MATH).is_ok());
    }
    _ => {
      let math_ty = math_mode.gpu_data_ty();
      assert_eq!(conv_type.w_ty, math_ty);
      assert_eq!(conv_type.x_ty, math_ty);
      assert_eq!(conv_type.y_ty, math_ty);
    }
  }

  //let key = (dev, determinism, math_mode, conv_shape, conv_type);
  let key = XGPUConvKey{
    dev, determinism, math_mode, conv_shape, conv_type,
  };
  {
    let cache = CACHED_CONV_FWD_ALGOS.lock().unwrap();
    if let Some(algo) = cache.get(&key) {
      return Some((algo.clone(), XGPUConvState::Cudnn(CudnnGPUConvState{
        kernel_desc, src_desc, dst_desc, conv_desc,
      })));
    }
  }

  let maybe_algo = {
    let workspace_limit = query_gpu_workspace_limit();
    conn.sync();

    let mut algo_count: i32 = 0;
    let mut algo_results: [cudnnConvolutionFwdAlgoPerf_t; 10] = unsafe { uninitialized() };
    {
      let mut stream = conn.cuda_stream();
      let mut cudnn_h = conn.cudnn();
      assert!(cudnn_h.set_stream(&mut stream).is_ok());
      let status = unsafe { cudnnFindConvolutionForwardAlgorithm(
          cudnn_h.as_mut_ptr(),
          src_desc.as_mut_ptr(),
          kernel_desc.as_mut_ptr(),
          conv_desc.as_mut_ptr(),
          dst_desc.as_mut_ptr(),
          10,
          &mut algo_count as *mut _,
          (&mut algo_results).as_mut_ptr(),
      ) };
      assert_eq!(status, cudnnStatus_t_CUDNN_STATUS_SUCCESS);
      assert!(stream.synchronize().is_ok());
    }
    let mut found_k = None;
    for k in 0 .. algo_count as usize {
      if algo_results[k].status != cudnnStatus_t_CUDNN_STATUS_SUCCESS {
        continue;
      }
      if algo_results[k].memory > workspace_limit {
        continue;
      }
      if determinism == GPUMathDeterminism::Deterministic {
        if algo_results[k].determinism != cudnnDeterminism_t_CUDNN_DETERMINISTIC {
          continue;
        }
      }
      match math_mode {
        GPUMathMode::Fp16MMA => {
          if  algo_results[k].algo != cudnnConvolutionFwdAlgo_t_CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM &&
              algo_results[k].algo != cudnnConvolutionFwdAlgo_t_CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED
          {
            continue;
          }
        }
        _ => {}
      }
      if (CUDNN_MAJOR, CUDNN_MINOR) >= (5, 1) {
        if algo_results[k].algo == cudnnConvolutionFwdAlgo_t_CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING {
          continue;
        }
      }
      if CUDNN_MAJOR < 7 {
        if algo_results[k].algo == cudnnConvolutionFwdAlgo_t_CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED {
          continue;
        }
      }
      found_k = Some(k);
      break;
    }
    found_k.map(|k| XGPUConvFwdConfig::Cudnn(CudnnGPUConvFwdConfig{
      algo_desc:  algo_results[k].algo,
      workspace:  algo_results[k].memory
    }))
  };

  if let Some(ref algo) = maybe_algo {
    let mut cache = CACHED_CONV_FWD_ALGOS.lock().unwrap();
    if !cache.contains_key(&key) {
      cache.insert(key, algo.clone());
    }
  }
  maybe_algo.map(|algo| (algo, XGPUConvState::Cudnn(CudnnGPUConvState{
    kernel_desc, src_desc, dst_desc, conv_desc,
  })))
}

pub fn query_gpu_conv_bwd_w_algo<WTy, XTy, YTy>(dev: GPUDeviceId, maybe_determinism: Option<GPUMathDeterminism>, maybe_math_mode: Option<GPUMathMode>, conv_shape: XConvFullShape, conn: GPUDeviceConn) -> Option<XGPUConvBwdWConfig>
where WTy: GPUDataTyped + CudnnDataTypeExt,
      XTy: GPUDataTyped + CudnnDataTypeExt,
      YTy: GPUDataTyped + CudnnDataTypeExt,
{
  let determinism = maybe_determinism.unwrap_or(GPUMathDeterminism::default());
  let math_mode = maybe_math_mode.unwrap_or(GPUMathMode::default());
  let conv_type = XConvType{
    w_ty:   WTy::gpu_data_ty(),
    x_ty:   XTy::gpu_data_ty(),
    y_ty:   YTy::gpu_data_ty(),
  };

  let mut kernel_desc = CudnnFilterDesc::<WTy>::create().unwrap();
  let mut src_desc = CudnnTensorDesc::<XTy>::create().unwrap();
  let mut dst_desc = CudnnTensorDesc::<YTy>::create().unwrap();
  let mut conv_desc = CudnnConvDesc::create().unwrap();
  match conv_shape {
    XConvFullShape::Conv2d(shape) => {
      // TODO: configure tensor layout.
      assert!(kernel_desc.set_4d_nchw(
          sz2int(shape.dst_size[2]),
          sz2int(shape.src_size[2]),
          sz2int(shape.ker_size[1]),
          sz2int(shape.ker_size[0]),
      ).is_ok());
      assert!(src_desc.set_4d_nchw(
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
      assert!(conv_desc.set_2d(
          sz2int(shape.zero_pad[1]),  sz2int(shape.zero_pad[0]),
          sz2int(shape.stride[1]),    sz2int(shape.stride[0]),
          sz2int(shape.dilation[1]),  sz2int(shape.dilation[0]),
          match shape.cross {
            false => cudnnConvolutionMode_t_CUDNN_CONVOLUTION,
            true  => cudnnConvolutionMode_t_CUDNN_CROSS_CORRELATION,
          },
          match math_mode {
            GPUMathMode::Fp32 => f32::cudnn_data_ty(),
            GPUMathMode::Fp64 => f64::cudnn_data_ty(),
            GPUMathMode::Fp16 | GPUMathMode::Fp16MMA => f16_stub::cudnn_data_ty(),
          },
      ).is_ok());
      if shape.groups > 1 {
        assert!(conv_desc.set_group_count(sz2int(shape.groups)).is_ok());
      }
    }
    _ => unimplemented!(),
  }
  match math_mode {
    GPUMathMode::Fp16MMA => {
      assert!(CUDNN_MAJOR >= 7);
      assert_eq!(conv_type.w_ty, GPUDataType::Fp16);
      assert_eq!(conv_type.x_ty, GPUDataType::Fp16);
      assert_eq!(conv_type.y_ty, GPUDataType::Fp16);
      assert!(conv_desc.set_math_type(cudnnMathType_t_CUDNN_TENSOR_OP_MATH).is_ok());
    }
    _ => {
      let math_ty = math_mode.gpu_data_ty();
      assert_eq!(conv_type.w_ty, math_ty);
      assert_eq!(conv_type.x_ty, math_ty);
      assert_eq!(conv_type.y_ty, math_ty);
    }
  }

  //let key = (dev, determinism, math_mode, conv_shape, conv_type);
  let key = XGPUConvKey{
    dev, determinism, math_mode, conv_shape, conv_type,
  };
  {
    let cache = CACHED_CONV_BWD_W_ALGOS.lock().unwrap();
    if let Some(algo) = cache.get(&key) {
      return Some(algo.clone())
    }
  }

  let maybe_algo = {
    let workspace_limit = query_gpu_workspace_limit();
    conn.sync();

    let mut algo_count: i32 = 0;
    let mut algo_results: [cudnnConvolutionBwdFilterAlgoPerf_t; 10] = unsafe { uninitialized() };
    {
      let mut stream = conn.cuda_stream();
      let mut cudnn_h = conn.cudnn();
      assert!(cudnn_h.set_stream(&mut stream).is_ok());
      let status = unsafe { cudnnFindConvolutionBackwardFilterAlgorithm(
          cudnn_h.as_mut_ptr(),
          src_desc.as_mut_ptr(),
          dst_desc.as_mut_ptr(),
          conv_desc.as_mut_ptr(),
          kernel_desc.as_mut_ptr(),
          10,
          &mut algo_count as *mut _,
          (&mut algo_results).as_mut_ptr(),
      ) };
      assert_eq!(status, cudnnStatus_t_CUDNN_STATUS_SUCCESS);
    }
    let mut found_k = None;
    for k in 0 .. algo_count as usize {
      if algo_results[k].status != cudnnStatus_t_CUDNN_STATUS_SUCCESS {
        continue;
      }
      if algo_results[k].memory > workspace_limit {
        continue;
      }
      if determinism == GPUMathDeterminism::Deterministic {
        if algo_results[k].determinism != cudnnDeterminism_t_CUDNN_DETERMINISTIC {
          continue;
        }
      }
      match math_mode {
        GPUMathMode::Fp16MMA => {
          // TODO: check which bwd filter algos are available w/ mma.
          if  algo_results[k].algo != cudnnConvolutionBwdFilterAlgo_t_CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1 &&
              algo_results[k].algo != cudnnConvolutionBwdFilterAlgo_t_CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED
          {
            continue;
          }
        }
        _ => {}
      }
      /*if (CUDNN_MAJOR, CUDNN_MINOR) >= (5, 1) {
        if algo_results[k].algo == cudnnConvolutionBwdFilterAlgo_t_CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING {
          continue;
        }
      }*/
      if CUDNN_MAJOR < 7 {
        if algo_results[k].algo == cudnnConvolutionBwdFilterAlgo_t_CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED {
          continue;
        }
      }
      found_k = Some(k);
      break;
    }
    found_k.map(|k| XGPUConvBwdWConfig::Cudnn(CudnnGPUConvBwdWConfig{
      algo_desc:  algo_results[k].algo,
      workspace:  algo_results[k].memory
    }))
  };

  if let Some(ref algo) = maybe_algo {
    let mut cache = CACHED_CONV_BWD_W_ALGOS.lock().unwrap();
    if !cache.contains_key(&key) {
      cache.insert(key, algo.clone());
    }
  }
  maybe_algo
}

pub fn query_gpu_conv_bwd_x_algo<WTy, XTy, YTy>(dev: GPUDeviceId, maybe_determinism: Option<GPUMathDeterminism>, maybe_math_mode: Option<GPUMathMode>, conv_shape: XConvFullShape, conn: GPUDeviceConn) -> Option<XGPUConvBwdXConfig>
where WTy: GPUDataTyped + CudnnDataTypeExt,
      XTy: GPUDataTyped + CudnnDataTypeExt,
      YTy: GPUDataTyped + CudnnDataTypeExt,
{
  let determinism = maybe_determinism.unwrap_or(GPUMathDeterminism::default());
  let math_mode = maybe_math_mode.unwrap_or(GPUMathMode::default());
  let conv_type = XConvType{
    w_ty:   WTy::gpu_data_ty(),
    x_ty:   XTy::gpu_data_ty(),
    y_ty:   YTy::gpu_data_ty(),
  };

  let mut kernel_desc = CudnnFilterDesc::<WTy>::create().unwrap();
  let mut src_desc = CudnnTensorDesc::<XTy>::create().unwrap();
  let mut dst_desc = CudnnTensorDesc::<YTy>::create().unwrap();
  let mut conv_desc = CudnnConvDesc::create().unwrap();
  match conv_shape {
    XConvFullShape::Conv2d(shape) => {
      // TODO: configure tensor layout.
      assert!(kernel_desc.set_4d_nchw(
          sz2int(shape.dst_size[2]),
          sz2int(shape.src_size[2]),
          sz2int(shape.ker_size[1]),
          sz2int(shape.ker_size[0]),
      ).is_ok());
      assert!(src_desc.set_4d_nchw(
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
      assert!(conv_desc.set_2d(
          sz2int(shape.zero_pad[1]),  sz2int(shape.zero_pad[0]),
          sz2int(shape.stride[1]),    sz2int(shape.stride[0]),
          sz2int(shape.dilation[1]),  sz2int(shape.dilation[0]),
          match shape.cross {
            false => cudnnConvolutionMode_t_CUDNN_CONVOLUTION,
            true  => cudnnConvolutionMode_t_CUDNN_CROSS_CORRELATION,
          },
          match math_mode {
            GPUMathMode::Fp32 => f32::cudnn_data_ty(),
            GPUMathMode::Fp64 => f64::cudnn_data_ty(),
            GPUMathMode::Fp16 | GPUMathMode::Fp16MMA => f16_stub::cudnn_data_ty(),
          },
      ).is_ok());
      if shape.groups > 1 {
        assert!(conv_desc.set_group_count(sz2int(shape.groups)).is_ok());
      }
    }
    _ => unimplemented!(),
  }
  match math_mode {
    GPUMathMode::Fp16MMA => {
      assert!(CUDNN_MAJOR >= 7);
      assert_eq!(conv_type.w_ty, GPUDataType::Fp16);
      assert_eq!(conv_type.x_ty, GPUDataType::Fp16);
      assert_eq!(conv_type.y_ty, GPUDataType::Fp16);
      assert!(conv_desc.set_math_type(cudnnMathType_t_CUDNN_TENSOR_OP_MATH).is_ok());
    }
    _ => {
      let math_ty = math_mode.gpu_data_ty();
      assert_eq!(conv_type.w_ty, math_ty);
      assert_eq!(conv_type.x_ty, math_ty);
      assert_eq!(conv_type.y_ty, math_ty);
    }
  }

  //let key = (dev, determinism, math_mode, conv_shape, conv_type);
  let key = XGPUConvKey{
    dev, determinism, math_mode, conv_shape, conv_type,
  };
  {
    let cache = CACHED_CONV_BWD_X_ALGOS.lock().unwrap();
    if let Some(algo) = cache.get(&key) {
      return Some(algo.clone())
    }
  }

  let maybe_algo = {
    let workspace_limit = query_gpu_workspace_limit();
    conn.sync();

    let mut algo_count: i32 = 0;
    let mut algo_results: [cudnnConvolutionBwdDataAlgoPerf_t; 10] = unsafe { uninitialized() };
    {
      let mut stream = conn.cuda_stream();
      let mut cudnn_h = conn.cudnn();
      assert!(cudnn_h.set_stream(&mut stream).is_ok());
      let status = unsafe { cudnnFindConvolutionBackwardDataAlgorithm(
          cudnn_h.as_mut_ptr(),
          kernel_desc.as_mut_ptr(),
          dst_desc.as_mut_ptr(),
          conv_desc.as_mut_ptr(),
          src_desc.as_mut_ptr(),
          10,
          &mut algo_count as *mut _,
          (&mut algo_results).as_mut_ptr(),
      ) };
      assert_eq!(status, cudnnStatus_t_CUDNN_STATUS_SUCCESS);
    }
    let mut found_k = None;
    for k in 0 .. algo_count as usize {
      if algo_results[k].status != cudnnStatus_t_CUDNN_STATUS_SUCCESS {
        continue;
      }
      if algo_results[k].memory > workspace_limit {
        continue;
      }
      if determinism == GPUMathDeterminism::Deterministic {
        if algo_results[k].determinism != cudnnDeterminism_t_CUDNN_DETERMINISTIC {
          continue;
        }
      }
      match math_mode {
        GPUMathMode::Fp16MMA => {
          // TODO: check which bwd data algos are available w/ mma.
          if  algo_results[k].algo != cudnnConvolutionBwdDataAlgo_t_CUDNN_CONVOLUTION_BWD_DATA_ALGO_1 &&
              algo_results[k].algo != cudnnConvolutionBwdDataAlgo_t_CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED
          {
            continue;
          }
        }
        _ => {}
      }
      /*if (CUDNN_MAJOR, CUDNN_MINOR) >= (5, 1) {
        if algo_results[k].algo == cudnnConvolutionBwdDataAlgo_t_CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING {
          continue;
        }
      }*/
      if CUDNN_MAJOR < 7 {
        if algo_results[k].algo == cudnnConvolutionBwdDataAlgo_t_CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED {
          continue;
        }
      }
      found_k = Some(k);
      break;
    }
    found_k.map(|k| XGPUConvBwdXConfig::Cudnn(CudnnGPUConvBwdXConfig{
      algo_desc:  algo_results[k].algo,
      workspace:  algo_results[k].memory
    }))
  };

  if let Some(ref algo) = maybe_algo {
    let mut cache = CACHED_CONV_BWD_X_ALGOS.lock().unwrap();
    if !cache.contains_key(&key) {
      cache.insert(key, algo.clone());
    }
  }
  maybe_algo
}

pub fn gpu_batch_conv2d<WTy, XTy, YTy>(
    cfg: &XGPUConvFwdConfig,
    state: &mut XGPUConvState<WTy, XTy, YTy>,
    w: GPUDeviceArrayView4d<WTy>,
    x: GPUDeviceArrayView4d<XTy>,
    y: GPUDeviceArrayViewMut4d<YTy>,
    workspace: GPUDeviceArrayViewMut1d<u8>,
    conn: GPUDeviceConn)
where /*WTy: GPUDataTyped + CudnnDataTypeExt,
      XTy: GPUDataTyped + CudnnDataTypeExt,
      YTy: GPUDataTyped + CudnnDataTypeExt,*/
      WTy: Copy,
      XTy: Copy,
      YTy: Copy,
      CudnnHandle: CudnnConvExt<WTy, XTy, YTy>,
      <CudnnHandle as CudnnConvExt<WTy, XTy, YTy>>::HostScalar: PseudoField,
{
  match (cfg, state) {
    (&XGPUConvFwdConfig::Cudnn(ref cfg), &mut XGPUConvState::Cudnn(ref mut state)) => {
      let mut stream = conn.cuda_stream();
      let mut cudnn_h = conn.cudnn();
      assert!(cudnn_h.set_stream(&mut stream).is_ok());
      /*// TODO: alpha beta types.
      let alpha: f32 = 1.0;
      let beta: f32 = 0.0;
      let status = unsafe { cudnnConvolutionForward(
          cudnn_h.as_mut_ptr(),
          &alpha as *const _ as *const _,
          state.src_desc.as_mut_ptr(),
          x.as_dptr() as *const _,
          state.kernel_desc.as_mut_ptr(),
          w.as_dptr() as *const _,
          state.conv_desc.as_mut_ptr(),
          cfg.algo_desc,
          workspace.as_mut_dptr() as *mut _,
          workspace.size(),
          &beta as *const _ as *const _,
          state.dst_desc.as_mut_ptr(),
          y.as_mut_dptr() as *mut _,
      ) };
      assert_eq!(status, cudnnStatus_t_CUDNN_STATUS_SUCCESS);*/
      let alpha: <CudnnHandle as CudnnConvExt<WTy, XTy, YTy>>::HostScalar = PseudoField::one();
      let beta: <CudnnHandle as CudnnConvExt<WTy, XTy, YTy>>::HostScalar = PseudoRing::zero();
      let status = unsafe { cudnn_h.conv_fwd(
          alpha,
          &mut state.src_desc,
          x.as_dptr(),
          &mut state.kernel_desc,
          w.as_dptr(),
          &mut state.conv_desc,
          cfg.algo_desc,
          workspace.as_mut_dptr(),
          workspace.size(),
          beta,
          &mut state.dst_desc,
          y.as_mut_dptr(),
      ) };
      assert!(status.is_ok());
    }
    _ => unimplemented!(),
  }
}

pub fn gpu_batch_left_transpose_conv2d<WTy, XTy, YTy>(
    cfg: &XGPUConvBwdXConfig,
    state: &mut XGPUConvState<WTy, XTy, YTy>,
    w: GPUDeviceArrayView4d<WTy>,
    y: GPUDeviceArrayView4d<YTy>,
    x: GPUDeviceArrayViewMut4d<XTy>,
    workspace: GPUDeviceArrayViewMut1d<u8>,
    conn: GPUDeviceConn)
where /*WTy: GPUDataTyped + CudnnDataTypeExt,
      XTy: GPUDataTyped + CudnnDataTypeExt,
      YTy: GPUDataTyped + CudnnDataTypeExt,*/
      WTy: Copy,
      XTy: Copy,
      YTy: Copy,
      CudnnHandle: CudnnConvExt<WTy, XTy, YTy>,
      <CudnnHandle as CudnnConvExt<WTy, XTy, YTy>>::HostScalar: PseudoField,
{
  match (cfg, state) {
    (&XGPUConvBwdXConfig::Cudnn(ref cfg), &mut XGPUConvState::Cudnn(ref mut state)) => {
      let mut stream = conn.cuda_stream();
      let mut cudnn_h = conn.cudnn();
      assert!(cudnn_h.set_stream(&mut stream).is_ok());
      /*// TODO: alpha beta types.
      let alpha: f32 = 1.0;
      let beta: f32 = 0.0;
      let status = unsafe { cudnnConvolutionBackwardData(
          cudnn_h.as_mut_ptr(),
          &alpha as *const _ as *const _,
          state.kernel_desc.as_mut_ptr(),
          w.as_dptr() as *const _,
          state.dst_desc.as_mut_ptr(),
          y.as_dptr() as *const _,
          state.conv_desc.as_mut_ptr(),
          cfg.algo_desc,
          workspace.as_mut_dptr() as *mut _,
          workspace.size(),
          &beta as *const _ as *const _,
          state.src_desc.as_mut_ptr(),
          x.as_mut_dptr() as *mut _,
      ) };
      assert_eq!(status, cudnnStatus_t_CUDNN_STATUS_SUCCESS);*/
      let alpha: <CudnnHandle as CudnnConvExt<WTy, XTy, YTy>>::HostScalar = PseudoField::one();
      let beta: <CudnnHandle as CudnnConvExt<WTy, XTy, YTy>>::HostScalar = PseudoRing::zero();
      let status = unsafe { cudnn_h.conv_bwd_data(
          alpha,
          &mut state.kernel_desc,
          w.as_dptr(),
          &mut state.dst_desc,
          y.as_dptr(),
          &mut state.conv_desc,
          cfg.algo_desc,
          workspace.as_mut_dptr(),
          workspace.size(),
          beta,
          &mut state.src_desc,
          x.as_mut_dptr(),
      ) };
      assert!(status.is_ok());
    }
    _ => unimplemented!(),
  }
}

pub fn gpu_batch_right_transpose_conv2d<WTy, XTy, YTy>(
    cfg: &XGPUConvBwdWConfig,
    state: &mut XGPUConvState<WTy, XTy, YTy>,
    y: GPUDeviceArrayView4d<YTy>,
    x: GPUDeviceArrayView4d<XTy>,
    w: GPUDeviceArrayViewMut4d<WTy>,
    workspace: GPUDeviceArrayViewMut1d<u8>,
    conn: GPUDeviceConn)
where /*WTy: GPUDataTyped + CudnnDataTypeExt,
      XTy: GPUDataTyped + CudnnDataTypeExt,
      YTy: GPUDataTyped + CudnnDataTypeExt,*/
      WTy: Copy,
      XTy: Copy,
      YTy: Copy,
      CudnnHandle: CudnnConvExt<WTy, XTy, YTy>,
      <CudnnHandle as CudnnConvExt<WTy, XTy, YTy>>::HostScalar: PseudoField,
{
  match (cfg, state) {
    (&XGPUConvBwdWConfig::Cudnn(ref cfg), &mut XGPUConvState::Cudnn(ref mut state)) => {
      let mut stream = conn.cuda_stream();
      let mut cudnn_h = conn.cudnn();
      assert!(cudnn_h.set_stream(&mut stream).is_ok());
      /*// TODO: alpha beta types.
      let alpha: f32 = 1.0;
      let beta: f32 = 0.0;
      let status = unsafe { cudnnConvolutionBackwardFilter(
          cudnn_h.as_mut_ptr(),
          &alpha as *const _ as *const _,
          state.src_desc.as_mut_ptr(),
          x.as_dptr() as *const _,
          state.dst_desc.as_mut_ptr(),
          y.as_dptr() as *const _,
          state.conv_desc.as_mut_ptr(),
          cfg.algo_desc,
          workspace.as_mut_dptr() as *mut _,
          workspace.size(),
          &beta as *const _ as *const _,
          state.kernel_desc.as_mut_ptr(),
          w.as_mut_dptr() as *mut _,
      ) };
      assert_eq!(status, cudnnStatus_t_CUDNN_STATUS_SUCCESS);*/
      let alpha: <CudnnHandle as CudnnConvExt<WTy, XTy, YTy>>::HostScalar = PseudoField::one();
      let beta: <CudnnHandle as CudnnConvExt<WTy, XTy, YTy>>::HostScalar = PseudoRing::zero();
      let status = unsafe { cudnn_h.conv_bwd_filter(
          alpha,
          &mut state.src_desc,
          x.as_dptr(),
          &mut state.dst_desc,
          y.as_dptr(),
          &mut state.conv_desc,
          cfg.algo_desc,
          workspace.as_mut_dptr(),
          workspace.size(),
          beta,
          &mut state.kernel_desc,
          w.as_mut_dptr(),
      ) };
      assert!(status.is_ok());
    }
    _ => unimplemented!(),
  }
}
