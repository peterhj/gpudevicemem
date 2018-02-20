use ::{GPUDeviceId, GPUDeviceConn};

use cuda_dnn::*;
use cuda_dnn::ffi::*;
use float::stub::*;

use std::collections::{HashMap};
use std::mem::{uninitialized};
use std::ptr::{null, null_mut};
use std::sync::{Arc, Mutex};

lazy_static! {
  static ref CACHED_CONV_FWD_ALGOS: Arc<Mutex<HashMap<(GPUDeviceId, GPUMathDeterminism, GPUMathMode, XConvFullShape, XConvType), XGPUConvFwdAlgo>>> = {
    Arc::new(Mutex::new(HashMap::new()))
  };
  static ref CACHED_CONV_BWD_W_ALGOS: Arc<Mutex<HashMap<(GPUDeviceId, GPUMathDeterminism, GPUMathMode, XConvFullShape, XConvType), XGPUConvBwdWAlgo>>> = {
    Arc::new(Mutex::new(HashMap::new()))
  };
  static ref CACHED_CONV_BWD_X_ALGOS: Arc<Mutex<HashMap<(GPUDeviceId, GPUMathDeterminism, GPUMathMode, XConvFullShape, XConvType), XGPUConvBwdXAlgo>>> = {
    Arc::new(Mutex::new(HashMap::new()))
  };
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
  pub fn gpu_ty(&self) -> GPUType {
    match *self {
      GPUMathMode::Fp32     => GPUType::Fp32,
      GPUMathMode::Fp64     => GPUType::Fp64,
      GPUMathMode::Fp16     |
      GPUMathMode::Fp16MMA  => GPUType::Fp16,
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
  pub fn preset_cudnn_nchw(n: usize, c: usize, h: usize, w: usize) -> Self {
    // TODO
    unimplemented!();
  }

  pub fn preset_cudnn_nwhc(n: usize, h: usize, w: usize, c: usize) -> Self {
    // TODO
    unimplemented!();
  }
}

pub type Conv1dFullShape = ConvFullShape<usize, [usize; 3]>;
pub type Conv2dFullShape = ConvFullShape<[usize; 2], [usize; 4]>;
pub type Conv3dFullShape = ConvFullShape<[usize; 3], [usize; 5]>;

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct ConvFullShape<WIdx, XIdx> {
  pub kernel:   WIdx,
  pub stride:   WIdx,
  pub zero_pad: WIdx,
  pub dilation: WIdx,
  pub ker_axes: (),
  pub src_axes: (),
  pub src_size: XIdx,
  pub dst_axes: (),
  pub dst_size: XIdx,
  pub blocks:   usize,
  pub cross:    bool,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum GPUType {
  Fp32,
  Fp64,
  Fp16,
}

pub trait GPUTyped {
  fn gpu_ty() -> GPUType;
}

impl GPUTyped for f32       { fn gpu_ty() -> GPUType { GPUType::Fp32 } }
impl GPUTyped for f64       { fn gpu_ty() -> GPUType { GPUType::Fp64 } }
impl GPUTyped for f16_stub  { fn gpu_ty() -> GPUType { GPUType::Fp16 } }

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct XConvType {
  pub w_ty: GPUType,
  pub x_ty: GPUType,
  pub y_ty: GPUType,
}

#[derive(Clone, Copy)]
pub enum XGPUConvFwdAlgo {
  Cudnn(CudnnGPUConvFwdAlgo),
}

#[derive(Clone, Copy)]
pub struct CudnnGPUConvFwdAlgo {
  // TODO
  pub desc:         cudnnConvolutionFwdAlgo_t,
  pub workspace:    usize,
}

#[derive(Clone, Copy)]
pub enum XGPUConvBwdWAlgo {
  Cudnn(CudnnGPUConvBwdXAlgo),
}

#[derive(Clone, Copy)]
pub struct CudnnGPUConvBwdWAlgo {
  // TODO
  pub desc:         cudnnConvolutionBwdFilterAlgo_t,
  pub workspace:    usize,
}

#[derive(Clone, Copy)]
pub enum XGPUConvBwdXAlgo {
  Cudnn(CudnnGPUConvBwdXAlgo),
}

#[derive(Clone, Copy)]
pub struct CudnnGPUConvBwdXAlgo {
  // TODO
  pub desc:         cudnnConvolutionBwdDataAlgo_t,
  pub workspace:    usize,
}

fn sz2int(sz: usize) -> i32 {
  assert!(sz <= i32::max_value() as usize);
  sz as i32
}

pub fn query_workspace_limit() -> usize {
  // TODO: configurable.
  3_000_000_000
}

pub fn query_conv_fwd_algo<WTy, XTy, YTy>(
    dev: GPUDeviceId,
    maybe_determinism: Option<GPUMathDeterminism>,
    maybe_math_mode: Option<GPUMathMode>,
    conv_shape: XConvFullShape,
    conn: GPUDeviceConn)
-> Option<XGPUConvFwdAlgo>
where WTy: GPUTyped + CudnnDataTypeExt,
      XTy: GPUTyped + CudnnDataTypeExt,
      YTy: GPUTyped + CudnnDataTypeExt,
{
  let determinism = maybe_determinism.unwrap_or(GPUMathDeterminism::default());
  let math_mode = maybe_math_mode.unwrap_or(GPUMathMode::default());
  let conv_type = XConvType{
    w_ty:   WTy::gpu_ty(),
    x_ty:   XTy::gpu_ty(),
    y_ty:   YTy::gpu_ty(),
  };
  let key = (dev, determinism, math_mode, conv_shape, conv_type);
  {
    let cache = CACHED_CONV_FWD_ALGOS.lock().unwrap();
    if let Some(algo) = cache.get(&key) {
      return Some(algo.clone())
    }
  }

  let workspace_limit = query_workspace_limit();
  conn.sync();

  let mut cudnn_h = conn.cudnn();
  let kernel_desc = CudnnFilterDesc::<WTy>::create().unwrap();
  let src_desc = CudnnTensorDesc::<XTy>::create().unwrap();
  let dst_desc = CudnnTensorDesc::<YTy>::create().unwrap();
  let conv_desc = CudnnConvDesc::create().unwrap();
  match conv_shape {
    XConvFullShape::Conv2d(shape) => {
      // TODO: configure tensor layout.
      assert!(kernel_desc.set_4d_nchw(
          sz2int(shape.dst_size[2]),
          sz2int(shape.src_size[2]),
          sz2int(shape.kernel[1]),
          sz2int(shape.kernel[0]),
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
      if shape.blocks > 1 {
        assert!(conv_desc.set_group_count(sz2int(shape.blocks)).is_ok());
      }
    }
    _ => unimplemented!(),
  }
  match math_mode {
    GPUMathMode::Fp16MMA => {
      assert!(CUDNN_MAJOR >= 7);
      assert_eq!(conv_type.w_ty, GPUType::Fp16);
      assert_eq!(conv_type.x_ty, GPUType::Fp16);
      assert_eq!(conv_type.y_ty, GPUType::Fp16);
      assert!(conv_desc.set_math_type(cudnnMathType_t_CUDNN_TENSOR_OP_MATH).is_ok());
    }
    _ => {
      let math_ty = math_mode.gpu_ty();
      assert_eq!(conv_type.w_ty, math_ty);
      assert_eq!(conv_type.x_ty, math_ty);
      assert_eq!(conv_type.y_ty, math_ty);
    }
  }
  let maybe_algo = {
    let mut algo_count: i32 = 0;
    let mut algo_results: [cudnnConvolutionFwdAlgoPerf_t; 10] = unsafe { uninitialized() };
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
    found_k.map(|k| XGPUConvFwdAlgo::Cudnn(CudnnGPUConvFwdAlgo{
      desc:       algo_results[k].algo,
      workspace:  algo_results[k].memory
    }))
  };

  if let Some(ref algo) = maybe_algo {
    let mut cache = CACHED_CONV_FWD_ALGOS.lock().unwrap();
    cache.insert(key, algo.clone());
  }
  maybe_algo
}

pub fn query_conv_bwd_w_algo<WTy, XTy, YTy>(dev: GPUDeviceId, maybe_determinism: Option<GPUMathDeterminism>, maybe_math_mode: Option<GPUMathMode>, conv_shape: XConvFullShape, conn: GPUDeviceConn) -> Option<XGPUConvBwdWAlgo>
where WTy: GPUTyped + CudnnDataTypeExt,
      XTy: GPUTyped + CudnnDataTypeExt,
      YTy: GPUTyped + CudnnDataTypeExt,
{
  let determinism = maybe_determinism.unwrap_or(GPUMathDeterminism::default());
  let math_mode = maybe_math_mode.unwrap_or(GPUMathMode::default());
  let conv_type = XConvType{
    w_ty:   WTy::gpu_ty(),
    x_ty:   XTy::gpu_ty(),
    y_ty:   YTy::gpu_ty(),
  };
  let key = (dev, determinism, math_mode, conv_shape, conv_type);
  {
    let cache = CACHED_CONV_BWD_W_ALGOS.lock().unwrap();
    if let Some(algo) = cache.get(&key) {
      return Some(algo.clone())
    }
  }
  // TODO: find conv algo.
  unimplemented!();
}

pub fn query_conv_bwd_x_algo<WTy, XTy, YTy>(dev: GPUDeviceId, maybe_determinism: Option<GPUMathDeterminism>, maybe_math_mode: Option<GPUMathMode>, conv_shape: XConvFullShape, conn: GPUDeviceConn) -> Option<XGPUConvBwdXAlgo>
where WTy: GPUTyped + CudnnDataTypeExt,
      XTy: GPUTyped + CudnnDataTypeExt,
      YTy: GPUTyped + CudnnDataTypeExt,
{
  let determinism = maybe_determinism.unwrap_or(GPUMathDeterminism::default());
  let math_mode = maybe_math_mode.unwrap_or(GPUMathMode::default());
  let conv_type = XConvType{
    w_ty:   WTy::gpu_ty(),
    x_ty:   XTy::gpu_ty(),
    y_ty:   YTy::gpu_ty(),
  };
  let key = (dev, determinism, math_mode, conv_shape, conv_type);
  {
    let cache = CACHED_CONV_BWD_X_ALGOS.lock().unwrap();
    if let Some(algo) = cache.get(&key) {
      return Some(algo.clone())
    }
  }
  // TODO: find conv algo.
  unimplemented!();
}
