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

use ::{GPUDeviceConn};
use ::array::*;
use ::ffi::routines_gpu::*;

use cuda_blas::*;

#[inline]
fn sz2int(sz: usize) -> i32 {
  assert!(sz <= i32::max_value() as _);
  sz as _
}

#[inline]
fn sz2uint(sz: usize) -> u32 {
  assert!(sz <= u32::max_value() as _);
  sz as _
}

pub trait GPUVectorOps<T> where T: Copy {
  fn sync_vector_norm(&self, conn: GPUDeviceConn) -> T;
}

pub trait GPUVectorMutOps<T> where T: Copy {
  fn reduce_sum(&mut self, x: GPUDeviceArrayView2d<T>, axis: isize, conn: GPUDeviceConn);

  fn matrix_vector_mult(&mut self,
      w: GPUDeviceArrayView2d<T>,
      x: GPUDeviceArrayView1d<T>,
      conn: GPUDeviceConn);
}

impl GPUVectorOps<f32> for GPUDeviceArrayView1d<f32> {
  fn sync_vector_norm(&self, conn: GPUDeviceConn) -> f32 {
    let mut stream = conn.cuda_stream();
    let mut cublas_h = conn.cublas();
    assert!(cublas_h.set_stream(&mut stream).is_ok());
    assert!(cublas_h.set_pointer_mode(CublasPointerMode::Host).is_ok());
    assert!(cublas_h.set_atomics_mode(CublasAtomicsMode::NotAllowed).is_ok());
    #[cfg(feature = "cuda9")] {
      assert!(cublas_h.set_math_mode(CublasMathMode::Default).is_ok());
    }
    let mut result: f32 = 0.0;
    let status = unsafe { cublas_h.nrm2(
        sz2int(self.size()),
        self.raw_dptr(), sz2int(self.stride()),
        &mut result as *mut _,
    ) };
    assert!(status.is_ok());
    result
  }
}

impl GPUVectorMutOps<f32> for GPUDeviceArrayViewMut1d<f32> {
  fn reduce_sum(&mut self, x: GPUDeviceArrayView2d<f32>, axis: isize, conn: GPUDeviceConn) {
    if self.is_packed() {
      let mut stream = conn.cuda_stream();
      match axis {
        0 => {
          assert_eq!(self.size(), x.size()[1]);
          unsafe { gpudevicemem_sum_I1ab_Ob_packed_deterministic_f32(
              sz2uint(x.size()[0]),
              sz2uint(x.size()[1]),
              x.raw_dptr(),
              self.raw_mut_dptr(),
              conn.cuda_kernel_cfg() as *const _,
              stream.as_mut_ptr(),
          ) };
        }
        1 => {
          assert_eq!(self.size(), x.size()[0]);
          unsafe { gpudevicemem_sum_I1ab_Oa_packed_deterministic_f32(
              sz2uint(x.size()[0]),
              sz2uint(x.size()[1]),
              x.raw_dptr(),
              self.raw_mut_dptr(),
              conn.cuda_kernel_cfg() as *const _,
              stream.as_mut_ptr(),
          ) };
        }
        _ => unreachable!(),
      }
    } else {
      unimplemented!();
    }
  }

  fn matrix_vector_mult(&mut self,
      w: GPUDeviceArrayView2d<f32>,
      x: GPUDeviceArrayView1d<f32>,
      conn: GPUDeviceConn)
  {
    let mut stream = conn.cuda_stream();
    let mut cublas_h = conn.cublas();
    assert!(cublas_h.set_stream(&mut stream).is_ok());
    assert!(cublas_h.set_pointer_mode(CublasPointerMode::Host).is_ok());
    assert!(cublas_h.set_atomics_mode(CublasAtomicsMode::NotAllowed).is_ok());
    #[cfg(feature = "cuda9")] {
      assert!(cublas_h.set_math_mode(CublasMathMode::Default).is_ok());
    }
    assert_eq!(w.size()[0], self.size());
    assert_eq!(w.size()[1], x.size());
    assert_eq!(w.stride()[0], 1);
    let alpha: f32 = 1.0;
    let beta: f32 = 0.0;
    let status = unsafe { cublas_h.gemv(
        CublasTranspose::N,
        sz2int(w.size()[0]),
        sz2int(w.size()[1]),
        &alpha,
        w.raw_dptr(), sz2int(w.stride()[1]),
        x.raw_dptr(), sz2int(x.stride()),
        &beta,
        self.raw_mut_dptr(), sz2int(self.stride()),
    ) };
    assert!(status.is_ok());
  }
}

pub fn gpu_matrix_vector_mult<T>(
    w: GPUDeviceArrayView2d<T>,
    x: GPUDeviceArrayView1d<T>,
    mut y: GPUDeviceArrayViewMut1d<T>,
    conn: GPUDeviceConn)
where T: Copy,
      GPUDeviceArrayViewMut1d<T>: GPUVectorMutOps<T>
{
  y.matrix_vector_mult(w, x, conn);
}

pub trait GPUMatrixOps<T> where T: Copy {
  fn broadcast_add_vector_inplace(&mut self, x: GPUDeviceArrayView1d<T>, axis: isize, conn: GPUDeviceConn);

  fn matrix_mult(&mut self,
      w: GPUDeviceArrayView2d<T>,
      x: GPUDeviceArrayView2d<T>,
      conn: GPUDeviceConn);
  fn left_transpose_matrix_mult(&mut self,
      w: GPUDeviceArrayView2d<T>,
      y: GPUDeviceArrayView2d<T>,
      conn: GPUDeviceConn);
  fn right_transpose_matrix_mult(&mut self,
      y: GPUDeviceArrayView2d<T>,
      x: GPUDeviceArrayView2d<T>,
      conn: GPUDeviceConn);
}

impl GPUMatrixOps<f32> for GPUDeviceArrayViewMut2d<f32> {
  fn broadcast_add_vector_inplace(&mut self, x: GPUDeviceArrayView1d<f32>, axis: isize, conn: GPUDeviceConn) {
    if self.is_packed() && x.is_packed() {
      let mut stream = conn.cuda_stream();
      match axis {
        0 => {
          assert_eq!(x.size(), self.size()[0]);
          unsafe { gpudevicemem_bcast_flat_add_I1a_IO2ab_inplace_packed_f32(
              sz2uint(self.size()[0]),
              sz2uint(self.size()[1]),
              x.raw_dptr(),
              self.raw_mut_dptr(),
              conn.cuda_kernel_cfg() as *const _,
              stream.as_mut_ptr(),
          ) };
        }
        1 => {
          // TODO
          unimplemented!();
        }
        _ => unreachable!(),
      }
    } else {
      unimplemented!();
    }
  }

  fn matrix_mult(&mut self,
      w: GPUDeviceArrayView2d<f32>,
      x: GPUDeviceArrayView2d<f32>,
      conn: GPUDeviceConn)
  {
    let mut stream = conn.cuda_stream();
    let mut cublas_h = conn.cublas();
    assert!(cublas_h.set_stream(&mut stream).is_ok());
    assert!(cublas_h.set_pointer_mode(CublasPointerMode::Host).is_ok());
    assert!(cublas_h.set_atomics_mode(CublasAtomicsMode::NotAllowed).is_ok());
    #[cfg(feature = "cuda9")] {
      assert!(cublas_h.set_math_mode(CublasMathMode::Default).is_ok());
    }
    assert_eq!(w.size()[0], self.size()[0]);
    assert_eq!(w.size()[1], x.size()[0]);
    assert_eq!(x.size()[1], self.size()[1]);
    assert_eq!(w.stride()[0], 1);
    assert_eq!(x.stride()[0], 1);
    assert_eq!(self.stride()[0], 1);
    let alpha: f32 = 1.0;
    let beta: f32 = 0.0;
    let status = unsafe { cublas_h.gemm(
        CublasTranspose::N,
        CublasTranspose::N,
        sz2int(w.size()[0]),
        sz2int(x.size()[1]),
        sz2int(w.size()[1]),
        &alpha,
        w.raw_dptr(), sz2int(w.stride()[1]),
        x.raw_dptr(), sz2int(x.stride()[1]),
        &beta,
        self.raw_mut_dptr(), sz2int(self.stride()[1]),
    ) };
    assert!(status.is_ok());
  }

  fn left_transpose_matrix_mult(&mut self,
      w: GPUDeviceArrayView2d<f32>,
      y: GPUDeviceArrayView2d<f32>,
      conn: GPUDeviceConn)
  {
    let mut stream = conn.cuda_stream();
    let mut cublas_h = conn.cublas();
    assert!(cublas_h.set_stream(&mut stream).is_ok());
    assert!(cublas_h.set_pointer_mode(CublasPointerMode::Host).is_ok());
    assert!(cublas_h.set_atomics_mode(CublasAtomicsMode::NotAllowed).is_ok());
    #[cfg(feature = "cuda9")] {
      assert!(cublas_h.set_math_mode(CublasMathMode::Default).is_ok());
    }
    assert_eq!(w.size()[1], self.size()[0]);
    assert_eq!(w.size()[0], y.size()[0]);
    assert_eq!(y.size()[1], self.size()[1]);
    assert_eq!(w.stride()[0], 1);
    assert_eq!(y.stride()[0], 1);
    assert_eq!(self.stride()[0], 1);
    let alpha: f32 = 1.0;
    let beta: f32 = 0.0;
    let status = unsafe { cublas_h.gemm(
        CublasTranspose::T,
        CublasTranspose::N,
        sz2int(w.size()[1]),
        sz2int(y.size()[1]),
        sz2int(w.size()[0]),
        &alpha,
        w.raw_dptr(), sz2int(w.stride()[1]),
        y.raw_dptr(), sz2int(y.stride()[1]),
        &beta,
        self.raw_mut_dptr(), sz2int(self.stride()[1]),
    ) };
    assert!(status.is_ok());
  }

  fn right_transpose_matrix_mult(&mut self,
      y: GPUDeviceArrayView2d<f32>,
      x: GPUDeviceArrayView2d<f32>,
      conn: GPUDeviceConn)
  {
    let mut stream = conn.cuda_stream();
    let mut cublas_h = conn.cublas();
    assert!(cublas_h.set_stream(&mut stream).is_ok());
    assert!(cublas_h.set_pointer_mode(CublasPointerMode::Host).is_ok());
    assert!(cublas_h.set_atomics_mode(CublasAtomicsMode::NotAllowed).is_ok());
    #[cfg(feature = "cuda9")] {
      assert!(cublas_h.set_math_mode(CublasMathMode::Default).is_ok());
    }
    assert_eq!(y.size()[0], self.size()[0]);
    assert_eq!(y.size()[1], x.size()[1]);
    assert_eq!(x.size()[0], self.size()[1]);
    assert_eq!(y.stride()[0], 1);
    assert_eq!(x.stride()[0], 1);
    assert_eq!(self.stride()[0], 1);
    let alpha: f32 = 1.0;
    let beta: f32 = 0.0;
    let status = unsafe { cublas_h.gemm(
        CublasTranspose::N,
        CublasTranspose::T,
        sz2int(y.size()[0]),
        sz2int(x.size()[0]),
        sz2int(y.size()[1]),
        &alpha,
        y.raw_dptr(), sz2int(y.stride()[1]),
        x.raw_dptr(), sz2int(x.stride()[1]),
        &beta,
        self.raw_mut_dptr(), sz2int(self.stride()[1]),
    ) };
    assert!(status.is_ok());
  }
}

/*pub fn gpu_matrix_mult<T>(
    w: GPUDeviceArrayView2d<T>,
    x: GPUDeviceArrayView2d<T>,
    mut y: GPUDeviceArrayViewMut2d<T>,
    conn: GPUDeviceConn)
where T: Copy,
      GPUDeviceArrayViewMut2d<T>: GPUMatrixOps<T>
{
  y.matrix_mult(w, x, conn);
}

pub fn gpu_left_transpose_matrix_mult<T>(
    w: GPUDeviceArrayView2d<T>,
    y: GPUDeviceArrayView2d<T>,
    mut x: GPUDeviceArrayViewMut2d<T>,
    conn: GPUDeviceConn)
where T: Copy,
      GPUDeviceArrayViewMut2d<T>: GPUMatrixOps<T>
{
  x.left_transpose_matrix_mult(w, y, conn);
}

pub fn gpu_right_transpose_matrix_mult<T>(
    y: GPUDeviceArrayView2d<T>,
    x: GPUDeviceArrayView2d<T>,
    mut w: GPUDeviceArrayViewMut2d<T>,
    conn: GPUDeviceConn)
where T: Copy,
      GPUDeviceArrayViewMut2d<T>: GPUMatrixOps<T>
{
  w.right_transpose_matrix_mult(y, x, conn);
}*/
