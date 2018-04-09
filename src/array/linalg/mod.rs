use ::{GPUDeviceConn};
use ::array::*;

use cuda_blas::*;

pub mod conv;

#[inline]
fn sz2int(sz: usize) -> i32 {
  assert!(sz <= i32::max_value() as _);
  sz as _
}

pub trait GPUVectorOps<T> where T: Copy {
  fn matrix_vector_mult(&mut self,
      w: GPUDeviceArrayView2d<T>,
      x: GPUDeviceArrayView1d<T>,
      conn: GPUDeviceConn);
}

impl GPUVectorOps<f32> for GPUDeviceArrayViewMut1d<f32> {
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
    assert!(cublas_h.set_math_mode(CublasMathMode::Default).is_ok());
    assert_eq!(w.size()[0], self.size());
    assert_eq!(w.size()[1], x.size());
    let alpha: f32 = 1.0;
    let beta: f32 = 0.0;
    let status = unsafe { cublas_h.gemv(
        CublasTranspose::N,
        sz2int(w.size()[0]),
        sz2int(w.size()[1]),
        &alpha,
        w.as_dptr(), sz2int(w.stride()[1]),
        x.as_dptr(), sz2int(x.stride()),
        &beta,
        self.as_mut_dptr(), sz2int(self.stride()),
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
      GPUDeviceArrayViewMut1d<T>: GPUVectorOps<T>
{
  y.matrix_vector_mult(w, x, conn);
}

pub trait GPUMatrixOps<T> where T: Copy {
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
    assert!(cublas_h.set_math_mode(CublasMathMode::Default).is_ok());
    assert_eq!(w.size()[0], self.size()[0]);
    assert_eq!(w.size()[1], x.size()[0]);
    assert_eq!(x.size()[1], self.size()[1]);
    let alpha: f32 = 1.0;
    let beta: f32 = 0.0;
    let status = unsafe { cublas_h.gemm(
        CublasTranspose::N,
        CublasTranspose::N,
        sz2int(w.size()[0]),
        sz2int(x.size()[1]),
        sz2int(w.size()[1]),
        &alpha,
        w.as_dptr(), sz2int(w.stride()[1]),
        x.as_dptr(), sz2int(x.stride()[1]),
        &beta,
        self.as_mut_dptr(), sz2int(self.stride()[1]),
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
    assert!(cublas_h.set_math_mode(CublasMathMode::Default).is_ok());
    assert_eq!(w.size()[1], self.size()[0]);
    assert_eq!(w.size()[0], y.size()[0]);
    assert_eq!(y.size()[1], self.size()[1]);
    let alpha: f32 = 1.0;
    let beta: f32 = 0.0;
    let status = unsafe { cublas_h.gemm(
        CublasTranspose::T,
        CublasTranspose::N,
        sz2int(w.size()[1]),
        sz2int(y.size()[1]),
        sz2int(w.size()[0]),
        &alpha,
        w.as_dptr(), sz2int(w.stride()[1]),
        y.as_dptr(), sz2int(y.stride()[1]),
        &beta,
        self.as_mut_dptr(), sz2int(self.stride()[1]),
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
    assert!(cublas_h.set_math_mode(CublasMathMode::Default).is_ok());
    assert_eq!(y.size()[0], self.size()[0]);
    assert_eq!(y.size()[1], x.size()[1]);
    assert_eq!(x.size()[0], self.size()[1]);
    let alpha: f32 = 1.0;
    let beta: f32 = 0.0;
    let status = unsafe { cublas_h.gemm(
        CublasTranspose::N,
        CublasTranspose::T,
        sz2int(y.size()[0]),
        sz2int(x.size()[0]),
        sz2int(y.size()[1]),
        &alpha,
        y.as_dptr(), sz2int(y.stride()[1]),
        x.as_dptr(), sz2int(x.stride()[1]),
        &beta,
        self.as_mut_dptr(), sz2int(self.stride()[1]),
    ) };
    assert!(status.is_ok());
  }
}

pub fn gpu_matrix_mult<T>(
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
}
