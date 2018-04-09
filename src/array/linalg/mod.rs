use ::{GPUDeviceConn};
use ::array::*;

use cuda_blas::*;
use memarray::*;

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
    // TODO: check sizes.
    let alpha: f32 = 1.0;
    let beta: f32 = 0.0;
    let status = unsafe { conn.cublas().gemv(
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

fn gpu_matrix_vector_mult<T>(
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
    // TODO: check sizes.
    let alpha: f32 = 1.0;
    let beta: f32 = 0.0;
    let status = unsafe { conn.cublas().gemm(
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
    // TODO: check sizes.
    let alpha: f32 = 1.0;
    let beta: f32 = 0.0;
    let status = unsafe { conn.cublas().gemm(
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
    // TODO: check sizes.
    let alpha: f32 = 1.0;
    let beta: f32 = 0.0;
    let status = unsafe { conn.cublas().gemm(
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

fn gpu_matrix_mult<T>(
    w: GPUDeviceArrayView2d<T>,
    x: GPUDeviceArrayView2d<T>,
    mut y: GPUDeviceArrayViewMut2d<T>,
    conn: GPUDeviceConn)
where T: Copy,
      GPUDeviceArrayViewMut2d<T>: GPUMatrixOps<T>
{
  y.matrix_mult(w, x, conn);
}

fn gpu_left_transpose_matrix_mult<T>(
    w: GPUDeviceArrayView2d<T>,
    y: GPUDeviceArrayView2d<T>,
    mut x: GPUDeviceArrayViewMut2d<T>,
    conn: GPUDeviceConn)
where T: Copy,
      GPUDeviceArrayViewMut2d<T>: GPUMatrixOps<T>
{
  x.left_transpose_matrix_mult(w, y, conn);
}

fn gpu_right_transpose_matrix_mult<T>(
    y: GPUDeviceArrayView2d<T>,
    x: GPUDeviceArrayView2d<T>,
    mut w: GPUDeviceArrayViewMut2d<T>,
    conn: GPUDeviceConn)
where T: Copy,
      GPUDeviceArrayViewMut2d<T>: GPUMatrixOps<T>
{
  w.right_transpose_matrix_mult(y, x, conn);
}
