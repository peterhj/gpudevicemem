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

pub mod batch_norm;
pub mod conv;
pub mod pool;
pub mod reduce;
pub mod softmax;

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

pub trait GPUTensorMutOps<T> where T: Copy {
  type ViewTy;

  fn broadcast_add_1d_inplace(&mut self, x: GPUDeviceArrayView1d<T>, axis: isize, conn: GPUDeviceConn);
  fn broadcast_mult_1d(&mut self, a: GPUDeviceArrayView1d<T>, x: Self::ViewTy, axis: isize, conn: GPUDeviceConn);
  fn broadcast_mult_add_1d(&mut self, a: GPUDeviceArrayView1d<T>, x: Self::ViewTy, b: GPUDeviceArrayView1d<T>, axis: isize, conn: GPUDeviceConn);
  fn broadcast_mult_add_1d_inplace(&mut self, a: GPUDeviceArrayView1d<T>, b: GPUDeviceArrayView1d<T>, axis: isize, conn: GPUDeviceConn);
}

pub trait GPUTensorOps<T> where T: Copy {
  fn reduce_sum_1d_to(&self, y: &mut GPUDeviceArrayViewMut1d<T>, axis: isize, conn: GPUDeviceConn);
  fn mult_then_reduce_sum_1d_to(&self, x: Self, y: &mut GPUDeviceArrayViewMut1d<T>, axis: isize, conn: GPUDeviceConn) where Self: Sized;
}

impl GPUTensorMutOps<f32> for GPUDeviceArrayViewMut2d<f32> {
  type ViewTy = GPUDeviceArrayView2d<f32>;

  fn broadcast_add_1d_inplace(&mut self, x: GPUDeviceArrayView1d<f32>, axis: isize, conn: GPUDeviceConn) {
    // FIXME
    unimplemented!();
  }

  fn broadcast_mult_1d(&mut self, a: GPUDeviceArrayView1d<f32>, x: GPUDeviceArrayView2d<f32>, axis: isize, conn: GPUDeviceConn) {
    // TODO: size checks.
    assert_eq!(x.size(), self.size());
    assert_eq!(x.size().index_at(axis), a.size());
    if self.is_packed() && a.is_packed() && x.is_packed() {
      let a = a.wait(conn.clone());
      let x = x.wait(conn.clone());
      let mut y = self.wait_mut(conn.clone());
      let mut stream = conn.cuda_stream();
      match axis {
        0 => {
          unsafe { gpudevicemem_bcast_flat_mult_I1a_I2ab_Oab_packed_f32(
              sz2uint(x.inner().size().index_at(0)),
              sz2uint(x.inner().size().index_cut(0)),
              a.as_dptr(),
              x.as_dptr(),
              y.as_mut_dptr(),
              conn.cuda_kernel_cfg() as *const _,
              stream.as_mut_ptr(),
          ) };
        }
        _ => unimplemented!(),
      }
    } else {
      unimplemented!();
    }
  }

  fn broadcast_mult_add_1d(&mut self, a: GPUDeviceArrayView1d<f32>, x: GPUDeviceArrayView2d<f32>, b: GPUDeviceArrayView1d<f32>, axis: isize, conn: GPUDeviceConn) {
    // FIXME
    unimplemented!();
  }

  fn broadcast_mult_add_1d_inplace(&mut self, a: GPUDeviceArrayView1d<f32>, b: GPUDeviceArrayView1d<f32>, axis: isize, conn: GPUDeviceConn) {
    // FIXME
    unimplemented!();
  }
}

impl GPUTensorMutOps<f32> for GPUDeviceArrayViewMut4d<f32> {
  type ViewTy = GPUDeviceArrayView4d<f32>;

  fn broadcast_add_1d_inplace(&mut self, x: GPUDeviceArrayView1d<f32>, axis: isize, conn: GPUDeviceConn) {
    // TODO: size checks.
    if self.is_packed() && x.is_packed() {
      let x = x.wait(conn.clone());
      let mut y = self.wait_mut(conn.clone());
      let mut stream = conn.cuda_stream();
      match axis {
        0 => {
          assert_eq!(x.inner().size(), y.inner().size().index_at(0));
          unsafe { gpudevicemem_bcast_flat_add_I1a_IO2ab_inplace_packed_f32(
              sz2uint(y.inner().size().index_at(0)),
              sz2uint(y.inner().size().index_cut(0).flat_len()),
              x.as_dptr(),
              y.as_mut_dptr(),
              conn.cuda_kernel_cfg() as *const _,
              stream.as_mut_ptr(),
          ) };
        }
        1 => {
          assert_eq!(x.inner().size(), y.inner().size().index_at(1));
          unsafe { gpudevicemem_bcast_flat_add_I1b_IO2abc_inplace_packed_f32(
              sz2uint(y.inner().size().index_at(0)),
              sz2uint(y.inner().size().index_at(1)),
              sz2uint(y.inner().size().index_cut(1).index_cut(0).flat_len()),
              x.as_dptr(),
              y.as_mut_dptr(),
              conn.cuda_kernel_cfg() as *const _,
              stream.as_mut_ptr(),
          ) };
        }
        2 => {
          assert_eq!(x.inner().size(), y.inner().size().index_at(2));
          unsafe { gpudevicemem_bcast_flat_add_I1b_IO2abc_inplace_packed_f32(
              sz2uint(y.inner().size().index_cut(3).index_cut(2).flat_len()),
              sz2uint(y.inner().size().index_at(2)),
              sz2uint(y.inner().size().index_at(3)),
              x.as_dptr(),
              y.as_mut_dptr(),
              conn.cuda_kernel_cfg() as *const _,
              stream.as_mut_ptr(),
          ) };
        }
        3 | -1 => {
          // TODO
          unimplemented!();
        }
        _ => unreachable!(),
      }
    } else {
      unimplemented!();
    }
  }

  fn broadcast_mult_1d(&mut self, a: GPUDeviceArrayView1d<f32>, x: GPUDeviceArrayView4d<f32>, axis: isize, conn: GPUDeviceConn) {
    // FIXME
    unimplemented!();
  }

  fn broadcast_mult_add_1d(&mut self, a: GPUDeviceArrayView1d<f32>, x: GPUDeviceArrayView4d<f32>, b: GPUDeviceArrayView1d<f32>, axis: isize, conn: GPUDeviceConn) {
    // FIXME
    unimplemented!();
  }

  fn broadcast_mult_add_1d_inplace(&mut self, a: GPUDeviceArrayView1d<f32>, b: GPUDeviceArrayView1d<f32>, axis: isize, conn: GPUDeviceConn) {
    // TODO: size checks.
    if self.is_packed() && a.is_packed() && b.is_packed() {
      let a = a.wait(conn.clone());
      let b = b.wait(conn.clone());
      let mut y = self.wait_mut(conn.clone());
      let mut stream = conn.cuda_stream();
      match axis {
        0 => {
          // TODO
          unimplemented!();
        }
        1 => {
          // TODO
          unimplemented!();
        }
        2 => {
          unsafe { gpudevicemem_bcast_flat_mult_add_I1b_I2abc_I3b_Oabc_packed_f32(
              sz2uint(y.inner().size().index_cut(3).index_cut(2).flat_len()),
              sz2uint(y.inner().size().index_at(2)),
              sz2uint(y.inner().size().index_at(3)),
              a.as_dptr(),
              y.as_dptr(),
              b.as_dptr(),
              y.as_mut_dptr(),
              conn.cuda_kernel_cfg() as *const _,
              stream.as_mut_ptr(),
          ) }
        }
        3 | -1 => {
          // TODO
          unimplemented!();
        }
        _ => unreachable!(),
      }
    } else {
      unimplemented!();
    }
  }
}

impl GPUTensorMutOps<f32> for GPUDeviceArrayViewMut5d<f32> {
  type ViewTy = GPUDeviceArrayView5d<f32>;

  fn broadcast_add_1d_inplace(&mut self, x: GPUDeviceArrayView1d<f32>, axis: isize, conn: GPUDeviceConn) {
    // TODO: size checks.
    assert_eq!(self.size().index_at(axis), x.size());
    if self.is_packed() && x.is_packed() {
      let x = x.wait(conn.clone());
      let mut y = self.wait_mut(conn.clone());
      let mut stream = conn.cuda_stream();
      match axis {
        0 => {
          unsafe { gpudevicemem_bcast_flat_add_I1a_IO2ab_inplace_packed_f32(
              sz2uint(y.inner().size().index_at(0)),
              sz2uint(y.inner().size().index_cut(0).flat_len()),
              x.as_dptr(),
              y.as_mut_dptr(),
              conn.cuda_kernel_cfg() as *const _,
              stream.as_mut_ptr(),
          ) };
        }
        1 => {
          unsafe { gpudevicemem_bcast_flat_add_I1b_IO2abc_inplace_packed_f32(
              sz2uint(y.inner().size().index_at(0)),
              sz2uint(y.inner().size().index_at(1)),
              sz2uint(y.inner().size().index_cut(1).index_cut(0).flat_len()),
              x.as_dptr(),
              y.as_mut_dptr(),
              conn.cuda_kernel_cfg() as *const _,
              stream.as_mut_ptr(),
          ) };
        }
        2 => {
          unsafe { gpudevicemem_bcast_flat_add_I1b_IO2abc_inplace_packed_f32(
              sz2uint(y.inner().size().index_cut(4).index_cut(3).index_cut(2).flat_len()),
              sz2uint(y.inner().size().index_at(2)),
              sz2uint(y.inner().size().index_cut(2).index_cut(1).index_cut(0).flat_len()),
              x.as_dptr(),
              y.as_mut_dptr(),
              conn.cuda_kernel_cfg() as *const _,
              stream.as_mut_ptr(),
          ) };
        }
        3 => {
          unsafe { gpudevicemem_bcast_flat_add_I1b_IO2abc_inplace_packed_f32(
              sz2uint(y.inner().size().index_cut(4).index_cut(3).flat_len()),
              sz2uint(y.inner().size().index_at(3)),
              sz2uint(y.inner().size().index_at(4)),
              x.as_dptr(),
              y.as_mut_dptr(),
              conn.cuda_kernel_cfg() as *const _,
              stream.as_mut_ptr(),
          ) };
        }
        4 | -1 => {
          // TODO
          unimplemented!();
        }
        _ => unreachable!(),
      }
    } else {
      unimplemented!();
    }
  }

  fn broadcast_mult_1d(&mut self, a: GPUDeviceArrayView1d<f32>, x: GPUDeviceArrayView5d<f32>, axis: isize, conn: GPUDeviceConn) {
    // FIXME
    unimplemented!();
  }

  fn broadcast_mult_add_1d(&mut self, a: GPUDeviceArrayView1d<f32>, x: GPUDeviceArrayView5d<f32>, b: GPUDeviceArrayView1d<f32>, axis: isize, conn: GPUDeviceConn) {
    // TODO: size checks.
    assert_eq!(self.size(), x.size());
    assert_eq!(self.size().index_at(axis), a.size());
    assert_eq!(self.size().index_at(axis), b.size());
    if self.is_packed() && a.is_packed() && x.is_packed() && b.is_packed() {
      let a = a.wait(conn.clone());
      let x = x.wait(conn.clone());
      let b = b.wait(conn.clone());
      let mut y = self.wait_mut(conn.clone());
      let mut stream = conn.cuda_stream();
      match axis {
        3 => {
          unsafe { gpudevicemem_bcast_flat_mult_add_I1b_I2abc_I3b_Oabc_packed_f32(
              sz2uint(y.inner().size().index_cut(4).index_cut(3).flat_len()),
              sz2uint(y.inner().size().index_at(3)),
              sz2uint(y.inner().size().index_at(4)),
              a.as_dptr(),
              x.as_dptr(),
              b.as_dptr(),
              y.as_mut_dptr(),
              conn.cuda_kernel_cfg() as *const _,
              stream.as_mut_ptr(),
          ) };
        }
        _ => unimplemented!(),
      }
    } else {
      unimplemented!();
    }
  }

  fn broadcast_mult_add_1d_inplace(&mut self, a: GPUDeviceArrayView1d<f32>, b: GPUDeviceArrayView1d<f32>, axis: isize, conn: GPUDeviceConn) {
    // TODO: size checks.
    assert_eq!(self.size().index_at(axis), a.size());
    assert_eq!(self.size().index_at(axis), b.size());
    if self.is_packed() && a.is_packed() && b.is_packed() {
      let a = a.wait(conn.clone());
      let b = b.wait(conn.clone());
      let mut y = self.wait_mut(conn.clone());
      let mut stream = conn.cuda_stream();
      match axis {
        3 => {
          unsafe { gpudevicemem_bcast_flat_mult_add_I1b_I2abc_I3b_Oabc_packed_f32(
              sz2uint(y.inner().size().index_cut(4).index_cut(3).flat_len()),
              sz2uint(y.inner().size().index_at(3)),
              sz2uint(y.inner().size().index_at(4)),
              a.as_dptr(),
              y.as_dptr(),
              b.as_dptr(),
              y.as_mut_dptr(),
              conn.cuda_kernel_cfg() as *const _,
              stream.as_mut_ptr(),
          ) };
        }
        _ => unimplemented!(),
      }
    } else {
      unimplemented!();
    }
  }
}

impl GPUTensorOps<f32> for GPUDeviceArrayView4d<f32> {
  fn reduce_sum_1d_to(&self, y: &mut GPUDeviceArrayViewMut1d<f32>, axis: isize, conn: GPUDeviceConn) {
    // TODO: size checks.
    if self.is_packed() && y.is_packed() {
      let x = self.wait(conn.clone());
      let mut y = y.wait_mut(conn.clone());
      let mut stream = conn.cuda_stream();
      match axis {
        0 => {
          // TODO
          unimplemented!();
        }
        1 => {
          // TODO
          unimplemented!();
        }
        2 => {
          assert_eq!(x.inner().size().index_at(2), y.inner().size());
          unsafe { gpudevicemem_sum_I1abc_Ob_packed_deterministic_f32(
              sz2uint(x.inner().size().index_cut(3).index_cut(2).flat_len()),
              sz2uint(x.inner().size().index_at(2)),
              sz2uint(x.inner().size().index_at(3)),
              x.as_dptr(),
              y.as_mut_dptr(),
              conn.cuda_kernel_cfg() as *const _,
              stream.as_mut_ptr(),
          ) }
        }
        3 | -1 => {
          assert_eq!(x.inner().size().index_at(3), y.inner().size());
          // TODO: could use a better implementation for this.
          unsafe { gpudevicemem_sum_I1abc_Ob_packed_deterministic_f32(
              sz2uint(x.inner().size().index_cut(3).flat_len()),
              sz2uint(x.inner().size().index_at(3)),
              1,
              x.as_dptr(),
              y.as_mut_dptr(),
              conn.cuda_kernel_cfg() as *const _,
              stream.as_mut_ptr(),
          ) }
        }
        _ => unreachable!(),
      }
    } else {
      unimplemented!();
    }
  }

  // TODO: naming convention should be "flat_mut"?
  fn mult_then_reduce_sum_1d_to(&self, x: GPUDeviceArrayView4d<f32>, y: &mut GPUDeviceArrayViewMut1d<f32>, axis: isize, conn: GPUDeviceConn) {
    // TODO: size checks.
    if self.is_packed() && y.is_packed() {
      let w = self.wait(conn.clone());
      let x = x.wait(conn.clone());
      let mut y = y.wait_mut(conn.clone());
      let mut stream = conn.cuda_stream();
      match axis {
        0 => {
          // TODO
          unimplemented!();
        }
        1 => {
          // TODO
          unimplemented!();
        }
        2 => {
          assert_eq!(x.inner().size().index_at(2), y.inner().size());
          unsafe { gpudevicemem_mult_then_sum_I1abc_I2abc_Ob_packed_deterministic_f32(
              sz2uint(w.inner().size().index_cut(3).index_cut(2).flat_len()),
              sz2uint(w.inner().size().index_at(2)),
              sz2uint(w.inner().size().index_at(3)),
              w.as_dptr(),
              x.as_dptr(),
              y.as_mut_dptr(),
              conn.cuda_kernel_cfg() as *const _,
              stream.as_mut_ptr(),
          ) }
        }
        3 | -1 => {
          // TODO
          unimplemented!();
        }
        _ => unreachable!(),
      }
    } else {
      unimplemented!();
    }
  }
}

impl GPUTensorOps<f32> for GPUDeviceArrayView5d<f32> {
  fn reduce_sum_1d_to(&self, y: &mut GPUDeviceArrayViewMut1d<f32>, axis: isize, conn: GPUDeviceConn) {
    // TODO: size checks.
    assert_eq!(self.size().index_at(axis), y.size());
    if self.is_packed() && y.is_packed() {
      let x = self.wait(conn.clone());
      let mut y = y.wait_mut(conn.clone());
      let mut stream = conn.cuda_stream();
      match axis {
        3 => {
          assert_eq!(x.inner().size().index_at(3), y.inner().size());
          unsafe { gpudevicemem_sum_I1abc_Ob_packed_deterministic_f32(
              sz2uint(x.inner().size().index_cut(4).index_cut(3).flat_len()),
              sz2uint(x.inner().size().index_at(3)),
              sz2uint(x.inner().size().index_at(4)),
              x.as_dptr(),
              y.as_mut_dptr(),
              conn.cuda_kernel_cfg() as *const _,
              stream.as_mut_ptr(),
          ) }
        }
        _ => unimplemented!(),
      }
    } else {
      unimplemented!();
    }
  }

  // TODO: naming convention should be "flat_mut"?
  fn mult_then_reduce_sum_1d_to(&self, x: GPUDeviceArrayView5d<f32>, y: &mut GPUDeviceArrayViewMut1d<f32>, axis: isize, conn: GPUDeviceConn) {
    // TODO: size checks.
    assert_eq!(self.size().index_at(axis), y.size());
    if self.is_packed() && y.is_packed() {
      let a = self.wait(conn.clone());
      let x = x.wait(conn.clone());
      let mut y = y.wait_mut(conn.clone());
      let mut stream = conn.cuda_stream();
      match axis {
        3 => {
          assert_eq!(x.inner().size().index_at(3), y.inner().size());
          unsafe { gpudevicemem_mult_then_sum_I1abc_I2abc_Ob_packed_deterministic_f32(
              sz2uint(a.inner().size().index_cut(4).index_cut(3).flat_len()),
              sz2uint(a.inner().size().index_at(3)),
              sz2uint(a.inner().size().index_at(4)),
              a.as_dptr(),
              x.as_dptr(),
              y.as_mut_dptr(),
              conn.cuda_kernel_cfg() as *const _,
              stream.as_mut_ptr(),
          ) }
        }
        _ => unimplemented!(),
      }
    } else {
      unimplemented!();
    }
  }
}
