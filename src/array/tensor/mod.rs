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

pub mod conv;
pub mod pool;
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
  fn broadcast_add_1d_inplace(&mut self, x: GPUDeviceArrayView1d<T>, axis: isize, conn: GPUDeviceConn);
  fn broadcast_mult_add_1d_inplace(&mut self, a: GPUDeviceArrayView1d<T>, b: GPUDeviceArrayView1d<T>, axis: isize, conn: GPUDeviceConn);
}

pub trait GPUTensorOps<T> where T: Copy {
  fn reduce_sum_1d_to(&self, y: &mut GPUDeviceArrayViewMut1d<T>, axis: isize, conn: GPUDeviceConn);
  fn mult_then_reduce_sum_1d_to(&self, x: Self, y: &mut GPUDeviceArrayViewMut1d<T>, axis: isize, conn: GPUDeviceConn) where Self: Sized;
}

impl GPUTensorMutOps<f32> for GPUDeviceArrayViewMut4d<f32> {
  fn broadcast_add_1d_inplace(&mut self, x: GPUDeviceArrayView1d<f32>, axis: isize, conn: GPUDeviceConn) {
    if self.is_packed() && x.is_packed() {
      let mut stream = conn.cuda_stream();
      match axis {
        0 => {
          unsafe { gpudevicemem_bcast_flat_add_I1a_IO2ab_inplace_packed_f32(
              sz2uint(self.size()[0]),
              sz2uint(self.size()[1] * self.size()[2] * self.size()[3]),
              x.raw_dptr(),
              self.raw_mut_dptr(),
              conn.cuda_kernel_cfg() as *const _,
              stream.as_mut_ptr(),
          ) };
        }
        1 => {
          unsafe { gpudevicemem_bcast_flat_add_I1b_IO2abc_inplace_packed_f32(
              sz2uint(self.size()[0]),
              sz2uint(self.size()[1]),
              sz2uint(self.size()[2] * self.size()[3]),
              x.raw_dptr(),
              self.raw_mut_dptr(),
              conn.cuda_kernel_cfg() as *const _,
              stream.as_mut_ptr(),
          ) };
        }
        2 => {
          unsafe { gpudevicemem_bcast_flat_add_I1b_IO2abc_inplace_packed_f32(
              sz2uint(self.size()[0] * self.size()[1]),
              sz2uint(self.size()[2]),
              sz2uint(self.size()[3]),
              x.raw_dptr(),
              self.raw_mut_dptr(),
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

  fn broadcast_mult_add_1d_inplace(&mut self, a: GPUDeviceArrayView1d<f32>, b: GPUDeviceArrayView1d<f32>, axis: isize, conn: GPUDeviceConn) {
    // TODO
    if self.is_packed() && a.is_packed() && b.is_packed() {
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
              sz2uint(self.size()[0] * self.size()[1]),
              sz2uint(self.size()[2]),
              sz2uint(self.size()[3]),
              a.raw_dptr(),
              self.raw_dptr(),
              b.raw_dptr(),
              self.raw_mut_dptr(),
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
  fn broadcast_add_1d_inplace(&mut self, x: GPUDeviceArrayView1d<f32>, axis: isize, conn: GPUDeviceConn) {
    if self.is_packed() && x.is_packed() {
      let mut stream = conn.cuda_stream();
      match axis {
        0 => {
          unsafe { gpudevicemem_bcast_flat_add_I1a_IO2ab_inplace_packed_f32(
              sz2uint(self.size()[0]),
              sz2uint(self.size()[1] * self.size()[2] * self.size()[3] * self.size()[4]),
              x.raw_dptr(),
              self.raw_mut_dptr(),
              conn.cuda_kernel_cfg() as *const _,
              stream.as_mut_ptr(),
          ) };
        }
        1 => {
          unsafe { gpudevicemem_bcast_flat_add_I1b_IO2abc_inplace_packed_f32(
              sz2uint(self.size()[0]),
              sz2uint(self.size()[1]),
              sz2uint(self.size()[2] * self.size()[3] * self.size()[4]),
              x.raw_dptr(),
              self.raw_mut_dptr(),
              conn.cuda_kernel_cfg() as *const _,
              stream.as_mut_ptr(),
          ) };
        }
        2 => {
          unsafe { gpudevicemem_bcast_flat_add_I1b_IO2abc_inplace_packed_f32(
              sz2uint(self.size()[0] * self.size()[1]),
              sz2uint(self.size()[2]),
              sz2uint(self.size()[3] * self.size()[4]),
              x.raw_dptr(),
              self.raw_mut_dptr(),
              conn.cuda_kernel_cfg() as *const _,
              stream.as_mut_ptr(),
          ) };
        }
        3 => {
          unsafe { gpudevicemem_bcast_flat_add_I1b_IO2abc_inplace_packed_f32(
              sz2uint(self.size()[0] * self.size()[1] * self.size()[2]),
              sz2uint(self.size()[3]),
              sz2uint(self.size()[4]),
              x.raw_dptr(),
              self.raw_mut_dptr(),
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

  fn broadcast_mult_add_1d_inplace(&mut self, a: GPUDeviceArrayView1d<f32>, b: GPUDeviceArrayView1d<f32>, axis: isize, conn: GPUDeviceConn) {
    // TODO
    unimplemented!();
  }
}

impl GPUTensorOps<f32> for GPUDeviceArrayView4d<f32> {
  fn reduce_sum_1d_to(&self, y: &mut GPUDeviceArrayViewMut1d<f32>, axis: isize, conn: GPUDeviceConn) {
    // TODO
    if self.is_packed() && y.is_packed() {
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
          unsafe { gpudevicemem_sum_I1abc_Ob_packed_deterministic_f32(
              sz2uint(self.size()[0] * self.size()[1]),
              sz2uint(self.size()[2]),
              sz2uint(self.size()[3]),
              self.raw_dptr(),
              y.raw_mut_dptr(),
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

  fn mult_then_reduce_sum_1d_to(&self, x: GPUDeviceArrayView4d<f32>, y: &mut GPUDeviceArrayViewMut1d<f32>, axis: isize, conn: GPUDeviceConn) {
    // TODO
    if self.is_packed() && y.is_packed() {
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
          unsafe { gpudevicemem_mult_then_sum_I1abc_I2abc_Ob_packed_deterministic_f32(
              sz2uint(self.size()[0] * self.size()[1]),
              sz2uint(self.size()[2]),
              sz2uint(self.size()[3]),
              self.raw_dptr(),
              x.raw_dptr(),
              y.raw_mut_dptr(),
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
