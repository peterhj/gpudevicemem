/*
Copyright 2017 the devicemem_gpu authors

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

use ::{GPUDeviceConn, GPUDeviceMem, GPUDeviceRawMem};
use ffi::routines_gpu::*;

use arrayidx::*;
use cuda::runtime::*;
use memarray::{Array, ZeroBits, MemArray};

use std::sync::{Arc};

pub mod linalg;

/*pub trait Array {
  type Idx;

  fn size(&self) -> Self::Idx;
}*/

//pub struct BatchWrap<T>(pub T);

pub trait BatchArray: Array {
  fn batch_size(&self) -> usize;
  fn set_batch_size(&mut self, new_batch_sz: usize);
}

pub trait GPUDeviceArrayZeros: Array {
  fn zeros(size: Self::Idx, conn: GPUDeviceConn) -> Self where Self: Sized;
  //fn zeros_with_storage(size: Self::Idx, mem: Arc<GPUDeviceMem<T>>, conn: GPUDeviceConn) -> Self where Self: Sized;
}

pub trait GPUDeviceBatchArrayZeros: BatchArray {
  fn zeros(size: Self::Idx, batch_sz: usize, conn: GPUDeviceConn) -> Self where Self: Sized;
  //fn zeros_with_storage(size: Self::Idx, batch_sz: usize, mem: Arc<GPUDeviceMem<T>>, conn: GPUDeviceConn) -> Self where Self: Sized;
}

pub trait AsView {
  type ViewTy;

  fn as_view(&self) -> Self::ViewTy;
}

pub trait AsViewMut: AsView {
  type ViewMutTy;

  fn as_view_mut(&mut self) -> Self::ViewMutTy;
}

pub trait FlatView {
  type FlatViewTy;

  fn flat_view(&self) -> Option<Self::FlatViewTy>;
}

pub trait FlatViewMut: FlatView {
  type FlatViewMutTy;

  fn flat_view_mut(&mut self) -> Option<Self::FlatViewMutTy>;
}

// TODO: waiting on trait aliases.
//pub trait GPUFlatView<T> = FlatView<FlatViewTy=GPUDeviceArrayView1d<T>> where T: Copy;
//pub trait GPUFlatViewMut<T> = GPUFlatView<T> + FlatViewMut<FlatViewMutTy=GPUDeviceArrayViewMut1d<T>> where T: Copy;

pub trait View<Idx> {
  fn view(self, idx: Idx) -> Self where Self: Sized;
}

#[derive(Clone)]
pub struct GPUDeviceArray<Idx, T> where T: Copy {
  size:     Idx,
  offset:   Idx,
  stride:   Idx,
  mem:      Arc<GPUDeviceMem<T>>,
}

pub type GPUDeviceScalar<T>  = GPUDeviceArray<Index0d, T>;
pub type GPUDeviceArray1d<T> = GPUDeviceArray<Index1d, T>;
pub type GPUDeviceArray2d<T> = GPUDeviceArray<Index2d, T>;
pub type GPUDeviceArray3d<T> = GPUDeviceArray<Index3d, T>;
pub type GPUDeviceArray4d<T> = GPUDeviceArray<Index4d, T>;
pub type GPUDeviceArray5d<T> = GPUDeviceArray<Index5d, T>;

impl<Idx, T> GPUDeviceArrayZeros for GPUDeviceArray<Idx, T> where Idx: ArrayIndex, T: Copy + 'static {
  fn zeros(size: Idx, conn: GPUDeviceConn) -> Self {
    GPUDeviceArray{
      size:     size.clone(),
      offset:   Idx::zero(),
      stride:   size.to_packed_stride(),
      mem:      Arc::new(unsafe { GPUDeviceRawMem::<T>::alloc(size.flat_len(), conn) }),
    }
  }
}

impl<Idx, T> Array for GPUDeviceArray<Idx, T> where Idx: Clone, T: Copy {
  type Idx = Idx;

  fn size(&self) -> Idx {
    self.size.clone()
  }
}

impl<Idx, T> AsView for GPUDeviceArray<Idx, T> where Idx: Clone, T: Copy {
  type ViewTy = GPUDeviceArrayView<Idx, T>;

  fn as_view(&self) -> GPUDeviceArrayView<Idx, T> {
    GPUDeviceArrayView{
      size:     self.size.clone(),
      offset:   self.offset.clone(),
      stride:   self.stride.clone(),
      mem:      self.mem.clone(),
    }
  }
}

impl<Idx, T> AsViewMut for GPUDeviceArray<Idx, T> where Idx: Clone, T: Copy {
  type ViewMutTy = GPUDeviceArrayViewMut<Idx, T>;

  fn as_view_mut(&mut self) -> GPUDeviceArrayViewMut<Idx, T> {
    GPUDeviceArrayViewMut{
      size:     self.size.clone(),
      offset:   self.offset.clone(),
      stride:   self.stride.clone(),
      mem:      self.mem.clone(),
    }
  }
}

impl<Idx, T> FlatView for GPUDeviceArray<Idx, T> where Idx: ArrayIndex, T: Copy {
  type FlatViewTy = GPUDeviceArrayView1d<T>;

  fn flat_view(&self) -> Option<GPUDeviceArrayView1d<T>> {
    if !self.size.is_packed(&self.stride) {
      None
    } else {
      let flat_size = self.size.flat_len();
      Some(GPUDeviceArrayView{
        size:   flat_size,
        offset: 0,
        stride: flat_size.to_packed_stride(),
        mem:    self.mem.clone(),
      })
    }
  }
}

impl<Idx, T> FlatViewMut for GPUDeviceArray<Idx, T> where Idx: ArrayIndex, T: Copy {
  type FlatViewMutTy = GPUDeviceArrayViewMut1d<T>;

  fn flat_view_mut(&mut self) -> Option<GPUDeviceArrayViewMut1d<T>> {
    if self.size.is_packed(&self.stride) {
      let flat_size = self.size.flat_len();
      Some(GPUDeviceArrayViewMut{
        size:   flat_size,
        offset: 0,
        stride: flat_size.to_packed_stride(),
        mem:    self.mem.clone(),
      })
    } else {
      None
    }
  }
}

pub struct GPUDeviceInnerBatchArray<Idx, T> where T: Copy {
  size:         Idx,
  offset:       Idx,
  stride:       Idx,
  batch_sz:     usize,
  max_batch_sz: usize,
  mem:          Arc<GPUDeviceMem<T>>,
}

pub type GPUDeviceInnerBatchScalar<T>  = GPUDeviceInnerBatchArray<Index0d, T>;
pub type GPUDeviceInnerBatchArray1d<T> = GPUDeviceInnerBatchArray<Index1d, T>;
pub type GPUDeviceInnerBatchArray2d<T> = GPUDeviceInnerBatchArray<Index2d, T>;
pub type GPUDeviceInnerBatchArray3d<T> = GPUDeviceInnerBatchArray<Index3d, T>;
pub type GPUDeviceInnerBatchArray4d<T> = GPUDeviceInnerBatchArray<Index4d, T>;

impl<Idx, T> Array for GPUDeviceInnerBatchArray<Idx, T> where Idx: Clone, T: Copy {
  type Idx = Idx;

  fn size(&self) -> Idx {
    self.size.clone()
  }
}

impl<Idx, T> BatchArray for GPUDeviceInnerBatchArray<Idx, T> where Idx: Clone, T: Copy {
  fn batch_size(&self) -> usize {
    self.batch_sz
  }

  fn set_batch_size(&mut self, new_batch_sz: usize) {
    self.batch_sz = new_batch_sz;
  }
}

impl<Idx, T> AsView for GPUDeviceInnerBatchArray<Idx, T> where Idx: ArrayIndex, T: Copy {
  type ViewTy = GPUDeviceArrayView<Idx::Above, T>;

  fn as_view(&self) -> GPUDeviceArrayView<Idx::Above, T> {
    let view_size = self.size.prepend(self.batch_sz);
    // TODO
    unimplemented!();
  }
}

pub struct GPUDeviceOuterBatchArray<Idx, T> where T: Copy {
  size:         Idx,
  offset:       Idx,
  stride:       Idx,
  batch_sz:     usize,
  max_batch_sz: usize,
  mem:          Arc<GPUDeviceMem<T>>,
}

pub type GPUDeviceOuterBatchScalar<T>  = GPUDeviceOuterBatchArray<Index0d, T>;
pub type GPUDeviceOuterBatchArray1d<T> = GPUDeviceOuterBatchArray<Index1d, T>;
pub type GPUDeviceOuterBatchArray2d<T> = GPUDeviceOuterBatchArray<Index2d, T>;
pub type GPUDeviceOuterBatchArray3d<T> = GPUDeviceOuterBatchArray<Index3d, T>;
pub type GPUDeviceOuterBatchArray4d<T> = GPUDeviceOuterBatchArray<Index4d, T>;

impl<Idx, T> GPUDeviceBatchArrayZeros for GPUDeviceOuterBatchArray<Idx, T> where Idx: ArrayIndex, T: Copy {
  fn zeros(size: Idx, batch_sz: usize, conn: GPUDeviceConn) -> Self {
    // TODO
    unimplemented!();
  }
}

impl<Idx, T> Array for GPUDeviceOuterBatchArray<Idx, T> where Idx: Clone, T: Copy {
  type Idx = Idx;

  fn size(&self) -> Idx {
    self.size.clone()
  }
}

impl<Idx, T> BatchArray for GPUDeviceOuterBatchArray<Idx, T> where Idx: Clone, T: Copy {
  fn batch_size(&self) -> usize {
    self.batch_sz
  }

  fn set_batch_size(&mut self, new_batch_sz: usize) {
    self.batch_sz = new_batch_sz;
  }
}

impl<Idx, T> AsView for GPUDeviceOuterBatchArray<Idx, T> where Idx: ArrayIndex, T: Copy {
  type ViewTy = GPUDeviceArrayView<Idx::Above, T>;

  fn as_view(&self) -> GPUDeviceArrayView<Idx::Above, T> {
    let view_size = self.size.append(self.batch_sz);
    let view_offset = self.offset.append(0);
    let view_stride = self.stride.stride_append_packed(self.size.outside());
    GPUDeviceArrayView{
      size:     view_size,
      offset:   view_offset,
      stride:   view_stride,
      mem:      self.mem.clone(),
    }
  }
}

impl<Idx, T> FlatView for GPUDeviceOuterBatchArray<Idx, T> where Idx: ArrayIndex, T: Copy {
  type FlatViewTy = GPUDeviceArrayView1d<T>;

  fn flat_view(&self) -> Option<GPUDeviceArrayView1d<T>> {
    if self.size.is_packed(&self.stride) {
      let flat_size = self.size.flat_len() * self.batch_sz;
      Some(GPUDeviceArrayView{
        size:   flat_size,
        offset: 0,
        stride: flat_size.to_packed_stride(),
        mem:    self.mem.clone(),
      })
    } else {
      None
    }
  }
}

#[derive(Clone)]
pub struct GPUDeviceArrayView<Idx, T> where T: Copy {
  size:     Idx,
  offset:   Idx,
  stride:   Idx,
  mem:      Arc<GPUDeviceMem<T>>,
}

pub type GPUDeviceScalarView<T>  = GPUDeviceArrayView<Index0d, T>;
pub type GPUDeviceArrayView1d<T> = GPUDeviceArrayView<Index1d, T>;
pub type GPUDeviceArrayView2d<T> = GPUDeviceArrayView<Index2d, T>;
pub type GPUDeviceArrayView3d<T> = GPUDeviceArrayView<Index3d, T>;
pub type GPUDeviceArrayView4d<T> = GPUDeviceArrayView<Index4d, T>;
pub type GPUDeviceArrayView5d<T> = GPUDeviceArrayView<Index5d, T>;

#[derive(Clone)]
pub struct GPUDeviceArrayViewMut<Idx, T> where T: Copy {
  size:     Idx,
  offset:   Idx,
  stride:   Idx,
  mem:      Arc<GPUDeviceMem<T>>,
}

pub type GPUDeviceScalarViewMut<T>  = GPUDeviceArrayViewMut<Index0d, T>;
pub type GPUDeviceArrayViewMut1d<T> = GPUDeviceArrayViewMut<Index1d, T>;
pub type GPUDeviceArrayViewMut2d<T> = GPUDeviceArrayViewMut<Index2d, T>;
pub type GPUDeviceArrayViewMut3d<T> = GPUDeviceArrayViewMut<Index3d, T>;
pub type GPUDeviceArrayViewMut4d<T> = GPUDeviceArrayViewMut<Index4d, T>;

impl<Idx, T> GPUDeviceArrayView<Idx, T> where Idx: ArrayIndex, T: Copy {
  pub unsafe fn as_dptr(&self) -> *const T {
    self.mem.as_dptr().offset(self.offset.flat_index(&self.stride) as _)
  }

  pub fn stride(&self) -> Idx {
    self.stride.clone()
  }
}

impl<Idx, T> Array for GPUDeviceArrayView<Idx, T> where Idx: ArrayIndex, T: Copy {
  type Idx = Idx;

  fn size(&self) -> Idx {
    self.size.clone()
  }
}

impl<Idx, T> GPUDeviceArrayView<Idx, T> where Idx: ArrayIndex, T: Copy {
  pub fn dump_mem(&self, dst: &mut MemArray<Idx, T>, conn: GPUDeviceConn) {
    assert_eq!(self.size, dst.size());
    conn.sync();
    if self.size.is_packed(&self.stride) {
      let len = self.size.flat_len();
      match unsafe { cuda_memcpy_async(
          dst.as_mut_ptr(),
          self.as_dptr(),
          len,
          CudaMemcpyKind::DeviceToHost,
          &mut conn.cuda_stream(),
      ) } {
        Err(_) => panic!(),
        Ok(_) => {}
      }
    } else {
      unimplemented!();
    }
    conn.sync();
  }
}

/*impl<Idx, T> GPUDeviceArrayView<Idx, T> where Idx: ArrayIndex, T: ZeroBits + Copy {
  pub fn set_zeros(&self, conn: &GPUDeviceConn) {
    //println!("DEBUG: GPUDeviceArrayView: set_zeros");
    if self.size.is_packed(&self.stride) {
      let res = unsafe { cuda_memset_async(self.as_mut_dptr() as *mut u8, 0, self.mem.size_bytes(), &mut conn.cuda_stream()) };
      assert!(res.is_ok());
    } else {
      unimplemented!();
    }
  }
}*/

/*impl<Idx, T> GPUDeviceArrayView<Idx, T> where Idx: ArrayIndex, T: Copy {
  pub fn copy(&mut self, other: &GPUDeviceArrayView<Idx, T>, conn: &GPUDeviceConn) {
    // TODO
    unimplemented!();
  }

  pub fn add(&mut self, other: &GPUDeviceArrayView<Idx, T>, conn: &GPUDeviceConn) {
    // TODO
    unimplemented!();
  }
}*/

/*impl<Idx> GPUDeviceArrayView<Idx, f32> where Idx: ArrayIndex {
  pub fn set_constant(&self, c: f32, conn: &GPUDeviceConn) {
    if self.size.is_packed(&self.stride) {
      let len = self.size.flat_len();
      // TODO: error handling.
      let mut stream = conn.cuda_stream();
      unsafe { devicemem_gpu_set_constant_flat_map_f32(
          len as _,
          c,
          self.as_mut_dptr(),
          conn.cuda_kernel_cfg() as *const _,
          stream.as_ptr(),
      ) };
    } else {
      unimplemented!();
    }
  }
}*/

impl<Idx, T> GPUDeviceArrayViewMut<Idx, T> where Idx: ArrayIndex, T: Copy {
  pub fn copy_mem(&mut self, src: &MemArray<Idx, T>, conn: GPUDeviceConn) {
    assert_eq!(self.size, src.size());
    conn.sync();
    if self.size.is_packed(&self.stride) {
      let len = self.size.flat_len();
      match unsafe { cuda_memcpy_async(
          self.as_mut_dptr(),
          src.as_ptr(),
          len,
          CudaMemcpyKind::HostToDevice,
          &mut conn.cuda_stream(),
      ) } {
        Err(_) => panic!(),
        Ok(_) => {}
      }
    } else {
      unimplemented!();
    }
    conn.sync();
  }
}

impl<Idx, T> GPUDeviceArrayViewMut<Idx, T> where Idx: ArrayIndex, T: Copy {
  pub unsafe fn as_dptr(&self) -> *const T {
    self.mem.as_dptr().offset(self.offset.flat_index(&self.stride) as _)
  }

  pub unsafe fn as_mut_dptr(&self) -> *mut T {
    self.mem.as_mut_dptr().offset(self.offset.flat_index(&self.stride) as _)
  }

  pub fn stride(&self) -> Idx {
    self.stride.clone()
  }
}

impl<Idx, T> GPUDeviceArrayViewMut<Idx, T> where Idx: ArrayIndex, T: ZeroBits + Copy {
  pub fn set_zeros(&mut self, conn: GPUDeviceConn) {
    if self.size.is_packed(&self.stride) {
      let mut stream = conn.cuda_stream();
      let res = unsafe { cuda_memset_async(
          self.as_mut_dptr() as *mut u8,
          0,
          self.mem.size_bytes(),
          &mut stream,
      ) };
      assert!(res.is_ok());
    } else {
      unimplemented!();
    }
  }
}

impl<Idx> GPUDeviceArrayViewMut<Idx, f32> where Idx: ArrayIndex {
  pub fn set_constant(&mut self, c: f32, conn: GPUDeviceConn) {
    if self.size.is_packed(&self.stride) {
      let len = self.size.flat_len();
      // TODO: error handling.
      let mut stream = conn.cuda_stream();
      unsafe { devicemem_gpu_set_constant_flat_map_f32(
          len as _,
          c,
          self.as_mut_dptr(),
          conn.cuda_kernel_cfg() as *const _,
          stream.as_ptr(),
      ) };
    } else {
      unimplemented!();
    }
  }

  pub fn mult_constant(&mut self, c: f32, x: GPUDeviceArrayView<Idx, f32>, conn: GPUDeviceConn) {
    if self.size.is_packed(&self.stride) {
      let len = self.size.flat_len();
      // TODO: error handling.
      let mut stream = conn.cuda_stream();
      unsafe { devicemem_gpu_mult_constant_flat_map_f32(
          len as _,
          c,
          x.as_dptr(),
          self.as_mut_dptr(),
          conn.cuda_kernel_cfg() as *const _,
          stream.as_ptr(),
      ) };
    } else {
      unimplemented!();
    }
  }
}

pub trait GPUViewMutExt {
  type ViewTy;

  fn copy(&mut self, src: Self::ViewTy, conn: GPUDeviceConn);
  fn add(&mut self, x: Self::ViewTy, conn: GPUDeviceConn);
}

impl<Idx, T> GPUViewMutExt for GPUDeviceArrayViewMut<Idx, T> where Idx: ArrayIndex, T: Copy {
  type ViewTy = GPUDeviceArrayView<Idx, T>;

  fn copy(&mut self, src: GPUDeviceArrayView<Idx, T>, conn: GPUDeviceConn) {
    assert_eq!(self.size, src.size());
    if self.size.is_packed(&self.stride) {
      let len = self.size.flat_len();
      match unsafe { cuda_memcpy_async(
          self.as_mut_dptr(),
          src.as_dptr(),
          len,
          CudaMemcpyKind::DeviceToDevice,
          &mut conn.cuda_stream(),
      ) } {
        Err(_) => panic!(),
        Ok(_) => {}
      }
    } else {
      unimplemented!();
    }
  }

  default fn add(&mut self, x: GPUDeviceArrayView<Idx, T>, conn: GPUDeviceConn) {
    // TODO
    unimplemented!();
  }
}

impl<Idx> GPUViewMutExt for GPUDeviceArrayViewMut<Idx, f32> where Idx: ArrayIndex {
  fn add(&mut self, x: GPUDeviceArrayView<Idx, f32>, conn: GPUDeviceConn) {
    // TODO
    unimplemented!();
  }
}

/*impl<Idx, Range, T> View<Range> for GPUDeviceArrayView<Idx, T> where Idx: ArrayIndex, Range: ArrayRange<Idx> + Copy, T: Copy {
  fn view(self, range: Range) -> Self {
    // TODO: bounds check.
    let view_size = range.end(&self.size).sub(&range.start(&Idx::zero()));
    let view_offset = self.offset.add(&range.start(&Idx::zero()));
    let view_stride = self.stride;
    GPUDeviceArrayView{
      size:     view_size,
      offset:   view_offset,
      stride:   view_stride,
      mem:      self.mem.clone(),
    }
  }
}*/
