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

use super::{GPUDeviceConn, GPUDeviceMem, GPUDeviceRawMem, GPUDeviceStreamMem};

use arrayidx::*;

use std::sync::{Arc};

pub mod linalg;

#[derive(Clone)]
pub struct GPUDeviceArray<Idx, T> where T: Copy {
  size:     Idx,
  offset:   Idx,
  stride:   Idx,
  mem:      Arc<GPUDeviceMem<T>>,
}

pub trait Array {
  type Idx;

  fn size(&self) -> Self::Idx;
}

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

/*pub trait AsViewMut: AsView {
  type ViewMutTy;

  fn as_view_mut(&self) -> Self::ViewMutTy;
}*/

pub trait FlatView {
  type FlatViewTy;

  fn flat_view(&self) -> Option<Self::FlatViewTy>;
}

/*pub trait FlatViewMut: FlatView {
  type FlatViewMutTy;

  fn flat_view_mut(&self) -> Option<Self::FlatViewMutTy>;
}*/

pub trait View<Idx> {
  fn view(self, idx: Idx) -> Self where Self: Sized;
}

pub type GPUDeviceScalar<T>  = GPUDeviceArray<Index0d, T>;
pub type GPUDeviceArray1d<T> = GPUDeviceArray<Index1d, T>;
pub type GPUDeviceArray2d<T> = GPUDeviceArray<Index2d, T>;
pub type GPUDeviceArray3d<T> = GPUDeviceArray<Index3d, T>;
pub type GPUDeviceArray4d<T> = GPUDeviceArray<Index4d, T>;
pub type GPUDeviceArray5d<T> = GPUDeviceArray<Index5d, T>;

impl<Idx, T> GPUDeviceArrayZeros for GPUDeviceArray<Idx, T> where Idx: ArrayIndex + Copy, T: Copy + 'static {
  fn zeros(size: Idx, conn: GPUDeviceConn) -> Self {
    GPUDeviceArray{
      size:     size,
      offset:   Idx::zero(),
      stride:   size.to_packed_stride(),
      mem:      Arc::new(unsafe { GPUDeviceRawMem::<T>::alloc(size.flat_len(), conn) }),
      //mem:      Arc::new(unsafe { GPUDeviceStreamMem::<T>::alloc(size.flat_len(), conn) }),
    }
  }
}

impl<Idx, T> Array for GPUDeviceArray<Idx, T> where Idx: Copy, T: Copy {
  type Idx = Idx;

  fn size(&self) -> Idx {
    self.size
  }
}

impl<Idx, T> AsView for GPUDeviceArray<Idx, T> where Idx: Copy, T: Copy {
  type ViewTy = GPUDeviceArrayView<Idx, T>;

  fn as_view(&self) -> GPUDeviceArrayView<Idx, T> {
    // TODO
    unimplemented!();
  }
}

impl<Idx, T> FlatView for GPUDeviceArray<Idx, T> where Idx: ArrayIndex + Copy, T: Copy {
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

impl<Idx, T> Array for GPUDeviceInnerBatchArray<Idx, T> where Idx: Copy, T: Copy {
  type Idx = Idx;

  fn size(&self) -> Idx {
    self.size
  }
}

impl<Idx, T> BatchArray for GPUDeviceInnerBatchArray<Idx, T> where Idx: Copy, T: Copy {
  fn batch_size(&self) -> usize {
    self.batch_sz
  }

  fn set_batch_size(&mut self, new_batch_sz: usize) {
    self.batch_sz = new_batch_sz;
  }
}

impl<Idx, T> AsView for GPUDeviceInnerBatchArray<Idx, T> where Idx: ArrayIndex + Copy, T: Copy {
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

impl<Idx, T> GPUDeviceBatchArrayZeros for GPUDeviceOuterBatchArray<Idx, T> where Idx: Copy, T: Copy {
  fn zeros(size: Idx, batch_sz: usize, conn: GPUDeviceConn) -> Self {
    // TODO
    unimplemented!();
  }
}

impl<Idx, T> Array for GPUDeviceOuterBatchArray<Idx, T> where Idx: Copy, T: Copy {
  type Idx = Idx;

  fn size(&self) -> Idx {
    self.size
  }
}

impl<Idx, T> BatchArray for GPUDeviceOuterBatchArray<Idx, T> where Idx: Copy, T: Copy {
  fn batch_size(&self) -> usize {
    self.batch_sz
  }

  fn set_batch_size(&mut self, new_batch_sz: usize) {
    self.batch_sz = new_batch_sz;
  }
}

impl<Idx, T> AsView for GPUDeviceOuterBatchArray<Idx, T> where Idx: ArrayIndex + Copy, T: Copy {
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

impl<Idx, T> FlatView for GPUDeviceOuterBatchArray<Idx, T> where Idx: ArrayIndex + Copy, T: Copy {
  type FlatViewTy = GPUDeviceArrayView1d<T>;

  fn flat_view(&self) -> Option<GPUDeviceArrayView1d<T>> {
    if !self.size.is_packed(&self.stride) {
      None
    } else {
      let flat_size = self.size.flat_len() * self.batch_sz;
      Some(GPUDeviceArrayView{
        size:   flat_size,
        offset: 0,
        stride: flat_size.to_packed_stride(),
        mem:    self.mem.clone(),
      })
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

/*pub struct GPUDeviceArrayViewMut<Idx, T> where T: Copy {
  size:     Idx,
  stride:   Idx,
  mem:      Arc<GPUDeviceMem<T>>,
}

pub type GPUDeviceScalarViewMut<T>  = GPUDeviceArrayViewMut<Index0d, T>;
pub type GPUDeviceArrayViewMut1d<T> = GPUDeviceArrayViewMut<Index1d, T>;
pub type GPUDeviceArrayViewMut2d<T> = GPUDeviceArrayViewMut<Index2d, T>;
pub type GPUDeviceArrayViewMut3d<T> = GPUDeviceArrayViewMut<Index3d, T>;
pub type GPUDeviceArrayViewMut4d<T> = GPUDeviceArrayViewMut<Index4d, T>;*/

impl<Idx, T> GPUDeviceArrayView<Idx, T> where Idx: ArrayIndex + Copy, T: Copy {
  pub unsafe fn as_dptr(&self) -> *const T {
    self.mem.as_dptr().offset(self.offset.flat_index(&self.stride) as _)
  }

  pub unsafe fn as_mut_dptr(&self) -> *mut T {
    self.mem.as_mut_dptr().offset(self.offset.flat_index(&self.stride) as _)
  }

  pub fn stride(&self) -> Idx {
    self.stride
  }
}

impl<Idx, T> Array for GPUDeviceArrayView<Idx, T> where Idx: ArrayIndex + Copy, T: Copy {
  type Idx = Idx;

  fn size(&self) -> Idx {
    self.size
  }
}

impl<Idx, T> GPUDeviceArrayView<Idx, T> where Idx: ArrayIndex + Copy, T: Copy {
  pub fn copy(&mut self, other: &GPUDeviceArrayView<Idx, T>, conn: &GPUDeviceConn) {
    // TODO
    unimplemented!();
  }

  pub fn add(&mut self, other: &GPUDeviceArrayView<Idx, T>, conn: &GPUDeviceConn) {
    // TODO
    unimplemented!();
  }
}

impl<Idx, Range, T> View<Range> for GPUDeviceArrayView<Idx, T> where Idx: ArrayIndex + Copy, Range: ArrayRange<Idx> + Copy, T: Copy {
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
}