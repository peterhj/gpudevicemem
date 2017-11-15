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

use super::{GPUDeviceConn, GPUDeviceMem, GPUDeviceAllocMem};

use std::ops::{Range, RangeFrom, RangeTo, RangeFull};
use std::sync::{Arc};

pub type Index0d = ();
pub type Index1d = usize;
pub type Index2d = [usize; 2];
pub type Index3d = [usize; 3];
pub type Index4d = [usize; 4];
pub type Index5d = [usize; 5];
pub struct UnimplIndex;

pub type Range0d = ();
pub type Range1d = Range<usize>;
pub type Range2d = [Range<usize>; 2];
pub type Range3d = [Range<usize>; 3];
pub type Range4d = [Range<usize>; 4];
pub type Range5d = [Range<usize>; 5];

pub trait ArrayIndex {
  type Range;
  type Above: Sized;

  fn zero() -> Self where Self: Sized;
  fn add(&self, shift: &Self) -> Self where Self: Sized;
  fn sub(&self, shift: &Self) -> Self where Self: Sized;

  fn prepend(&self, new_inside: usize) -> Self::Above;
  fn append(&self, new_outside: usize) -> Self::Above;

  fn to_packed_stride(&self) -> Self where Self: Sized;
  fn is_packed(&self, stride: &Self) -> bool where Self: Sized;
  fn stride_append_packed(&self, outside: usize) -> Self::Above where Self: Sized {
    self.append(self.outside() * outside)
  }

  fn flat_len(&self) -> usize;
  fn flat_index(&self, stride: &Self) -> usize;

  fn inside(&self) -> usize;
  fn outside(&self) -> usize;

  fn dim(&self) -> usize;
}

pub trait ArrayRange<Idx> {
  fn start(&self, offset: &Idx) -> Idx;
  fn end(&self, limit: &Idx) -> Idx;
}

impl ArrayIndex for Index0d {
  type Range = Range0d;
  type Above = Index1d;

  fn zero() -> Self {
    ()
  }

  fn add(&self, shift: &Self) -> Self {
    ()
  }

  fn sub(&self, shift: &Self) -> Self {
    ()
  }

  fn to_packed_stride(&self) -> Self {
    ()
  }

  fn is_packed(&self, stride: &Self) -> bool {
    true
  }

  fn prepend(&self, major: usize) -> Index1d {
    major
  }

  fn append(&self, minor: usize) -> Index1d {
    minor
  }

  fn flat_len(&self) -> usize {
    1
  }

  fn flat_index(&self, stride: &Self) -> usize {
    0
  }

  fn inside(&self) -> usize {
    1
  }

  fn outside(&self) -> usize {
    1
  }

  fn dim(&self) -> usize {
    0
  }
}

impl ArrayIndex for Index1d {
  type Range = Range1d;
  type Above = Index2d;

  fn zero() -> Self {
    1
  }

  fn add(&self, shift: &Self) -> Self {
    *self + *shift
  }

  fn sub(&self, shift: &Self) -> Self {
    *self - *shift
  }

  fn to_packed_stride(&self) -> Self {
    1
  }

  fn is_packed(&self, stride: &Self) -> bool {
    self.to_packed_stride() == *stride
  }

  fn prepend(&self, major: usize) -> Index2d {
    [major, *self]
  }

  fn append(&self, minor: usize) -> Index2d {
    [*self, minor]
  }

  fn flat_len(&self) -> usize {
    *self
  }

  fn flat_index(&self, stride: &Self) -> usize {
    (*self * *stride) as _
  }

  fn inside(&self) -> usize {
    *self
  }

  fn outside(&self) -> usize {
    *self
  }

  fn dim(&self) -> usize {
    1
  }
}

impl ArrayIndex for Index2d {
  type Range = Range2d;
  type Above = Index3d;

  fn zero() -> Self {
    [0, 0]
  }

  fn add(&self, shift: &Self) -> Self {
    [ self[0] + shift[0],
      self[1] + shift[1], ]
  }

  fn sub(&self, shift: &Self) -> Self {
    [ self[0] - shift[0],
      self[1] - shift[1], ]
  }

  fn to_packed_stride(&self) -> Self {
    let mut s = [0, 0];
    s[0] = 1;
    s[1] = s[0] * self[0];
    s
  }

  fn is_packed(&self, stride: &Self) -> bool {
    self.to_packed_stride() == *stride
  }

  fn prepend(&self, major: usize) -> Index3d {
    [major, self[0], self[1]]
  }

  fn append(&self, minor: usize) -> Index3d {
    [self[0], self[1], minor]
  }

  fn flat_len(&self) -> usize {
    self[0] * self[1]
  }

  fn flat_index(&self, stride: &Self) -> usize {
    ( self[0] * stride[0] +
      self[1] * stride[1] ) as _
  }

  fn inside(&self) -> usize {
    self[0]
  }

  fn outside(&self) -> usize {
    self[1]
  }

  fn dim(&self) -> usize {
    2
  }
}

impl ArrayIndex for Index3d {
  type Range = Range3d;
  type Above = Index4d;

  fn zero() -> Self {
    [0, 0, 0]
  }

  fn add(&self, shift: &Self) -> Self {
    [ self[0] + shift[0],
      self[1] + shift[1],
      self[2] + shift[2], ]
  }

  fn sub(&self, shift: &Self) -> Self {
    [ self[0] - shift[0],
      self[1] - shift[1],
      self[2] - shift[2], ]
  }

  fn to_packed_stride(&self) -> Self {
    let mut s = [0, 0, 0];
    s[0] = 1;
    s[1] = s[0] * self[0];
    s[2] = s[1] * self[1];
    s
  }

  fn is_packed(&self, stride: &Self) -> bool {
    self.to_packed_stride() == *stride
  }

  fn prepend(&self, major: usize) -> Index4d {
    [major, self[0], self[1], self[2]]
  }

  fn append(&self, minor: usize) -> Index4d {
    [self[0], self[1], self[2], minor]
  }

  fn flat_len(&self) -> usize {
    self[0] * self[1] * self[2]
  }

  fn flat_index(&self, stride: &Self) -> usize {
    ( self[0] * stride[0] +
      self[1] * stride[1] +
      self[2] * stride[2] ) as _
  }

  fn inside(&self) -> usize {
    self[0]
  }

  fn outside(&self) -> usize {
    self[2]
  }

  fn dim(&self) -> usize {
    3
  }
}

impl ArrayIndex for Index4d {
  type Range = Range4d;
  type Above = Index5d;

  fn add(&self, shift: &Self) -> Self {
    [ self[0] + shift[0],
      self[1] + shift[1],
      self[2] + shift[2],
      self[3] + shift[3], ]
  }

  fn sub(&self, shift: &Self) -> Self {
    [ self[0] - shift[0],
      self[1] - shift[1],
      self[2] - shift[2],
      self[3] - shift[3], ]
  }

  fn to_packed_stride(&self) -> Self {
    let mut s = [0, 0, 0, 0];
    s[0] = 1;
    s[1] = s[0] * self[0];
    s[2] = s[1] * self[1];
    s[3] = s[2] * self[2];
    s
  }

  fn zero() -> Self {
    [0, 0, 0, 0]
  }

  fn is_packed(&self, stride: &Self) -> bool {
    self.to_packed_stride() == *stride
  }

  fn prepend(&self, major: usize) -> Index5d {
    [major, self[0], self[1], self[2], self[3]]
  }

  fn append(&self, minor: usize) -> Index5d {
    [self[0], self[1], self[2], self[3], minor]
  }

  fn flat_len(&self) -> usize {
    self[0] * self[1] * self[2] * self[3]
  }

  fn flat_index(&self, stride: &Self) -> usize {
    ( self[0] * stride[0] +
      self[1] * stride[1] +
      self[2] * stride[2] +
      self[3] * stride[3] ) as _
  }

  fn inside(&self) -> usize {
    self[0]
  }

  fn outside(&self) -> usize {
    self[3]
  }

  fn dim(&self) -> usize {
    4
  }
}

impl ArrayIndex for Index5d {
  type Range = Range5d;
  type Above = UnimplIndex;

  fn zero() -> Self {
    [0, 0, 0, 0, 0]
  }

  fn add(&self, shift: &Self) -> Self {
    [ self[0] + shift[0],
      self[1] + shift[1],
      self[2] + shift[2],
      self[3] + shift[3],
      self[4] + shift[4], ]
  }

  fn sub(&self, shift: &Self) -> Self {
    [ self[0] - shift[0],
      self[1] - shift[1],
      self[2] - shift[2],
      self[3] - shift[3],
      self[4] - shift[4], ]
  }

  fn to_packed_stride(&self) -> Self {
    let mut s = [0, 0, 0, 0, 0];
    s[0] = 1;
    s[1] = s[0] * self[0];
    s[2] = s[1] * self[1];
    s[3] = s[2] * self[2];
    s[4] = s[3] * self[3];
    s
  }

  fn is_packed(&self, stride: &Self) -> bool {
    self.to_packed_stride() == *stride
  }

  fn prepend(&self, major: usize) -> UnimplIndex {
    unimplemented!();
  }

  fn append(&self, minor: usize) -> UnimplIndex {
    unimplemented!();
  }

  fn flat_len(&self) -> usize {
    self[0] * self[1] * self[2] * self[3] * self[4]
  }

  fn flat_index(&self, stride: &Self) -> usize {
    ( self[0] * stride[0] +
      self[1] * stride[1] +
      self[2] * stride[2] +
      self[3] * stride[3] +
      self[4] * stride[4] ) as _
  }

  fn inside(&self) -> usize {
    self[0]
  }

  fn outside(&self) -> usize {
    self[4]
  }

  fn dim(&self) -> usize {
    5
  }
}

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

pub struct BatchWrap<T>(pub T);

pub trait BatchArray: Array {
  fn batch_size(&self) -> usize;
  fn set_batch_size(&mut self, new_batch_sz: usize);
}

pub trait GPUDeviceArrayZeros: Array {
  fn zeros(size: Self::Idx, conn: GPUDeviceConn) -> Self where Self: Sized;
  //fn zeros_with_offset_stride(size: Self::Idx, offset: Self::Idx, stride: Self::Idx, conn: &GPUDeviceConn) -> Self where Self: Sized;
}

pub trait GPUDeviceBatchArrayZeros: BatchArray {
  fn zeros(size: Self::Idx, batch_sz: usize, conn: GPUDeviceConn) -> Self where Self: Sized;
  //fn zeros_with_offset_stride(size: Self::Idx, offset: Self::Idx, stride: Self::Idx, batch_sz: usize, conn: &GPUDeviceConn) -> Self where Self: Sized;
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

/*impl<Idx, T> Clone for GPUDeviceArray<Idx, T> where Idx: Copy, T: Copy {
  fn clone(&self) -> Self {
    // TODO
    panic!("unimplemented: cloning arrays");
  }
}*/

impl<Idx, T> GPUDeviceArrayZeros for GPUDeviceArray<Idx, T> where Idx: ArrayIndex + Copy, T: Copy + 'static {
  fn zeros(size: Idx, conn: GPUDeviceConn) -> Self {
    //let mem: Arc<GPUDeviceAllocMem<T>> = ;
    GPUDeviceArray{
      size:     size,
      offset:   Idx::zero(),
      stride:   size.to_packed_stride(),
      mem:      Arc::new(unsafe { GPUDeviceAllocMem::<T>::alloc(size.flat_len(), conn) }),
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
