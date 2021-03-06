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

use ::*;
use ffi::routines_gpu::*;

use arrayidx::*;
use cuda::runtime::*;
use cuda_rand::{CurandGenerator};
use memarray::*;
use parking_lot::{Mutex};

use std::marker::{PhantomData};
use std::mem::{size_of};
use std::ops::{RangeBounds};
use std::sync::{Arc};

pub mod linalg;
pub mod parallel;
pub mod tensor;

#[inline]
fn sz2uint(sz: usize) -> u32 {
  assert!(sz <= u32::max_value() as _);
  sz as _
}

//pub struct BatchWrap<T>(pub T);

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

pub trait SqueezeView<ViewTy> {
  fn as_view_or_squeeze(&self, axis: isize) -> ViewTy;
}

impl<ViewTy, A> SqueezeView<ViewTy> for A where A: AsView<ViewTy=ViewTy> {
  default fn as_view_or_squeeze(&self, _axis: isize) -> ViewTy {
    self.as_view()
  }
}

// TODO: waiting on trait aliases.
//pub trait GPUFlatView<T> = FlatView<FlatViewTy=GPUDeviceArrayView1d<T>> where T: Copy;
//pub trait GPUFlatViewMut<T> = GPUFlatView<T> + FlatViewMut<FlatViewMutTy=GPUDeviceArrayViewMut1d<T>> where T: Copy;

/*pub trait View<Idx> {
  fn view(self, idx: Idx) -> Self where Self: Sized;
}*/

pub trait GPUDeviceZerosShape<T>: Shape where T: Copy {
  fn zeros_shape(shape: Self::Shape, conn: GPUDeviceConn) -> Self where Self: Sized;
  fn zeros_shape_with_alloc<Alloc>(allocator: Alloc, shape: Self::Shape, conn: GPUDeviceConn) -> Self where Self: Sized, T: Copy + 'static, Alloc: GPUDeviceAlloc<T> + Sized;
}

pub trait GPUDeviceZerosNd<Idx, T> where Idx: ArrayIndex, T: Copy {
  fn zeros_nd(nd_shape: Vec<usize>, conn: GPUDeviceConn) -> Self where Self: Sized;
  fn zeros_nd_with_alloc<Alloc>(allocator: Alloc, nd_shape: Vec<usize>, conn: GPUDeviceConn) -> Self where Self: Sized, T: Copy + 'static, Alloc: GPUDeviceAlloc<T> + Sized;
}

pub trait GPUDeviceArrayZeros<T>: Array where T: Copy {
  fn zeros(size: Self::Idx, conn: GPUDeviceConn) -> Self where Self: Sized;
  fn zeros_with_alloc<Alloc>(allocator: Alloc, size: Self::Idx, conn: GPUDeviceConn) -> Self where Self: Sized, T: Copy + 'static, Alloc: GPUDeviceAlloc<T> + Sized;
}

pub trait GPUDeviceBatchArrayZeros<T>: BatchArray where T: Copy {
  fn zeros(size: Self::Idx, batch_sz: usize, conn: GPUDeviceConn) -> Self where Self: Sized;
  //fn zeros_with_storage(size: Self::Idx, batch_sz: usize, mem: Arc<GPUDeviceMem<T>>, conn: GPUDeviceConn) -> Self where Self: Sized;
}

pub type GPUHostScalar<T>  = GPUHostArray0d<T>;
pub type GPUHostArray0d<T> = MemArray<Index0d, T, GPUHostMem<T>>;
pub type GPUHostArray1d<T> = MemArray<Index1d, T, GPUHostMem<T>>;
pub type GPUHostArray2d<T> = MemArray<Index2d, T, GPUHostMem<T>>;
pub type GPUHostArray3d<T> = MemArray<Index3d, T, GPUHostMem<T>>;
pub type GPUHostArray4d<T> = MemArray<Index4d, T, GPUHostMem<T>>;

#[derive(Clone)]
pub struct GPUDeviceArray<Idx, T> where T: Copy {
  base:     *mut T,
  size:     Idx,
  offset:   Idx,
  stride:   Idx,
  mem:      Arc<GPUDeviceAsyncMem<T>>,
}

pub type GPUDeviceScalar<T>  = GPUDeviceArray0d<T>;
pub type GPUDeviceArray0d<T> = GPUDeviceArray<Index0d, T>;
pub type GPUDeviceArray1d<T> = GPUDeviceArray<Index1d, T>;
pub type GPUDeviceArray2d<T> = GPUDeviceArray<Index2d, T>;
pub type GPUDeviceArray3d<T> = GPUDeviceArray<Index3d, T>;
pub type GPUDeviceArray4d<T> = GPUDeviceArray<Index4d, T>;
pub type GPUDeviceArray5d<T> = GPUDeviceArray<Index5d, T>;

impl<Idx, T> GPUDeviceArray<Idx, T> where Idx: ArrayIndex, T: Copy + 'static {
  pub unsafe fn alloc_default(size: Idx, conn: GPUDeviceConn) -> Self {
    Self::alloc(GPUDeviceDefaultAlloc::default(), size, conn)
  }

  pub unsafe fn alloc<Alloc>(allocator: Alloc, size: Idx, conn: GPUDeviceConn) -> Self where Alloc: GPUDeviceAlloc<T> + Sized {
    let mem = unsafe { allocator.alloc(size.flat_len(), conn) };
    GPUDeviceArray{
      base:     mem.raw_mut_dptr(),
      size:     size.clone(),
      offset:   Idx::zero(),
      stride:   size.to_packed_stride(),
      mem:      Arc::new(mem),
      //mem:      Arc::new(Mutex::new(mem)),
    }
  }

  pub unsafe fn alloc_shaped(size: Idx, offset: Idx, stride: Idx, conn: GPUDeviceConn) -> Self {
    let ph_flat_len = stride.outside() * size.outside();
    // TODO
    unimplemented!();
    /*let mem = Arc::new(unsafe { GPUDeviceRawMem::<T>::alloc(ph_flat_len, conn) });
    GPUDeviceArray{
      size:     size,
      offset:   offset,
      stride:   stride,
      mem:      mem,
    }*/
  }
}

impl<Idx, T> GPUDeviceAsync for GPUDeviceArray<Idx, T> where Idx: ArrayIndex, T: Copy {
  fn async_state(&self) -> Arc<Mutex<GPUAsyncState>> {
    self.mem.async_state()
  }
}

impl<Idx, T> GPUDeviceZerosShape<T> for GPUDeviceArray<Idx, T> where Idx: ArrayIndex, T: ZeroBits + Copy + 'static {
  fn zeros_shape(size: Idx, conn: GPUDeviceConn) -> Self {
    let mut arr = unsafe { GPUDeviceArray::<Idx, T>::alloc_default(size, conn.clone()) };
    arr.as_view_mut().set_zeros(conn);
    arr
  }

  fn zeros_shape_with_alloc<Alloc>(allocator: Alloc, size: Idx, conn: GPUDeviceConn) -> Self where Alloc: GPUDeviceAlloc<T> + Sized {
    let mut arr = unsafe { GPUDeviceArray::<Idx, T>::alloc(allocator, size, conn.clone()) };
    arr.as_view_mut().set_zeros(conn);
    arr
  }
}

impl<Idx, T> GPUDeviceZerosNd<Idx, T> for GPUDeviceArray<Idx, T> where Idx: ArrayIndex, T: ZeroBits + Copy + 'static {
  fn zeros_nd(nd_size: Vec<usize>, conn: GPUDeviceConn) -> Self {
    let size = Idx::from_nd(nd_size);
    let mut arr = unsafe { GPUDeviceArray::<Idx, T>::alloc_default(size, conn.clone()) };
    arr.as_view_mut().set_zeros(conn);
    arr
  }

  fn zeros_nd_with_alloc<Alloc>(allocator: Alloc, nd_size: Vec<usize>, conn: GPUDeviceConn) -> Self where Alloc: GPUDeviceAlloc<T> + Sized {
    let size = Idx::from_nd(nd_size);
    let mut arr = unsafe { GPUDeviceArray::<Idx, T>::alloc(allocator, size, conn.clone()) };
    arr.as_view_mut().set_zeros(conn);
    arr
  }
}

impl<Idx, T> GPUDeviceArrayZeros<T> for GPUDeviceArray<Idx, T> where Idx: ArrayIndex, T: ZeroBits + Copy + 'static {
  fn zeros(size: Idx, conn: GPUDeviceConn) -> Self {
    let mut arr = unsafe { GPUDeviceArray::<Idx, T>::alloc_default(size, conn.clone()) };
    arr.as_view_mut().set_zeros(conn);
    arr
  }

  fn zeros_with_alloc<Alloc>(allocator: Alloc, size: Idx, conn: GPUDeviceConn) -> Self where Alloc: GPUDeviceAlloc<T> + Sized {
    let mut arr = unsafe { GPUDeviceArray::<Idx, T>::alloc(allocator, size, conn.clone()) };
    arr.as_view_mut().set_zeros(conn);
    arr
  }
}

impl<Idx, T> Shape for GPUDeviceArray<Idx, T> where Idx: ArrayIndex, T: Copy {
  type Shape = Idx;

  fn shape(&self) -> Idx {
    self.size.clone()
  }
}

impl<Idx, T> Reshape for GPUDeviceArray<Idx, T> where Idx: ArrayIndex, T: Copy {
  fn reshape(&mut self, new_size: Idx) {
    // FIXME: allow changing the array size.
    assert_eq!(self.size, new_size);
  }
}

impl<Idx, T> Array for GPUDeviceArray<Idx, T> where Idx: ArrayIndex, T: Copy {
  type Idx = Idx;

  fn size(&self) -> Idx {
    self.size.clone()
  }
}

impl<Idx, T> DenseArray for GPUDeviceArray<Idx, T> where Idx: ArrayIndex, T: Copy {
  fn offset(&self) -> Idx {
    self.offset.clone()
  }

  fn stride(&self) -> Idx {
    self.stride.clone()
  }
}

impl<Idx, T> AsView for GPUDeviceArray<Idx, T> where Idx: Clone, T: Copy {
  type ViewTy = GPUDeviceArrayView<Idx, T>;

  fn as_view(&self) -> GPUDeviceArrayView<Idx, T> {
    GPUDeviceArrayView{
      base:     self.base,
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
      base:     self.base,
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
    if self.is_packed() {
      let flat_size = self.flat_size();
      Some(GPUDeviceArrayView{
        base:   self.base,
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

impl<Idx, T> FlatViewMut for GPUDeviceArray<Idx, T> where Idx: ArrayIndex, T: Copy {
  type FlatViewMutTy = GPUDeviceArrayViewMut1d<T>;

  fn flat_view_mut(&mut self) -> Option<GPUDeviceArrayViewMut1d<T>> {
    if self.is_packed() {
      let flat_size = self.flat_size();
      Some(GPUDeviceArrayViewMut{
        base:   self.base,
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
  base:         *mut T,
  size:         Idx,
  offset:       Idx,
  stride:       Idx,
  batch_sz:     usize,
  max_batch_sz: usize,
  mem:          Arc<GPUDeviceAsyncMem<T>>,
  //mem:          Arc<Mutex<GPUDeviceAsyncMem<T>>>,
}

pub type GPUDeviceInnerBatchScalar<T>  = GPUDeviceInnerBatchArray<Index0d, T>;
pub type GPUDeviceInnerBatchArray1d<T> = GPUDeviceInnerBatchArray<Index1d, T>;
pub type GPUDeviceInnerBatchArray2d<T> = GPUDeviceInnerBatchArray<Index2d, T>;
pub type GPUDeviceInnerBatchArray3d<T> = GPUDeviceInnerBatchArray<Index3d, T>;
pub type GPUDeviceInnerBatchArray4d<T> = GPUDeviceInnerBatchArray<Index4d, T>;

impl<Idx, T> Shape for GPUDeviceInnerBatchArray<Idx, T> where Idx: ArrayIndex, T: Copy {
  type Shape = (usize, Idx);

  fn shape(&self) -> (usize, Idx) {
    (self.batch_sz, self.size.clone())
  }
}

impl<Idx, T> Array for GPUDeviceInnerBatchArray<Idx, T> where Idx: ArrayIndex, T: Copy {
  type Idx = Idx;

  fn size(&self) -> Idx {
    self.size.clone()
  }
}

impl<Idx, T> BatchArray for GPUDeviceInnerBatchArray<Idx, T> where Idx: ArrayIndex, T: Copy {
  fn max_batch_size(&self) -> usize {
    self.max_batch_sz
  }

  fn batch_size(&self) -> usize {
    self.batch_sz
  }

  fn set_batch_size(&mut self, new_batch_sz: usize) {
    assert!(new_batch_sz <= self.max_batch_sz);
    self.batch_sz = new_batch_sz;
  }
}

impl<Idx, T> AsView for GPUDeviceInnerBatchArray<Idx, T> where Idx: ArrayIndex, T: Copy {
  type ViewTy = GPUDeviceArrayView<Idx::Above, T>;

  fn as_view(&self) -> GPUDeviceArrayView<Idx::Above, T> {
    let view_size = self.size.index_prepend(self.batch_sz);
    // TODO
    unimplemented!();
  }
}

pub struct GPUDeviceOuterBatchArray<Idx, T> where T: Copy {
  base:         *mut T,
  size:         Idx,
  offset:       Idx,
  stride:       Idx,
  batch_sz:     usize,
  max_batch_sz: usize,
  mem:          Arc<GPUDeviceAsyncMem<T>>,
  //mem:          Arc<Mutex<GPUDeviceAsyncMem<T>>>,
}

pub type GPUDeviceOuterBatchScalar<T>  = GPUDeviceOuterBatchArray<Index0d, T>;
pub type GPUDeviceOuterBatchArray1d<T> = GPUDeviceOuterBatchArray<Index1d, T>;
pub type GPUDeviceOuterBatchArray2d<T> = GPUDeviceOuterBatchArray<Index2d, T>;
pub type GPUDeviceOuterBatchArray3d<T> = GPUDeviceOuterBatchArray<Index3d, T>;
pub type GPUDeviceOuterBatchArray4d<T> = GPUDeviceOuterBatchArray<Index4d, T>;

impl<Idx, T> GPUDeviceOuterBatchArray<Idx, T> where Idx: ArrayIndex, T: Copy + 'static {
  pub unsafe fn alloc_default(size: Idx, max_batch_sz: usize, conn: GPUDeviceConn) -> Self {
    Self::alloc(GPUDeviceDefaultAlloc::default(), size, max_batch_sz, conn)
  }

  pub unsafe fn alloc<Alloc>(allocator: Alloc, size: Idx, max_batch_sz: usize, conn: GPUDeviceConn) -> Self where Alloc: GPUDeviceAlloc<T> + Sized {
    let mem = unsafe { allocator.alloc(size.flat_len() * max_batch_sz, conn) };
    GPUDeviceOuterBatchArray{
      base:     mem.raw_mut_dptr(),
      size:     size.clone(),
      offset:   Idx::zero(),
      stride:   size.to_packed_stride(),
      batch_sz:     max_batch_sz,
      max_batch_sz: max_batch_sz,
      mem:      Arc::new(mem),
    }
  }
}

impl<Idx, T> GPUDeviceZerosShape<T> for GPUDeviceOuterBatchArray<Idx, T> where Idx: ArrayIndex, T: ZeroBits + Copy + 'static {
  fn zeros_shape(shape: (Idx, usize), conn: GPUDeviceConn) -> Self {
    let mut arr = unsafe { GPUDeviceOuterBatchArray::<Idx, T>::alloc_default(shape.0, shape.1, conn.clone()) };
    arr.flat_view_mut().unwrap().set_zeros(conn);
    arr
  }

  fn zeros_shape_with_alloc<Alloc>(allocator: Alloc, shape: (Idx, usize), conn: GPUDeviceConn) -> Self where Alloc: GPUDeviceAlloc<T> + Sized {
    let mut arr = unsafe { GPUDeviceOuterBatchArray::<Idx, T>::alloc(allocator, shape.0, shape.1, conn.clone()) };
    arr.flat_view_mut().unwrap().set_zeros(conn);
    arr
  }
}

impl<Idx, T> GPUDeviceZerosNd<Idx, T> for GPUDeviceOuterBatchArray<<<Idx as ArrayIndex>::Above as ArrayIndex>::Below, T> where Idx: ArrayIndex, T: ZeroBits + Copy + 'static {
  fn zeros_nd(nd_shape: Vec<usize>, conn: GPUDeviceConn) -> Self {
    let ndim = nd_shape.len();
    let max_batch_sz = nd_shape[ndim - 1];
    let shape = Idx::Above::from_nd(nd_shape);
    let size = shape.index_cut((ndim - 1) as _);
    let mut arr = unsafe { GPUDeviceOuterBatchArray::<<<Idx as ArrayIndex>::Above as ArrayIndex>::Below, T>::alloc_default(size, max_batch_sz, conn.clone()) };
    arr.as_view_mut().set_zeros(conn);
    arr
  }

  fn zeros_nd_with_alloc<Alloc>(allocator: Alloc, nd_shape: Vec<usize>, conn: GPUDeviceConn) -> Self where Alloc: GPUDeviceAlloc<T> + Sized {
    let ndim = nd_shape.len();
    let max_batch_sz = nd_shape[ndim - 1];
    let shape = Idx::Above::from_nd(nd_shape);
    let size = shape.index_cut((ndim - 1) as _);
    let mut arr = unsafe { GPUDeviceOuterBatchArray::<<<Idx as ArrayIndex>::Above as ArrayIndex>::Below, T>::alloc(allocator, size, max_batch_sz, conn.clone()) };
    arr.as_view_mut().set_zeros(conn);
    arr
  }
}

impl<Idx, T> GPUDeviceBatchArrayZeros<T> for GPUDeviceOuterBatchArray<Idx, T>
where Idx: ArrayIndex,
      T: ZeroBits + Copy + 'static,
      //GPUDeviceArrayViewMut<Idx::Above, T>: GPUDeviceArrayViewMutOpsExt,
{
  fn zeros(size: Idx, max_batch_sz: usize, conn: GPUDeviceConn) -> Self {
    let mut arr = unsafe { GPUDeviceOuterBatchArray::<Idx, T>::alloc_default(size, max_batch_sz, conn.clone()) };
    /*arr.as_view_mut().set_zeros(conn);*/
    arr.flat_view_mut().unwrap().set_zeros(conn);
    arr
  }
}

impl<Idx, T> GPUDeviceAsync for GPUDeviceOuterBatchArray<Idx, T> where Idx: ArrayIndex, T: Copy {
  fn async_state(&self) -> Arc<Mutex<GPUAsyncState>> {
    self.mem.async_state()
  }
}

impl<Idx, T> Shape for GPUDeviceOuterBatchArray<Idx, T> where Idx: ArrayIndex, T: Copy {
  type Shape = (Idx, usize);

  fn shape(&self) -> (Idx, usize) {
    (self.size.clone(), self.batch_sz)
  }
}

impl<Idx, T> Reshape for GPUDeviceOuterBatchArray<Idx, T> where Idx: ArrayIndex, T: Copy {
  fn reshape(&mut self, new_shape: (Idx, usize)) {
    // FIXME: allow changing the array size.
    assert_eq!(self.size, new_shape.0);
    self.set_batch_size(new_shape.1);
  }
}

impl<Idx, T> Array for GPUDeviceOuterBatchArray<Idx, T> where Idx: ArrayIndex, T: Copy {
  type Idx = Idx;

  fn size(&self) -> Idx {
    self.size.clone()
  }
}

impl<Idx, T> DenseArray for GPUDeviceOuterBatchArray<Idx, T> where Idx: ArrayIndex, T: Copy {
  fn offset(&self) -> Idx {
    self.offset.clone()
  }

  fn stride(&self) -> Idx {
    self.stride.clone()
  }
}

impl<Idx, T> BatchArray for GPUDeviceOuterBatchArray<Idx, T> where Idx: ArrayIndex, T: Copy {
  fn max_batch_size(&self) -> usize {
    self.max_batch_sz
  }

  fn batch_size(&self) -> usize {
    self.batch_sz
  }

  fn set_batch_size(&mut self, new_batch_sz: usize) {
    assert!(new_batch_sz <= self.max_batch_sz);
    self.batch_sz = new_batch_sz;
  }
}

impl<Idx, T> AsView for GPUDeviceOuterBatchArray<Idx, T> where Idx: ArrayIndex, T: Copy {
  type ViewTy = GPUDeviceArrayView<Idx::Above, T>;

  fn as_view(&self) -> GPUDeviceArrayView<Idx::Above, T> {
    let view_size = self.size.index_append(self.batch_sz);
    let view_offset = self.offset.index_append(0);
    // TODO: support for a batch stride.
    let view_stride = self.stride.stride_append_packed(self.size.outside());
    GPUDeviceArrayView{
      base:     self.base,
      size:     view_size,
      offset:   view_offset,
      stride:   view_stride,
      mem:      self.mem.clone(),
    }
  }
}

impl<Idx, T> AsViewMut for GPUDeviceOuterBatchArray<Idx, T> where Idx: ArrayIndex, T: Copy {
  type ViewMutTy = GPUDeviceArrayViewMut<Idx::Above, T>;

  fn as_view_mut(&mut self) -> GPUDeviceArrayViewMut<Idx::Above, T> {
    let view_size = self.size.index_append(self.batch_sz);
    let view_offset = self.offset.index_append(0);
    // TODO: support for a batch stride.
    let view_stride = self.stride.stride_append_packed(self.size.outside());
    GPUDeviceArrayViewMut{
      base:     self.base,
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
    if self.is_packed() {
      let flat_size = self.flat_size() * self.batch_sz;
      Some(GPUDeviceArrayView{
        base:   self.base,
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

impl<Idx, T> FlatViewMut for GPUDeviceOuterBatchArray<Idx, T> where Idx: ArrayIndex, T: Copy {
  type FlatViewMutTy = GPUDeviceArrayViewMut1d<T>;

  fn flat_view_mut(&mut self) -> Option<GPUDeviceArrayViewMut1d<T>> {
    if self.is_packed() {
      let flat_size = self.flat_size() * self.batch_sz;
      Some(GPUDeviceArrayViewMut{
        base:   self.base,
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
  base:     *mut T,
  size:     Idx,
  offset:   Idx,
  stride:   Idx,
  mem:      Arc<GPUDeviceAsyncMem<T>>,
  //mem:      Arc<Mutex<GPUDeviceAsyncMem<T>>>,
}

pub type GPUDeviceScalarView<T>  = GPUDeviceArrayView<Index0d, T>;
pub type GPUDeviceArrayView1d<T> = GPUDeviceArrayView<Index1d, T>;
pub type GPUDeviceArrayView2d<T> = GPUDeviceArrayView<Index2d, T>;
pub type GPUDeviceArrayView3d<T> = GPUDeviceArrayView<Index3d, T>;
pub type GPUDeviceArrayView4d<T> = GPUDeviceArrayView<Index4d, T>;
pub type GPUDeviceArrayView5d<T> = GPUDeviceArrayView<Index5d, T>;

impl<T> SqueezeView<GPUDeviceArrayView4d<T>> for GPUDeviceArrayView5d<T> where T: Copy {
  fn as_view_or_squeeze(&self, axis: isize) -> GPUDeviceArrayView4d<T> {
    if self.is_packed() && self.size().index_at(axis) == 1 {
      assert_eq!(0, self.offset().index_at(axis));
      let squeeze_size = self.size().index_cut(axis);
      let squeeze_offset = self.offset().index_cut(axis);
      let squeeze_stride = squeeze_size.to_packed_stride();
      GPUDeviceArrayView{
        base:     self.base,
        size:     squeeze_size,
        offset:   squeeze_offset,
        stride:   squeeze_stride,
        mem:      self.mem.clone(),
      }
    } else {
      unimplemented!();
    }
  }
}

impl<Idx, T> GPUDeviceMem<T> for GPUDeviceArrayView<Idx, T> where Idx: ArrayIndex, T: Copy {
  unsafe fn raw_dptr(&self) -> *const T {
    self.base.offset(self.flat_offset() as _)
  }

  unsafe fn raw_mut_dptr(&self) -> *mut T {
    unreachable!();
  }
}

impl<Idx, T> GPUDevicePlace for GPUDeviceArrayView<Idx, T> where Idx: ArrayIndex, T: Copy {
  fn device(&self) -> GPUDeviceId {
    self.mem.device()
  }
}

impl<Idx, T> GPUDeviceAsync for GPUDeviceArrayView<Idx, T> where Idx: ArrayIndex, T: Copy {
  fn async_state(&self) -> Arc<Mutex<GPUAsyncState>> {
    self.mem.async_state()
  }
}

impl<Idx, T> GPUDeviceAsyncMem<T> for GPUDeviceArrayView<Idx, T> where Idx: ArrayIndex, T: Copy + 'static {
  fn wait(&self, conn: GPUDeviceConn) -> GPUDeviceAsyncWaitGuard<T, Self> {
    let frame = thread_gpu_async_frame();
    frame.borrow_mut()._wait(self.async_state(), conn);
    GPUDeviceAsyncWaitGuard{mem: self, _mrk: PhantomData}
  }

  fn wait_mut(&mut self, conn: GPUDeviceConn) -> GPUDeviceAsyncWaitMutGuard<T, Self> {
    unreachable!();
  }
}

impl<Idx, T> Shape for GPUDeviceArrayView<Idx, T> where Idx: ArrayIndex, T: Copy {
  type Shape = Idx;

  fn shape(&self) -> Idx {
    self.size.clone()
  }
}

impl<Idx, T> Array for GPUDeviceArrayView<Idx, T> where Idx: ArrayIndex, T: Copy {
  type Idx = Idx;

  fn size(&self) -> Idx {
    self.size.clone()
  }
}

impl<Idx, T> DenseArray for GPUDeviceArrayView<Idx, T> where Idx: ArrayIndex, T: Copy {
  fn offset(&self) -> Idx {
    self.offset.clone()
  }

  fn stride(&self) -> Idx {
    self.stride.clone()
  }
}

impl<T> GPUDeviceArrayView1d<T> where T: Copy {
  pub fn view<R>(self, r: R) -> GPUDeviceArrayView1d<T>
  where R: RangeBounds<usize>,
  {
    let (start_idx, end_idx) = range2idxs_1d(r, self.size);
    let view_size = end_idx - start_idx;
    let view_offset = self.offset + start_idx;
    GPUDeviceArrayView{
      base:     self.base,
      size:     view_size,
      offset:   view_offset,
      stride:   self.stride,
      mem:      self.mem.clone(),
    }
  }
}

impl<Idx, T> GPUDeviceArrayView<Idx, T> where Idx: ArrayIndex, T: Copy {
  pub fn sync_dump_mem(&self, mut dst: MemArrayViewMut<Idx, T>, conn: GPUDeviceConn) {
    assert_eq!(self.size, dst.size());
    if self.is_packed() && dst.is_packed() {
      let len = self.flat_size();
      conn.sync();
      {
        let src = self.wait(conn.clone());
        let mut stream = conn.cuda_stream();
        match unsafe { cuda_memcpy_async(
            dst.as_mut_ptr(),
            src.as_dptr(),
            len,
            CudaMemcpyKind::DeviceToHost,
            &mut stream,
        ) } {
          Err(_) => panic!(),
          Ok(_) => {}
        }
      }
      conn.sync();
    } else {
      unimplemented!();
    }
  }
}

#[derive(Clone)]
pub struct GPUDeviceArrayViewMut<Idx, T> where T: Copy {
  base:     *mut T,
  size:     Idx,
  offset:   Idx,
  stride:   Idx,
  mem:      Arc<GPUDeviceAsyncMem<T>>,
  //mem:      Arc<GPUDeviceAsyncMem<T>>,
}

pub type GPUDeviceScalarViewMut<T>  = GPUDeviceArrayViewMut<Index0d, T>;
pub type GPUDeviceArrayViewMut1d<T> = GPUDeviceArrayViewMut<Index1d, T>;
pub type GPUDeviceArrayViewMut2d<T> = GPUDeviceArrayViewMut<Index2d, T>;
pub type GPUDeviceArrayViewMut3d<T> = GPUDeviceArrayViewMut<Index3d, T>;
pub type GPUDeviceArrayViewMut4d<T> = GPUDeviceArrayViewMut<Index4d, T>;
pub type GPUDeviceArrayViewMut5d<T> = GPUDeviceArrayViewMut<Index5d, T>;

impl<Idx, T> GPUDeviceMem<T> for GPUDeviceArrayViewMut<Idx, T> where Idx: ArrayIndex, T: Copy {
  unsafe fn raw_dptr(&self) -> *const T {
    self.base.offset(self.flat_offset() as _)
  }

  unsafe fn raw_mut_dptr(&self) -> *mut T {
    self.base.offset(self.flat_offset() as _)
  }
}

impl<Idx, T> GPUDevicePlace for GPUDeviceArrayViewMut<Idx, T> where Idx: ArrayIndex, T: Copy {
  fn device(&self) -> GPUDeviceId {
    self.mem.device()
  }
}

impl<Idx, T> GPUDeviceAsync for GPUDeviceArrayViewMut<Idx, T> where Idx: ArrayIndex, T: Copy {
  fn async_state(&self) -> Arc<Mutex<GPUAsyncState>> {
    self.mem.async_state()
  }
}

impl<Idx, T> GPUDeviceAsyncMem<T> for GPUDeviceArrayViewMut<Idx, T> where Idx: ArrayIndex, T: Copy + 'static {
  fn wait(&self, conn: GPUDeviceConn) -> GPUDeviceAsyncWaitGuard<T, Self> {
    let frame = thread_gpu_async_frame();
    frame.borrow_mut()._wait(self.async_state(), conn);
    GPUDeviceAsyncWaitGuard{mem: self, _mrk: PhantomData}
  }

  fn wait_mut(&mut self, conn: GPUDeviceConn) -> GPUDeviceAsyncWaitMutGuard<T, Self> {
    let frame = thread_gpu_async_frame();
    frame.borrow_mut()._wait(self.async_state(), conn);
    GPUDeviceAsyncWaitMutGuard{mem: self, _mrk: PhantomData}
  }
}

impl<Idx, T> Shape for GPUDeviceArrayViewMut<Idx, T> where Idx: ArrayIndex, T: Copy {
  type Shape = Idx;

  fn shape(&self) -> Idx {
    self.size.clone()
  }
}

impl<Idx, T> Array for GPUDeviceArrayViewMut<Idx, T> where Idx: ArrayIndex, T: Copy {
  type Idx = Idx;

  fn size(&self) -> Idx {
    self.size.clone()
  }
}

impl<Idx, T> DenseArray for GPUDeviceArrayViewMut<Idx, T> where Idx: ArrayIndex, T: Copy {
  fn offset(&self) -> Idx {
    self.offset.clone()
  }

  fn stride(&self) -> Idx {
    self.stride.clone()
  }
}

impl<Idx, T> FlatView for GPUDeviceArrayViewMut<Idx, T> where Idx: ArrayIndex, T: Copy {
  type FlatViewTy = GPUDeviceArrayView1d<T>;

  fn flat_view(&self) -> Option<GPUDeviceArrayView1d<T>> {
    if self.is_packed() {
      // FIXME: use correct flat offset.
      assert_eq!(self.offset, Idx::zero());
      let flat_size = self.flat_size();
      Some(GPUDeviceArrayView{
        base:   self.base,
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

impl<T> GPUDeviceArrayViewMut1d<T> where T: Copy {
  pub fn view_mut<R>(self, r: R) -> GPUDeviceArrayViewMut1d<T>
  where R: RangeBounds<usize>,
  {
    let (start_idx, end_idx) = range2idxs_1d(r, self.size);
    let view_size = end_idx - start_idx;
    let view_offset = self.offset + start_idx;
    GPUDeviceArrayViewMut{
      base:     self.base,
      size:     view_size,
      offset:   view_offset,
      stride:   self.stride,
      mem:      self.mem.clone(),
    }
  }
}

impl<Idx, T> GPUDeviceArrayViewMut<Idx, T> where Idx: ArrayIndex, T: Copy {
  pub fn sync_copy_mem(&mut self, src: MemArrayView<Idx, T>, conn: GPUDeviceConn) {
    assert_eq!(self.size, src.size());
    if self.is_packed() && src.is_packed() {
      let len = self.flat_size();
      conn.sync();
      {
        let mut dst = self.wait_mut(conn.clone());
        let mut stream = conn.cuda_stream();
        match unsafe { cuda_memcpy_async(
            dst.as_mut_dptr(),
            src.as_ptr(),
            len,
            CudaMemcpyKind::HostToDevice,
            &mut stream,
        ) } {
          Err(_) => panic!(),
          Ok(_) => {}
        }
      }
      conn.sync();
    } else {
      unimplemented!();
    }
  }
}

pub trait GPUDeviceArrayViewMutOpsExt<T> where T: Copy {
  type ViewTy;

  fn set_zeros(&mut self, conn: GPUDeviceConn);
  fn fill_random(&mut self, rng: &mut CurandGenerator, conn: GPUDeviceConn);
  fn fill_uniform(&mut self, rng: &mut CurandGenerator, conn: GPUDeviceConn);
  fn copy(&mut self, src: Self::ViewTy, conn: GPUDeviceConn);
  fn add(&mut self, x: Self::ViewTy, conn: GPUDeviceConn);
  fn mult(&mut self, x: Self::ViewTy, conn: GPUDeviceConn);
  fn div(&mut self, x: Self::ViewTy, conn: GPUDeviceConn);
  fn is_nonzero(&mut self, x: Self::ViewTy, conn: GPUDeviceConn);
}

pub trait GPUDeviceArrayViewMutConstantOpsExt<T>: GPUDeviceArrayViewMutOpsExt<T> where T: Copy {
  fn set_constant(&mut self, c: T, conn: GPUDeviceConn);
  fn add_constant_inplace(&mut self, c: T, conn: GPUDeviceConn);
  fn add_constant(&mut self, c: T, x: Self::ViewTy, conn: GPUDeviceConn);
  fn mult_constant(&mut self, c: T, x: Self::ViewTy, conn: GPUDeviceConn);
  fn div_constant(&mut self, x: Self::ViewTy, c: T, conn: GPUDeviceConn);
  fn online_add(&mut self, c: T, x: Self::ViewTy, conn: GPUDeviceConn);
  fn online_discount(&mut self, c: T, x: Self::ViewTy, conn: GPUDeviceConn);
  fn online_average(&mut self, c: T, x: Self::ViewTy, conn: GPUDeviceConn);
}

pub trait GPUDeviceArrayViewHalo1dOpsExt<T> where T: Copy {
  fn pack_left_boundary(&self, halo_radius: usize, axis: isize, dst_buf: &mut GPUDeviceArrayViewMut1d<T>, conn: GPUDeviceConn);
  fn pack_right_boundary(&self, halo_radius: usize, axis: isize, dst_buf: &mut GPUDeviceArrayViewMut1d<T>, conn: GPUDeviceConn);
  fn pack_left_ghost(&self, halo_radius: usize, axis: isize, dst_buf: &mut GPUDeviceArrayViewMut1d<T>, conn: GPUDeviceConn);
  fn pack_right_ghost(&self, halo_radius: usize, axis: isize, dst_buf: &mut GPUDeviceArrayViewMut1d<T>, conn: GPUDeviceConn);
}

pub trait GPUDeviceArrayViewMutHalo1dOpsExt<T>: GPUDeviceArrayViewMutOpsExt<T> where T: Copy {
  fn zero_ghost(&mut self, halo_radius: usize, axis: isize, conn: GPUDeviceConn);
  fn copy_pad(&mut self, halo_radius: usize, axis: isize, src: Self::ViewTy, conn: GPUDeviceConn);
  fn copy_unpad(&mut self, halo_radius: usize, axis: isize, src: Self::ViewTy, conn: GPUDeviceConn);
  fn unpack_into_left_ghost(&mut self, halo_radius: usize, axis: isize, src: GPUDeviceArrayView1d<T>, conn: GPUDeviceConn);
  fn unpack_into_right_ghost(&mut self, halo_radius: usize, axis: isize, src: GPUDeviceArrayView1d<T>, conn: GPUDeviceConn);
  fn unpack_accumulate_into_left_boundary(&mut self, halo_radius: usize, axis: isize, src: GPUDeviceArrayView1d<T>, conn: GPUDeviceConn);
  fn unpack_accumulate_into_right_boundary(&mut self, halo_radius: usize, axis: isize, src: GPUDeviceArrayView1d<T>, conn: GPUDeviceConn);
}

impl<Idx, T> GPUDeviceArrayViewMutOpsExt<T> for GPUDeviceArrayViewMut<Idx, T> where Idx: ArrayIndex, T: Copy + 'static {
  type ViewTy = GPUDeviceArrayView<Idx, T>;

  default fn set_zeros(&mut self, conn: GPUDeviceConn) {
    // TODO: how to gracefully handle?
    unimplemented!();
  }

  default fn fill_random(&mut self, rng: &mut CurandGenerator, conn: GPUDeviceConn) {
    unimplemented!();
  }

  default fn fill_uniform(&mut self, rng: &mut CurandGenerator, conn: GPUDeviceConn) {
    unimplemented!();
  }

  default fn copy(&mut self, src: GPUDeviceArrayView<Idx, T>, conn: GPUDeviceConn) {
    assert_eq!(self.size(), src.size());
    if self.is_packed() && src.is_packed() {
      let len = self.flat_size();
      assert_eq!(len, src.flat_size());
      let src = src.wait(conn.clone());
      let mut dst = self.wait_mut(conn.clone());
      let mut stream = conn.cuda_stream();
      match unsafe { cuda_memcpy_async(
          dst.as_mut_dptr(),
          src.as_dptr(),
          len,
          CudaMemcpyKind::DeviceToDevice,
          &mut stream,
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

  default fn mult(&mut self, x: GPUDeviceArrayView<Idx, T>, conn: GPUDeviceConn) {
    // TODO
    unimplemented!();
  }

  default fn div(&mut self, x: GPUDeviceArrayView<Idx, T>, conn: GPUDeviceConn) {
    // TODO
    unimplemented!();
  }

  default fn is_nonzero(&mut self, x: GPUDeviceArrayView<Idx, T>, conn: GPUDeviceConn) {
    // TODO
    unimplemented!();
  }
}

impl<Idx, T> GPUDeviceArrayViewMutOpsExt<T> for GPUDeviceArrayViewMut<Idx, T> where Idx: ArrayIndex, T: ZeroBits + 'static {
  default fn set_zeros(&mut self, conn: GPUDeviceConn) {
    if self.is_packed() {
      let mut dst = self.wait_mut(conn.clone());
      let mut stream = conn.cuda_stream();
      let res = unsafe { cuda_memset_async(
          dst.as_mut_dptr() as *mut u8,
          0,
          // TODO: need the footprint of the array w/ stride and offset.
          /*dst.inner().size().outside() * dst.inner().stride().outside() * size_of::<T>(),*/
          dst.inner().flat_size() * size_of::<T>(),
          &mut stream,
      ) };
      assert!(res.is_ok());
    } else {
      unimplemented!();
    }
  }

  default fn add(&mut self, x: GPUDeviceArrayView<Idx, T>, conn: GPUDeviceConn) {
    unimplemented!();
  }

  default fn mult(&mut self, x: GPUDeviceArrayView<Idx, T>, conn: GPUDeviceConn) {
    unimplemented!();
  }

  default fn div(&mut self, x: GPUDeviceArrayView<Idx, T>, conn: GPUDeviceConn) {
    unimplemented!();
  }
}

impl<Idx> GPUDeviceArrayViewMutOpsExt<f32> for GPUDeviceArrayViewMut<Idx, f32> where Idx: ArrayIndex {
  fn fill_uniform(&mut self, rng: &mut CurandGenerator, conn: GPUDeviceConn) {
    if self.is_packed() {
      let len = self.flat_size();
      let mut dst = self.wait_mut(conn.clone());
      let mut stream = conn.cuda_stream();
      // TODO: error handling.
      assert!(rng.set_stream(&mut stream).is_ok());
      let res = unsafe { rng.generate_uniform(dst.as_mut_dptr(), len) };
      assert!(res.is_ok());
    } else {
      unimplemented!();
    }
  }

  fn add(&mut self, x: GPUDeviceArrayView<Idx, f32>, conn: GPUDeviceConn) {
    assert_eq!(x.size(), self.size());
    if x.is_packed() && self.is_packed() {
      // TODO: size checks.
      let len = self.flat_size();
      let x = x.wait(conn.clone());
      let mut y = self.wait_mut(conn.clone());
      let mut stream = conn.cuda_stream();
      // TODO: error handling.
      unsafe { gpudevicemem_flat_add_inplace_f32(
          sz2uint(len),
          x.as_dptr(),
          y.as_mut_dptr(),
          conn.cuda_kernel_cfg() as *const _,
          stream.as_mut_ptr(),
      ) };
    } else {
      unimplemented!();
    }
  }

  fn mult(&mut self, x: GPUDeviceArrayView<Idx, f32>, conn: GPUDeviceConn) {
    assert_eq!(x.size(), self.size());
    if x.is_packed() && self.is_packed() {
      // TODO: size checks.
      let len = self.flat_size();
      let x = x.wait(conn.clone());
      let mut y = self.wait_mut(conn.clone());
      let mut stream = conn.cuda_stream();
      // TODO: error handling.
      unsafe { gpudevicemem_flat_mult_inplace_f32(
          sz2uint(len),
          x.as_dptr(),
          y.as_mut_dptr(),
          conn.cuda_kernel_cfg() as *const _,
          stream.as_mut_ptr(),
      ) };
    } else {
      unimplemented!();
    }
  }

  fn div(&mut self, x: GPUDeviceArrayView<Idx, f32>, conn: GPUDeviceConn) {
    assert_eq!(x.size(), self.size());
    if x.is_packed() && self.is_packed() {
      // TODO: size checks.
      let len = self.flat_size();
      let x = x.wait(conn.clone());
      let mut y = self.wait_mut(conn.clone());
      let mut stream = conn.cuda_stream();
      // TODO: error handling.
      unsafe { gpudevicemem_flat_rdiv_inplace_f32(
          sz2uint(len),
          x.as_dptr(),
          y.as_mut_dptr(),
          conn.cuda_kernel_cfg() as *const _,
          stream.as_mut_ptr(),
      ) };
    } else {
      unimplemented!();
    }
  }

  fn is_nonzero(&mut self, x: GPUDeviceArrayView<Idx, f32>, conn: GPUDeviceConn) {
    assert_eq!(x.size(), self.size());
    if x.is_packed() && self.is_packed() {
      // TODO: size checks.
      let len = self.flat_size();
      let x = x.wait(conn.clone());
      let mut y = self.wait_mut(conn.clone());
      let mut stream = conn.cuda_stream();
      // TODO: error handling.
      unsafe { gpudevicemem_is_nonzero_flat_map_f32(
          sz2uint(len),
          x.as_dptr(),
          y.as_mut_dptr(),
          conn.cuda_kernel_cfg() as *const _,
          stream.as_mut_ptr(),
      ) };
    } else {
      unimplemented!();
    }
  }
}

impl<Idx> GPUDeviceArrayViewMutOpsExt<f64> for GPUDeviceArrayViewMut<Idx, f64> where Idx: ArrayIndex {
  fn fill_uniform(&mut self, rng: &mut CurandGenerator, conn: GPUDeviceConn) {
    if self.is_packed() {
      let len = self.flat_size();
      let mut dst = self.wait_mut(conn.clone());
      let mut stream = conn.cuda_stream();
      // TODO: error handling.
      assert!(rng.set_stream(&mut stream).is_ok());
      let res = unsafe { rng.generate_uniform64(dst.as_mut_dptr(), len) };
      assert!(res.is_ok());
    } else {
      unimplemented!();
    }
  }
}

impl<Idx> GPUDeviceArrayViewMutOpsExt<u32> for GPUDeviceArrayViewMut<Idx, u32> where Idx: ArrayIndex {
  fn fill_random(&mut self, rng: &mut CurandGenerator, conn: GPUDeviceConn) {
    if self.is_packed() {
      let len = self.flat_size();
      let mut dst = self.wait_mut(conn.clone());
      let mut stream = conn.cuda_stream();
      // TODO: error handling.
      assert!(rng.set_stream(&mut stream).is_ok());
      let res = unsafe { rng.generate(dst.as_mut_dptr(), len) };
      assert!(res.is_ok());
    } else {
      unimplemented!();
    }
  }
}

impl<Idx, T> GPUDeviceArrayViewMutConstantOpsExt<T> for GPUDeviceArrayViewMut<Idx, T> where Idx: ArrayIndex, T: Copy + 'static {
  default fn set_constant(&mut self, c: T, conn: GPUDeviceConn) {
    // TODO
    unimplemented!();
  }

  default fn add_constant_inplace(&mut self, c: T, conn: GPUDeviceConn) {
    // TODO
    unimplemented!();
  }

  default fn add_constant(&mut self, c: T, x: GPUDeviceArrayView<Idx, T>, conn: GPUDeviceConn) {
    // TODO
    unimplemented!();
  }

  default fn mult_constant(&mut self, c: T, x: GPUDeviceArrayView<Idx, T>, conn: GPUDeviceConn) {
    // TODO
    unimplemented!();
  }

  default fn div_constant(&mut self, x: GPUDeviceArrayView<Idx, T>, c: T, conn: GPUDeviceConn) {
    // TODO
    unimplemented!();
  }

  default fn online_add(&mut self, c: T, x: GPUDeviceArrayView<Idx, T>, conn: GPUDeviceConn) {
    // TODO
    unimplemented!();
  }

  default fn online_discount(&mut self, c: T, x: GPUDeviceArrayView<Idx, T>, conn: GPUDeviceConn) {
    // TODO
    unimplemented!();
  }

  default fn online_average(&mut self, c: T, x: GPUDeviceArrayView<Idx, T>, conn: GPUDeviceConn) {
    // TODO
    unimplemented!();
  }
}

impl<Idx> GPUDeviceArrayViewMutConstantOpsExt<u8> for GPUDeviceArrayViewMut<Idx, u8> where Idx: ArrayIndex {
  fn set_constant(&mut self, c: u8, conn: GPUDeviceConn) {
    if self.is_packed() {
      let len = self.flat_size();
      let mut y = self.wait_mut(conn.clone());
      let mut stream = conn.cuda_stream();
      let status = unsafe { cuda_memset_async(
          y.as_mut_dptr(),
          c as i32,
          len,
          &mut *stream,
      ) };
      assert!(status.is_ok());
    } else {
      unimplemented!();
    }
  }

  fn add_constant_inplace(&mut self, c: u8, conn: GPUDeviceConn) {
    // TODO
    unimplemented!();
  }

  fn add_constant(&mut self, c: u8, x: GPUDeviceArrayView<Idx, u8>, conn: GPUDeviceConn) {
    // TODO
    unimplemented!();
  }

  fn mult_constant(&mut self, c: u8, x: GPUDeviceArrayView<Idx, u8>, conn: GPUDeviceConn) {
    // TODO
    unimplemented!();
  }

  fn div_constant(&mut self, x: GPUDeviceArrayView<Idx, u8>, c: u8, conn: GPUDeviceConn) {
    // TODO
    unimplemented!();
  }

  fn online_add(&mut self, c: u8, x: GPUDeviceArrayView<Idx, u8>, conn: GPUDeviceConn) {
    // TODO
    unimplemented!();
  }

  fn online_discount(&mut self, c: u8, x: GPUDeviceArrayView<Idx, u8>, conn: GPUDeviceConn) {
    // TODO
    unimplemented!();
  }

  fn online_average(&mut self, c: u8, x: GPUDeviceArrayView<Idx, u8>, conn: GPUDeviceConn) {
    // TODO
    unimplemented!();
  }
}

impl<Idx> GPUDeviceArrayViewMutConstantOpsExt<f32> for GPUDeviceArrayViewMut<Idx, f32> where Idx: ArrayIndex {
  fn set_constant(&mut self, c: f32, conn: GPUDeviceConn) {
    if self.is_packed() {
      let len = self.flat_size();
      let mut y = self.wait_mut(conn.clone());
      let mut stream = conn.cuda_stream();
      // TODO: error handling.
      unsafe { gpudevicemem_set_constant_flat_map_inplace_f32(
          sz2uint(len),
          c,
          y.as_mut_dptr(),
          conn.cuda_kernel_cfg() as *const _,
          stream.as_mut_ptr(),
      ) };
    } else {
      unimplemented!();
    }
  }

  fn add_constant_inplace(&mut self, c: f32, conn: GPUDeviceConn) {
    if self.is_packed() {
      let len = self.flat_size();
      let mut y = self.wait_mut(conn.clone());
      let mut stream = conn.cuda_stream();
      // TODO: error handling.
      unsafe { gpudevicemem_add_constant_flat_map_inplace_f32(
          sz2uint(len),
          c,
          y.as_mut_dptr(),
          conn.cuda_kernel_cfg() as *const _,
          stream.as_mut_ptr(),
      ) };
    } else {
      unimplemented!();
    }
  }

  fn add_constant(&mut self, c: f32, x: GPUDeviceArrayView<Idx, f32>, conn: GPUDeviceConn) {
    if self.is_packed() {
      // TODO: size checks.
      let len = self.flat_size();
      let x = x.wait(conn.clone());
      let mut y = self.wait_mut(conn.clone());
      let mut stream = conn.cuda_stream();
      // TODO: error handling.
      unsafe { gpudevicemem_add_constant_flat_map_f32(
          sz2uint(len),
          c,
          x.as_dptr(),
          y.as_mut_dptr(),
          conn.cuda_kernel_cfg() as *const _,
          stream.as_mut_ptr(),
      ) };
    } else {
      unimplemented!();
    }
  }

  fn mult_constant(&mut self, c: f32, x: GPUDeviceArrayView<Idx, f32>, conn: GPUDeviceConn) {
    if self.is_packed() {
      // TODO: size checks.
      let len = self.flat_size();
      let x = x.wait(conn.clone());
      let mut y = self.wait_mut(conn.clone());
      let mut stream = conn.cuda_stream();
      // TODO: error handling.
      unsafe { gpudevicemem_mult_constant_flat_map_f32(
          sz2uint(len),
          c,
          x.as_dptr(),
          y.as_mut_dptr(),
          conn.cuda_kernel_cfg() as *const _,
          stream.as_mut_ptr(),
      ) };
    } else {
      unimplemented!();
    }
  }

  fn div_constant(&mut self, x: GPUDeviceArrayView<Idx, f32>, c: f32, conn: GPUDeviceConn) {
    if self.is_packed() {
      // TODO: size checks.
      let len = self.flat_size();
      let x = x.wait(conn.clone());
      let mut y = self.wait_mut(conn.clone());
      let mut stream = conn.cuda_stream();
      // TODO: error handling.
      unsafe { gpudevicemem_rdiv_constant_flat_map_f32(
          sz2uint(len),
          c,
          x.as_dptr(),
          y.as_mut_dptr(),
          conn.cuda_kernel_cfg() as *const _,
          stream.as_mut_ptr(),
      ) };
    } else {
      unimplemented!();
    }
  }

  fn online_add(&mut self, c: f32, x: GPUDeviceArrayView<Idx, f32>, conn: GPUDeviceConn) {
    if self.is_packed() {
      // TODO: size checks.
      let len = self.flat_size();
      let x = x.wait(conn.clone());
      let mut y = self.wait_mut(conn.clone());
      let mut stream = conn.cuda_stream();
      // TODO: error handling.
      unsafe { gpudevicemem_online_add_flat_map_accum_f32(
          sz2uint(len),
          c,
          x.as_dptr(),
          y.as_mut_dptr(),
          conn.cuda_kernel_cfg() as *const _,
          stream.as_mut_ptr(),
      ) };
    } else {
      unimplemented!();
    }
  }

  fn online_discount(&mut self, c: f32, x: GPUDeviceArrayView<Idx, f32>, conn: GPUDeviceConn) {
    if self.is_packed() {
      // TODO: size checks.
      let len = self.flat_size();
      let x = x.wait(conn.clone());
      let mut y = self.wait_mut(conn.clone());
      let mut stream = conn.cuda_stream();
      // TODO: error handling.
      unsafe { gpudevicemem_online_discount_flat_map_accum_f32(
          sz2uint(len),
          c,
          x.as_dptr(),
          y.as_mut_dptr(),
          conn.cuda_kernel_cfg() as *const _,
          stream.as_mut_ptr(),
      ) };
    } else {
      unimplemented!();
    }
  }

  fn online_average(&mut self, c: f32, x: GPUDeviceArrayView<Idx, f32>, conn: GPUDeviceConn) {
    if self.is_packed() {
      // TODO: size checks.
      let len = self.flat_size();
      let x = x.wait(conn.clone());
      let mut y = self.wait_mut(conn.clone());
      let mut stream = conn.cuda_stream();
      // TODO: error handling.
      unsafe { gpudevicemem_online_average_flat_map_accum_f32(
          sz2uint(len),
          c,
          x.as_dptr(),
          y.as_mut_dptr(),
          conn.cuda_kernel_cfg() as *const _,
          stream.as_mut_ptr(),
      ) };
    } else {
      unimplemented!();
    }
  }
}

impl GPUDeviceArrayViewHalo1dOpsExt<f32> for GPUDeviceArrayView4d<f32> {
  fn pack_left_boundary(&self, halo_radius: usize, axis: isize, dst_buf: &mut GPUDeviceArrayViewMut1d<f32>, conn: GPUDeviceConn) {
    // TODO: size checks.
    assert!(self.size().index_at(axis) > 2 * halo_radius);
    let (prefix_ndsize, axis_ndsize, suffix_ndsize) = self.size()._to_nd().splice_at(axis);
    assert_eq!(prefix_ndsize.flat_len() * halo_radius * suffix_ndsize.flat_len(), dst_buf.size());
    let packed = self.is_packed() && dst_buf.is_packed();
    if packed {
      let src_arr = self.wait(conn.clone());
      let mut dst_buf = dst_buf.wait_mut(conn.clone());
      match axis {
        // TODO
        2 => {
          assert_eq!(prefix_ndsize.flat_len(), src_arr.inner().size().index_cut(3).index_cut(2).flat_len());
          assert_eq!(axis_ndsize.flat_len(), src_arr.inner().size().index_at(2));
          assert_eq!(suffix_ndsize.flat_len(), src_arr.inner().size().index_at(3));
          let mut stream = conn.cuda_stream();
          unsafe { gpudevicemem_halo_ring_3d1_copy_lboundary_to_buf_f32(
              sz2uint(halo_radius),
              sz2uint(prefix_ndsize.flat_len()),
              sz2uint(axis_ndsize.flat_len()),
              sz2uint(suffix_ndsize.flat_len()),
              src_arr.as_dptr() as *mut f32,
              dst_buf.as_mut_dptr(),
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

  fn pack_right_boundary(&self, halo_radius: usize, axis: isize, dst_buf: &mut GPUDeviceArrayViewMut1d<f32>, conn: GPUDeviceConn) {
    // TODO: size checks.
    assert!(self.size().index_at(axis) > 2 * halo_radius);
    let (prefix_ndsize, axis_ndsize, suffix_ndsize) = self.size()._to_nd().splice_at(axis);
    assert_eq!(prefix_ndsize.flat_len() * halo_radius * suffix_ndsize.flat_len(), dst_buf.size());
    let packed = self.is_packed() && dst_buf.is_packed();
    if packed {
      let src_arr = self.wait(conn.clone());
      let mut dst_buf = dst_buf.wait_mut(conn.clone());
      match axis {
        // TODO
        2 => {
          assert_eq!(prefix_ndsize.flat_len(), src_arr.inner().size().index_cut(3).index_cut(2).flat_len());
          assert_eq!(axis_ndsize.flat_len(), src_arr.inner().size().index_at(2));
          assert_eq!(suffix_ndsize.flat_len(), src_arr.inner().size().index_at(3));
          let mut stream = conn.cuda_stream();
          unsafe { gpudevicemem_halo_ring_3d1_copy_rboundary_to_buf_f32(
              sz2uint(halo_radius),
              sz2uint(prefix_ndsize.flat_len()),
              sz2uint(axis_ndsize.flat_len()),
              sz2uint(suffix_ndsize.flat_len()),
              src_arr.as_dptr() as *mut f32,
              dst_buf.as_mut_dptr(),
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

  fn pack_left_ghost(&self, halo_radius: usize, axis: isize, dst_buf: &mut GPUDeviceArrayViewMut1d<f32>, conn: GPUDeviceConn) {
    // TODO: size checks.
    assert!(self.size().index_at(axis) > 2 * halo_radius);
    let (prefix_ndsize, axis_ndsize, suffix_ndsize) = self.size()._to_nd().splice_at(axis);
    assert_eq!(prefix_ndsize.flat_len() * halo_radius * suffix_ndsize.flat_len(), dst_buf.size());
    let packed = self.is_packed() && dst_buf.is_packed();
    if packed {
      let src_arr = self.wait(conn.clone());
      let mut dst_buf = dst_buf.wait_mut(conn.clone());
      match axis {
        // TODO
        2 => {
          assert_eq!(prefix_ndsize.flat_len(), src_arr.inner().size().index_cut(3).index_cut(2).flat_len());
          assert_eq!(axis_ndsize.flat_len(), src_arr.inner().size().index_at(2));
          assert_eq!(suffix_ndsize.flat_len(), src_arr.inner().size().index_at(3));
          let mut stream = conn.cuda_stream();
          unsafe { gpudevicemem_halo_ring_3d1_copy_lghost_to_buf_f32(
              sz2uint(halo_radius),
              sz2uint(prefix_ndsize.flat_len()),
              sz2uint(axis_ndsize.flat_len()),
              sz2uint(suffix_ndsize.flat_len()),
              src_arr.as_dptr() as *mut f32,
              dst_buf.as_mut_dptr(),
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

  fn pack_right_ghost(&self, halo_radius: usize, axis: isize, dst_buf: &mut GPUDeviceArrayViewMut1d<f32>, conn: GPUDeviceConn) {
    // TODO: size checks.
    assert!(self.size().index_at(axis) > 2 * halo_radius);
    let (prefix_ndsize, axis_ndsize, suffix_ndsize) = self.size()._to_nd().splice_at(axis);
    assert_eq!(prefix_ndsize.flat_len() * halo_radius * suffix_ndsize.flat_len(), dst_buf.size());
    let packed = self.is_packed() && dst_buf.is_packed();
    if packed {
      let src_arr = self.wait(conn.clone());
      let mut dst_buf = dst_buf.wait_mut(conn.clone());
      match axis {
        // TODO
        2 => {
          assert_eq!(prefix_ndsize.flat_len(), src_arr.inner().size().index_cut(3).index_cut(2).flat_len());
          assert_eq!(axis_ndsize.flat_len(), src_arr.inner().size().index_at(2));
          assert_eq!(suffix_ndsize.flat_len(), src_arr.inner().size().index_at(3));
          let mut stream = conn.cuda_stream();
          unsafe { gpudevicemem_halo_ring_3d1_copy_rghost_to_buf_f32(
              sz2uint(halo_radius),
              sz2uint(prefix_ndsize.flat_len()),
              sz2uint(axis_ndsize.flat_len()),
              sz2uint(suffix_ndsize.flat_len()),
              src_arr.as_dptr() as *mut f32,
              dst_buf.as_mut_dptr(),
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

impl GPUDeviceArrayViewMutHalo1dOpsExt<f32> for GPUDeviceArrayViewMut4d<f32> {
  fn zero_ghost(&mut self, halo_radius: usize, axis: isize, conn: GPUDeviceConn) {
    // TODO: size checks.
    let (prefix_ndsize, axis_ndsize, suffix_ndsize) = self.size()._to_nd().splice_at(axis);
    assert!(axis_ndsize.flat_len() > 2 * halo_radius);
    let packed = self.is_packed();
    if packed {
      let mut dst_arr = self.wait_mut(conn.clone());
      match axis {
        // TODO
        2 => {
          assert_eq!(prefix_ndsize.flat_len(), dst_arr.inner().size().index_cut(3).index_cut(2).flat_len());
          assert_eq!(axis_ndsize.flat_len(), dst_arr.inner().size().index_at(2));
          assert_eq!(suffix_ndsize.flat_len(), dst_arr.inner().size().index_at(3));
          let mut stream = conn.cuda_stream();
          unsafe { gpudevicemem_halo_ring_3d1_zero_lghost_f32(
              sz2uint(halo_radius),
              sz2uint(prefix_ndsize.flat_len()),
              sz2uint(axis_ndsize.flat_len()),
              sz2uint(suffix_ndsize.flat_len()),
              dst_arr.as_mut_dptr(),
              conn.cuda_kernel_cfg() as *const _,
              stream.as_mut_ptr(),
          ) };
          unsafe { gpudevicemem_halo_ring_3d1_zero_rghost_f32(
              sz2uint(halo_radius),
              sz2uint(prefix_ndsize.flat_len()),
              sz2uint(axis_ndsize.flat_len()),
              sz2uint(suffix_ndsize.flat_len()),
              dst_arr.as_mut_dptr(),
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

  fn copy_pad(&mut self, halo_radius: usize, axis: isize, src: GPUDeviceArrayView4d<f32>, conn: GPUDeviceConn) {
    // TODO
  }

  fn copy_unpad(&mut self, halo_radius: usize, axis: isize, src: GPUDeviceArrayView4d<f32>, conn: GPUDeviceConn) {
    // TODO
  }

  fn unpack_into_left_ghost(&mut self, halo_radius: usize, axis: isize, src: GPUDeviceArrayView1d<f32>, conn: GPUDeviceConn) {
    // TODO
  }

  fn unpack_into_right_ghost(&mut self, halo_radius: usize, axis: isize, src: GPUDeviceArrayView1d<f32>, conn: GPUDeviceConn) {
    // TODO
  }

  fn unpack_accumulate_into_left_boundary(&mut self, halo_radius: usize, axis: isize, src: GPUDeviceArrayView1d<f32>, conn: GPUDeviceConn) {
    // TODO
  }

  fn unpack_accumulate_into_right_boundary(&mut self, halo_radius: usize, axis: isize, src: GPUDeviceArrayView1d<f32>, conn: GPUDeviceConn) {
    // TODO
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
