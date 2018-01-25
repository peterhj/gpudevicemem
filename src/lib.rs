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

extern crate arrayidx;
extern crate cuda;
extern crate cuda_blas;
extern crate cuda_dnn;

use cuda::ffi::runtime::{cudaError_t, cudaStream_t, cudaDeviceProp};
use cuda::runtime::*;
use cuda_blas::{CublasHandle};
use cuda_dnn::{CudnnHandle};

//use libc::{c_void};
use std::mem::{size_of, transmute};
use std::os::raw::{c_void};
use std::rc::{Rc};
use std::sync::{Arc, Mutex, RwLock};
use std::sync::atomic::{AtomicUsize, Ordering};

pub mod array;

//static STREAM_POOL_UID_COUNTER: AtomicU64 = ATOMIC_U64_INIT;

#[derive(Clone, Copy)]
pub struct GPUDeviceId(pub i32);

impl GPUDeviceId {
  /*pub fn enumerate() -> Vec<DeviceId> {
    // TODO
    unimplemented!();
  }*/

  pub fn rank(&self) -> usize {
    self.0 as _
  }
}

pub struct GPUDeviceRawStream {
  dev_id:       GPUDeviceId,
  raw_stream:   Arc<CudaStream>,
  sync_event:   Arc<CudaEvent>,
}

impl GPUDeviceRawStream {
  pub fn new(dev_id: GPUDeviceId) -> Self {
    let prev_dev = CudaDevice::get_current().unwrap();
    CudaDevice::set_current(dev_id.0).unwrap();
    let raw_stream = Arc::new(CudaStream::create().unwrap());
    let sync_event = Arc::new(CudaEvent::create_fastest().unwrap());
    CudaDevice::set_current(prev_dev).unwrap();
    GPUDeviceRawStream{
      dev_id:       dev_id,
      raw_stream:   raw_stream,
      sync_event:   sync_event,
    }
  }

  pub fn sync_event(&self) -> Arc<CudaEvent> {
    self.sync_event.clone()
  }

  pub fn wait_on_event(&self, event: &CudaEvent) {
    self.raw_stream.wait_event(event).unwrap();
  }
}

#[derive(Clone, Copy, Debug)]
pub struct GPUDeviceArchSummary {
  pub mp_count:             usize,
  pub sharedmem_sz_per_mp:  usize,
  pub register_sz_per_mp:   usize,
}

#[derive(Clone)]
pub struct GPUDeviceStreamPool {
  dev_id:   GPUDeviceId,
  arch_sum: GPUDeviceArchSummary,
  stream:   Arc<GPUDeviceRawStream>,
  cublas_h: Arc<CublasHandle>,
  //cublas_h: Arc<Mutex<Option<CublasHandle>>>,
  cudnn_h:  Arc<Mutex<Option<CudnnHandle>>>,
  //workspace_sz: Arc<AtomicUsize>,
  //workspace:    Arc<Mutex<Option<GPUDeviceRawMem<u8>>>>,
}

impl GPUDeviceStreamPool {
  pub fn new(dev_id: GPUDeviceId/*, pool_size: usize*/) -> GPUDeviceStreamPool {
    let dev = dev_id.0;
    let dev_prop = Arc::new(CudaDevice::get_properties(dev as usize).unwrap());
    /*println!("DEBUG: cuda: device: index: {} smp count: {}", dev, dev_prop.multiprocessor_count);
    println!("DEBUG: cuda: device: index: {} shared mem per smp: {}", dev, dev_prop.shared_mem_per_multiprocessor);
    println!("DEBUG: cuda: device: index: {} registers per smp: {}", dev, dev_prop.regs_per_multiprocessor);*/
    let arch_sum = GPUDeviceArchSummary{
      mp_count:             dev_prop.multiProcessorCount as _,
      sharedmem_sz_per_mp:  dev_prop.sharedMemPerMultiprocessor as _,
      register_sz_per_mp:   dev_prop.regsPerMultiprocessor as _,
    };
    println!("DEBUG: GPUDeviceStreamPool: dev: {} arch: {:?}", dev, arch_sum);
    let stream = Arc::new(GPUDeviceRawStream::new(dev_id));
    let cublas_h = Arc::new(CublasHandle::create().unwrap());
    GPUDeviceStreamPool{
      dev_id:   dev_id,
      arch_sum: arch_sum,
      stream:   stream,
      cublas_h: cublas_h,
      //cublas_h: Arc::new(Mutex::new(None)),
      cudnn_h:  Arc::new(Mutex::new(None)),
      //workspace_sz: Arc::new(AtomicUsize::new(0)),
      //workspace:    Arc::new(Mutex::new(None)),
    }
  }

  pub fn conn<'a>(&'a self) -> GPUDeviceConn<'a> {
    let prev_dev = CudaDevice::get_current().unwrap();
    CudaDevice::set_current(self.dev_id.0).unwrap();
    GPUDeviceConn{
      dev:      self.dev_id,
      pop_dev:  GPUDeviceId(prev_dev),
      stream:   self.stream.clone(),
      cublas_h: self.cublas_h.clone(),
      cudnn_h:  self.cudnn_h.clone(),
      borrow:   &(),
    }
  }
}

#[derive(Clone)]
pub struct GPUDeviceConn<'a> {
  dev:      GPUDeviceId,
  pop_dev:  GPUDeviceId,
  stream:   Arc<GPUDeviceRawStream>,
  cublas_h: Arc<CublasHandle>,
  //cublas_h: Arc<Mutex<Option<CublasHandle>>>,
  cudnn_h:  Arc<Mutex<Option<CudnnHandle>>>,
  borrow:   &'a (),
}

impl<'a> Drop for GPUDeviceConn<'a> {
  fn drop(&mut self) {
    CudaDevice::set_current(self.pop_dev.0).unwrap();
  }
}

impl<'a> GPUDeviceConn<'a> {
  pub fn device(&self) -> GPUDeviceId {
    self.dev
  }

  pub fn stream(&self) -> Arc<GPUDeviceRawStream> {
    self.stream.clone()
  }

  pub fn cublas(&self) -> Arc<CublasHandle> {
    self.cublas_h.clone()
  }
}

pub trait GPUDeviceAllocator {
  unsafe fn alloc<T>(&self, len: usize, conn: GPUDeviceConn) -> Arc<GPUDeviceMem<T>> where T: Copy + 'static;
}

pub struct GPUDeviceRawAlloc {
}

impl GPUDeviceAllocator for GPUDeviceRawAlloc {
  unsafe fn alloc<T>(&self, len: usize, conn: GPUDeviceConn) -> Arc<GPUDeviceMem<T>> where T: Copy + 'static {
    Arc::new(GPUDeviceRawMem::<T>::alloc(len, conn))
  }
}

/*pub struct GPUDeviceStreamAlloc {
}

impl GPUDeviceAllocator for GPUDeviceStreamAlloc {
  unsafe fn alloc<T>(&self, len: usize, conn: GPUDeviceConn) -> Arc<GPUDeviceMem<T>> where T: Copy + 'static {
    Arc::new(GPUDeviceStreamMem::<T>::alloc(len, conn))
  }
}

pub struct GPUDeviceCachingStreamAlloc {
}

impl GPUDeviceAllocator for GPUDeviceCachingStreamAlloc {
  unsafe fn alloc<T>(&self, len: usize, conn: GPUDeviceConn) -> Arc<GPUDeviceMem<T>> where T: Copy + 'static {
    // TODO
    Arc::new(GPUDeviceStreamMem::<T>::alloc(len, conn))
  }
}*/

pub trait GPUDeviceMem<T> where T: Copy {
  fn as_dptr(&self) -> *const T;
  fn as_mut_dptr(&self) -> *mut T;
  fn len(&self) -> usize;
}

pub struct GPUDeviceRawMem<T> {
  dev:  GPUDeviceId,
  dptr: *mut T,
  len:  usize,
  psz:  usize,
}

impl<T> GPUDeviceRawMem<T> where T: Copy {
  pub unsafe fn alloc(len: usize, conn: GPUDeviceConn) -> GPUDeviceRawMem<T> where T: Copy {
    println!("DEBUG: GPUDeviceRawMem: alloc len: {}", len);
    assert!(len <= <i32>::max_value() as usize,
        "device memory size should not exceed 2**31-1 elements");
    let dptr = match cuda_alloc_device::<T>(len) {
      Err(e) => panic!("GPUDeviceRawMem allocation failed: {:?}", e),
      Ok(dptr) => dptr,
    };
    assert!(!dptr.is_null());
    GPUDeviceRawMem{
      dev:  conn.device(),
      dptr: dptr,
      len:  len,
      psz:  len * size_of::<T>(),
    }
  }
}

impl<T> GPUDeviceMem<T> for GPUDeviceRawMem<T> where T: Copy {
  fn as_dptr(&self) -> *const T {
    self.dptr
  }

  fn as_mut_dptr(&self) -> *mut T {
    self.dptr
  }

  fn len(&self) -> usize {
    self.len
  }
}

pub struct GPUDeviceResizableMem<T> {
  dev:  GPUDeviceId,
  dptr: *mut T,
  len:  usize,
  psz:  usize,
}

impl<T> GPUDeviceResizableMem<T> where T: Copy {
  pub fn new(conn: GPUDeviceConn) -> GPUDeviceResizableMem<T> where T: Copy {
    // TODO
    unimplemented!();
  }

  pub fn reserve(&self, len: usize) {
    // TODO
    unimplemented!();
  }
}

pub struct GPUDeviceStreamMem<T> {
  dev:  GPUDeviceId,
  stream:   Arc<GPUDeviceRawStream>,
  dptr: *mut T,
  len:  usize,
  psz:  usize,
}

impl<T> Drop for GPUDeviceStreamMem<T> {
  fn drop(&mut self) {
    // TODO
    unimplemented!();
  }
}

/*impl<T> GPUDeviceStreamMem<T> where T: Copy {
  pub unsafe fn alloc(len: usize, conn: GPUDeviceConn) -> GPUDeviceStreamMem<T> where T: Copy {
    println!("DEBUG: GPUDeviceStreamMem: alloc len: {}", len);
    assert!(len <= <u32>::max_value() as usize,
        "device memory size should not exceed 2**31-1 elements");
    let dev = conn.device();
    let (dptr, psz) = match cuda_alloc_managed::<T>(len) {
      Err(e) => panic!("GPUDeviceStreamMem allocation failed: {:?}", e),
      Ok((dptr, psz)) => (dptr, psz),
    };
    assert!(!dptr.is_null());
    // TODO: set hints.
    assert!(cuda_mem_advise_set_preferred_location(dptr, psz, dev.0).is_ok());
    // TODO: attach mem to stream.
    let stream = conn.stream();
    assert!(stream.attach_single_mem_async(dptr, psz).is_ok());
    GPUDeviceStreamMem{
      dev:  dev,
      stream:   stream,
      dptr: dptr,
      len:  len,
      psz:  psz,
    }
  }
}*/

pub struct GPUDeviceToken {
  //producers:    Arc<AtomicArcList<GPUDeviceRawStream>>,
  producers:    Arc<RwLock<Vec<Arc<GPUDeviceRawStream>>>>,
}

impl GPUDeviceToken {
  pub fn post_excl(&self, stream: Arc<GPUDeviceRawStream>) {
    self.producers.write().unwrap().push(stream);
  }

  pub fn wait_excl(&self, stream: Arc<GPUDeviceRawStream>) {
    let mut producers_lock = self.producers.write().unwrap();
    let producers: Vec<_> = producers_lock.drain(..).collect();
    for producer in producers.iter() {
      // TODO
      /*if producer == stream {
        continue;
      }
      producer.record_sync_event();
      stream.wait_on_event(producer.sync_event());*/
    }
  }
}

pub struct GPUDevicePost {
  stream:   Arc<GPUDeviceRawStream>,
  xtokens:  Vec<GPUDeviceToken>,
  stokens:  Vec<GPUDeviceToken>,
}

pub struct GPUDeviceWait {
  stream:   Arc<GPUDeviceRawStream>,
  xtokens:  Vec<GPUDeviceToken>,
  stokens:  Vec<GPUDeviceToken>,
}

extern "C" fn dataflow_post(stream: cudaStream_t, status: cudaError_t, post_raw_data: *mut c_void) {
  // TODO
  let post: Arc<GPUDevicePost> = unsafe { Arc::from_raw(transmute(post_raw_data)) };
  for xtoken in post.xtokens.iter() {
    xtoken.post_excl(post.stream.clone());
  }
  assert!(post.stokens.is_empty(), "shared tokens are not supported yet");
}

extern "C" fn dataflow_wait(stream: cudaStream_t, status: cudaError_t, wait_raw_data: *mut c_void) {
  // TODO
  let wait: Arc<GPUDeviceWait> = unsafe { Arc::from_raw(transmute(wait_raw_data)) };
  for xtoken in wait.xtokens.iter() {
    xtoken.wait_excl(wait.stream.clone());
  }
  assert!(wait.stokens.is_empty(), "shared tokens are not supported yet");
}

/*pub struct GPUDeviceMemRef<T> where T: Copy {
  mem:  Rc<GPUDeviceMem<T>>,
  dptr: *mut T,
  len:  usize,
}

impl<T> GPUDeviceMemRef<T> where T: Copy {
  pub unsafe fn as_ptr(&self) -> *const T {
    self.dptr
  }

  pub fn len(&self) -> usize {
    self.len
  }

  pub fn size_bytes(&self) -> usize {
    self.len * size_of::<T>()
  }
}

pub struct GPUDeviceMemRefMut<T> where T: Copy {
  mem:  Rc<GPUDeviceMem<T>>,
  dptr: *mut T,
  len:  usize,
}

impl<T> GPUDeviceMemRefMut<T> where T: Copy {
  pub unsafe fn as_ptr(&self) -> *const T {
    self.dptr
  }

  pub unsafe fn as_mut_ptr(&self) -> *mut T {
    self.dptr
  }

  pub fn len(&self) -> usize {
    self.len
  }

  pub fn size_bytes(&self) -> usize {
    self.len * size_of::<T>()
  }
}*/
