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
extern crate float;
#[macro_use] extern crate lazy_static;
extern crate memarray;

use ffi::routines_gpu::{KernelConfig};

//use cuda::ffi::runtime::{cudaError_t, cudaStream_t, cudaDeviceProp};
use cuda::runtime::*;
use cuda_blas::{CublasHandle};
use cuda_dnn::{CudnnHandle};

use std::cmp::{max};
use std::collections::{HashMap};
use std::marker::{PhantomData};
use std::mem::{size_of};
use std::ops::{Deref, DerefMut};
use std::sync::{Arc, Mutex, MutexGuard};
//use std::sync::atomic::{AtomicUsize, Ordering};

pub mod array;
pub mod ffi;

//static STREAM_POOL_UID_COUNTER: AtomicU64 = ATOMIC_U64_INIT;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct GPUDeviceId(pub i32);

impl GPUDeviceId {
  pub fn count() -> usize {
    CudaDevice::count().unwrap()
  }

  /*pub fn enumerate() -> Vec<DeviceId> {
    // TODO
    unimplemented!();
  }*/

  pub fn rank(&self) -> usize {
    self.0 as _
  }
}

#[derive(Default)]
pub struct LazyCudaStream {
  h:    Option<CudaStream>,
}

impl Deref for LazyCudaStream {
  type Target = CudaStream;

  fn deref(&self) -> &CudaStream {
    unreachable!();
  }
}

impl DerefMut for LazyCudaStream {
  fn deref_mut(&mut self) -> &mut CudaStream {
    if self.h.is_none() {
      self.h = Some(CudaStream::create().unwrap());
    }
    self.h.as_mut().unwrap()
  }
}

#[derive(Default)]
pub struct LazyCublasHandle {
  h:    Option<CublasHandle>,
}

impl Deref for LazyCublasHandle {
  type Target = CublasHandle;

  fn deref(&self) -> &CublasHandle {
    unreachable!();
  }
}

impl DerefMut for LazyCublasHandle {
  fn deref_mut(&mut self) -> &mut CublasHandle {
    if self.h.is_none() {
      self.h = Some(CublasHandle::create().unwrap());
    }
    self.h.as_mut().unwrap()
  }
}

#[derive(Default)]
pub struct LazyCudnnHandle {
  h:    Option<CudnnHandle>,
}

impl Deref for LazyCudnnHandle {
  type Target = CudnnHandle;

  fn deref(&self) -> &CudnnHandle {
    unreachable!();
  }
}

impl DerefMut for LazyCudnnHandle {
  fn deref_mut(&mut self) -> &mut CudnnHandle {
    if self.h.is_none() {
      self.h = Some(CudnnHandle::create().unwrap());
    }
    self.h.as_mut().unwrap()
  }
}

/*pub struct GPUDeviceRawStream {
  dev_id:       GPUDeviceId,
  cuda_stream:  Arc<Mutex<LazyCudaStream>>,
  //sync_event:   Arc<CudaEvent>,
}

impl GPUDeviceRawStream {
  pub fn new(dev_id: GPUDeviceId) -> Self {
    //let prev_dev = CudaDevice::get_current().unwrap();
    //CudaDevice::set_current(dev_id.0).unwrap();
    //let cuda_stream = Arc::new(CudaStream::create().unwrap());
    //let sync_event = Arc::new(CudaEvent::create_fastest().unwrap());
    //CudaDevice::set_current(prev_dev).unwrap();
    let lazy_stream = LazyCudaStream::default();
    let cuda_stream = Arc::new(Mutex::new(lazy_stream));
    GPUDeviceRawStream{
      dev_id:       dev_id,
      cuda_stream:  cuda_stream,
      //sync_event:   sync_event,
    }
  }

  pub fn cuda_stream(&self) -> MutexGuard<LazyCudaStream> {
    self.cuda_stream.lock().unwrap()
  }

  /*pub fn sync_event(&self) -> Arc<CudaEvent> {
    self.sync_event.clone()
  }

  pub fn wait_on_event(&self, event: &CudaEvent) {
    self.cuda_stream.wait_event(event).unwrap();
  }*/
}*/

#[derive(Clone, Copy, Debug)]
pub struct GPUDeviceArchSummary {
  pub mp_count:             usize,
  pub sharedmem_sz_per_mp:  usize,
  pub register_sz_per_mp:   usize,
}

#[derive(Clone)]
pub struct GPUDeviceStreamPool {
  dev_id:       GPUDeviceId,
  arch_sum:     GPUDeviceArchSummary,
  kernel_cfg:   KernelConfig,
  cuda_stream:  Arc<Mutex<LazyCudaStream>>,
  cublas_h:     Arc<Mutex<LazyCublasHandle>>,
  cudnn_h:      Arc<Mutex<LazyCudnnHandle>>,
  burst_arena:  GPUDeviceBurstArena,
}

impl GPUDeviceStreamPool {
  pub fn new(dev_id: GPUDeviceId/*, pool_size: usize*/) -> GPUDeviceStreamPool {
    let dev = dev_id.0;
    let dev_prop = CudaDevice::get_properties(dev as usize).unwrap();
    /*println!("DEBUG: cuda: device: index: {} smp count: {}", dev, dev_prop.multiprocessor_count);
    println!("DEBUG: cuda: device: index: {} shared mem per smp: {}", dev, dev_prop.shared_mem_per_multiprocessor);
    println!("DEBUG: cuda: device: index: {} registers per smp: {}", dev, dev_prop.regs_per_multiprocessor);*/
    let arch_sum = GPUDeviceArchSummary{
      mp_count:             dev_prop.multiProcessorCount as _,
      sharedmem_sz_per_mp:  dev_prop.sharedMemPerMultiprocessor as _,
      register_sz_per_mp:   dev_prop.regsPerMultiprocessor as _,
    };
    println!("DEBUG: GPUDeviceStreamPool: dev: {} arch: {:?}", dev, arch_sum);
    let kernel_cfg = KernelConfig::new(&arch_sum);
    //let stream = Arc::new(GPUDeviceRawStream::new(dev_id));
    GPUDeviceStreamPool{
      dev_id:       dev_id,
      arch_sum:     arch_sum,
      kernel_cfg:   kernel_cfg,
      cuda_stream:  Arc::new(Mutex::new(LazyCudaStream::default())),
      cublas_h:     Arc::new(Mutex::new(LazyCublasHandle::default())),
      cudnn_h:      Arc::new(Mutex::new(LazyCudnnHandle::default())),
      // TODO: configurable arena limit.
      burst_arena:  GPUDeviceBurstArena::with_limit(dev_id, 3_000_000_000),
    }
  }

  pub fn conn<'a>(&'a self) -> GPUDeviceConn<'a> {
    let prev_dev = CudaDevice::get_current().unwrap();
    CudaDevice::set_current(self.dev_id.0).unwrap();
    GPUDeviceConn{
      dev:          self.dev_id,
      pop_dev:      GPUDeviceId(prev_dev),
      kernel_cfg:   self.kernel_cfg,
      cuda_stream:  self.cuda_stream.clone(),
      cublas_h:     self.cublas_h.clone(),
      cudnn_h:      self.cudnn_h.clone(),
      burst_arena:  self.burst_arena.clone(),
      borrow:       &(),
    }
  }
}

#[must_use]
#[derive(Clone)]
pub struct GPUDeviceConn<'a> {
  dev:          GPUDeviceId,
  pop_dev:      GPUDeviceId,
  kernel_cfg:   KernelConfig,
  cuda_stream:  Arc<Mutex<LazyCudaStream>>,
  cublas_h:     Arc<Mutex<LazyCublasHandle>>,
  cudnn_h:      Arc<Mutex<LazyCudnnHandle>>,
  burst_arena:  GPUDeviceBurstArena,
  borrow:       &'a (),
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

  pub fn cuda_kernel_cfg(&self) -> &KernelConfig {
    &self.kernel_cfg
  }

  /*pub fn stream(&self) -> Arc<GPUDeviceRawStream> {
    self.stream.clone()
  }*/

  pub fn cuda_stream(&self) -> MutexGuard<LazyCudaStream> {
    self.cuda_stream.lock().unwrap()
  }

  pub fn cublas(&self) -> MutexGuard<LazyCublasHandle> {
    self.cublas_h.lock().unwrap()
  }

  pub fn cudnn(&self) -> MutexGuard<LazyCudnnHandle> {
    self.cudnn_h.lock().unwrap()
  }

  /*pub fn burst_reserve_bytes(&self, reserve: usize) {
    self.burst_arena.reserve_bytes(reserve);
  }

  pub unsafe fn burst_alloc<T>(&self, len: usize) -> GPUDeviceRegionMem<T> where T: Copy {
    self.burst_arena.alloc::<T>(len, self.clone())
  }*/

  pub fn burst_arena(&self) -> GPUDeviceBurstArena {
    self.burst_arena.clone()
  }

  pub fn sync(&self) {
    let res = self.cuda_stream().synchronize();
    assert!(res.is_ok());
  }
}

pub struct GPUDeviceRegionMem<T> where T: Copy {
  raw:  Arc<GPUDeviceMem<u8>>,
  len:  usize,
  _m:   PhantomData<T>,
}

impl<T> GPUDeviceMem<T> for GPUDeviceRegionMem<T> where T: Copy {
  fn as_dptr(&self) -> *const T {
    self.raw.as_dptr() as *const T
  }

  fn as_mut_dptr(&self) -> *mut T {
    self.raw.as_mut_dptr() as *mut T
  }

  fn len(&self) -> usize {
    self.len
  }

  fn size_bytes(&self) -> usize {
    self.len * size_of::<T>()
  }
}

pub struct GPUDeviceRegionSliceMem {
  dev:  GPUDeviceId,
  dptr: *mut u8,
  phsz: usize,
  _mem: Arc<GPUDeviceRegionRawMem>,
}

impl GPUDeviceMem<u8> for GPUDeviceRegionSliceMem {
  fn as_dptr(&self) -> *const u8 {
    self.dptr
  }

  fn as_mut_dptr(&self) -> *mut u8 {
    self.dptr
  }

  fn len(&self) -> usize {
    self.phsz
  }

  fn size_bytes(&self) -> usize {
    self.phsz
  }
}

pub struct GPUDeviceRegionRawMem {
  dev:  GPUDeviceId,
  dptr: *mut u8,
  phsz: usize,
  rdup: usize,
}

impl Drop for GPUDeviceRegionRawMem {
  fn drop(&mut self) {
    let pop_dev = CudaDevice::get_current().unwrap();
    CudaDevice::synchronize().unwrap();
    CudaDevice::set_current(self.dev.0).unwrap();
    match unsafe { cuda_free_device::<u8>(self.dptr) } {
      Err(_) => panic!(),
      Ok(_) => {}
    }
    CudaDevice::set_current(pop_dev).unwrap();
  }
}

impl GPUDeviceMem<u8> for GPUDeviceRegionRawMem {
  fn as_dptr(&self) -> *const u8 {
    self.dptr
  }

  fn as_mut_dptr(&self) -> *mut u8 {
    self.dptr
  }

  fn len(&self) -> usize {
    self.phsz
  }

  fn size_bytes(&self) -> usize {
    self.phsz
  }
}

// TODO:
// Very aggressive region-reclaiming arena.
// Designed for "bursts" of allocations, usually for scratch memory,
// where _all_ active regions are reclaimed shortly after creation.
#[derive(Clone)]
pub struct GPUDeviceBurstArena {
  inner:    Arc<Mutex<GPUDeviceBurstArenaInner>>,
}

impl GPUDeviceBurstArena {
  pub fn with_limit(dev: GPUDeviceId, max_phsz: usize) -> Self {
    GPUDeviceBurstArena{
      inner:    Arc::new(Mutex::new(GPUDeviceBurstArenaInner{
        dev:        dev,
        max_phsz:   max_phsz,
        regions:    vec![],
        used0:      0,
        used_ext:   0,
        reserved:   0,
      })),
    }
  }

  pub fn reserve<T>(&self, req_len: usize) where T: Copy {
    let req_phsz = req_len * size_of::<T>();
    self.reserve_bytes(req_phsz);
  }

  pub fn reserve_bytes(&self, req_phsz: usize) {
    let mut inner = self.inner.lock().unwrap();
    inner.reserve_bytes(req_phsz);
  }

  pub fn prealloc<T>(&self, len: usize, conn: GPUDeviceConn) where T: Copy {
    let _reg = unsafe { self.alloc::<T>(len, conn) };
  }

  pub unsafe fn alloc<T>(&self, len: usize, conn: GPUDeviceConn) -> GPUDeviceRegionMem<T> where T: Copy {
    let mut inner = self.inner.lock().unwrap();
    inner.alloc::<T>(len, conn)
  }
}

pub struct GPUDeviceBurstArenaInner {
  dev:      GPUDeviceId,
  max_phsz: usize,
  regions:  Vec<Arc<GPUDeviceRegionRawMem>>,
  used0:    usize,
  used_ext: usize,
  reserved: usize,
}

impl GPUDeviceBurstArenaInner {
  pub fn _check_all_regions_free(&self) -> bool {
    for reg in self.regions.iter().rev() {
      if Arc::strong_count(reg) > 1 {
        return false;
      }
    }
    true
  }

  pub unsafe fn _merge_all_regions(&mut self, req_phsz: usize) {
    assert!(req_phsz <= self.max_phsz);
    let total_phsz = self.used0 + self.used_ext;
    self.used0 = 0;
    self.used_ext = 0;
    if req_phsz <= total_phsz && self.regions.len() == 1 {
      // Do nothing.
    } else {
      let merge_phsz = max(max(total_phsz, self.reserved), req_phsz);
      assert!(merge_phsz <= self.max_phsz);
      self.regions.clear();
      let dptr = match cuda_alloc_device::<u8>(merge_phsz) {
        Err(_) => panic!(),
        Ok(dptr) => dptr,
      };
      let reg = Arc::new(GPUDeviceRegionRawMem{
        dev:    self.dev,
        dptr:   dptr,
        phsz:   merge_phsz,
        rdup:   merge_phsz,
      });
      self.regions.push(reg.clone());
    }
  }

  pub fn reserve_bytes(&mut self, req_phsz: usize) {
    self.reserved = max(self.reserved, req_phsz);
  }

  pub unsafe fn alloc<T>(&mut self, len: usize, conn: GPUDeviceConn) -> GPUDeviceRegionMem<T> where T: Copy {
    assert_eq!(self.dev, conn.device());
    let phsz = len * size_of::<T>();
    if self._check_all_regions_free() {
      self._merge_all_regions(phsz);
    }
    // TODO: alignment/padding.
    let mem: Arc<GPUDeviceMem<u8>> = if self.used0 + phsz <= self.regions[0].phsz {
      let dptr = self.regions[0].dptr.offset(self.used0 as _);
      let slice = Arc::new(GPUDeviceRegionSliceMem{
        dev:    self.dev,
        dptr:   dptr,
        phsz:   phsz,
        _mem:   self.regions[0].clone(),
      });
      self.used0 += phsz;
      slice
    } else {
      let dptr = match cuda_alloc_device::<u8>(phsz) {
        Err(_) => panic!(),
        Ok(dptr) => dptr,
      };
      let reg = Arc::new(GPUDeviceRegionRawMem{
        dev:    self.dev,
        dptr:   dptr,
        phsz:   phsz,
        rdup:   phsz,
      });
      self.regions.push(reg.clone());
      self.used_ext += phsz;
      reg
    };
    assert!(self.regions[0].phsz + self.used_ext <= self.max_phsz);
    GPUDeviceRegionMem{
      raw:  mem,
      len:  len,
      _m:   PhantomData,
    }
  }
}

pub struct GPUDeviceArena {
  inner:    Arc<Mutex<GPUDeviceArenaInner>>,
}

impl GPUDeviceArena {
  pub fn new(dev: GPUDeviceId) -> Self {
    // TODO
    unimplemented!();
  }
}

pub struct GPUDeviceArenaInner {
  dev:      GPUDeviceId,
  chunks:   HashMap<usize, Vec<Arc<GPUDeviceRegionRawMem>>>,
  //free:     HashMap<usize, Vec<Arc<GPUDeviceRegionRawMem>>>,
}

impl GPUDeviceArenaInner {
  pub unsafe fn alloc(&mut self, phsz: usize, conn: GPUDeviceConn) -> Arc<GPUDeviceRegionRawMem> {
    // TODO: round up the alloc size to a nice round number.
    let rdup_phsz = if phsz <= 512 {
      (phsz + 512 - 1) / 512 * 512
    } else if phsz <= 32768 {
      (phsz + 32768 - 1) / 32768 * 32768
    } else {
      (phsz + 1048576 - 1) / 1048576 * 1048576
    };

    if self.chunks.get(&rdup_phsz).is_none() {
      self.chunks.insert(rdup_phsz, vec![]);
    }

    if !self.chunks.get(&rdup_phsz).unwrap().is_empty() {
      let chunks = self.chunks.get(&rdup_phsz).unwrap();
      for chunk in chunks.iter() {
        if Arc::strong_count(chunk) == 1 {
          return chunk.clone()
        }
      }
    }

    let dptr = match cuda_alloc_device::<u8>(rdup_phsz) {
      Err(_) => {
        self.collect_garbage();
        match cuda_alloc_device::<u8>(phsz) {
          Err(_) => panic!("GPUDeviceArena allocation failed"),
          Ok(dptr) => dptr,
        }
      }
      Ok(dptr) => dptr,
    };
    let chunk = Arc::new(GPUDeviceRegionRawMem{
      dev:  self.dev,
      dptr: dptr,
      phsz: phsz,
      rdup: rdup_phsz,
    });
    self.chunks.get_mut(&rdup_phsz).unwrap().push(chunk.clone());
    chunk
  }

  pub fn collect_garbage(&mut self) {
    for (_, mut chunks) in self.chunks.iter_mut() {
      let mut free_idxs = vec![];
      for (idx, chunk) in chunks.iter().enumerate() {
        if Arc::strong_count(chunk) == 1 {
          free_idxs.push(idx);
        }
      }
      for &idx in free_idxs.iter().rev() {
        chunks.remove(idx);
      }
    }
  }
}

/*pub trait GPUDeviceAllocator {
  unsafe fn alloc<T>(&self, len: usize, conn: GPUDeviceConn) -> Arc<GPUDeviceMem<T>> where T: Copy + 'static;
}

pub struct GPUDeviceRawAlloc {
}

impl GPUDeviceAllocator for GPUDeviceRawAlloc {
  unsafe fn alloc<T>(&self, len: usize, conn: GPUDeviceConn) -> Arc<GPUDeviceMem<T>> where T: Copy + 'static {
    Arc::new(GPUDeviceRawMem::<T>::alloc(len, conn))
  }
}*/

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
  fn size_bytes(&self) -> usize;
}

pub struct GPUDeviceRawMem<T> where T: Copy {
  dev:  GPUDeviceId,
  dptr: *mut T,
  len:  usize,
  phsz: usize,
}

impl<T> Drop for GPUDeviceRawMem<T> where T: Copy {
  fn drop(&mut self) {
    let pop_dev = CudaDevice::get_current().unwrap();
    CudaDevice::synchronize().unwrap();
    CudaDevice::set_current(self.dev.0).unwrap();
    match unsafe { cuda_free_device::<T>(self.dptr) } {
      Err(_) => panic!(),
      Ok(_) => {}
    }
    CudaDevice::set_current(pop_dev).unwrap();
  }
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
      phsz: len * size_of::<T>(),
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

  fn size_bytes(&self) -> usize {
    self.phsz
  }
}

/*pub struct GPUDeviceResizableMem<T> {
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
}*/

/*pub struct GPUDeviceStreamMem<T> {
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
}*/

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

/*pub struct GPUDeviceToken {
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
}*/

/*pub struct GPUDevicePost {
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
}*/

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
