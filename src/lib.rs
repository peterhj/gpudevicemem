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

extern crate cuda;
extern crate cuda_blas;
extern crate cuda_dnn;

use cuda::bind_ffi::runtime::{cudaError_t, cudaStream_t, cudaDeviceProp};
use cuda::runtime_new::*;
use cuda_blas::new::{CublasHandle};
use cuda_dnn::v5::{CudnnHandle};
//use densearray::prelude::*;

//use libc::{c_void};
use std::mem::{size_of, transmute};
use std::os::raw::{c_void};
use std::rc::{Rc};
use std::sync::{Arc, RwLock};
use std::sync::atomic::{AtomicUsize, Ordering};

//static STREAM_POOL_UID_COUNTER: AtomicU64 = ATOMIC_U64_INIT;

#[derive(Clone, Copy)]
pub struct DeviceId(pub i32);

impl DeviceId {
  /*pub fn enumerate() -> Vec<DeviceId> {
    // TODO
    unimplemented!();
  }*/

  pub fn rank(&self) -> usize {
    self.0 as _
  }
}

#[derive(Clone)]
pub struct DevicePlacement {
  dev_id:   DeviceId,
}

impl DevicePlacement {
}

pub struct DeviceStream {
  dev_id:       DeviceId,
  raw_stream:   Arc<CudaStream>,
  sync_event:   Arc<CudaEvent>,
  workspace_sz: Arc<AtomicUsize>,
  workspace:    Option<Arc<DeviceAllocMem<u8>>>,
}

impl DeviceStream {
  pub fn new(dev_id: DeviceId) -> Self {
    let prev_dev = CudaDevice::get_current().unwrap();
    CudaDevice::set_current(dev_id.0).unwrap();
    let raw_stream = Arc::new(CudaStream::create().unwrap());
    let sync_event = Arc::new(CudaEvent::create_fastest().unwrap());
    CudaDevice::set_current(prev_dev).unwrap();
    DeviceStream{
      dev_id:       dev_id,
      raw_stream:   raw_stream,
      sync_event:   sync_event,
      workspace_sz: Arc::new(AtomicUsize::new(0)),
      workspace:    None,
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
pub struct DeviceArchSummary {
  pub mp_count:             usize,
  pub sharedmem_sz_per_mp:  usize,
  pub register_sz_per_mp:   usize,
}

pub struct DeviceStreamPool {
  //uid:      u64,
  dev_id:   DeviceId,
  dev_prop: Arc<cudaDeviceProp>,
  arch_sum: DeviceArchSummary,
  stream:   Arc<DeviceStream>,
  cublas_h: Arc<CublasHandle>,
  cudnn_h:  Option<Arc<CudnnHandle>>,
}

impl DeviceStreamPool {
  pub fn implicit() -> Self {
    // FIXME
    unimplemented!();
  }

  pub fn new(dev_id: DeviceId, pool_size: usize) -> DeviceStreamPool {
    //let uid = STREAM_POOL_UID_COUNTER.fetch_add(1, Ordering::AcqRel) + 1;
    //assert!(0 != uid);
    let dev = dev_id.0;
    let dev_prop = Arc::new(CudaDevice::get_properties(dev as usize).unwrap());
    /*println!("DEBUG: cuda: device: index: {} smp count: {}", dev, dev_prop.multiprocessor_count);
    println!("DEBUG: cuda: device: index: {} shared mem per smp: {}", dev, dev_prop.shared_mem_per_multiprocessor);
    println!("DEBUG: cuda: device: index: {} registers per smp: {}", dev, dev_prop.regs_per_multiprocessor);*/
    let arch_sum = DeviceArchSummary{
      mp_count:             dev_prop.multiProcessorCount as _,
      sharedmem_sz_per_mp:  dev_prop.sharedMemPerMultiprocessor as _,
      register_sz_per_mp:   dev_prop.regsPerMultiprocessor as _,
    };
    println!("DEBUG: DeviceStreamPool: dev: {} arch: {:?}", dev, arch_sum);
    let stream = Arc::new(DeviceStream::new(dev_id));
    let cublas_h = Arc::new(CublasHandle::create().unwrap());
    DeviceStreamPool{
      //uid:      uid,
      dev_id:   dev_id,
      dev_prop: dev_prop,
      arch_sum: arch_sum,
      stream:   stream,
      cublas_h: cublas_h,
      cudnn_h:  None,
    }
  }

  pub fn conn(&self) -> DeviceConn {
    let prev_dev = CudaDevice::get_current().unwrap();
    CudaDevice::set_current(self.dev_id.0).unwrap();
    DeviceConn{
      dev:      self.dev_id,
      pop_dev:  DeviceId(prev_dev),
      stream:   self.stream.clone(),
      cublas_h: self.cublas_h.clone(),
    }
  }
}

#[derive(Clone)]
pub struct DeviceConn {
  dev:      DeviceId,
  pop_dev:  DeviceId,
  stream:   Arc<DeviceStream>,
  cublas_h: Arc<CublasHandle>,
}

impl Drop for DeviceConn {
  fn drop(&mut self) {
    CudaDevice::set_current(self.pop_dev.0).unwrap();
  }
}

impl DeviceConn {
  pub fn device(&self) -> DeviceId {
    self.dev
  }

  pub fn stream(&self) -> Arc<DeviceStream> {
    self.stream.clone()
  }

  pub fn cublas(&self) -> Arc<CublasHandle> {
    self.cublas_h.clone()
  }
}

pub trait DeviceMem<T> where T: Copy {
  fn as_dptr(&self) -> *const T;
  fn as_mut_dptr(&self) -> *mut T;
  fn len(&self) -> usize;
}

pub struct DeviceAllocMem<T> {
  dev:  DeviceId,
  dptr: *mut T,
  len:  usize,
  psz:  usize,
}

impl<T> DeviceAllocMem<T> where T: Copy {
  pub unsafe fn alloc(len: usize, conn: DeviceConn) -> DeviceAllocMem<T> where T: Copy {
    assert!(len <= <u32>::max_value() as usize,
        "device memory size should not exceed 2**31-1 bytes");
    let dptr = match cuda_alloc_device::<T>(len) {
      Err(e) => panic!("DeviceAllocMem allocation failed: {:?}", e),
      Ok(dptr) => dptr,
    };
    assert!(!dptr.is_null());
    DeviceAllocMem{
      dev:  conn.device(),
      dptr: dptr,
      len:  len,
      psz:  len * size_of::<T>(),
    }
  }
}

impl<T> DeviceMem<T> for DeviceAllocMem<T> where T: Copy {
  fn as_dptr(&self) -> *const T {
    self.dptr
  }

  fn as_mut_dptr(&self) -> *mut T {
    self.dptr
  }

  fn len(&self) -> usize{
    self.len
  }
}

pub struct DeviceToken {
  //producers:    Arc<AtomicArcList<DeviceStream>>,
  producers:    Arc<RwLock<Vec<Arc<DeviceStream>>>>,
}

impl DeviceToken {
  pub fn post_excl(&self, stream: Arc<DeviceStream>) {
    self.producers.write().unwrap().push(stream);
  }

  pub fn wait_excl(&self, stream: Arc<DeviceStream>) {
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

pub struct DevicePost {
  stream:   Arc<DeviceStream>,
  xtokens:  Vec<DeviceToken>,
  stokens:  Vec<DeviceToken>,
}

pub struct DeviceWait {
  stream:   Arc<DeviceStream>,
  xtokens:  Vec<DeviceToken>,
  stokens:  Vec<DeviceToken>,
}

extern "C" fn dataflow_post(stream: cudaStream_t, status: cudaError_t, post_raw_data: *mut c_void) {
  // TODO
  let post: Arc<DevicePost> = unsafe { Arc::from_raw(transmute(post_raw_data)) };
  for xtoken in post.xtokens.iter() {
    xtoken.post_excl(post.stream.clone());
  }
  assert!(post.stokens.is_empty(), "shared tokens are not supported yet");
}

extern "C" fn dataflow_wait(stream: cudaStream_t, status: cudaError_t, wait_raw_data: *mut c_void) {
  // TODO
  let wait: Arc<DeviceWait> = unsafe { Arc::from_raw(transmute(wait_raw_data)) };
  for xtoken in wait.xtokens.iter() {
    xtoken.wait_excl(wait.stream.clone());
  }
  assert!(wait.stokens.is_empty(), "shared tokens are not supported yet");
}

/*pub struct DeviceMemRef<T> where T: Copy {
  mem:  Rc<DeviceMem<T>>,
  dptr: *mut T,
  len:  usize,
}

impl<T> DeviceMemRef<T> where T: Copy {
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

pub struct DeviceMemRefMut<T> where T: Copy {
  mem:  Rc<DeviceMem<T>>,
  dptr: *mut T,
  len:  usize,
}

impl<T> DeviceMemRefMut<T> where T: Copy {
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
