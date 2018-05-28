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

#![feature(collections_range)]
#![feature(specialization)]
//#![feature(trait_alias)]

extern crate arithmetic;
extern crate arrayidx;
extern crate cuda;
extern crate cuda_blas;
extern crate cuda_dnn;
extern crate cuda_rand;
extern crate float;
#[macro_use] extern crate lazy_static;
extern crate memarray;
extern crate parking_lot;

use ffi::routines_gpu::{KernelConfig};

//use cuda::ffi::runtime::{cudaError_t, cudaStream_t, cudaDeviceProp};
use cuda::runtime::*;
use cuda_blas::{CublasHandle};
use cuda_dnn::{CudnnHandle};
use cuda_rand::{CurandGenerator};
use parking_lot::{Mutex, MutexGuard};

use std::cell::{Cell, RefCell, RefMut};
use std::cmp::{max};
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::marker::{PhantomData};
use std::mem::{size_of};
use std::ops::{Deref, DerefMut};
use std::rc::{Rc};
use std::sync::{Arc};
//use std::sync::atomic::{AtomicUsize, Ordering};

pub mod array;
pub mod ffi;
pub mod utils;

//static STREAM_POOL_UID_COUNTER: AtomicU64 = ATOMIC_U64_INIT;

pub struct GPUHostMem<T> where T: Copy {
  buf:  *mut T,
  len:  usize,
  phsz: usize,
}

impl<T> GPUHostMem<T> where T: Copy {
  pub unsafe fn alloc(len: usize) -> Self {
    let ptr = match unsafe { cuda_alloc_host(len) } {
      Err(_) => panic!(),
      Ok(ptr) => ptr,
    };
    GPUHostMem{
      buf:  ptr,
      len:  len,
      phsz: len * size_of::<T>(),
    }
  }
}

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
    assert!(!self.h.is_none());
    self.h.as_ref().unwrap()
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

#[derive(Default)]
pub struct LazyCurandGenerator {
  h:    Option<CurandGenerator>,
}

impl LazyCurandGenerator {
  pub fn default_shared_local() -> Rc<RefCell<Self>> {
    Rc::new(RefCell::new(Self::default()))
  }
}

impl Deref for LazyCurandGenerator {
  type Target = CurandGenerator;

  fn deref(&self) -> &CurandGenerator {
    unreachable!();
  }
}

impl DerefMut for LazyCurandGenerator {
  fn deref_mut(&mut self) -> &mut CurandGenerator {
    if self.h.is_none() {
      self.h = Some(CurandGenerator::create().unwrap());
    }
    self.h.as_mut().unwrap()
  }
}

#[derive(Clone, Copy, Debug)]
pub struct GPUDeviceArchSummary {
  pub capability_major:     usize,
  pub mp_count:             usize,
  pub sharedmem_sz_per_mp:  usize,
  pub register_sz_per_mp:   usize,
}

thread_local! {
  static CONN_DEV:      Rc<Cell<GPUDeviceId>> = Rc::new(Cell::new(GPUDeviceId(CudaDevice::get_current().unwrap())));
  static CUBLAS_HS_TLS: RefCell<HashMap<GPUDeviceId, Rc<RefCell<LazyCublasHandle>>>> = RefCell::new(HashMap::new());
  static CUDNN_HS_TLS:  RefCell<HashMap<GPUDeviceId, Rc<RefCell<LazyCudnnHandle>>>> = RefCell::new(HashMap::new());
}

#[derive(Clone)]
pub struct GPUDeviceStreamPool {
  dev_id:       GPUDeviceId,
  arch_sum:     GPUDeviceArchSummary,
  kernel_cfg:   KernelConfig,
  cuda_s_uid:   usize,
  cuda_stream:  Arc<Mutex<LazyCudaStream>>,
  burst_arena:  GPUDeviceBurstArena,
}

impl GPUDeviceStreamPool {
  pub fn new(dev_id: GPUDeviceId/*, pool_size: usize*/) -> GPUDeviceStreamPool {
    let dev = dev_id.0;
    let dev_prop = CudaDevice::get_properties(dev as usize).unwrap();
    let arch_sum = GPUDeviceArchSummary{
      capability_major:     dev_prop.major as _,
      mp_count:             dev_prop.multiProcessorCount as _,
      sharedmem_sz_per_mp:  dev_prop.sharedMemPerMultiprocessor as _,
      register_sz_per_mp:   dev_prop.regsPerMultiprocessor as _,
    };
    println!("DEBUG: GPUDeviceStreamPool: dev: {} arch: {:?}", dev, arch_sum);
    let kernel_cfg = KernelConfig::new(&arch_sum);
    let mut cuda_stream = LazyCudaStream::default();
    let cuda_s_uid = (&mut *cuda_stream).unique_id();
    GPUDeviceStreamPool{
      dev_id:       dev_id,
      arch_sum:     arch_sum,
      kernel_cfg:   kernel_cfg,
      cuda_s_uid:   cuda_s_uid,
      cuda_stream:  Arc::new(Mutex::new(cuda_stream)),
      // TODO: configurable arena limit.
      burst_arena:  GPUDeviceBurstArena::with_limit(dev_id, i32::max_value() as _),
    }
  }

  pub fn device(&self) -> GPUDeviceId {
    self.dev_id
  }

  pub fn conn<'a>(&'a mut self) -> GPUDeviceConn<'a> {
    let conn_dev = CONN_DEV.with(|conn_dev| conn_dev.clone());
    if Rc::strong_count(&conn_dev) <= 2 || conn_dev.get() == self.dev_id {
      // OK.
    } else {
      panic!();
    }
    /*let prev_dev = CudaDevice::get_current().unwrap();
    assert_eq!(prev_dev, conn_dev.get().0);*/
    let prev_dev = conn_dev.get();
    CudaDevice::set_current(self.dev_id.0).unwrap();
    conn_dev.set(self.dev_id);
    GPUDeviceConn{
      pop:          Rc::new(PopConn{pop_dev: prev_dev, conn_dev}),
      dev:          self.dev_id,
      kernel_cfg:   self.kernel_cfg,
      cuda_s_uid:   self.cuda_s_uid,
      cuda_stream:  self.cuda_stream.clone(),
      cublas_h_tls: {
        CUBLAS_HS_TLS.with(|hs| {
          let mut hs = hs.borrow_mut();
          hs.entry(self.dev_id)
            .or_insert_with(|| Rc::new(RefCell::new(LazyCublasHandle::default())))
            .clone()
        })
      },
      cudnn_h_tls:  {
        CUDNN_HS_TLS.with(|hs| {
          let mut hs = hs.borrow_mut();
          hs.entry(self.dev_id)
            .or_insert_with(|| Rc::new(RefCell::new(LazyCudnnHandle::default())))
            .clone()
        })
      },
      burst_arena:  self.burst_arena.clone(),
      borrow:       &(),
    }
  }
}

struct PopConn {
  pop_dev:  GPUDeviceId,
  conn_dev: Rc<Cell<GPUDeviceId>>,
}

impl Drop for PopConn {
  fn drop(&mut self) {
    CudaDevice::set_current(self.pop_dev.0).unwrap();
    self.conn_dev.set(self.pop_dev);
  }
}

#[must_use]
#[derive(Clone)]
pub struct GPUDeviceConn<'a> {
  pop:          Rc<PopConn>,
  dev:          GPUDeviceId,
  kernel_cfg:   KernelConfig,
  cuda_s_uid:   usize,
  cuda_stream:  Arc<Mutex<LazyCudaStream>>,
  cublas_h_tls: Rc<RefCell<LazyCublasHandle>>,
  cudnn_h_tls:  Rc<RefCell<LazyCudnnHandle>>,
  burst_arena:  GPUDeviceBurstArena,
  borrow:       &'a (),
}

impl<'a> GPUDeviceConn<'a> {
  pub fn device(&self) -> GPUDeviceId {
    self.dev
  }

  pub fn cuda_kernel_cfg(&self) -> &KernelConfig {
    self.cuda_kernel_config()
  }

  pub fn cuda_kernel_config(&self) -> &KernelConfig {
    &self.kernel_cfg
  }

  pub fn cuda_stream(&self) -> MutexGuard<LazyCudaStream> {
    self.cuda_stream.lock()
  }

  pub fn cuda_stream_uid(&self) -> usize {
    self.cuda_s_uid
  }

  pub fn cublas(&self) -> RefMut<LazyCublasHandle> {
    self.cublas_h_tls.borrow_mut()
  }

  pub fn cudnn(&self) -> RefMut<LazyCudnnHandle> {
    self.cudnn_h_tls.borrow_mut()
  }

  /*pub fn burst_reserve_bytes(&self, reserve: usize) {
    self.burst_arena.reserve_bytes(reserve);
  }

  pub unsafe fn burst_alloc<T>(&self, len: usize) -> GPUDeviceTypedMem<T> where T: Copy {
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

#[derive(Clone)]
pub struct GPUAsyncTicket {
  event:    Arc<Mutex<CudaEvent>>,
  //stream:   Arc<Mutex<CudaStream>>,
  e_uid:    usize,
  s_uid:    usize,
}

#[derive(Default)]
pub struct GPUAsyncState {
  tick: Option<GPUAsyncTicket>,
}

impl GPUAsyncState {
  pub fn swap_ticket(&mut self, ticket: GPUAsyncTicket) -> Option<GPUAsyncTicket> {
    let prev_tick = self.tick.take();
    self.tick = Some(ticket);
    prev_tick
  }

  pub fn take_ticket(&mut self) -> Option<GPUAsyncTicket> {
    self.tick.take()
  }
}

/*thread_local! {
  static SECTION_STACK: RefCell<Vec<Rc<RefCell<GPUAsyncSectionState>>>> = RefCell::new(Vec::new());
}*/

struct ArcKey<T: ?Sized>(pub Arc<T>);

impl<T: ?Sized> PartialEq for ArcKey<T> {
  fn eq(&self, other: &Self) -> bool {
    Arc::ptr_eq(&self.0, &other.0)
  }
}

impl<T: ?Sized> Eq for ArcKey<T> {}

impl<T: ?Sized> Hash for ArcKey<T> {
  fn hash<H>(&self, state: &mut H) where H: Hasher {
    // NOTE: This should be correct; see the `Arc` impl of `fmt::Pointer`.
    let ptr: *const T = &*self.0;
    ptr.hash(state);
  }
}

pub struct GPULazyAsyncSection {
  shared:   Rc<RefCell<Option<GPUAsyncSection>>>,
  cached:   Option<GPUAsyncSection>,
}

impl Default for GPULazyAsyncSection {
  fn default() -> Self {
    GPULazyAsyncSection{
      shared:   Rc::new(RefCell::new(None)),
      cached:   None,
    }
  }
}

impl Clone for GPULazyAsyncSection {
  fn clone(&self) -> Self {
    GPULazyAsyncSection{
      shared:   self.shared.clone(),
      cached:   None,
    }
  }
}

impl GPULazyAsyncSection {
  pub fn enter<'a>(&'a mut self, conn: GPUDeviceConn<'a>) -> GPUAsyncSectionGuard<'a> {
    if self.shared.borrow().is_none() {
      *self.shared.borrow_mut() = Some(GPUAsyncSection::new(conn.clone()));
    }
    if self.cached.is_none() {
      self.cached = self.shared.borrow().clone();
    }
    self.cached.as_ref().unwrap().enter(conn)
  }
}

#[derive(Clone)]
pub struct GPUAsyncSection {
  dev:      GPUDeviceId,
  event:    Arc<Mutex<CudaEvent>>,
}

impl GPUAsyncSection {
  pub fn new(conn: GPUDeviceConn) -> Self {
    let dev = conn.device();
    let event = CudaEvent::create_fastest().unwrap();
    GPUAsyncSection{
      dev:      dev,
      event:    Arc::new(Mutex::new(event)),
    }
  }

  pub fn enter<'a>(&'a self, conn: GPUDeviceConn<'a>) -> GPUAsyncSectionGuard<'a> {
    //println!("DEBUG: GPUAsyncSection::enter()");
    assert_eq!(self.dev, conn.device());
    let evcopy = self.event.clone();
    GPUAsyncSectionGuard{
      event:    self.event.lock(),
      conn:     conn,
      evcopy:   evcopy,
      state:    GPUAsyncSectionState::default(),
    }
  }
}

#[derive(Default)]
struct GPUAsyncSectionState {
  wait_keys:    HashSet<(usize, usize)>,
  reg_data:     HashSet<ArcKey<Mutex<GPUAsyncState>>>,
}

impl GPUAsyncSectionState {
  pub fn wait_ticket(&mut self, ticket: GPUAsyncTicket, conn: GPUDeviceConn) {
    let key = (ticket.e_uid, ticket.s_uid);
    if self.wait_keys.insert(key) {
      if conn.cuda_stream_uid() != ticket.s_uid {
        assert!(conn.cuda_stream().wait_event(&mut *ticket.event.lock()).is_ok());
      }
    }
  }

  pub fn register(&mut self, data: Arc<Mutex<GPUAsyncState>>) {
    self.reg_data.insert(ArcKey(data));
  }
}

pub struct GPUAsyncSectionGuard<'a> {
  event:    MutexGuard<'a, CudaEvent>,
  conn:     GPUDeviceConn<'a>,
  evcopy:   Arc<Mutex<CudaEvent>>,
  state:    GPUAsyncSectionState,
}

impl<'a> Drop for GPUAsyncSectionGuard<'a> {
  fn drop(&mut self) {
    // Create a ticket from an event recording.
    let (e_uid, s_uid) = {
      let mut stream = self.conn.cuda_stream();
      assert!(self.event.record(&mut *stream).is_ok());
      (self.event.unique_id(), stream.unique_id())
    };
    let ticket = GPUAsyncTicket{
      event:    self.evcopy.clone(),
      e_uid:    e_uid,
      s_uid:    s_uid,
    };

    // Post ticket to each registered data.
    for data in self.state.reg_data.drain() {
      let prev_tick = data.0.lock().swap_ticket(ticket.clone());
      assert!(prev_tick.is_none());
    }
  }
}

impl<'a> GPUAsyncSectionGuard<'a> {
  // TODO: This is a stopgap API. It should be replaced by access to a
  // thread-local stack of section guard states. APIs that create views to
  // GPU device memory should then call `wait` on itself for the active section.
  pub fn _wait(&mut self, data: Arc<Mutex<GPUAsyncState>>, /*conn: GPUDeviceConn*/) {
    // If the data has a ticket, wait on it.
    if let Some(ticket) = data.lock().take_ticket() {
      self.state.wait_ticket(ticket, self.conn.clone());
    }

    // Register the data for a future post.
    self.state.register(data);
  }
}

pub trait GPUDevicePlace {
  fn device(&self) -> GPUDeviceId;
}

pub trait GPUDeviceAsync {
  fn async_state(&self) -> Arc<Mutex<GPUAsyncState>>;
}

pub trait GPUDeviceMem<T>: GPUDevicePlace + GPUDeviceAsync where T: Copy {
  //fn device(&self) -> GPUDeviceId;
  unsafe fn as_dptr(&self) -> *const T;
  unsafe fn as_mut_dptr(&self) -> *mut T;
  fn len(&self) -> usize;
  fn size_bytes(&self) -> usize;
}

pub trait GPUDeviceAlloc<T> where T: Copy + 'static {
  type Mem: GPUDeviceMem<T> + 'static;

  unsafe fn alloc(&self, len: usize, conn: GPUDeviceConn) -> Self::Mem;
}

pub struct GPUDeviceTypedMem<T> where T: Copy {
  dptr: *mut T,
  len:  usize,
  _mem: Arc<GPUDeviceMem<u8>>,
  _mrk: PhantomData<T>,
}

unsafe impl<T> Send for GPUDeviceTypedMem<T> where T: Copy {}
unsafe impl<T> Sync for GPUDeviceTypedMem<T> where T: Copy {}

impl<T> GPUDeviceAsync for GPUDeviceTypedMem<T> where T: Copy {
  fn async_state(&self) -> Arc<Mutex<GPUAsyncState>> {
    self._mem.async_state()
  }
}

impl<T> GPUDevicePlace for GPUDeviceTypedMem<T> where T: Copy {
  fn device(&self) -> GPUDeviceId {
    self._mem.device()
  }
}

impl<T> GPUDeviceMem<T> for GPUDeviceTypedMem<T> where T: Copy {
  unsafe fn as_dptr(&self) -> *const T {
    self.dptr
  }

  unsafe fn as_mut_dptr(&self) -> *mut T {
    self.dptr
  }

  fn len(&self) -> usize {
    self.len
  }

  fn size_bytes(&self) -> usize {
    self.len * size_of::<T>()
  }
}

pub struct GPUDeviceSliceMem<T> where T: Copy {
  dptr: *mut T,
  len:  usize,
  phsz: usize,
  _mem: Arc<GPUDeviceMem<T>>,
}

unsafe impl<T> Send for GPUDeviceSliceMem<T> where T: Copy {}
unsafe impl<T> Sync for GPUDeviceSliceMem<T> where T: Copy {}

impl<T> GPUDeviceAsync for GPUDeviceSliceMem<T> where T: Copy {
  fn async_state(&self) -> Arc<Mutex<GPUAsyncState>> {
    self._mem.async_state()
  }
}

impl<T> GPUDevicePlace for GPUDeviceSliceMem<T> where T: Copy {
  fn device(&self) -> GPUDeviceId {
    self._mem.device()
  }
}

impl<T> GPUDeviceMem<T> for GPUDeviceSliceMem<T> where T: Copy {
  unsafe fn as_dptr(&self) -> *const T {
    self.dptr
  }

  unsafe fn as_mut_dptr(&self) -> *mut T {
    self.dptr
  }

  fn len(&self) -> usize {
    self.phsz
  }

  fn size_bytes(&self) -> usize {
    self.phsz
  }
}

const BURST_MEM_ALIGN: usize = 16;

fn round_up(sz: usize, alignment: usize) -> usize {
  assert!(alignment >= 1);
  (sz + alignment - 1) / alignment * alignment
}

fn check_alignment(sz: usize, alignment: usize) -> bool {
  assert!(alignment >= 1);
  sz % alignment == 0
}

/// Very aggressive region-reclaiming arena.
/// Designed for "bursts" of allocations, usually for scratch memory,
/// where _all_ active regions are reclaimed shortly after creation.
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
    let mut inner = self.inner.lock();
    inner.reserve_bytes(req_phsz);
  }

  /*pub fn prealloc<T>(&self, len: usize, conn: GPUDeviceConn) where T: Copy {
    let _reg = unsafe { self.alloc::<T>(len, conn) };
  }*/
}

impl<T> GPUDeviceAlloc<T> for GPUDeviceBurstArena where T: Copy + 'static {
  type Mem = GPUDeviceTypedMem<T>;

  unsafe fn alloc(&self, len: usize, conn: GPUDeviceConn) -> GPUDeviceTypedMem<T> where T: Copy {
    let mut inner = self.inner.lock();
    inner.alloc::<T>(len, conn)
  }
}

pub struct GPUDeviceBurstArenaInner {
  dev:      GPUDeviceId,
  max_phsz: usize,
  regions:  Vec<Arc<GPUDeviceRawMem<u8>>>,
  used0:    usize,
  used_ext: usize,
  reserved: usize,
}

impl GPUDeviceBurstArenaInner {
  fn _check_all_regions_free(&self) -> bool {
    for reg in self.regions.iter().rev() {
      if Arc::strong_count(reg) > 1 {
        return false;
      }
    }
    true
  }

  unsafe fn _merge_all_regions(&mut self, req_phsz: usize) {
    assert!(req_phsz <= self.max_phsz, "GPUDeviceBurstArena exceeded allocation limit");
    assert!(check_alignment(req_phsz, BURST_MEM_ALIGN));
    let total_phsz = self.used0 + self.used_ext;
    assert!(check_alignment(total_phsz, BURST_MEM_ALIGN));
    self.used0 = 0;
    self.used_ext = 0;
    if req_phsz <= total_phsz && self.regions.len() == 1 {
      // Do nothing.
    } else {
      let merge_phsz = max(max(total_phsz, self.reserved), req_phsz);
      assert!(merge_phsz <= self.max_phsz, "GPUDeviceBurstArena exceeded allocation limit");
      assert!(check_alignment(merge_phsz, BURST_MEM_ALIGN));
      self.regions.clear();
      let dptr = match cuda_alloc_device::<u8>(merge_phsz) {
        Err(e) => panic!("GPUDeviceBurstArena: failed to alloc size {}: {:?}", merge_phsz, e),
        Ok(dptr) => dptr,
      };
      let reg = Arc::new(GPUDeviceRawMem{
        dev:    self.dev,
        dptr:   dptr,
        len:    merge_phsz,
        phsz:   merge_phsz,
        asd:    Arc::new(Mutex::new(GPUAsyncState::default())),
      });
      self.regions.push(reg.clone());
    }
  }

  pub fn reserve_bytes(&mut self, bare_phsz: usize) {
    let rdup_phsz = round_up(bare_phsz, BURST_MEM_ALIGN);
    self.reserved = max(self.reserved, rdup_phsz);
  }

  pub unsafe fn alloc<T>(&mut self, len: usize, conn: GPUDeviceConn) -> GPUDeviceTypedMem<T> where T: Copy {
    assert_eq!(self.dev, conn.device());
    let bare_phsz = len * size_of::<T>();
    let rdup_phsz = round_up(bare_phsz, BURST_MEM_ALIGN);
    if self._check_all_regions_free() {
      self._merge_all_regions(rdup_phsz);
    }
    let mem: Arc<GPUDeviceMem<u8>> = if self.used0 + rdup_phsz <= self.regions[0].phsz {
      let dptr = self.regions[0].dptr.offset(self.used0 as _);
      assert!(check_alignment(dptr as usize, BURST_MEM_ALIGN));
      let slice = Arc::new(GPUDeviceSliceMem{
        dptr:   dptr,
        len:    rdup_phsz,
        phsz:   rdup_phsz,
        _mem:   self.regions[0].clone(),
      });
      self.used0 += rdup_phsz;
      slice
    } else {
      let dptr = match cuda_alloc_device::<u8>(rdup_phsz) {
        Err(e) => panic!("GPUDeviceBurstArena: failed to alloc size {}: {:?}", rdup_phsz, e),
        Ok(dptr) => dptr,
      };
      assert!(check_alignment(dptr as usize, BURST_MEM_ALIGN));
      let reg = Arc::new(GPUDeviceRawMem{
        dev:    self.dev,
        dptr:   dptr,
        len:    rdup_phsz,
        phsz:   rdup_phsz,
        asd:    Arc::new(Mutex::new(GPUAsyncState::default())),
      });
      self.regions.push(reg.clone());
      self.used_ext += rdup_phsz;
      reg
    };
    assert!(self.regions[0].phsz + self.used_ext <= self.max_phsz);
    assert!(mem.size_bytes() >= bare_phsz);
    GPUDeviceTypedMem{
      dptr: mem.as_mut_dptr() as *mut _,
      len:  len,
      _mem: mem,
      _mrk: PhantomData,
    }
  }
}

pub struct GPUDeviceRawAlloc;
pub type GPUDeviceDefaultAlloc = GPUDeviceRawAlloc;

impl Default for GPUDeviceRawAlloc {
  fn default() -> Self {
    GPUDeviceRawAlloc
  }
}

impl<T> GPUDeviceAlloc<T> for GPUDeviceRawAlloc where T: Copy + 'static {
  type Mem = GPUDeviceRawMem<T>;

  unsafe fn alloc(&self, len: usize, conn: GPUDeviceConn) -> GPUDeviceRawMem<T> where T: Copy {
    GPUDeviceRawMem::<T>::alloc(len, conn)
  }
}

pub struct GPUDeviceRawMem<T> where T: Copy {
  dev:  GPUDeviceId,
  dptr: *mut T,
  len:  usize,
  phsz: usize,
  asd:  Arc<Mutex<GPUAsyncState>>,
}

unsafe impl<T> Send for GPUDeviceRawMem<T> where T: Copy {}
unsafe impl<T> Sync for GPUDeviceRawMem<T> where T: Copy {}

impl<T> Drop for GPUDeviceRawMem<T> where T: Copy {
  fn drop(&mut self) {
    let pop_dev = CudaDevice::get_current().unwrap();
    // TODO: need to synchronize on the ticket if one exists.
    //CudaDevice::synchronize().unwrap();
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
        "for safety, device memory size should not exceed 2**31-1 elements");
    let dptr = match cuda_alloc_device::<T>(len) {
      Err(e) => panic!("GPUDeviceRawMem: failed to alloc len {} elemsz {}: {:?}", len, size_of::<T>(), e),
      Ok(dptr) => dptr,
    };
    assert!(!dptr.is_null());
    GPUDeviceRawMem{
      dev:  conn.device(),
      dptr: dptr,
      len:  len,
      phsz: len * size_of::<T>(),
      asd:  Arc::new(Mutex::new(GPUAsyncState::default())),
    }
  }
}

impl<T> GPUDeviceAsync for GPUDeviceRawMem<T> where T: Copy {
  fn async_state(&self) -> Arc<Mutex<GPUAsyncState>> {
    self.asd.clone()
  }
}

impl<T> GPUDevicePlace for GPUDeviceRawMem<T> where T: Copy {
  fn device(&self) -> GPUDeviceId {
    self.dev
  }
}

impl<T> GPUDeviceMem<T> for GPUDeviceRawMem<T> where T: Copy {
  unsafe fn as_dptr(&self) -> *const T {
    self.dptr
  }

  unsafe fn as_mut_dptr(&self) -> *mut T {
    self.dptr
  }

  fn len(&self) -> usize {
    self.len
  }

  fn size_bytes(&self) -> usize {
    self.phsz
  }
}
