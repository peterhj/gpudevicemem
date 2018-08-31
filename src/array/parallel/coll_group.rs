/*
Copyright 2018 Peter Jin

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
use ::array::*;
use ::ffi::routines_gpu::*;

#[cfg(feature = "nccl")] use cuda_coll::*;
use parking_lot::{Mutex};

use std::ptr::{null_mut};

pub struct FlatGroupConfig {
  pub num_workers:  usize,
  pub root_rank:    usize,
}

pub trait FlatGroupExt<T: Copy> {
  // FIXME: conns below should be pools.
  fn group_broadcast(
      &mut self,
      send_buf: GPUDeviceArrayView1d<T>,
      recv_bufs: Vec<GPUDeviceArrayViewMut1d<T>>,
      conns: Vec<GPUDeviceConn>);
  fn group_reduce(
      &mut self,
      send_bufs: Vec<GPUDeviceArrayView1d<T>>,
      recv_buf: GPUDeviceArrayViewMut1d<T>,
      conns: Vec<GPUDeviceConn>);
  fn group_allreduce(
      &mut self,
      send_bufs: Vec<GPUDeviceArrayView1d<T>>,
      recv_bufs: Vec<GPUDeviceArrayViewMut1d<T>>,
      conns: Vec<GPUDeviceConn>);
}

#[cfg(feature = "nccl")] static NCCL_GROUP_MUTEX: Mutex<()> = Mutex::new(());

#[cfg(feature = "nccl")]
struct NcclState {
  comm_id:  NcclUniqueId,
  comm:     NcclComm,
  rank:     i32,
  max_rank: i32,
}

#[cfg(feature = "nccl")]
pub struct NcclFlatGroup {
  cfg:      FlatGroupConfig,
  sections: Vec<GPULazyAsyncSection>,
  states:   Vec<NcclState>,
}

#[cfg(feature = "nccl")]
impl NcclFlatGroup {
  pub fn new(cfg: FlatGroupConfig, mut pools: Vec<GPUDeviceStreamPool>) -> NcclFlatGroup {
    let mut sections = vec![];
    let mut states = vec![];
    let comm_id = NcclUniqueId::create().unwrap();
    NCCL_GROUP_MUTEX.raw_lock();
    unsafe { NcclComm::group_start() };
    for rank in 0 .. cfg.num_workers {
      let section = GPULazyAsyncSection::default();
      sections.push(section);
      let state = {
        let conn = pools[rank].conn();
        let comm = NcclComm::init_rank(
            rank as _,
            cfg.num_workers as _,
            comm_id.clone(),
        ).unwrap();
        NcclState{
          comm_id:  comm_id.clone(),
          comm:     comm,
          rank:     rank as _,
          max_rank: cfg.num_workers as _,
        }
      };
      states.push(state);
    }
    unsafe { NcclComm::group_end() };
    unsafe { NCCL_GROUP_MUTEX.raw_unlock() };
    NcclFlatGroup{
      cfg,
      sections,
      states,
    }
  }
}

#[cfg(feature = "nccl")]
impl<T> FlatGroupExt<T> for NcclFlatGroup where T: NcclDataType + 'static {
  fn group_broadcast(
      &mut self,
      send_buf: GPUDeviceArrayView1d<T>,
      mut recv_bufs: Vec<GPUDeviceArrayViewMut1d<T>>,
      conns: Vec<GPUDeviceConn>)
  {
    assert_eq!(recv_bufs.len(), self.cfg.num_workers - 1);
    NCCL_GROUP_MUTEX.raw_lock();
    unsafe { NcclComm::group_start(); }
    for rank in 0 .. self.cfg.num_workers {
      assert_eq!(send_buf.size(), recv_bufs[rank].size());
      let conn = conns[rank].clone();
      if rank == self.cfg.root_rank {
        let packed = send_buf.is_packed();
        assert!(packed, "NCCL-based collectives require packed layout");
        let send_buf = send_buf.wait(conn.clone());
        let mut stream = conn.cuda_stream();
        let res = unsafe { self.states[rank].comm.broadcast(
            send_buf.as_dptr() as *mut T,
            send_buf.inner().size(),
            self.cfg.root_rank as i32, // FIXME: "our" rank to "nccl" rank.
            stream.as_mut_ptr(),
        ) };
        assert!(res.is_ok());
      } else {
        let recv_idx = if rank < self.cfg.root_rank { rank } else { rank - 1 };
        let packed = recv_bufs[recv_idx].is_packed();
        assert!(packed, "NCCL-based collectives require packed layout");
        let mut recv_buf = recv_bufs[recv_idx].wait_mut(conn.clone());
        let mut stream = conn.cuda_stream();
        let res = unsafe { self.states[rank].comm.broadcast(
            recv_buf.as_mut_dptr(),
            recv_buf.inner().size(),
            self.cfg.root_rank as i32, // FIXME: "our" rank to "nccl" rank.
            stream.as_mut_ptr(),
        ) };
        assert!(res.is_ok());
      }
    }
    unsafe { NcclComm::group_end(); }
    unsafe { NCCL_GROUP_MUTEX.raw_unlock(); }
  }

  fn group_reduce(
      &mut self,
      send_bufs: Vec<GPUDeviceArrayView1d<T>>,
      mut recv_buf: GPUDeviceArrayViewMut1d<T>,
      conns: Vec<GPUDeviceConn>)
  {
    assert_eq!(send_bufs.len(), self.cfg.num_workers);
    NCCL_GROUP_MUTEX.raw_lock();
    unsafe { NcclComm::group_start(); }
    for rank in 0 .. self.cfg.num_workers {
      assert_eq!(recv_buf.size(), send_bufs[rank].size());
      let packed = send_bufs[rank].is_packed() && recv_buf.is_packed();
      assert!(packed, "NCCL-based collectives require packed layout");
      let conn = conns[rank].clone();
      if rank == self.cfg.root_rank {
        let send_buf = send_bufs[rank].wait(conn.clone());
        let mut recv_buf = recv_buf.wait_mut(conn.clone());
        let mut stream = conn.cuda_stream();
        let res = unsafe { self.states[rank].comm.reduce(
            send_buf.as_dptr(),
            recv_buf.as_mut_dptr(),
            send_buf.inner().size(),
            NcclReduceOp::Sum,
            self.cfg.root_rank as i32, // FIXME: "our" rank to "nccl" rank.
            stream.as_mut_ptr(),
        ) };
        assert!(res.is_ok());
      } else {
        let send_buf = send_bufs[rank].wait(conn.clone());
        let mut stream = conn.cuda_stream();
        let res = unsafe { self.states[rank].comm.reduce(
            send_buf.as_dptr(),
            null_mut(),
            send_buf.inner().size(),
            NcclReduceOp::Sum,
            self.cfg.root_rank as i32, // FIXME: "our" rank to "nccl" rank.
            stream.as_mut_ptr(),
        ) };
        assert!(res.is_ok());
      }
    }
    unsafe { NcclComm::group_end(); }
    unsafe { NCCL_GROUP_MUTEX.raw_unlock(); }
  }

  fn group_allreduce(
      &mut self,
      send_bufs: Vec<GPUDeviceArrayView1d<T>>,
      mut recv_bufs: Vec<GPUDeviceArrayViewMut1d<T>>,
      conns: Vec<GPUDeviceConn>)
  {
    assert_eq!(send_bufs.len(), self.cfg.num_workers);
    assert_eq!(recv_bufs.len(), self.cfg.num_workers);
    NCCL_GROUP_MUTEX.raw_lock();
    unsafe { NcclComm::group_start(); }
    for rank in 0 .. self.cfg.num_workers {
      assert_eq!(send_bufs[0].size(), send_bufs[rank].size());
      assert_eq!(send_bufs[0].size(), recv_bufs[rank].size());
      let packed = send_bufs[rank].is_packed() && recv_bufs[rank].is_packed();
      assert!(packed, "NCCL-based collectives require packed layout");
      let conn = conns[rank].clone();
      let send_buf = send_bufs[rank].wait(conn.clone());
      let mut recv_buf = recv_bufs[rank].wait_mut(conn.clone());
      let mut stream = conn.cuda_stream();
      let res = unsafe { self.states[rank].comm.all_reduce(
          send_buf.as_dptr(),
          recv_buf.as_mut_dptr(),
          send_buf.inner().size(),
          NcclReduceOp::Sum,
          stream.as_mut_ptr(),
      ) };
      assert!(res.is_ok());
    }
    unsafe { NcclComm::group_end(); }
    unsafe { NCCL_GROUP_MUTEX.raw_unlock(); }
  }
}

pub struct RingGroupConfig {
  pub num_workers:  usize,
  pub wraparound:   bool,
}

pub trait RingGroupExt<T: Copy> {
  // FIXME: conns below should be pools.
  fn group_ring_exchange(
      &mut self,
      lsend_bufs: Vec<GPUDeviceArrayView1d<T>>,
      rsend_bufs: Vec<GPUDeviceArrayView1d<T>>,
      lrecv_bufs: Vec<GPUDeviceArrayViewMut1d<T>>,
      rrecv_bufs: Vec<GPUDeviceArrayViewMut1d<T>>,
      conns: Vec<GPUDeviceConn>);
}

pub struct RingGroup {
  cfg:      RingGroupConfig,
  sections: Vec<GPULazyAsyncSection>,
}

impl RingGroupExt<f32> for RingGroup {
  fn group_ring_exchange(
      &mut self,
      lsend_bufs: Vec<GPUDeviceArrayView1d<f32>>,
      rsend_bufs: Vec<GPUDeviceArrayView1d<f32>>,
      mut lrecv_bufs: Vec<GPUDeviceArrayViewMut1d<f32>>,
      mut rrecv_bufs: Vec<GPUDeviceArrayViewMut1d<f32>>,
      conns: Vec<GPUDeviceConn>)
  {
    assert_eq!(self.cfg.num_workers, lsend_bufs.len());
    assert_eq!(self.cfg.num_workers, rsend_bufs.len());
    assert_eq!(self.cfg.num_workers, lrecv_bufs.len());
    assert_eq!(self.cfg.num_workers, rrecv_bufs.len());
    assert_eq!(self.cfg.num_workers, conns.len());
    for src_rank in 0 .. self.cfg.num_workers {
      let lrecv_rank = (src_rank + 1) % self.cfg.num_workers;
      let rrecv_rank = (src_rank + self.cfg.num_workers - 1) % self.cfg.num_workers;
      let asguard = self.sections[src_rank].push(conns[src_rank].clone());
      if src_rank >= lrecv_rank && !self.cfg.wraparound {
        // Do nothing.
      } else {
        lrecv_bufs[lrecv_rank].copy(
            rsend_bufs[src_rank].clone(),
            conns[src_rank].clone());
      }
      if rrecv_rank >= src_rank && !self.cfg.wraparound {
        // Do nothing.
      } else {
        rrecv_bufs[rrecv_rank].copy(
            lsend_bufs[src_rank].clone(),
            conns[src_rank].clone());
      }
    }
  }
}
