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

pub struct Halo1dConfig {
  pub halo_radius:  usize,
  pub halo_axis:    isize,
}

pub struct RingConfig {
  pub num_workers:  usize,
  pub wraparound:   bool,
}

pub struct RingGroup {
  ring_cfg: RingConfig,
  sections: Vec<GPULazyAsyncSection>,
}

pub trait RingGroupExt<T: Copy> {
  fn group_ring_exchange(
      &mut self,
      lsend_bufs: Vec<GPUDeviceArrayView1d<T>>,
      rsend_bufs: Vec<GPUDeviceArrayView1d<T>>,
      lrecv_bufs: Vec<GPUDeviceArrayViewMut1d<T>>,
      rrecv_bufs: Vec<GPUDeviceArrayViewMut1d<T>>,
      conns: Vec<GPUDeviceConn>);
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
    assert_eq!(self.ring_cfg.num_workers, lsend_bufs.len());
    assert_eq!(self.ring_cfg.num_workers, rsend_bufs.len());
    assert_eq!(self.ring_cfg.num_workers, lrecv_bufs.len());
    assert_eq!(self.ring_cfg.num_workers, rrecv_bufs.len());
    assert_eq!(self.ring_cfg.num_workers, conns.len());
    for src_rank in 0 .. self.ring_cfg.num_workers {
      let lrecv_rank = (src_rank + 1) % self.ring_cfg.num_workers;
      let rrecv_rank = (src_rank + self.ring_cfg.num_workers - 1) % self.ring_cfg.num_workers;
      let asguard = self.sections[src_rank].push(conns[src_rank].clone());
      if src_rank >= lrecv_rank && !self.ring_cfg.wraparound {
        // Do nothing.
      } else {
        lrecv_bufs[lrecv_rank].copy(
            rsend_bufs[src_rank].clone(),
            conns[src_rank].clone());
      }
      if rrecv_rank >= src_rank && !self.ring_cfg.wraparound {
        // Do nothing.
      } else {
        rrecv_bufs[rrecv_rank].copy(
            lsend_bufs[src_rank].clone(),
            conns[src_rank].clone());
      }
    }
  }
}
