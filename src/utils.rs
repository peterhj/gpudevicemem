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

#![allow(non_upper_case_globals)]

use ::*;

use cuda::ffi::driver_types::*;
use cuda::ffi::runtime::*;

pub fn enable_gpu_peer_access(pools: &mut Vec<GPUDeviceStreamPool>) {
  let other_pools = pools.clone();
  for pool in pools.iter_mut() {
    for other_pool in other_pools.iter() {
      if pool.device() == other_pool.device() {
        continue;
      }
      let mut access: i32 = -1;
      let status = unsafe { cudaDeviceCanAccessPeer(
          &mut access as *mut _,
          pool.device().0,
          other_pool.device().0,
      ) };
      assert_eq!(status, cudaError_cudaSuccess);
      match access {
        0 => continue,
        1 => {
          let conn = pool.conn();
          let status = unsafe { cudaDeviceEnablePeerAccess(
              other_pool.device().0,
              0,
          ) };
          match status {
            cudaError_cudaSuccess |
            cudaError_cudaErrorPeerAccessAlreadyEnabled => {}
            e => panic!(),
          }
        }
        _ => panic!(),
      }
    }
  }
}
