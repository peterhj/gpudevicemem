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

#![allow(non_camel_case_types)]
#![allow(non_upper_case_globals)]
#![allow(non_snake_case)]

pub mod routines_gpu {

use ::{GPUDeviceArchSummary};

use cuda::ffi::driver::{CUstream_st};

#[derive(Clone, Copy)]
#[repr(C)]
pub struct KernelConfig {
  pub block_sz:     u32,
  pub max_block_ct: u32,
}

impl KernelConfig {
  pub fn new(arch_sum: &GPUDeviceArchSummary) -> Self {
    let max_thblk_sz = match arch_sum.capability_major {
      0 | 4 => unreachable!(),
      1 => 512, // TODO: does 1.0 still even work?
      _ => 1024,
    };
    let max_thblks_per_mp = match arch_sum.capability_major {
      0 | 4 => unreachable!(),
      1 | 2 | 3 => 16,
      _ => 32,
    };
    KernelConfig{
      // TODO
      block_sz:     max_thblk_sz,
      max_block_ct: (max_thblks_per_mp * arch_sum.mp_count) as u32,
    }
  }
}

include!(concat!(env!("OUT_DIR"), "/routines_gpu_bind.rs"));

}
