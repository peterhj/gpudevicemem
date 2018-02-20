#![allow(non_camel_case_types)]
#![allow(non_upper_case_globals)]
#![allow(non_snake_case)]

pub mod routines_gpu {

use ::{GPUDeviceArchSummary};

use cuda::ffi::runtime::{CUstream_st};

#[derive(Clone, Copy)]
#[repr(C)]
pub struct KernelConfig {
  pub block_sz:     u32,
  pub max_block_ct: u32,
}

impl KernelConfig {
  pub fn new(arch_sum: &GPUDeviceArchSummary) -> Self {
    KernelConfig{
      // TODO
      block_sz:     1024,
      max_block_ct: 16 * arch_sum.mp_count as u32,
    }
  }
}

include!(concat!(env!("OUT_DIR"), "/routines_gpu_bind.rs"));

}
