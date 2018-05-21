use ::*;

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
            cudaError_cudaSuccess => {}
            cudaError_cudaErrorPeerAccessAlreadyEnabled => {}
            _ => panic!(),
          }
        }
        _ => panic!(),
      }
    }
  }
}
