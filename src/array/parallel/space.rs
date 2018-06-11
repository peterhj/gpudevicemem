use ::*;
use ::array::*;
use ::ffi::routines_gpu::*;

pub fn group_halo_exchange_3d1(
    halo_size: usize,
    bufs: &mut Vec<GPUDeviceArrayViewMut3d<f32>>,
    lo_send: &mut Vec<GPUDeviceArrayViewMut1d<f32>>,
    hi_send: &mut Vec<GPUDeviceArrayViewMut1d<f32>>,
    lo_recv: &mut Vec<GPUDeviceArrayViewMut1d<f32>>,
    hi_recv: &mut Vec<GPUDeviceArrayViewMut1d<f32>>,
    pools: &mut Vec<GPUDeviceStreamPool>)
{
  let num_ranks = bufs.len();
  for (rank, buf) in bufs.iter_mut().enumerate() {
    assert!(buf.size()[1] > 2 * halo_size);
    if rank > 0 {
      assert_eq!(lo_send[rank - 1].size(), buf.size()[0] * halo_size * 2 * buf.size()[2]);
      let conn = pools[rank].conn();
      let mut stream = conn.cuda_stream();
      unsafe { gpudevicemem_halo_pack_lo_packed3d1_f32(
          buf.size()[0] as _,
          (buf.size()[1] - 2 * halo_size) as _,
          buf.size()[2] as _,
          halo_size as _,
          buf.as_dptr(),
          lo_send[rank - 1].as_mut_dptr(),
          conn.cuda_kernel_cfg() as *const _,
          stream.as_mut_ptr(),
      ) };
    }
    if rank < num_ranks - 1 {
      assert_eq!(hi_send[rank].size(), buf.size()[0] * halo_size * 2 * buf.size()[2]);
      let conn = pools[rank].conn();
      let mut stream = conn.cuda_stream();
      unsafe { gpudevicemem_halo_pack_hi_packed3d1_f32(
          buf.size()[0] as _,
          (buf.size()[1] - 2 * halo_size) as _,
          buf.size()[2] as _,
          halo_size as _,
          buf.as_dptr(),
          hi_send[rank].as_mut_dptr(),
          conn.cuda_kernel_cfg() as *const _,
          stream.as_mut_ptr(),
      ) };
    }
  }
  for rank in 0 .. num_ranks {
    let conn = pools[rank].conn();
    conn.sync();
  }
  for rank in 0 .. num_ranks {
    if rank > 0 {
      assert_eq!(lo_recv[rank - 1].size(), hi_send[rank - 1].size());
      let conn = pools[rank].conn();
      lo_recv[rank - 1].copy(hi_send[rank - 1].to_view(), conn);
    }
    if rank < num_ranks - 1 {
      assert_eq!(hi_recv[rank].size(), lo_send[rank].size());
      let conn = pools[rank].conn();
      hi_recv[rank].copy(lo_send[rank].to_view(), conn);
    }
  }
  for rank in 0 .. num_ranks {
    let conn = pools[rank].conn();
    conn.sync();
  }
  for (rank, buf) in bufs.iter_mut().enumerate() {
    if rank > 0 {
      assert_eq!(lo_recv[rank - 1].size(), buf.size()[0] * halo_size * 2 * buf.size()[2]);
      let conn = pools[rank].conn();
      let mut stream = conn.cuda_stream();
      unsafe { gpudevicemem_halo_unpack_lo_packed3d1_f32(
          buf.size()[0] as _,
          (buf.size()[1] - 2 * halo_size) as _,
          buf.size()[2] as _,
          halo_size as _,
          lo_recv[rank - 1].as_dptr(),
          buf.as_mut_dptr(),
          conn.cuda_kernel_cfg() as *const _,
          stream.as_mut_ptr(),
      ) };
    }
    if rank < num_ranks - 1 {
      assert_eq!(hi_recv[rank].size(), buf.size()[0] * halo_size * 2 * buf.size()[2]);
      let conn = pools[rank].conn();
      let mut stream = conn.cuda_stream();
      unsafe { gpudevicemem_halo_unpack_hi_packed3d1_f32(
          buf.size()[0] as _,
          (buf.size()[1] - 2 * halo_size) as _,
          buf.size()[2] as _,
          halo_size as _,
          hi_recv[rank].as_dptr(),
          buf.as_mut_dptr(),
          conn.cuda_kernel_cfg() as *const _,
          stream.as_mut_ptr(),
      ) };
    }
  }
}

pub fn group_halo_reduce_3d1(
    halo_size: usize,
    bufs: &mut Vec<GPUDeviceArrayViewMut3d<f32>>,
    lo_send: &mut Vec<GPUDeviceArrayViewMut1d<f32>>,
    hi_send: &mut Vec<GPUDeviceArrayViewMut1d<f32>>,
    lo_recv: &mut Vec<GPUDeviceArrayViewMut1d<f32>>,
    hi_recv: &mut Vec<GPUDeviceArrayViewMut1d<f32>>,
    pools: &mut Vec<GPUDeviceStreamPool>)
{
  let num_ranks = bufs.len();
  for (rank, buf) in bufs.iter_mut().enumerate() {
    assert!(buf.size()[1] > 2 * halo_size);
    if rank > 0 {
      assert_eq!(lo_send[rank - 1].size(), buf.size()[0] * halo_size * 2 * buf.size()[2]);
      let conn = pools[rank].conn();
      let mut stream = conn.cuda_stream();
      unsafe { gpudevicemem_halo_ghost_pack_lo_packed3d1_f32(
          buf.size()[0] as _,
          (buf.size()[1] - 2 * halo_size) as _,
          buf.size()[2] as _,
          halo_size as _,
          buf.as_dptr(),
          lo_send[rank - 1].as_mut_dptr(),
          conn.cuda_kernel_cfg() as *const _,
          stream.as_mut_ptr(),
      ) };
    }
    if rank < num_ranks - 1 {
      assert_eq!(hi_send[rank].size(), buf.size()[0] * halo_size * 2 * buf.size()[2]);
      let conn = pools[rank].conn();
      let mut stream = conn.cuda_stream();
      unsafe { gpudevicemem_halo_ghost_pack_hi_packed3d1_f32(
          buf.size()[0] as _,
          (buf.size()[1] - 2 * halo_size) as _,
          buf.size()[2] as _,
          halo_size as _,
          buf.as_dptr(),
          hi_send[rank].as_mut_dptr(),
          conn.cuda_kernel_cfg() as *const _,
          stream.as_mut_ptr(),
      ) };
    }
  }
  for rank in 0 .. num_ranks {
    let conn = pools[rank].conn();
    conn.sync();
  }
  for rank in 0 .. num_ranks {
    if rank > 0 {
      assert_eq!(lo_recv[rank - 1].size(), hi_send[rank - 1].size());
      let conn = pools[rank].conn();
      lo_recv[rank - 1].copy(hi_send[rank - 1].to_view(), conn);
    }
    if rank < num_ranks - 1 {
      assert_eq!(hi_recv[rank].size(), lo_send[rank].size());
      let conn = pools[rank].conn();
      hi_recv[rank].copy(lo_send[rank].to_view(), conn);
    }
  }
  for rank in 0 .. num_ranks {
    let conn = pools[rank].conn();
    conn.sync();
  }
  for (rank, buf) in bufs.iter_mut().enumerate() {
    if rank > 0 {
      assert_eq!(lo_recv[rank - 1].size(), buf.size()[0] * halo_size * 2 * buf.size()[2]);
      let conn = pools[rank].conn();
      let mut stream = conn.cuda_stream();
      unsafe { gpudevicemem_halo_edge_reduce_lo_packed3d1_f32(
          buf.size()[0] as _,
          (buf.size()[1] - 2 * halo_size) as _,
          buf.size()[2] as _,
          halo_size as _,
          lo_recv[rank - 1].as_dptr(),
          buf.as_mut_dptr(),
          conn.cuda_kernel_cfg() as *const _,
          stream.as_mut_ptr(),
      ) };
    }
    if rank < num_ranks - 1 {
      assert_eq!(hi_recv[rank].size(), buf.size()[0] * halo_size * 2 * buf.size()[2]);
      let conn = pools[rank].conn();
      let mut stream = conn.cuda_stream();
      unsafe { gpudevicemem_halo_edge_reduce_hi_packed3d1_f32(
          buf.size()[0] as _,
          (buf.size()[1] - 2 * halo_size) as _,
          buf.size()[2] as _,
          halo_size as _,
          hi_recv[rank].as_dptr(),
          buf.as_mut_dptr(),
          conn.cuda_kernel_cfg() as *const _,
          stream.as_mut_ptr(),
      ) };
    }
  }
}
