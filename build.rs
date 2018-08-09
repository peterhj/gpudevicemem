extern crate bindgen;
extern crate cc;
extern crate walkdir;

use walkdir::{WalkDir};

use std::env;
use std::fs;
use std::path::{PathBuf};

fn main() {
  let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
  let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

  println!("cargo:rustc-link-search=native={}", out_dir.display());
  println!("cargo:rustc-link-lib=static=gpudevicemem_routines_gpu");

  println!("cargo:rerun-if-changed=build.rs");
  let routines_gpu_src_dir = PathBuf::from(manifest_dir.clone()).join("routines_gpu");
  for entry in WalkDir::new(routines_gpu_src_dir.to_str().unwrap()) {
    let entry = entry.unwrap();
    println!("cargo:rerun-if-changed={}", entry.path().display());
  }

  fs::remove_file(out_dir.join("libgpudevicemem_routines_gpu.a")).ok();

  cc::Build::new()
    .cuda(true)
    .opt_level(2)
    .pic(true)
    .flag("-gencode").flag("arch=compute_35,code=sm_35")
    .flag("-gencode").flag("arch=compute_37,code=sm_37")
    .flag("-gencode").flag("arch=compute_52,code=sm_52")
    .flag("-gencode").flag("arch=compute_60,code=sm_60")
    .flag("-gencode").flag("arch=compute_61,code=sm_61")
    .flag("-gencode").flag("arch=compute_70,code=sm_70")
    .flag("-prec-div=true")
    .flag("-prec-sqrt=true")
    .flag("-std=c++11")
    .flag("-Xcompiler").flag("-fno-strict-aliasing")
    .flag("-Xcompiler").flag("-Werror")
    .include("routines_gpu")
    .include("/usr/local/cuda/include")
    .file("routines_gpu/bcast.cu")
    .file("routines_gpu/bcast_flat_linear.cu")
    .file("routines_gpu/flat_linear.cu")
    .file("routines_gpu/flat_map.cu")
    .file("routines_gpu/halo_ring.cu")
    .file("routines_gpu/reduce.cu")
    .compile("libgpudevicemem_routines_gpu.a");

  fs::remove_file(out_dir.join("routines_gpu_bind.rs")).ok();

  bindgen::Builder::default()
    .header("routines_gpu/lib.h")
    .whitelist_recursively(false)
    // "bcast.cu"
    .whitelist_function("gpudevicemem_bcast_packed_f32")
    .whitelist_function("gpudevicemem_bcast_packed_accumulate_f32")
    .whitelist_function("gpudevicemem_bcast_Ib_Oab_packed_f32")
    .whitelist_function("gpudevicemem_bcast_Ib_Oab_packed_accumulate_f32")
    // "bcast_flat_linear.cu"
    .whitelist_function("gpudevicemem_bcast_flat_add_I1a_I2ab_Oab_packed_f32")
    .whitelist_function("gpudevicemem_bcast_flat_add_I1a_IO2ab_inplace_packed_f32")
    .whitelist_function("gpudevicemem_bcast_flat_add_I1b_I2abc_Oabc_packed_f32")
    .whitelist_function("gpudevicemem_bcast_flat_add_I1b_IO2abc_inplace_packed_f32")
    .whitelist_function("gpudevicemem_bcast_flat_mult_I1b_I2ab_Oab_packed_f32")
    .whitelist_function("gpudevicemem_bcast_flat_mult_add_I1b_I2ab_I3b_Oab_packed_f32")
    .whitelist_function("gpudevicemem_bcast_flat_mult_I1b_I2abc_Oabc_packed_f32")
    .whitelist_function("gpudevicemem_bcast_flat_mult_add_I1b_I2abc_I3b_Oabc_packed_f32")
    .whitelist_function("gpudevicemem_flat_bcast_rdiv_I1ab_I2b_Oab_packed_f32")
    // "flat_linear.cu"
    .whitelist_function("gpudevicemem_flat_add_inplace_f32")
    .whitelist_function("gpudevicemem_flat_mult_inplace_f32")
    .whitelist_function("gpudevicemem_flat_mult_f32")
    .whitelist_function("gpudevicemem_flat_mult_add_f32")
    .whitelist_function("gpudevicemem_flat_rdiv_inplace_f32")
    // "flat_map.cu"
    .whitelist_function("gpudevicemem_set_constant_flat_map_inplace_f32")
    .whitelist_function("gpudevicemem_add_constant_flat_map_inplace_f32")
    .whitelist_function("gpudevicemem_add_constant_flat_map_f32")
    .whitelist_function("gpudevicemem_mult_constant_flat_map_f32")
    .whitelist_function("gpudevicemem_rdiv_constant_flat_map_f32")
    .whitelist_function("gpudevicemem_ldiv_constant_flat_map_f32")
    .whitelist_function("gpudevicemem_online_add_flat_map_accum_f32")
    .whitelist_function("gpudevicemem_online_discount_flat_map_accum_f32")
    .whitelist_function("gpudevicemem_online_average_flat_map_accum_f32")
    .whitelist_function("gpudevicemem_is_nonzero_flat_map_f32")
    .whitelist_function("gpudevicemem_is_zero_flat_map_f32")
    // "halo_ring.cu"
    .whitelist_function("gpudevicemem_halo_ring_3d1_zero_lghost_f32")
    .whitelist_function("gpudevicemem_halo_ring_3d1_zero_rghost_f32")
    .whitelist_function("gpudevicemem_halo_ring_3d1_copy_ledge_to_buf_f32")
    .whitelist_function("gpudevicemem_halo_ring_3d1_copy_redge_to_buf_f32")
    .whitelist_function("gpudevicemem_halo_ring_3d1_copy_buf_to_lghost_f32")
    .whitelist_function("gpudevicemem_halo_ring_3d1_copy_buf_to_rghost_f32")
    .whitelist_function("gpudevicemem_halo_ring_3d1_copy_lghost_to_buf_f32")
    .whitelist_function("gpudevicemem_halo_ring_3d1_copy_rghost_to_buf_f32")
    .whitelist_function("gpudevicemem_halo_ring_3d1_accumulate_buf_to_ledge_f32")
    .whitelist_function("gpudevicemem_halo_ring_3d1_accumulate_buf_to_redge_f32")
    // "reduce.cu"
    .whitelist_function("gpudevicemem_sum_packed_deterministic_f32")
    .whitelist_function("gpudevicemem_sum_packed_accumulate_deterministic_f32")
    .whitelist_function("gpudevicemem_sum_I1ab_Oa_packed_deterministic_f32")
    .whitelist_function("gpudevicemem_sum_I1ab_Ob_packed_deterministic_f32")
    //.whitelist_function("gpudevicemem_square_sum_I1ab_Ob_packed_deterministic_f32")
    .whitelist_function("gpudevicemem_sum_I1abc_Ob_packed_deterministic_f32")
    .whitelist_function("gpudevicemem_square_sum_I1abc_Ob_packed_deterministic_f32")
    .whitelist_function("gpudevicemem_mult_then_sum_I1abc_I2abc_Ob_packed_deterministic_f32")
    .generate()
    .expect("bindgen failed to generate cuda kernel bindings")
    .write_to_file(out_dir.join("routines_gpu_bind.rs"))
    .expect("bindgen failed to write cuda kernel bindings");
}
