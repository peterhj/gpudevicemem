extern crate bindgen;
extern crate cc;
extern crate walkdir;

use walkdir::{WalkDir};

use std::env;
use std::path::{PathBuf};

fn main() {
  let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
  let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

  println!("cargo:rustc-link-search=native={}", out_dir.display());
  println!("cargo:rustc-link-lib=static=devicemem_routines_gpu");

  let mut routines_gpu_src_dir = PathBuf::from(manifest_dir.clone());
  routines_gpu_src_dir.push("routines_gpu");
  for entry in WalkDir::new(routines_gpu_src_dir.to_str().unwrap()) {
    let entry = entry.unwrap();
    println!("cargo:rerun-if-changed={}", entry.path().display());
  }

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
    .file("routines_gpu/bcast_flat_linear.cu")
    .file("routines_gpu/flat_linear.cu")
    .file("routines_gpu/flat_map.cu")
    .file("routines_gpu/reduce.cu")
    .compile("libdevicemem_routines_gpu.a");

  bindgen::Builder::default()
    .header("routines_gpu/lib.h")
    .whitelist_recursively(false)
    // "bcast_flat_linear.cu"
    .whitelist_function("devicemem_gpu_bcast_flat_mult_I1b_I2ab_Oab_packed_f32")
    .whitelist_function("devicemem_gpu_bcast_flat_mult_add_I1b_I2ab_I3b_Oab_packed_f32")
    .whitelist_function("devicemem_gpu_bcast_flat_mult_I1b_I2abc_Oabc_packed_f32")
    .whitelist_function("devicemem_gpu_bcast_flat_mult_add_I1b_I2abc_I3b_Oabc_packed_f32")
    // "flat_linear.cu"
    .whitelist_function("devicemem_gpu_flat_mult_f32")
    .whitelist_function("devicemem_gpu_flat_mult_add_f32")
    // "flat_map.cu"
    .whitelist_function("devicemem_gpu_set_constant_flat_map_f32")
    .whitelist_function("devicemem_gpu_mult_constant_flat_map_f32")
    // "reduce.cu"
    .whitelist_function("devicemem_gpu_sum_I1ab_Ob_packed_deterministic_f32")
    //.whitelist_function("devicemem_gpu_square_sum_I1ab_Ob_packed_deterministic_f32")
    .whitelist_function("devicemem_gpu_sum_I1abc_Ob_packed_deterministic_f32")
    .whitelist_function("devicemem_gpu_square_sum_I1abc_Ob_packed_deterministic_f32")
    .generate()
    .expect("bindgen failed to generate cuda kernel bindings")
    .write_to_file(out_dir.join("routines_gpu_bind.rs"))
    .expect("bindgen failed to write cuda kernel bindings");
}
