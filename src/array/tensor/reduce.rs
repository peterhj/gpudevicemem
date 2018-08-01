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

use ::{GPUDeviceId, GPUDeviceConn};
use ::array::*;
use ::array::tensor::conv::*;

use cuda_dnn::*;
use cuda_dnn::ffi::*;
use float::stub::*;
use num_traits::identities::*;

use std::collections::{HashMap};
//use std::mem::{uninitialized};
//use std::ptr::{null, null_mut};
use std::sync::{Arc, Mutex};

// TODO

pub trait GPUBatchReduce4dOps<T: Copy> {
  fn batch_reduce(&mut self,
      //state: &mut XGPUReduceState<T>,
      x: GPUDeviceArrayView4d<T>,
      conn: GPUDeviceConn);
}
