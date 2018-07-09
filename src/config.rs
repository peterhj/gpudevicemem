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

use std::env;

fn parse_bool(key: &str, default: bool) -> bool {
  env::var(key).ok()
    .and_then(|value| match value.parse() {
      Err(_) => {
        println!("WARNING: config: failed to parse: key: '{}' value: '{}'", key, value);
        None
      }
      Ok(x) => Some(x),
    })
    .unwrap_or(default)
}

lazy_static! {
  pub static ref CONFIG: Config = Config::default();
}

pub struct Config {
  pub default_stream:   bool,
}

impl Default for Config {
  fn default() -> Self {
    Config{
      default_stream:   parse_bool("GPUDEV_CFG_DEFAULT_STREAM", false),
    }
  }
}
