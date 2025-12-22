// SPDX-License-Identifier: (Apache-2.0 OR MIT)

mod backend;
mod cache;
mod deserializer;
mod error;
mod pyobject;
mod utf8;

pub use backend::DeserializeResult;
pub use cache::{KeyMap, KEY_MAP};
pub use deserializer::{deserialize, deserialize_next};
pub use error::DeserializeError;
