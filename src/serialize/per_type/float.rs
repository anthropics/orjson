// SPDX-License-Identifier: MPL-2.0
// Copyright ijl (2018-2026)

use crate::ffi::PyFloatRef;
use crate::opt::{Opt, DISALLOW_NAN};
use crate::serialize::error::SerializeError;
use serde::ser::{Serialize, Serializer};

pub(crate) struct FloatSerializer {
    ob: PyFloatRef,
    opts: Opt,
}

impl FloatSerializer {
    pub fn new(ptr: PyFloatRef, opts: Opt) -> Self {
        FloatSerializer { ob: ptr, opts: opts }
    }
}

impl Serialize for FloatSerializer {
    #[inline(always)]
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let value = self.ob.value();
        if opt_enabled!(self.opts, DISALLOW_NAN) && !value.is_finite() {
            err!(SerializeError::FloatNotFinite)
        }
        serializer.serialize_f64(value)
    }
}
