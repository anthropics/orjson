// SPDX-License-Identifier: (Apache-2.0 OR MIT)

use crate::opt::{Opt, DISALLOW_NAN};
use serde::ser::{Serialize, Serializer};

pub struct FloatSerializer {
    ptr: *mut pyo3_ffi::PyObject,
    opts: Opt,
}

impl FloatSerializer {
    pub fn new(ptr: *mut pyo3_ffi::PyObject, opts: Opt) -> Self {
        FloatSerializer { ptr: ptr, opts: opts }
    }
}

impl Serialize for FloatSerializer {
    #[inline(always)]
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let value = ffi!(PyFloat_AS_DOUBLE(self.ptr));
        if unlikely!(opt_enabled!(self.opts, DISALLOW_NAN)) && !value.is_finite() {
            serializer.serialize_none()
        } else {
            serializer.serialize_f64(value)
        }
    }
}
