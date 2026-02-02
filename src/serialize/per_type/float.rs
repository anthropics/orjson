// SPDX-License-Identifier: (Apache-2.0 OR MIT)

use crate::opt::{Opt, DISALLOW_NAN};
use crate::serialize::error::SerializeError;
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
        #[cfg(yyjson_allow_inf_and_nan)]
        {
            if unlikely!(opt_enabled!(self.opts, DISALLOW_NAN)) && !value.is_finite() {
                err!(SerializeError::FloatNotFinite)
            }
            serializer.serialize_f64(value)
        }
        #[cfg(not(yyjson_allow_inf_and_nan))]
        {
            if value.is_finite() {
                serializer.serialize_f64(value)
            } else {
                Err(serde::ser::Error::custom("Cannot serialize Infinity or NaN"))
            }
        }
    }
}
