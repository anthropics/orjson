// SPDX-License-Identifier: (Apache-2.0 OR MIT)

use crate::deserialize::backend::DeserializeResult;
use crate::deserialize::utf8::read_input_to_buf;
use crate::deserialize::DeserializeError;
use crate::typeref::EMPTY_UNICODE;
use core::ptr::NonNull;

pub fn deserialize(
    ptr: *mut pyo3_ffi::PyObject,
) -> Result<NonNull<pyo3_ffi::PyObject>, DeserializeError<'static>> {
    let result = deserialize_impl(ptr, false)?;
    Ok(result.obj)
}

pub fn deserialize_next(
    ptr: *mut pyo3_ffi::PyObject,
) -> Result<DeserializeResult, DeserializeError<'static>> {
    deserialize_impl(ptr, true)
}

fn deserialize_impl(
    ptr: *mut pyo3_ffi::PyObject,
    stop_when_done: bool,
) -> Result<DeserializeResult, DeserializeError<'static>> {
    debug_assert!(ffi!(Py_REFCNT(ptr)) >= 1);
    let buffer = read_input_to_buf(ptr)?;

    if unlikely!(buffer.len() == 2 && !stop_when_done) {
        if buffer == b"[]" {
            return Ok(DeserializeResult {
                obj: nonnull!(ffi!(PyList_New(0))),
                bytes_read: 2,
            });
        } else if buffer == b"{}" {
            return Ok(DeserializeResult {
                obj: nonnull!(ffi!(PyDict_New())),
                bytes_read: 2,
            });
        } else if buffer == b"\"\"" {
            unsafe {
                return Ok(DeserializeResult {
                    obj: nonnull!(use_immortal!(EMPTY_UNICODE)),
                    bytes_read: 2,
                });
            }
        }
    }

    let buffer_str = unsafe { std::str::from_utf8_unchecked(buffer) };

    crate::deserialize::backend::deserialize(buffer_str, stop_when_done)
}
