// SPDX-License-Identifier: (Apache-2.0 OR MIT)
// Anthropic: PyTorch tensor serialization support

use crate::ffi::PyObject;
use crate::serialize::error::SerializeError;
use crate::serialize::per_type::{DefaultSerializer, NumpySerializer};
use crate::serialize::serializer::PyObjectSerializer;
use core::ffi::c_char;
use serde::ser::{Serialize, Serializer};

#[repr(transparent)]
pub(crate) struct PyTorchSerializer<'a> {
    previous: &'a PyObjectSerializer,
}

impl<'a> PyTorchSerializer<'a> {
    pub fn new(previous: &'a PyObjectSerializer) -> Self {
        Self { previous }
    }
}

impl Serialize for PyTorchSerializer<'_> {
    #[cold]
    #[inline(never)]
    #[cfg_attr(feature = "optimize", optimize(size))]
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        unsafe {
            let ptr = self.previous.ptr;

            // Approach: detach -> cpu -> numpy

            // Get detach() method from tensor if it requires grad
            let detach_method = crate::ffi::PyUnicode_InternFromString(
                "detach\0".as_ptr() as *const c_char,
            );
            let detached = crate::ffi::PyObject_CallMethodObjArgs(
                ptr,
                detach_method,
                core::ptr::null_mut::<PyObject>(),
            );
            crate::ffi::Py_DECREF(detach_method);

            // Get cpu() method to ensure tensor is on CPU
            let cpu_method =
                crate::ffi::PyUnicode_InternFromString("cpu\0".as_ptr() as *const c_char);
            let cpu_tensor = if detached.is_null() {
                crate::ffi::PyObject_CallMethodObjArgs(
                    ptr,
                    cpu_method,
                    core::ptr::null_mut::<PyObject>(),
                )
            } else {
                let result = crate::ffi::PyObject_CallMethodObjArgs(
                    detached,
                    cpu_method,
                    core::ptr::null_mut::<PyObject>(),
                );
                crate::ffi::Py_DECREF(detached);
                result
            };
            crate::ffi::Py_DECREF(cpu_method);

            // Get numpy() method from CPU tensor
            let numpy_method =
                crate::ffi::PyUnicode_InternFromString("numpy\0".as_ptr() as *const c_char);
            let numpy_array = if !cpu_tensor.is_null() {
                let result = crate::ffi::PyObject_CallMethodObjArgs(
                    cpu_tensor,
                    numpy_method,
                    core::ptr::null_mut::<PyObject>(),
                );
                crate::ffi::Py_DECREF(cpu_tensor);
                result
            } else {
                core::ptr::null_mut()
            };
            crate::ffi::Py_DECREF(numpy_method);

            if numpy_array.is_null() {
                // If conversion fails, try default serializer or error
                crate::ffi::PyErr_Clear();
                if self.previous.default.is_some() {
                    DefaultSerializer::new(self.previous).serialize(serializer)
                } else {
                    err!(SerializeError::PyTorchTensorConversion)
                }
            } else {
                // Create a PyObjectSerializer for the numpy array
                let numpy_serializer = PyObjectSerializer {
                    ptr: numpy_array,
                    default: self.previous.default,
                    state: self.previous.state,
                };

                // Use NumpySerializer directly for better performance
                let result = NumpySerializer::new(&numpy_serializer).serialize(serializer);
                crate::ffi::Py_DECREF(numpy_array);
                result
            }
        }
    }
}
