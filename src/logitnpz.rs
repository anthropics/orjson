// SPDX-License-Identifier: (Apache-2.0 OR MIT)

//! logitnpz format: NPZ-like archive format using zstd compression instead of deflate
//!
//! This module provides functions to save and load numpy arrays in a zip archive
//! where each array is stored as a .npy file compressed with zstd.

use core::ffi::c_char;
use pyo3_ffi::*;
use std::fs::File;
use std::io::{Cursor, Read, Write};
use zip::write::SimpleFileOptions;
use zip::{CompressionMethod, ZipArchive, ZipWriter};

use crate::typeref::{load_numpy_types, NUMPY_TYPES};

/// Error type for logitnpz operations
#[derive(Debug)]
pub enum LogitNpzError {
    IoError(std::io::Error),
    ZipError(zip::result::ZipError),
    InvalidFormat(String),
    NumpyNotAvailable,
    PythonError,
}

impl From<std::io::Error> for LogitNpzError {
    fn from(err: std::io::Error) -> Self {
        LogitNpzError::IoError(err)
    }
}

impl From<zip::result::ZipError> for LogitNpzError {
    fn from(err: zip::result::ZipError) -> Self {
        LogitNpzError::ZipError(err)
    }
}

impl LogitNpzError {
    fn to_py_error(&self) -> *mut PyObject {
        unsafe {
            let msg = match self {
                LogitNpzError::IoError(e) => format!("IO error: {}", e),
                LogitNpzError::ZipError(e) => format!("Zip error: {}", e),
                LogitNpzError::InvalidFormat(s) => format!("Invalid format: {}", s),
                LogitNpzError::NumpyNotAvailable => "numpy is not available".to_string(),
                LogitNpzError::PythonError => return std::ptr::null_mut(),
            };
            let msg_obj =
                PyUnicode_FromStringAndSize(msg.as_ptr() as *const c_char, msg.len() as isize);
            PyErr_SetObject(PyExc_ValueError, msg_obj);
            Py_DECREF(msg_obj);
            std::ptr::null_mut()
        }
    }
}

/// Get numpy.save function
unsafe fn get_numpy_save() -> Option<*mut PyObject> {
    let numpy_types = NUMPY_TYPES.get_or_init(load_numpy_types);
    if numpy_types.is_none() {
        return None;
    }

    let numpy_str = "numpy\0";
    let numpy_mod = PyImport_ImportModule(numpy_str.as_ptr() as *const c_char);
    if numpy_mod.is_null() {
        PyErr_Clear();
        return None;
    }

    let save_str = "save\0";
    let save_func = PyObject_GetAttrString(numpy_mod, save_str.as_ptr() as *const c_char);
    Py_DECREF(numpy_mod);

    if save_func.is_null() {
        PyErr_Clear();
        return None;
    }

    Some(save_func)
}

/// Get numpy.load function
unsafe fn get_numpy_load() -> Option<*mut PyObject> {
    let numpy_types = NUMPY_TYPES.get_or_init(load_numpy_types);
    if numpy_types.is_none() {
        return None;
    }

    let numpy_str = "numpy\0";
    let numpy_mod = PyImport_ImportModule(numpy_str.as_ptr() as *const c_char);
    if numpy_mod.is_null() {
        PyErr_Clear();
        return None;
    }

    let load_str = "load\0";
    let load_func = PyObject_GetAttrString(numpy_mod, load_str.as_ptr() as *const c_char);
    Py_DECREF(numpy_mod);

    if load_func.is_null() {
        PyErr_Clear();
        return None;
    }

    Some(load_func)
}

/// Get io.BytesIO class
unsafe fn get_bytesio() -> Option<*mut PyObject> {
    let io_str = "io\0";
    let io_mod = PyImport_ImportModule(io_str.as_ptr() as *const c_char);
    if io_mod.is_null() {
        PyErr_Clear();
        return None;
    }

    let bytesio_str = "BytesIO\0";
    let bytesio_class = PyObject_GetAttrString(io_mod, bytesio_str.as_ptr() as *const c_char);
    Py_DECREF(io_mod);

    if bytesio_class.is_null() {
        PyErr_Clear();
        return None;
    }

    Some(bytesio_class)
}

/// Serialize a numpy array to bytes in .npy format
unsafe fn array_to_npy_bytes(arr: *mut PyObject) -> Result<Vec<u8>, LogitNpzError> {
    let save_func = get_numpy_save().ok_or(LogitNpzError::NumpyNotAvailable)?;
    let bytesio_class = get_bytesio().ok_or(LogitNpzError::NumpyNotAvailable)?;

    // Create a BytesIO object
    let bytesio_args = PyTuple_New(0);
    let bytesio_obj = PyObject_Call(bytesio_class, bytesio_args, std::ptr::null_mut());
    Py_DECREF(bytesio_args);
    Py_DECREF(bytesio_class);

    if bytesio_obj.is_null() {
        Py_DECREF(save_func);
        return Err(LogitNpzError::PythonError);
    }

    // Call numpy.save(bytesio, arr)
    let save_args = PyTuple_New(2);
    Py_INCREF(bytesio_obj);
    PyTuple_SET_ITEM(save_args, 0, bytesio_obj);
    Py_INCREF(arr);
    PyTuple_SET_ITEM(save_args, 1, arr);

    let result = PyObject_Call(save_func, save_args, std::ptr::null_mut());
    Py_DECREF(save_args);
    Py_DECREF(save_func);

    if result.is_null() {
        Py_DECREF(bytesio_obj);
        return Err(LogitNpzError::PythonError);
    }
    Py_DECREF(result);

    // Get the bytes from BytesIO
    let getvalue_str = "getvalue\0";
    let getvalue_func =
        PyObject_GetAttrString(bytesio_obj, getvalue_str.as_ptr() as *const c_char);
    if getvalue_func.is_null() {
        Py_DECREF(bytesio_obj);
        return Err(LogitNpzError::PythonError);
    }

    let empty_args = PyTuple_New(0);
    let bytes_obj = PyObject_Call(getvalue_func, empty_args, std::ptr::null_mut());
    Py_DECREF(empty_args);
    Py_DECREF(getvalue_func);
    Py_DECREF(bytesio_obj);

    if bytes_obj.is_null() {
        return Err(LogitNpzError::PythonError);
    }

    // Extract bytes from Python bytes object
    let mut size: Py_ssize_t = 0;
    let mut buf: *mut c_char = std::ptr::null_mut();
    if PyBytes_AsStringAndSize(bytes_obj, &mut buf, &mut size) < 0 {
        Py_DECREF(bytes_obj);
        return Err(LogitNpzError::PythonError);
    }

    let bytes = std::slice::from_raw_parts(buf as *const u8, size as usize).to_vec();
    Py_DECREF(bytes_obj);

    Ok(bytes)
}

/// Deserialize bytes in .npy format to a numpy array
unsafe fn npy_bytes_to_array(data: &[u8]) -> Result<*mut PyObject, LogitNpzError> {
    let load_func = get_numpy_load().ok_or(LogitNpzError::NumpyNotAvailable)?;
    let bytesio_class = get_bytesio().ok_or(LogitNpzError::NumpyNotAvailable)?;

    // Create a BytesIO object with the data
    let bytes_obj = PyBytes_FromStringAndSize(data.as_ptr() as *const c_char, data.len() as isize);
    if bytes_obj.is_null() {
        Py_DECREF(load_func);
        Py_DECREF(bytesio_class);
        return Err(LogitNpzError::PythonError);
    }

    let bytesio_args = PyTuple_New(1);
    PyTuple_SET_ITEM(bytesio_args, 0, bytes_obj);
    let bytesio_obj = PyObject_Call(bytesio_class, bytesio_args, std::ptr::null_mut());
    Py_DECREF(bytesio_args);
    Py_DECREF(bytesio_class);

    if bytesio_obj.is_null() {
        Py_DECREF(load_func);
        return Err(LogitNpzError::PythonError);
    }

    // Call numpy.load(bytesio, allow_pickle=False)
    let load_args = PyTuple_New(1);
    PyTuple_SET_ITEM(load_args, 0, bytesio_obj);

    let kwargs = PyDict_New();
    let allow_pickle_str = "allow_pickle\0";
    PyDict_SetItemString(kwargs, allow_pickle_str.as_ptr() as *const c_char, crate::typeref::FALSE);

    let arr = PyObject_Call(load_func, load_args, kwargs);
    Py_DECREF(load_args);
    Py_DECREF(kwargs);
    Py_DECREF(load_func);

    if arr.is_null() {
        return Err(LogitNpzError::PythonError);
    }

    Ok(arr)
}

/// Save a dict of numpy arrays to a logitnpz file
pub unsafe fn save_logitnpz(
    path: *mut PyObject,
    arrays: *mut PyObject,
    compression_level: i64,
) -> Result<(), LogitNpzError> {
    // Get path as string
    let path_str = if PyUnicode_Check(path) != 0 {
        let mut size: Py_ssize_t = 0;
        let ptr = PyUnicode_AsUTF8AndSize(path, &mut size);
        if ptr.is_null() {
            return Err(LogitNpzError::PythonError);
        }
        std::str::from_utf8_unchecked(std::slice::from_raw_parts(ptr as *const u8, size as usize))
    } else {
        return Err(LogitNpzError::InvalidFormat(
            "path must be a string".to_string(),
        ));
    };

    // Verify arrays is a dict
    if PyDict_Check(arrays) == 0 {
        return Err(LogitNpzError::InvalidFormat(
            "arrays must be a dict".to_string(),
        ));
    }

    let file = File::create(path_str)?;
    let mut zip = ZipWriter::new(file);

    let options = SimpleFileOptions::default()
        .compression_method(CompressionMethod::Zstd)
        .compression_level(Some(compression_level));

    // Iterate over dict items
    let mut pos: Py_ssize_t = 0;
    let mut key: *mut PyObject = std::ptr::null_mut();
    let mut value: *mut PyObject = std::ptr::null_mut();

    while PyDict_Next(arrays, &mut pos, &mut key, &mut value) != 0 {
        // Get key as string
        let key_str = if PyUnicode_Check(key) != 0 {
            let mut size: Py_ssize_t = 0;
            let ptr = PyUnicode_AsUTF8AndSize(key, &mut size);
            if ptr.is_null() {
                return Err(LogitNpzError::PythonError);
            }
            std::str::from_utf8_unchecked(std::slice::from_raw_parts(
                ptr as *const u8,
                size as usize,
            ))
        } else {
            return Err(LogitNpzError::InvalidFormat(
                "dict keys must be strings".to_string(),
            ));
        };

        // Convert array to npy bytes
        let npy_bytes = array_to_npy_bytes(value)?;

        // Write to zip with .npy extension
        let filename = format!("{}.npy", key_str);
        zip.start_file(&filename, options)?;
        zip.write_all(&npy_bytes)?;
    }

    zip.finish()?;
    Ok(())
}

/// Load a dict of numpy arrays from a logitnpz file
pub unsafe fn load_logitnpz(path: *mut PyObject) -> Result<*mut PyObject, LogitNpzError> {
    // Get path as string
    let path_str = if PyUnicode_Check(path) != 0 {
        let mut size: Py_ssize_t = 0;
        let ptr = PyUnicode_AsUTF8AndSize(path, &mut size);
        if ptr.is_null() {
            return Err(LogitNpzError::PythonError);
        }
        std::str::from_utf8_unchecked(std::slice::from_raw_parts(ptr as *const u8, size as usize))
    } else {
        return Err(LogitNpzError::InvalidFormat(
            "path must be a string".to_string(),
        ));
    };

    let file = File::open(path_str)?;
    let mut archive = ZipArchive::new(file)?;

    let result_dict = PyDict_New();
    if result_dict.is_null() {
        return Err(LogitNpzError::PythonError);
    }

    for i in 0..archive.len() {
        let mut file = archive.by_index(i)?;
        let name = file.name().to_string();

        // Only process .npy files
        if !name.ends_with(".npy") {
            continue;
        }

        // Read file contents
        let mut data = Vec::new();
        file.read_to_end(&mut data)?;

        // Convert to numpy array
        let arr = npy_bytes_to_array(&data)?;

        // Get key name (remove .npy extension)
        let key_name = &name[..name.len() - 4];
        let key_obj =
            PyUnicode_FromStringAndSize(key_name.as_ptr() as *const c_char, key_name.len() as isize);

        if PyDict_SetItem(result_dict, key_obj, arr) < 0 {
            Py_DECREF(key_obj);
            Py_DECREF(arr);
            Py_DECREF(result_dict);
            return Err(LogitNpzError::PythonError);
        }

        Py_DECREF(key_obj);
        Py_DECREF(arr);
    }

    Ok(result_dict)
}

/// Save a dict of numpy arrays to bytes in logitnpz format
pub unsafe fn save_logitnpz_bytes(
    arrays: *mut PyObject,
    compression_level: i64,
) -> Result<*mut PyObject, LogitNpzError> {
    // Verify arrays is a dict
    if PyDict_Check(arrays) == 0 {
        return Err(LogitNpzError::InvalidFormat(
            "arrays must be a dict".to_string(),
        ));
    }

    let mut buffer = Cursor::new(Vec::new());
    {
        let mut zip = ZipWriter::new(&mut buffer);

        let options = SimpleFileOptions::default()
            .compression_method(CompressionMethod::Zstd)
            .compression_level(Some(compression_level));

        // Iterate over dict items
        let mut pos: Py_ssize_t = 0;
        let mut key: *mut PyObject = std::ptr::null_mut();
        let mut value: *mut PyObject = std::ptr::null_mut();

        while PyDict_Next(arrays, &mut pos, &mut key, &mut value) != 0 {
            // Get key as string
            let key_str = if PyUnicode_Check(key) != 0 {
                let mut size: Py_ssize_t = 0;
                let ptr = PyUnicode_AsUTF8AndSize(key, &mut size);
                if ptr.is_null() {
                    return Err(LogitNpzError::PythonError);
                }
                std::str::from_utf8_unchecked(std::slice::from_raw_parts(
                    ptr as *const u8,
                    size as usize,
                ))
            } else {
                return Err(LogitNpzError::InvalidFormat(
                    "dict keys must be strings".to_string(),
                ));
            };

            // Convert array to npy bytes
            let npy_bytes = array_to_npy_bytes(value)?;

            // Write to zip with .npy extension
            let filename = format!("{}.npy", key_str);
            zip.start_file(&filename, options)?;
            zip.write_all(&npy_bytes)?;
        }

        zip.finish()?;
    }

    let data = buffer.into_inner();
    let bytes_obj = PyBytes_FromStringAndSize(data.as_ptr() as *const c_char, data.len() as isize);
    if bytes_obj.is_null() {
        return Err(LogitNpzError::PythonError);
    }

    Ok(bytes_obj)
}

/// Load a dict of numpy arrays from bytes in logitnpz format
pub unsafe fn load_logitnpz_bytes(data: *mut PyObject) -> Result<*mut PyObject, LogitNpzError> {
    // Get bytes from Python bytes object
    let mut size: Py_ssize_t = 0;
    let mut buf: *mut c_char = std::ptr::null_mut();
    if PyBytes_AsStringAndSize(data, &mut buf, &mut size) < 0 {
        return Err(LogitNpzError::PythonError);
    }

    let bytes = std::slice::from_raw_parts(buf as *const u8, size as usize);
    let cursor = Cursor::new(bytes);
    let mut archive = ZipArchive::new(cursor)?;

    let result_dict = PyDict_New();
    if result_dict.is_null() {
        return Err(LogitNpzError::PythonError);
    }

    for i in 0..archive.len() {
        let mut file = archive.by_index(i)?;
        let name = file.name().to_string();

        // Only process .npy files
        if !name.ends_with(".npy") {
            continue;
        }

        // Read file contents
        let mut data = Vec::new();
        file.read_to_end(&mut data)?;

        // Convert to numpy array
        let arr = npy_bytes_to_array(&data)?;

        // Get key name (remove .npy extension)
        let key_name = &name[..name.len() - 4];
        let key_obj =
            PyUnicode_FromStringAndSize(key_name.as_ptr() as *const c_char, key_name.len() as isize);

        if PyDict_SetItem(result_dict, key_obj, arr) < 0 {
            Py_DECREF(key_obj);
            Py_DECREF(arr);
            Py_DECREF(result_dict);
            return Err(LogitNpzError::PythonError);
        }

        Py_DECREF(key_obj);
        Py_DECREF(arr);
    }

    Ok(result_dict)
}

// ============================================================================
// Python-callable functions
// ============================================================================

const DEFAULT_COMPRESSION_LEVEL: i64 = 3;

#[unsafe(no_mangle)]
pub unsafe extern "C" fn logitnpz_save(
    _self: *mut PyObject,
    args: *const *mut PyObject,
    nargs: Py_ssize_t,
    kwnames: *mut PyObject,
) -> *mut PyObject {
    let num_args = PyVectorcall_NARGS(nargs as usize);

    if num_args < 2 {
        let msg = "logitnpz_save() requires at least 2 arguments: path, arrays\0";
        PyErr_SetString(PyExc_TypeError, msg.as_ptr() as *const c_char);
        return std::ptr::null_mut();
    }

    let path = *args.offset(0);
    let arrays = *args.offset(1);
    let mut compression_level = DEFAULT_COMPRESSION_LEVEL;

    if num_args >= 3 {
        let level_obj = *args.offset(2);
        if PyLong_Check(level_obj) != 0 {
            compression_level = PyLong_AsLong(level_obj) as i64;
        }
    }

    // Check for keyword arguments
    if !kwnames.is_null() {
        let kwcount = Py_SIZE(kwnames);
        for i in 0..kwcount {
            let kwname = PyTuple_GET_ITEM(kwnames, i as Py_ssize_t);
            let mut size: Py_ssize_t = 0;
            let ptr = PyUnicode_AsUTF8AndSize(kwname, &mut size);
            if !ptr.is_null() {
                let name = std::str::from_utf8_unchecked(std::slice::from_raw_parts(
                    ptr as *const u8,
                    size as usize,
                ));
                if name == "compression_level" {
                    let level_obj = *args.offset(num_args + i);
                    if PyLong_Check(level_obj) != 0 {
                        compression_level = PyLong_AsLong(level_obj) as i64;
                    }
                }
            }
        }
    }

    match save_logitnpz(path, arrays, compression_level) {
        Ok(()) => {
            Py_INCREF(crate::typeref::NONE);
            crate::typeref::NONE
        }
        Err(e) => e.to_py_error(),
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn logitnpz_load(
    _self: *mut PyObject,
    path: *mut PyObject,
) -> *mut PyObject {
    match load_logitnpz(path) {
        Ok(dict) => dict,
        Err(e) => e.to_py_error(),
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn logitnpz_dumps(
    _self: *mut PyObject,
    args: *const *mut PyObject,
    nargs: Py_ssize_t,
    kwnames: *mut PyObject,
) -> *mut PyObject {
    let num_args = PyVectorcall_NARGS(nargs as usize);

    if num_args < 1 {
        let msg = "logitnpz_dumps() requires at least 1 argument: arrays\0";
        PyErr_SetString(PyExc_TypeError, msg.as_ptr() as *const c_char);
        return std::ptr::null_mut();
    }

    let arrays = *args.offset(0);
    let mut compression_level = DEFAULT_COMPRESSION_LEVEL;

    if num_args >= 2 {
        let level_obj = *args.offset(1);
        if PyLong_Check(level_obj) != 0 {
            compression_level = PyLong_AsLong(level_obj) as i64;
        }
    }

    // Check for keyword arguments
    if !kwnames.is_null() {
        let kwcount = Py_SIZE(kwnames);
        for i in 0..kwcount {
            let kwname = PyTuple_GET_ITEM(kwnames, i as Py_ssize_t);
            let mut size: Py_ssize_t = 0;
            let ptr = PyUnicode_AsUTF8AndSize(kwname, &mut size);
            if !ptr.is_null() {
                let name = std::str::from_utf8_unchecked(std::slice::from_raw_parts(
                    ptr as *const u8,
                    size as usize,
                ));
                if name == "compression_level" {
                    let level_obj = *args.offset(num_args + i);
                    if PyLong_Check(level_obj) != 0 {
                        compression_level = PyLong_AsLong(level_obj) as i64;
                    }
                }
            }
        }
    }

    match save_logitnpz_bytes(arrays, compression_level) {
        Ok(bytes) => bytes,
        Err(e) => e.to_py_error(),
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn logitnpz_loads(
    _self: *mut PyObject,
    data: *mut PyObject,
) -> *mut PyObject {
    match load_logitnpz_bytes(data) {
        Ok(dict) => dict,
        Err(e) => e.to_py_error(),
    }
}
