# SPDX-License-Identifier: (Apache-2.0 OR MIT)

import os
import tempfile

import pytest

import orjson

try:
    import numpy
except ImportError:
    numpy = None  # type: ignore


@pytest.mark.skipif(numpy is None, reason="numpy is not installed")
class TestLogitNpz:
    def test_save_load_single_array(self):
        """Test saving and loading a single array."""
        arr = numpy.array([1.0, 2.0, 3.0], dtype=numpy.float32)

        with tempfile.NamedTemporaryFile(suffix=".logitnpz", delete=False) as f:
            path = f.name

        try:
            orjson.logitnpz_save(path, {"data": arr})
            result = orjson.logitnpz_load(path)

            assert "data" in result
            numpy.testing.assert_array_equal(result["data"], arr)
        finally:
            os.unlink(path)

    def test_save_load_multiple_arrays(self):
        """Test saving and loading multiple arrays."""
        arrays = {
            "logits": numpy.random.randn(10, 5).astype(numpy.float32),
            "labels": numpy.array([0, 1, 2, 3, 4], dtype=numpy.int64),
            "mask": numpy.array([True, False, True, True, False], dtype=numpy.bool_),
        }

        with tempfile.NamedTemporaryFile(suffix=".logitnpz", delete=False) as f:
            path = f.name

        try:
            orjson.logitnpz_save(path, arrays)
            result = orjson.logitnpz_load(path)

            assert set(result.keys()) == set(arrays.keys())
            for key in arrays:
                numpy.testing.assert_array_equal(result[key], arrays[key])
        finally:
            os.unlink(path)

    def test_dumps_loads_single_array(self):
        """Test serializing and deserializing to/from bytes."""
        arr = numpy.array([[1, 2, 3], [4, 5, 6]], dtype=numpy.int32)

        data = orjson.logitnpz_dumps({"matrix": arr})
        assert isinstance(data, bytes)

        result = orjson.logitnpz_loads(data)
        assert "matrix" in result
        numpy.testing.assert_array_equal(result["matrix"], arr)

    def test_dumps_loads_multiple_arrays(self):
        """Test serializing and deserializing multiple arrays to/from bytes."""
        arrays = {
            "float64": numpy.array([1.5, 2.5, 3.5], dtype=numpy.float64),
            "int8": numpy.array([-128, 0, 127], dtype=numpy.int8),
            "uint8": numpy.array([0, 128, 255], dtype=numpy.uint8),
        }

        data = orjson.logitnpz_dumps(arrays)
        result = orjson.logitnpz_loads(data)

        for key in arrays:
            numpy.testing.assert_array_equal(result[key], arrays[key])

    def test_compression_level(self):
        """Test different compression levels produce valid output."""
        arr = numpy.random.randn(100, 100).astype(numpy.float32)

        # Test different compression levels
        for level in [1, 3, 9, 19]:
            data = orjson.logitnpz_dumps({"arr": arr}, compression_level=level)
            result = orjson.logitnpz_loads(data)
            numpy.testing.assert_array_almost_equal(result["arr"], arr)

    def test_compression_reduces_size(self):
        """Test that compression actually reduces size for compressible data."""
        # Create highly compressible data (all zeros)
        arr = numpy.zeros((1000, 1000), dtype=numpy.float32)

        data = orjson.logitnpz_dumps({"arr": arr})

        # Uncompressed size would be 1000*1000*4 = 4MB
        # Compressed should be much smaller
        assert len(data) < 4_000_000

    def test_various_dtypes(self):
        """Test all common numpy dtypes."""
        arrays = {
            "float16": numpy.array([1.0, 2.0], dtype=numpy.float16),
            "float32": numpy.array([1.0, 2.0], dtype=numpy.float32),
            "float64": numpy.array([1.0, 2.0], dtype=numpy.float64),
            "int8": numpy.array([1, 2], dtype=numpy.int8),
            "int16": numpy.array([1, 2], dtype=numpy.int16),
            "int32": numpy.array([1, 2], dtype=numpy.int32),
            "int64": numpy.array([1, 2], dtype=numpy.int64),
            "uint8": numpy.array([1, 2], dtype=numpy.uint8),
            "uint16": numpy.array([1, 2], dtype=numpy.uint16),
            "uint32": numpy.array([1, 2], dtype=numpy.uint32),
            "uint64": numpy.array([1, 2], dtype=numpy.uint64),
            "bool": numpy.array([True, False], dtype=numpy.bool_),
        }

        data = orjson.logitnpz_dumps(arrays)
        result = orjson.logitnpz_loads(data)

        for key in arrays:
            numpy.testing.assert_array_equal(result[key], arrays[key])
            assert result[key].dtype == arrays[key].dtype

    def test_multidimensional_arrays(self):
        """Test multi-dimensional arrays."""
        arrays = {
            "1d": numpy.arange(10),
            "2d": numpy.arange(20).reshape(4, 5),
            "3d": numpy.arange(24).reshape(2, 3, 4),
            "4d": numpy.arange(120).reshape(2, 3, 4, 5),
        }

        data = orjson.logitnpz_dumps(arrays)
        result = orjson.logitnpz_loads(data)

        for key in arrays:
            numpy.testing.assert_array_equal(result[key], arrays[key])
            assert result[key].shape == arrays[key].shape

    def test_empty_dict(self):
        """Test saving/loading empty dict."""
        data = orjson.logitnpz_dumps({})
        result = orjson.logitnpz_loads(data)
        assert result == {}

    def test_roundtrip_preserves_data(self):
        """Test that roundtrip preserves data exactly."""
        # Use a fixed seed for reproducibility
        rng = numpy.random.RandomState(42)
        arr = rng.randn(50, 50).astype(numpy.float32)

        data = orjson.logitnpz_dumps({"arr": arr})
        result = orjson.logitnpz_loads(data)

        # Check exact equality (not approximate)
        assert numpy.array_equal(result["arr"], arr)

    def test_save_load_file_roundtrip(self):
        """Test file-based save/load roundtrip."""
        arrays = {
            "logits": numpy.random.randn(32, 100).astype(numpy.float32),
            "tokens": numpy.arange(32, dtype=numpy.int64),
        }

        with tempfile.NamedTemporaryFile(suffix=".logitnpz", delete=False) as f:
            path = f.name

        try:
            orjson.logitnpz_save(path, arrays)
            result = orjson.logitnpz_load(path)

            for key in arrays:
                numpy.testing.assert_array_equal(result[key], arrays[key])
        finally:
            os.unlink(path)

    def test_error_invalid_path_type(self):
        """Test error on invalid path type."""
        with pytest.raises((TypeError, ValueError)):
            orjson.logitnpz_save(123, {"arr": numpy.array([1, 2, 3])})

    def test_error_invalid_arrays_type(self):
        """Test error on invalid arrays type."""
        with tempfile.NamedTemporaryFile(suffix=".logitnpz", delete=False) as f:
            path = f.name

        try:
            with pytest.raises((TypeError, ValueError)):
                orjson.logitnpz_save(path, [1, 2, 3])  # Should be dict
        finally:
            if os.path.exists(path):
                os.unlink(path)
