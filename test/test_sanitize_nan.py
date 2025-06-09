# SPDX-License-Identifier: (Apache-2.0 OR MIT)

import math

import pytest

import orjson


class TestSanitizeNaN:
    """
    Test suite for OPT_SANITIZE_NAN option to replace NaN/Infinity with null
    """

    def test_nan_float_dumps_with_sanitize(self):
        """
        NaN should be replaced with null when OPT_SANITIZE_NAN is set
        """
        assert orjson.dumps(float("nan"), option=orjson.OPT_SANITIZE_NAN) == b"null"
        assert orjson.dumps(math.nan, option=orjson.OPT_SANITIZE_NAN) == b"null"

    def test_infinity_float_dumps_with_sanitize(self):
        """
        Infinity should be replaced with null when OPT_SANITIZE_NAN is set
        """
        assert orjson.dumps(float("inf"), option=orjson.OPT_SANITIZE_NAN) == b"null"
        assert orjson.dumps(float("-inf"), option=orjson.OPT_SANITIZE_NAN) == b"null"
        assert orjson.dumps(math.inf, option=orjson.OPT_SANITIZE_NAN) == b"null"
        assert orjson.dumps(-math.inf, option=orjson.OPT_SANITIZE_NAN) == b"null"

    def test_nan_in_list_dumps_with_sanitize(self):
        """
        NaN values in lists should be replaced with null when OPT_SANITIZE_NAN is set
        """
        data = [1.0, float("nan"), 3.0]
        assert orjson.dumps(data, option=orjson.OPT_SANITIZE_NAN) == b"[1.0,null,3.0]"

    def test_infinity_in_dict_dumps_with_sanitize(self):
        """
        Infinity values in dicts should be replaced with null when OPT_SANITIZE_NAN is set
        """
        data = {"a": float("inf"), "b": float("-inf"), "c": 1.0}
        assert orjson.dumps(data, option=orjson.OPT_SANITIZE_NAN) == b'{"a":null,"b":null,"c":1.0}'

    def test_mixed_special_values_with_sanitize(self):
        """
        Mixed NaN and Infinity values should all be replaced with null
        """
        data = {
            "nan": float("nan"),
            "inf": float("inf"),
            "neg_inf": float("-inf"),
            "normal": 42.0,
            "list": [float("nan"), float("inf"), -1.0, float("-inf")]
        }
        expected = b'{"nan":null,"inf":null,"neg_inf":null,"normal":42.0,"list":[null,null,-1.0,null]}'
        assert orjson.dumps(data, option=orjson.OPT_SANITIZE_NAN) == expected

    def test_option_combination(self):
        """
        OPT_SANITIZE_NAN should work with other options
        """
        data = {"value": float("nan")}
        # With indent
        assert orjson.dumps(
            data, 
            option=orjson.OPT_SANITIZE_NAN | orjson.OPT_INDENT_2
        ) == b'{\n  "value": null\n}'
        
        # With sort keys
        data = {"b": float("inf"), "a": float("nan")}
        assert orjson.dumps(
            data,
            option=orjson.OPT_SANITIZE_NAN | orjson.OPT_SORT_KEYS
        ) == b'{"a":null,"b":null}'

    def test_default_behavior_unchanged(self):
        """
        Without OPT_SANITIZE_NAN, NaN and Infinity should still serialize as before
        """
        assert orjson.dumps(float("nan")) == b"NaN"
        assert orjson.dumps(float("inf")) == b"Infinity"
        assert orjson.dumps(float("-inf")) == b"-Infinity"

    def test_numpy_array_with_sanitize(self):
        """
        NaN and Infinity in numpy arrays should be replaced with null when OPT_SANITIZE_NAN is set
        """
        try:
            import numpy as np
        except ImportError:
            pytest.skip("numpy not available")
        
        arr = np.array([1.0, np.nan, np.inf, -np.inf, 2.0])
        assert orjson.dumps(
            arr, 
            option=orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_SANITIZE_NAN
        ) == b"[1.0,null,null,null,2.0]"

    def test_numpy_scalar_with_sanitize(self):
        """
        numpy scalar NaN and Infinity should be replaced with null
        """
        try:
            import numpy as np
        except ImportError:
            pytest.skip("numpy not available")
        
        assert orjson.dumps(
            np.float64(np.nan),
            option=orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_SANITIZE_NAN
        ) == b"null"
        assert orjson.dumps(
            np.float32(np.inf),
            option=orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_SANITIZE_NAN
        ) == b"null"

    def test_pytorch_tensor_with_sanitize(self):
        """
        NaN and Infinity in PyTorch tensors should be replaced with null when OPT_SANITIZE_NAN is set
        """
        try:
            import torch
        except ImportError:
            pytest.skip("torch not available")
        
        tensor = torch.tensor([1.0, float("nan"), float("inf"), float("-inf"), 2.0])
        assert orjson.dumps(
            tensor,
            option=orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_SANITIZE_NAN
        ) == b"[1.0,null,null,null,2.0]"

    def test_pytorch_scalar_with_sanitize(self):
        """
        PyTorch scalar tensors with NaN/Infinity should be replaced with null
        """
        try:
            import torch
        except ImportError:
            pytest.skip("torch not available")
        
        assert orjson.dumps(
            torch.tensor(float("nan")),
            option=orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_SANITIZE_NAN
        ) == b"null"
        assert orjson.dumps(
            torch.tensor(float("inf")),
            option=orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_SANITIZE_NAN
        ) == b"null"

    def test_zero_dim_array_with_sanitize(self):
        """
        Zero-dimensional arrays with NaN/Infinity should be replaced with null
        """
        try:
            import numpy as np
        except ImportError:
            pytest.skip("numpy not available")
        
        assert orjson.dumps(
            np.array(np.nan),
            option=orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_SANITIZE_NAN
        ) == b"null"
        assert orjson.dumps(
            np.array(np.inf),
            option=orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_SANITIZE_NAN
        ) == b"null"

    def test_deeply_nested_with_sanitize(self):
        """
        Deeply nested structures should have all NaN/Infinity replaced
        """
        data = {
            "level1": {
                "level2": {
                    "values": [float("nan"), {"level3": float("inf")}],
                    "neg_inf": float("-inf")
                }
            }
        }
        expected = b'{"level1":{"level2":{"values":[null,{"level3":null}],"neg_inf":null}}}'
        assert orjson.dumps(data, option=orjson.OPT_SANITIZE_NAN) == expected