import unittest
from unittest.mock import MagicMock

import orjson
import pytest

try:
    import torch
    import numpy as np
    HAVE_PYTORCH = True
    del np
except ImportError:
    HAVE_PYTORCH = False

@pytest.mark.skipif(not HAVE_PYTORCH, reason="pytorch not installed")
class PyTorchTests(unittest.TestCase):
    def test_tensor_1d(self):
        """
        torch.Tensor, 1-dimensional
        """
        tensor = torch.tensor([1, 2, 3])
        self.assertEqual(orjson.dumps(tensor, option=orjson.OPT_SERIALIZE_NUMPY), b'[1,2,3]')

    def test_tensor_2d(self):
        """
        torch.Tensor, 2-dimensional
        """
        tensor = torch.tensor([[1, 2], [3, 4]])
        self.assertEqual(orjson.dumps(tensor, option=orjson.OPT_SERIALIZE_NUMPY), b'[[1,2],[3,4]]')

    def test_tensor_float(self):
        """
        torch.Tensor, float
        """
        tensor = torch.tensor([1.1, 2.2, 3.3])
        self.assertEqual(orjson.dumps(tensor, option=orjson.OPT_SERIALIZE_NUMPY), b'[1.1,2.2,3.3]')

    def test_tensor_bool(self):
        """
        torch.Tensor, bool
        """
        tensor = torch.tensor([True, False, True])
        self.assertEqual(orjson.dumps(tensor, option=orjson.OPT_SERIALIZE_NUMPY), b'[true,false,true]')

    def test_tensor_empty(self):
        """
        torch.Tensor, empty
        """
        tensor = torch.tensor([])
        self.assertEqual(orjson.dumps(tensor, option=orjson.OPT_SERIALIZE_NUMPY), b'[]')

    def test_tensor_without_numpy_opt(self):
        """
        torch.Tensor without OPT_SERIALIZE_NUMPY
        """
        tensor = torch.tensor([1, 2, 3])
        with self.assertRaises(orjson.JSONEncodeError):
            orjson.dumps(tensor)

    def test_tensor_requires_grad(self):
        """
        torch.Tensor with requires_grad=True
        """
        tensor = torch.tensor([1., 2., 3.], requires_grad=True)
        self.assertEqual(orjson.dumps(tensor, option=orjson.OPT_SERIALIZE_NUMPY), b'[1.0,2.0,3.0]')

    def test_tensor_on_gpu(self):
        """
        torch.Tensor on GPU if available
        """
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        tensor = torch.tensor([1, 2, 3]).cuda()
        self.assertEqual(orjson.dumps(tensor, option=orjson.OPT_SERIALIZE_NUMPY), b'[1,2,3]')

    def test_tensor_on_gpu_and_requires_grad(self):
        """
        torch.Tensor on GPU if available AND requires_grad=True
        """
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        tensor = torch.tensor([1., 2., 3.], requires_grad=True).cuda()
        self.assertEqual(orjson.dumps(tensor, option=orjson.OPT_SERIALIZE_NUMPY), b'[1.0,2.0,3.0]')

    def test_tensor_zero_dim(self):
        """
        Test 0-dimensional tensors are properly serialized as scalar values
        """
        tensor_float = torch.tensor(0.03)
        self.assertEqual(orjson.dumps(tensor_float, option=orjson.OPT_SERIALIZE_NUMPY), b'0.03')

        tensor_int = torch.tensor(42)
        self.assertEqual(orjson.dumps(tensor_int, option=orjson.OPT_SERIALIZE_NUMPY), b'42')

        data = {
            "scalar_float": torch.tensor(0.03),
            "scalar_int": torch.tensor(42),
            "array": torch.tensor([1, 2, 3]),
        }
        self.assertEqual(
            orjson.dumps(data, option=orjson.OPT_SERIALIZE_NUMPY),
            b'{"scalar_float":0.03,"scalar_int":42,"array":[1,2,3]}'
        )

    def test_tensor_special_values(self):
        """
        Test that special values (nan, inf) are properly serialized
        """
        tensor_nan = torch.tensor(float('nan'))
        self.assertEqual(orjson.dumps(tensor_nan, option=orjson.OPT_SERIALIZE_NUMPY), b'NaN')

        tensor_inf = torch.tensor(float('inf'))
        self.assertEqual(orjson.dumps(tensor_inf, option=orjson.OPT_SERIALIZE_NUMPY), b'Infinity')
        tensor_neg_inf = torch.tensor(float('-inf'))
        self.assertEqual(orjson.dumps(tensor_neg_inf, option=orjson.OPT_SERIALIZE_NUMPY), b'-Infinity')

        data = {
            "nan": torch.tensor(float('nan')),
            "inf": torch.tensor(float('inf')),
            "neg_inf": torch.tensor(float('-inf')),
            "mixed": torch.tensor([1.0, float('nan'), float('inf'), float('-inf')]),
        }
        self.assertEqual(
            orjson.dumps(data, option=orjson.OPT_SERIALIZE_NUMPY),
            b'{"nan":NaN,"inf":Infinity,"neg_inf":-Infinity,"mixed":[1.0,NaN,Infinity,-Infinity]}'
        )

    def test_tensor_in_list(self):
        """PyTorch tensor as element in a Python list"""
        assert orjson.dumps([torch.tensor([1, 2])], option=orjson.OPT_SERIALIZE_NUMPY) == b'[[1,2]]'

    def test_tensor_3d(self):
        """3D tensor"""
        tensor = torch.zeros(2, 3, 4)
        result = orjson.loads(orjson.dumps(tensor, option=orjson.OPT_SERIALIZE_NUMPY))
        assert len(result) == 2 and len(result[0]) == 3 and len(result[0][0]) == 4

    def test_tensor_dtypes(self):
        """Various tensor dtypes"""
        for dtype in [torch.float16, torch.float64, torch.int8, torch.int16, torch.int32]:
            tensor = torch.tensor([1, 2, 3], dtype=dtype)
            result = orjson.loads(orjson.dumps(tensor, option=orjson.OPT_SERIALIZE_NUMPY))
            for i, v in enumerate(result):
                assert abs(v - [1, 2, 3][i]) < 0.01

    def test_non_torch_duck_type(self):
        """Object with numpy/cpu/detach but __module__ not 'torch' is not treated as tensor"""
        class FakeTensor:
            def numpy(self): return [1, 2]
            def cpu(self): return self
            def detach(self): return self
        with self.assertRaises(orjson.JSONEncodeError):
            orjson.dumps(FakeTensor(), option=orjson.OPT_SERIALIZE_NUMPY)

    def test_magicmock_not_tensor(self):
        """MagicMock not detected as PyTorch tensor (post4 fix)"""
        with self.assertRaises(orjson.JSONEncodeError):
            orjson.dumps(MagicMock(), option=orjson.OPT_SERIALIZE_NUMPY)

    def test_tensor_pretty(self):
        """PyTorch tensor with OPT_INDENT_2"""
        tensor = torch.tensor([[1, 2], [3, 4]])
        result = orjson.dumps(tensor, option=orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_INDENT_2)
        assert result == b'[\n  [\n    1,\n    2\n  ],\n  [\n    3,\n    4\n  ]\n]'

    def test_tensor_conversion_failure(self):
        """Sparse tensor fails numpy conversion - PyTorchTensorConversion error"""
        t = torch.sparse_coo_tensor(torch.tensor([[0, 1]]), torch.tensor([1.0, 2.0]), (3,))
        with self.assertRaises(orjson.JSONEncodeError) as cm:
            orjson.dumps(t, option=orjson.OPT_SERIALIZE_NUMPY)
        assert "failed to convert PyTorch tensor to numpy array" in str(cm.exception)

    def test_tensor_conversion_failure_with_default(self):
        """Sparse tensor with default callback falls back to default"""
        t = torch.sparse_coo_tensor(torch.tensor([[0, 1]]), torch.tensor([1.0, 2.0]), (3,))
        result = orjson.dumps(t, option=orjson.OPT_SERIALIZE_NUMPY, default=lambda x: "fallback")
        assert result == b'"fallback"'

    def test_tensor_unsupported_numpy_dtype(self):
        """Complex tensor: numpy() succeeds but numpy dtype is unsupported"""
        tensor = torch.tensor([1+2j, 3+4j])
        with self.assertRaises(orjson.JSONEncodeError) as cm:
            orjson.dumps(tensor, option=orjson.OPT_SERIALIZE_NUMPY)
        assert "unsupported datatype in numpy array" in str(cm.exception)

    def test_tensor_unsupported_numpy_dtype_with_default(self):
        """Complex tensor with default: falls back to default via numpy unsupported path"""
        tensor = torch.tensor([1+2j, 3+4j])
        result = orjson.dumps(tensor, option=orjson.OPT_SERIALIZE_NUMPY, default=lambda x: str(x))
        assert len(result) > 0
