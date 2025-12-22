# SPDX-License-Identifier: (Apache-2.0 OR MIT)

import pytest

import orjson


class TestLoadsNext:
    def test_loads_next_simple_object(self):
        """
        loads_next() parses a single JSON object
        """
        data = b'{"key": "value"}'
        obj, consumed = orjson.loads_next(data)
        assert obj == {"key": "value"}
        assert consumed == len(data)

    def test_loads_next_simple_array(self):
        """
        loads_next() parses a single JSON array
        """
        data = b'[1, 2, 3]'
        obj, consumed = orjson.loads_next(data)
        assert obj == [1, 2, 3]
        assert consumed == len(data)

    def test_loads_next_with_trailing_data(self):
        """
        loads_next() stops after first JSON document
        """
        data = b'{"a": 1}{"b": 2}'
        obj, consumed = orjson.loads_next(data)
        assert obj == {"a": 1}
        assert consumed == 8  # len('{"a": 1}')

    def test_loads_next_jsonl(self):
        """
        loads_next() can be used to parse JSONL
        """
        data = b'{"line": 1}\n{"line": 2}\n{"line": 3}\n'
        results = []
        offset = 0
        while offset < len(data):
            remaining = data[offset:]
            # Skip leading whitespace
            stripped = remaining.lstrip()
            if not stripped:
                break
            offset += len(remaining) - len(stripped)
            obj, consumed = orjson.loads_next(data[offset:])
            results.append(obj)
            offset += consumed
        assert results == [{"line": 1}, {"line": 2}, {"line": 3}]

    def test_loads_next_ndjson(self):
        """
        loads_next() can be used to parse NDJSON (newline-delimited JSON)
        """
        data = b'{"id":1}\n{"id":2}\n{"id":3}'
        results = []
        offset = 0
        while offset < len(data):
            remaining = data[offset:]
            stripped = remaining.lstrip()
            if not stripped:
                break
            offset += len(remaining) - len(stripped)
            obj, consumed = orjson.loads_next(data[offset:])
            results.append(obj)
            offset += consumed
        assert results == [{"id": 1}, {"id": 2}, {"id": 3}]

    def test_loads_next_concatenated_json(self):
        """
        loads_next() handles concatenated JSON without separators
        """
        data = b'123"hello"[1,2]{"a":1}'
        results = []
        offset = 0
        while offset < len(data):
            obj, consumed = orjson.loads_next(data[offset:])
            results.append(obj)
            offset += consumed
        assert results == [123, "hello", [1, 2], {"a": 1}]

    def test_loads_next_with_whitespace(self):
        """
        loads_next() handles leading and trailing whitespace
        """
        data = b'  {"key": "value"}  '
        obj, consumed = orjson.loads_next(data)
        assert obj == {"key": "value"}
        # consumed includes leading whitespace but not trailing
        assert consumed == 20  # '  {"key": "value"}'

    def test_loads_next_primitives(self):
        """
        loads_next() handles primitive JSON values
        """
        # Integer
        obj, consumed = orjson.loads_next(b'42extra')
        assert obj == 42
        assert consumed == 2

        # Float
        obj, consumed = orjson.loads_next(b'3.14extra')
        assert obj == 3.14
        assert consumed == 4

        # String
        obj, consumed = orjson.loads_next(b'"hello"extra')
        assert obj == "hello"
        assert consumed == 7

        # Boolean true
        obj, consumed = orjson.loads_next(b'trueextra')
        assert obj is True
        assert consumed == 4

        # Boolean false
        obj, consumed = orjson.loads_next(b'falseextra')
        assert obj is False
        assert consumed == 5

        # Null
        obj, consumed = orjson.loads_next(b'nullextra')
        assert obj is None
        assert consumed == 4

    def test_loads_next_rejects_str(self):
        """
        loads_next() rejects str input (only accepts bytes, bytearray, memoryview)
        """
        with pytest.raises(orjson.JSONDecodeError):
            orjson.loads_next('{"key": "value"}')  # type: ignore

    def test_loads_next_bytearray(self):
        """
        loads_next() accepts bytearray input
        """
        data = bytearray(b'{"key": "value"}')
        obj, consumed = orjson.loads_next(data)
        assert obj == {"key": "value"}
        assert consumed == len(data)

    def test_loads_next_memoryview(self):
        """
        loads_next() accepts memoryview input
        """
        data = memoryview(b'{"key": "value"}')
        obj, consumed = orjson.loads_next(data)
        assert obj == {"key": "value"}
        assert consumed == len(data)

    def test_loads_next_empty_raises(self):
        """
        loads_next() raises on empty input
        """
        with pytest.raises(orjson.JSONDecodeError):
            orjson.loads_next(b'')

    def test_loads_next_whitespace_only_raises(self):
        """
        loads_next() raises on whitespace-only input
        """
        with pytest.raises(orjson.JSONDecodeError):
            orjson.loads_next(b'   ')

    def test_loads_next_invalid_json_raises(self):
        """
        loads_next() raises on invalid JSON
        """
        with pytest.raises(orjson.JSONDecodeError):
            orjson.loads_next(b'{invalid}')

    def test_loads_next_nested(self):
        """
        loads_next() handles deeply nested structures
        """
        data = b'{"a": {"b": {"c": [1, 2, 3]}}}more'
        obj, consumed = orjson.loads_next(data)
        assert obj == {"a": {"b": {"c": [1, 2, 3]}}}
        assert consumed == 28

    def test_loads_next_return_type(self):
        """
        loads_next() returns a tuple of (object, int)
        """
        result = orjson.loads_next(b'{}')
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[1], int)

    def test_loads_next_unicode(self):
        """
        loads_next() handles unicode correctly
        """
        data = '{"emoji": "🎉"}'.encode('utf-8')
        obj, consumed = orjson.loads_next(data)
        assert obj == {"emoji": "🎉"}
        assert consumed == len(data)

    def test_loads_next_large_number(self):
        """
        loads_next() handles large numbers
        """
        data = b'12345678901234567890extra'
        obj, consumed = orjson.loads_next(data)
        assert obj == 12345678901234567890
        assert consumed == 20

    def test_loads_next_inf_nan(self):
        """
        loads_next() handles Infinity and NaN
        """
        import math

        obj, _ = orjson.loads_next(b'Infinity')
        assert math.isinf(obj) and obj > 0

        obj, _ = orjson.loads_next(b'-Infinity')
        assert math.isinf(obj) and obj < 0

        obj, _ = orjson.loads_next(b'NaN')
        assert math.isnan(obj)
