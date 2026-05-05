# SPDX-License-Identifier: (Apache-2.0 OR MIT)
# Copyright ijl (2018-2025), hauntsaninja (2020)

import datetime
import inspect
import json
import re

import pytest

import orjson

SIMPLE_TYPES = (1, 1.0, -1, None, "str", True, False)

LOADS_RECURSION_LIMIT = 2048


def default(obj):
    return str(obj)


class TestApi:
    def test_loads_trailing(self):
        """
        loads() handles trailing whitespace
        """
        assert orjson.loads("{}\n\t ") == {}

    def test_loads_trailing_invalid(self):
        """
        loads() handles trailing invalid
        """
        pytest.raises(orjson.JSONDecodeError, orjson.loads, "{}\n\t a")

    def test_simple_json(self):
        """
        dumps() equivalent to json on simple types
        """
        for obj in SIMPLE_TYPES:
            assert orjson.dumps(obj) == json.dumps(obj).encode("utf-8")

    def test_simple_round_trip(self):
        """
        dumps(), loads() round trip on simple types
        """
        for obj in SIMPLE_TYPES:
            assert orjson.loads(orjson.dumps(obj)) == obj

    def test_loads_type(self):
        """
        loads() invalid type
        """
        for val in (1, 3.14, [], {}, None):  # type: ignore
            pytest.raises(orjson.JSONDecodeError, orjson.loads, val)

    def test_loads_recursion_partial(self):
        """
        loads() recursion limit partial
        """
        pytest.raises(orjson.JSONDecodeError, orjson.loads, "[" * (1024 * 1024))

    @pytest.mark.parametrize(
        "doc",
        [
            b"[" * 2000 + b"]" * 2000,
            b'{"k":' * 2000 + b"1" + b"}" * 2000,
            b"[1," + b"[" * 2000 + b"]" * 2000 + b",2]",
        ],
        ids=["array", "object", "siblings"],
    )
    def test_loads_max_depth_exceeded(self, doc):
        """
        loads(max_depth=N) rejects nesting past N
        """
        with pytest.raises(orjson.JSONDecodeError, match="max_depth exceeded"):
            orjson.loads(doc, max_depth=1024)
        assert orjson.loads(doc, max_depth=None) is not None

    def test_loads_max_depth_under(self):
        """
        loads(max_depth=N) accepts nesting at or under N
        """
        assert isinstance(
            orjson.loads(b"[" * 500 + b"null" + b"]" * 500, max_depth=1024),
            list,
        )
        assert orjson.loads(b"[1]", max_depth=None) == [1]
        assert orjson.loads(b"42", max_depth=1) == 42

    def test_loads_max_depth_fencepost(self):
        """
        loads(max_depth=N) allows exactly N container levels.

        Empty leaf containers don't count toward max_depth (resource bound,
        not structural) — populate_* never recurses for an empty container.
        """
        assert orjson.loads(b"[1]", max_depth=1) == [1]
        assert orjson.loads(b"[[]]", max_depth=1) == [[]]
        assert orjson.loads(b'{"a":{}}', max_depth=1) == {"a": {}}
        with pytest.raises(orjson.JSONDecodeError, match="max_depth exceeded"):
            orjson.loads(b"[[1]]", max_depth=1)
        assert orjson.loads(b'{"a":1}', max_depth=1) == {"a": 1}
        with pytest.raises(orjson.JSONDecodeError, match="max_depth exceeded"):
            orjson.loads(b'{"a":{"b":1}}', max_depth=1)
        assert orjson.loads(b'{"a":[{"b":1}]}', max_depth=3) == {"a": [{"b": 1}]}
        with pytest.raises(orjson.JSONDecodeError, match="max_depth exceeded"):
            orjson.loads(b'{"a":[{"b":1}]}', max_depth=2)

    @pytest.mark.parametrize("bad", [True, False, 0, -1, -(2**100), "x", 2.0])
    def test_loads_max_depth_invalid_value(self, bad):
        """
        loads(max_depth=) with non-positive-int raises TypeError
        """
        with pytest.raises(TypeError, match="positive int"):
            orjson.loads(b"[]", max_depth=bad)  # type: ignore[arg-type]

    @pytest.mark.parametrize("big", [2**40, 2**63])
    def test_loads_max_depth_large_clamped(self, big):
        """
        loads(max_depth=) with huge value is clamped, not rejected
        """
        assert orjson.loads(b"[1]", max_depth=big) == [1]

    def test_loads_default_depth_under(self):
        """
        loads() with no kwarg accepts nesting at the default limit (1024)
        """
        n = 1024
        assert isinstance(orjson.loads(b"[" * n + b"1" + b"]" * n), list)
        assert isinstance(
            orjson.loads(b'{"k":' * (n - 1) + b"1" + b"}" * (n - 1)),
            dict,
        )

    def test_loads_default_depth_exceeded(self):
        """
        loads() with no kwarg rejects nesting past the default limit (matches upstream)
        """
        n = 1025
        with pytest.raises(orjson.JSONDecodeError, match="max_depth exceeded"):
            orjson.loads(b"[" * n + b"1" + b"]" * n)
        with pytest.raises(orjson.JSONDecodeError, match="max_depth exceeded"):
            orjson.loads(b'{"k":' * n + b"1" + b"}" * n)

    def test_loads_max_depth_none_unbounded(self):
        """
        loads(max_depth=None) is the unbounded opt-out
        """
        n = LOADS_RECURSION_LIMIT * 4
        assert isinstance(
            orjson.loads(b"[" * n + b"1" + b"]" * n, max_depth=None),
            list,
        )

    def test_loads_max_depth_higher_override(self):
        """
        loads(max_depth=N) with N > default raises the limit
        """
        n = 4096
        body = b"[" * n + b"1" + b"]" * n
        assert isinstance(orjson.loads(body, max_depth=n), list)
        with pytest.raises(orjson.JSONDecodeError):
            orjson.loads(body)

    def test_loads_arity(self):
        """
        loads() arg validation raises TypeError, matching METH_O behavior
        """
        with pytest.raises(TypeError):
            orjson.loads()  # type: ignore[call-arg]
        with pytest.raises(TypeError):
            orjson.loads(b"[]", b"[]")  # type: ignore[call-arg, misc, arg-type]
        with pytest.raises(TypeError, match="unexpected keyword"):
            orjson.loads(b"[]", foo=1)  # type: ignore[call-arg]

    def test_loads_recursion_valid_limit_array(self):
        """
        loads() deep nesting array
        """
        n = LOADS_RECURSION_LIMIT + 1
        value = b"[" * n + b"]" * n
        result = orjson.loads(value, max_depth=None)
        depth = 0
        v = result
        while isinstance(v, list) and len(v) > 0:
            v = v[0]
            depth += 1
        assert depth == n - 1

    def test_loads_recursion_valid_limit_object(self):
        """
        loads() deep nesting object
        """
        n = LOADS_RECURSION_LIMIT
        value = b'{"key":' * n + b'{"key":true}' + b"}" * n
        result = orjson.loads(value, max_depth=None)
        depth = 0
        v = result
        while isinstance(v, dict) and "key" in v:
            v = v["key"]
            depth += 1
        assert depth == n + 1

    def test_loads_recursion_valid_limit_mixed(self):
        """
        loads() deep nesting mixed
        """
        n = LOADS_RECURSION_LIMIT
        value = b"[" + b'{"key":' * n + b'{"key":true}' + b"}" * n + b"]"
        result = orjson.loads(value, max_depth=None)
        assert isinstance(result, list)

    def test_loads_recursion_valid_excessive_array(self):
        """
        loads(max_depth=None) opt-out allows nesting past the default limit
        """
        n = LOADS_RECURSION_LIMIT * 4
        value = b"[" * n + b"]" * n
        result = orjson.loads(value, max_depth=None)
        depth = 0
        v = result
        while isinstance(v, list) and len(v) > 0:
            v = v[0]
            depth += 1
        assert depth == n - 1

    def test_loads_recursion_valid_limit_array_pretty(self):
        """
        loads() deep nesting array pretty
        """
        n = LOADS_RECURSION_LIMIT + 1
        value = b"[\n  " * n + b"]" * n
        result = orjson.loads(value, max_depth=None)
        depth = 0
        v = result
        while isinstance(v, list) and len(v) > 0:
            v = v[0]
            depth += 1
        assert depth == n - 1

    def test_loads_recursion_valid_limit_object_pretty(self):
        """
        loads() deep nesting object pretty
        """
        n = LOADS_RECURSION_LIMIT
        value = b'{\n  "key":' * n + b'{"key":true}' + b"}" * n
        result = orjson.loads(value, max_depth=None)
        depth = 0
        v = result
        while isinstance(v, dict) and "key" in v:
            v = v["key"]
            depth += 1
        assert depth == n + 1

    def test_loads_recursion_valid_limit_mixed_pretty(self):
        """
        loads() deep nesting mixed pretty
        """
        n = LOADS_RECURSION_LIMIT
        value = b'[\n  {"key":' * n + b'{"key":true}' + b"}]" * n
        result = orjson.loads(value, max_depth=None)
        assert isinstance(result, list)

    def test_loads_recursion_valid_excessive_array_pretty(self):
        """
        loads(max_depth=None) opt-out allows pretty nesting past the default limit
        """
        n = LOADS_RECURSION_LIMIT * 4
        value = b"[\n  " * n + b"]" * n
        result = orjson.loads(value, max_depth=None)
        depth = 0
        v = result
        while isinstance(v, list) and len(v) > 0:
            v = v[0]
            depth += 1
        assert depth == n - 1

    def test_version(self):
        """
        __version__
        """
        assert re.match(r"^\d+\.\d+\.\d+(-\w+)?$", orjson.__version__)

    def test_valueerror(self):
        """
        orjson.JSONDecodeError is a subclass of ValueError
        """
        pytest.raises(orjson.JSONDecodeError, orjson.loads, "{")
        pytest.raises(ValueError, orjson.loads, "{")

    def test_optional_none(self):
        """
        dumps() option, default None
        """
        assert orjson.dumps([], option=None) == b"[]"
        assert orjson.dumps([], default=None) == b"[]"
        assert orjson.dumps([], option=None, default=None) == b"[]"
        assert orjson.dumps([], None, None) == b"[]"

    def test_option_not_int(self):
        """
        dumps() option not int or None
        """
        with pytest.raises(orjson.JSONEncodeError):
            orjson.dumps(True, option=True)

    def test_option_invalid_int(self):
        """
        dumps() option invalid 64-bit number
        """
        with pytest.raises(orjson.JSONEncodeError):
            orjson.dumps(True, option=9223372036854775809)

    def test_option_range_low(self):
        """
        dumps() option out of range low
        """
        with pytest.raises(orjson.JSONEncodeError):
            orjson.dumps(True, option=-1)

    def test_option_range_high(self):
        """
        dumps() option out of range high
        """
        with pytest.raises(orjson.JSONEncodeError):
            orjson.dumps(True, option=1 << 13)

    def test_opts_multiple(self):
        """
        dumps() multiple option
        """
        assert (
            orjson.dumps(
                [1, datetime.datetime(2000, 1, 1, 2, 3, 4)],
                option=orjson.OPT_STRICT_INTEGER | orjson.OPT_NAIVE_UTC,
            )
            == b'[1,"2000-01-01T02:03:04+00:00"]'
        )

    def test_default_positional(self):
        """
        dumps() positional arg
        """
        with pytest.raises(TypeError):
            orjson.dumps(__obj={})  # type: ignore
        with pytest.raises(TypeError):
            orjson.dumps(zxc={})  # type: ignore

    def test_default_unknown_kwarg(self):
        """
        dumps() unknown kwarg
        """
        with pytest.raises(TypeError):
            orjson.dumps({}, zxc=default)  # type: ignore

    def test_default_empty_kwarg(self):
        """
        dumps() empty kwarg
        """
        assert orjson.dumps(None) == b"null"

    def test_default_twice(self):
        """
        dumps() default twice
        """
        with pytest.raises(TypeError):
            orjson.dumps({}, default, default=default)  # type: ignore

    def test_option_twice(self):
        """
        dumps() option twice
        """
        with pytest.raises(TypeError):
            orjson.dumps({}, None, orjson.OPT_NAIVE_UTC, option=orjson.OPT_NAIVE_UTC)  # type: ignore

    def test_option_mixed(self):
        """
        dumps() option one arg, one kwarg
        """

        class Custom:
            def __str__(self):
                return "zxc"

        assert (
            orjson.dumps(
                [Custom(), datetime.datetime(2000, 1, 1, 2, 3, 4)],
                default,
                option=orjson.OPT_NAIVE_UTC,
            )
            == b'["zxc","2000-01-01T02:03:04+00:00"]'
        )

    def test_dumps_signature(self):
        """
        dumps() valid __text_signature__
        """
        assert (
            str(inspect.signature(orjson.dumps))
            == "(obj, /, default=None, option=None)"
        )
        inspect.signature(orjson.dumps).bind("str")
        inspect.signature(orjson.dumps).bind("str", default=default, option=1)
        inspect.signature(orjson.dumps).bind("str", default=None, option=None)

    def test_loads_signature(self):
        """
        loads() valid __text_signature__
        """
        assert str(inspect.signature(orjson.loads)), "(obj == /)"
        inspect.signature(orjson.loads).bind("[]")

    def test_dumps_module_str(self):
        """
        orjson.dumps.__module__ is a str
        """
        assert orjson.dumps.__module__ == "orjson"

    def test_loads_module_str(self):
        """
        orjson.loads.__module__ is a str
        """
        assert orjson.loads.__module__ == "orjson"

    def test_bytes_buffer(self):
        """
        dumps() trigger buffer growing where length is greater than growth
        """
        a = "a" * 900
        b = "b" * 4096
        c = "c" * 4096 * 4096
        assert orjson.dumps([a, b, c]) == f'["{a}","{b}","{c}"]'.encode("utf-8")

    def test_bytes_null_terminated(self):
        """
        dumps() PyBytesObject buffer is null-terminated
        """
        # would raise ValueError: invalid literal for int() with base 10: b'1596728892'
        int(orjson.dumps(1596728892))
