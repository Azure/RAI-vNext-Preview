# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import json

import numpy as np
import pytest

from src.responsibleai.rai_analyse.arg_helpers import (
    boolean_parser, float_or_json_parser, int_or_none_parser,
    json_empty_is_none_parser, str_or_int_parser, str_or_list_parser)


class TestBooleanParser:
    @pytest.mark.parametrize("value", ["True", "true"])
    def test_true(self, value):
        assert boolean_parser(value)

    @pytest.mark.parametrize("value", ["False", "false"])
    def test_false(self, value):
        assert not boolean_parser(value)

    @pytest.mark.parametrize("value", [None, "1", "0", "rand_string"])
    def test_bad_input(self, value):
        with pytest.raises(ValueError, match="Failed to parse to boolean:"):
            boolean_parser(value)


class TestFloatOrJSONParser:
    def test_is_float(self):
        flt = 1.25
        assert float_or_json_parser(str(flt)) == flt

    def test_is_json(self):
        target = [0, 1, 2]
        target_json = json.dumps(target)

        actual = float_or_json_parser(target_json)
        assert np.array_equal(target, actual)


class TestStrOrIntParser:
    @pytest.mark.parametrize("value", [0, -1, 10, 100])
    def test_is_int(self, value):
        assert str_or_int_parser(str(value)) == value

    @pytest.mark.parametrize("value", ["10.1", "a", "None"])
    def test_is_string(self, value):
        assert str_or_int_parser(value) == value


class TestStrOrListParser:
    def test_is_list(self):
        target = [10.25, 1, 2]
        target_json = json.dumps(target)

        actual = str_or_list_parser(target_json)
        assert np.array_equal(target, actual)

    @pytest.mark.parametrize("value", ["", "a"])
    def test_is_string(self, value):
        assert str_or_list_parser(value) == value

    def test_json_not_list(self):
        target = {"a": 1, "b": 2}
        target_json = json.dumps(target)

        with pytest.raises(ValueError, match="Supplied JSON string not list"):
            str_or_list_parser(target_json)


class TestIntOrNoneParser:
    def test_is_none(self):
        assert int_or_none_parser("None") is None

    @pytest.mark.parametrize("value", [1, 2, -1])
    def test_is_int(self, value):
        assert int_or_none_parser(str(value)) == value

    @pytest.mark.parametrize("value", ["a", "10.1", "[]"])
    def test_bad_value(self, value):
        with pytest.raises(ValueError, match="int_or_none_parser failed on:"):
            int_or_none_parser(value)


class TestJSONEmptyIsNoneParser:
    @pytest.mark.parametrize("value", [[1, 2], {"a": 1, "b": 2}])
    def test_json_strings(self, value):
        value_json = json.dumps(value)

        assert value == json_empty_is_none_parser(value_json)

    @pytest.mark.parametrize("value", [[], dict()])
    def test_empty(self, value):
        value_json = json.dumps(value)

        assert json_empty_is_none_parser(value_json) is None
