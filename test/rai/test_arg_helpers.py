# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import json

import numpy as np

import pytest

from src.responsibleai.rai_analyse.arg_helpers import (
    boolean_parser,
    float_or_json_parser,
    str_or_int_parser,
)


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
