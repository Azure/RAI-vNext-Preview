# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import pytest

from src.responsibleai.rai_analyse.arg_helpers import (
    boolean_parser
)


class TestBooleanParser:
    @pytest.mark.parametrize("value", ['True', 'true'])
    def test_true(self, value):
        assert boolean_parser(value)

    @pytest.mark.parametrize("value", ["False", 'false'])
    def test_false(self, value):
        assert not boolean_parser(value)

    @pytest.mark.parameterize("value", [None, "1", "0", "rand_string"])
    def test_bad_input(self, value):
        with pytest.raises(ValueError, match="ddd"):
            boolean_parser(value)