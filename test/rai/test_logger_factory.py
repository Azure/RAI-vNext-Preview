# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
import sys
import traceback

from src.responsibleai.rai_analyse._telemetry._loggerfactory import \
    _extract_and_filter_stack


class TestStackTraceExtraction:
    secret_string = "this_is_a_secret_that_should_not_be_printed"

    def level_0(self):
        self.level_1()

    def level_1(self):
        self.level_2()

    def level_2(self):
        raise ValueError(f"{self.secret_string}")

    def test_expect_full_stack_trace_do_not_contain_secret_value(self):
        extracted_stack = None
        try:
            self.level_0()
        except Exception as e:
            extracted_stack = _extract_and_filter_stack(
                e, traceback.extract_tb(sys.exc_info()[2])
            )

        target_exception_snippet = [
            "ValueError",
            "self.level_0()",
            "self.level_1()",
            "self.level_2()",
            "self.secret_string",
        ]

        assert extracted_stack is not None
        stack_trace = " ".join(extracted_stack)
        assert self.secret_string not in stack_trace
        assert all([i in stack_trace for i in target_exception_snippet])
