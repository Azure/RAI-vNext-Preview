# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import numpy as np
from src.responsibleai.rai_analyse._score_card._rai_insight_data import \
    AlphabetLabelIterator
from src.responsibleai.rai_analyse._score_card._rai_insight_data import PdfDataGen


class TestScorecard:
    def test_alphabet_generator_generates_expected_sequence(self):
        alphabet_generator = AlphabetLabelIterator()
        expected_labels = [
            "A",
            "B",
            "C",
            "D",
            "E",
            "F",
            "G",
            "H",
            "I",
            "J",
            "K",
            "L",
            "M",
            "N",
            "O",
            "P",
            "Q",
            "R",
            "S",
            "T",
            "U",
            "V",
            "W",
            "X",
            "Y",
            "Z",
            "AA",
            "AB",
            "AC",
            "AD",
        ]
        generated_labels = [
            next(alphabet_generator) for i in range(len(expected_labels))
        ]

        assert expected_labels == generated_labels

    def test_replace_labels_as_expected(self):
        other_label = "replaced"
        primary_label = "true_label"
        test_y = np.array([primary_label, "replace_me", "blahblah", primary_label])
        replaced_y = PdfDataGen._replace_labels(test_y, primary_label, other_label)

        assert replaced_y == [primary_label, other_label, other_label, primary_label]
