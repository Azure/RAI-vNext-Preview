# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

from src.responsibleai.rai_analyse._score_card._rai_insight_data import \
    AlphabetLabelIterator


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
