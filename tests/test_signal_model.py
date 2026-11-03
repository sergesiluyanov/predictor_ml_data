import unittest

import numpy as np
import pandas as pd

from ensemble_signal_model import build_ticker_mapping, lightgbm_frame
from signal_model import (
    calibrate_probabilities,
    evaluate_confidence_thresholds,
    prepare_binary_dataset,
    select_direction_thresholds,
)
from tabular_features import NUMERIC_FEATURES


class SignalModelTests(unittest.TestCase):
    def test_prepare_binary_dataset_removes_small_moves(self):
        frame = pd.DataFrame(
            {"strategy_return": [-0.01, -0.001, 0.0, 0.004]}
        )
        result = prepare_binary_dataset(frame, 0.003)
        self.assertEqual(result["signal_target"].tolist(), [0, 1])

    def test_calibration_keeps_probabilities_bounded(self):
        calibrated = calibrate_probabilities(
            np.asarray([0.1, 0.5, 0.9]), 1.0, 0.0
        )
        np.testing.assert_allclose(calibrated, [0.1, 0.5, 0.9])

    def test_threshold_selection_reports_coverage(self):
        probabilities = np.concatenate(
            [np.full(100, 0.8), np.full(100, 0.2)]
        )
        actual = np.concatenate(
            [np.ones(100), np.zeros(100)]
        ).astype(int)
        threshold, scores = evaluate_confidence_thresholds(
            probabilities, actual
        )
        self.assertEqual(threshold, 0.55)
        self.assertEqual(scores["0.55"]["precision"], 1.0)
        self.assertEqual(scores["0.55"]["coverage"], 1.0)

    def test_direction_without_precision_is_disabled(self):
        probabilities = np.concatenate(
            [np.full(100, 0.8), np.full(100, 0.2)]
        )
        # Up predictions are wrong, down predictions are correct.
        actual = np.zeros(200, dtype=int)
        up_threshold, down_threshold, _ = select_direction_thresholds(
            probabilities, actual
        )
        self.assertIsNone(up_threshold)
        self.assertEqual(down_threshold, 0.55)

    def test_lightgbm_unknown_ticker_has_fallback_code(self):
        numeric = {
            column: [0.0, 0.0]
            for column in NUMERIC_FEATURES
        }
        frame = pd.DataFrame(
            {"ticker": ["AAA", "BBB"], **numeric}
        )
        mapping = build_ticker_mapping(frame.iloc[[0]])
        encoded = lightgbm_frame(frame, mapping)
        self.assertEqual(encoded["ticker_code"].tolist(), [0, -1])


if __name__ == "__main__":
    unittest.main()
