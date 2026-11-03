import unittest

import numpy as np
import pandas as pd

from evaluation_metrics import (
    probability_metrics,
    quality_gate,
    signal_quality_metrics,
    trading_quality_metrics,
    warning_quality_gate,
)


class EvaluationMetricTests(unittest.TestCase):
    def test_probability_metrics_reward_correct_probabilities(self):
        good = probability_metrics(
            np.asarray([0, 0, 1, 1]),
            np.asarray([0.1, 0.2, 0.8, 0.9]),
        )
        bad = probability_metrics(
            np.asarray([0, 0, 1, 1]),
            np.asarray([0.9, 0.8, 0.2, 0.1]),
        )
        self.assertLess(good["log_loss"], bad["log_loss"])
        self.assertLess(good["brier_score"], bad["brier_score"])

    def test_signal_and_trading_metrics(self):
        frame = pd.DataFrame(
            {
                "ticker": ["AAA"] * 4,
                "target_date": pd.date_range("2026-01-01", periods=4),
                "strategy_return": [0.01, -0.01, 0.02, -0.02],
                "target_return": [0.01, -0.01, 0.02, -0.02],
                "signal": ["up", "down", "up", "down"],
                "probability": [0.8, 0.2, 0.9, 0.1],
                "position": [1, 0, 1, 0],
            }
        )
        signals = signal_quality_metrics(
            frame, "signal", "probability", 0.003
        )
        trading = trading_quality_metrics(
            frame, "position", commission_bps=10, slippage_bps=5
        )
        self.assertEqual(signals["precision"], 1.0)
        self.assertGreater(trading["median_ticker_return"], 0.0)

    def test_quality_gate_requires_positive_trading_result(self):
        signals = {
            "precision": 0.6,
            "precision_95_ci": [0.55, 0.65],
            "coverage": 0.1,
        }
        trading = {
            "median_ticker_return": 0.0,
            "profitable_ticker_share": 0.6,
        }
        self.assertFalse(quality_gate(signals, trading)["passed"])

    def test_warning_gate_does_not_require_long_return(self):
        signals = {
            "precision": 0.6,
            "precision_95_ci": [0.56, 0.64],
            "coverage": 0.1,
        }
        groups = {
            "fold": {
                "1": {"signals": 100},
                "2": {"signals": 100},
            },
            "market_regime": {
                "sideways": {"signals": 100, "precision": 0.6}
            },
        }
        self.assertTrue(warning_quality_gate(signals, groups)["passed"])


if __name__ == "__main__":
    unittest.main()
