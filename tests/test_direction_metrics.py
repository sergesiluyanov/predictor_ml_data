import unittest

import numpy as np
import pandas as pd

from backtest_direction import classification_metrics, trading_metrics


class DirectionMetricTests(unittest.TestCase):
    def test_classification_metrics(self):
        actual = np.asarray([0, 1, 2, 2])
        predicted = np.asarray([0, 1, 0, 2])
        metrics = classification_metrics(actual, predicted)
        self.assertEqual(metrics["count"], 4)
        self.assertEqual(metrics["accuracy"], 0.75)
        self.assertEqual(metrics["balanced_accuracy"], (1 + 1 + 0.5) / 3)

    def test_trading_metrics_include_commission(self):
        frame = pd.DataFrame(
            {
                "ticker": ["AAA", "AAA"],
                "target_date": ["2026-01-01", "2026-01-02"],
                "actual_return": [0.01, 0.0],
                "position": [1, 0],
            }
        )
        metrics = trading_metrics(frame, "position", commission_bps=10)
        expected = (1 + 0.01 - 0.001) * (1 - 0.001) - 1
        self.assertAlmostEqual(metrics["mean_ticker_return"], expected)


if __name__ == "__main__":
    unittest.main()
