import unittest

import numpy as np
import pandas as pd

from backtest_ranker import ranker_gate
from ranking_model import (
    ranking_policy_metrics,
    select_long_candidates,
)


class RankingModelTests(unittest.TestCase):
    def test_selects_top_liquid_candidate(self):
        frame = pd.DataFrame(
            {
                "entry_date": ["2026-01-01"] * 6,
                "log_turnover": [1, 2, 3, 4, 5, 6],
                "market_volatility_20": [0.01] * 6,
                "market_return_20": [0.02] * 6,
                "horizon_return": [0.0] * 6,
            }
        )
        selected = select_long_candidates(
            frame,
            np.asarray([10, 20, 30, 40, 50, 60]),
            top_fraction=0.10,
        )
        self.assertEqual(int(selected.sum()), 1)
        self.assertTrue(selected[-1])

    def test_disables_long_in_bad_market_regime(self):
        frame = pd.DataFrame(
            {
                "entry_date": ["2026-01-01"] * 3,
                "log_turnover": [1, 2, 3],
                "market_volatility_20": [0.01] * 3,
                "market_return_20": [-0.02] * 3,
                "horizon_return": [0.01] * 3,
            }
        )
        selected = select_long_candidates(
            frame, np.asarray([1, 2, 3]), top_fraction=0.10
        )
        self.assertFalse(selected.any())

    def test_policy_metrics_include_roundtrip_cost(self):
        frame = pd.DataFrame(
            {
                "entry_date": ["2026-01-01", "2026-01-02"],
                "horizon_return": [0.01, -0.01],
            }
        )
        metrics = ranking_policy_metrics(
            frame,
            np.asarray([True, True]),
            roundtrip_cost_bps=30,
        )
        self.assertEqual(metrics["precision"], 0.5)
        self.assertAlmostEqual(metrics["mean_signal_return"], -0.003)

    def test_gate_accepts_positive_expectancy_below_55_percent_win_rate(self):
        metrics = {
            "signals": 405,
            "mean_signal_return_95_ci": [0.001, 0.01],
            "median_daily_return_95_ci": [0.0001, 0.003],
            "profitable_date_share": 0.63,
            "max_drawdown": -0.04,
            "folds": {
                "1": {"signals": 171, "median_daily_return": 0.0003},
                "2": {"signals": 0, "median_daily_return": None},
                "3": {"signals": 234, "median_daily_return": 0.0017},
            },
        }
        self.assertTrue(ranker_gate(metrics)["passed"])

    def test_gate_rejects_negative_active_fold(self):
        metrics = {
            "signals": 405,
            "mean_signal_return_95_ci": [0.001, 0.01],
            "median_daily_return_95_ci": [0.0001, 0.003],
            "profitable_date_share": 0.63,
            "max_drawdown": -0.04,
            "folds": {
                "1": {"signals": 171, "median_daily_return": -0.0003},
                "2": {"signals": 0, "median_daily_return": None},
                "3": {"signals": 234, "median_daily_return": 0.0017},
            },
        }
        self.assertFalse(ranker_gate(metrics)["passed"])


if __name__ == "__main__":
    unittest.main()
