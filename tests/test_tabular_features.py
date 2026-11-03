import unittest

import numpy as np

from tabular_features import (
    FEATURE_COLUMNS,
    build_feature_frame,
    build_training_frame,
    latest_feature_row,
)


def make_rows(count: int) -> list[dict]:
    return [
        {
            "tradedate": f"2026-01-{index + 1:02d}"
            if index < 31
            else f"2026-02-{index - 30:02d}",
            "close": 100.0 + index * 0.2 + np.sin(index),
            "high": 101.0 + index * 0.2 + np.sin(index),
            "low": 99.0 + index * 0.2 + np.sin(index),
            "volume": 1_000_000.0 + index * 1000,
        }
        for index in range(count)
    ]


class TabularFeatureTests(unittest.TestCase):
    def test_latest_row_has_expected_columns(self):
        rows = make_rows(55)
        latest = latest_feature_row(rows, "TEST", None)
        self.assertEqual(list(latest.columns), FEATURE_COLUMNS)
        self.assertFalse(latest.isna().any().any())

    def test_future_change_does_not_change_previous_features(self):
        rows = make_rows(55)
        original = build_feature_frame(rows, "TEST", None)
        changed_rows = [dict(row) for row in rows]
        changed_rows[-1]["close"] *= 2
        changed = build_feature_frame(changed_rows, "TEST", None)
        np.testing.assert_allclose(
            original.iloc[-2][FEATURE_COLUMNS[1:]].astype(float),
            changed.iloc[-2][FEATURE_COLUMNS[1:]].astype(float),
        )

    def test_training_target_uses_next_day(self):
        rows = make_rows(55)
        frame = build_training_frame(rows, "TEST", None, threshold=0.003)
        last = frame.iloc[-1]
        self.assertEqual(last["target_date"], rows[-1]["tradedate"])
        expected = rows[-1]["close"] / rows[-2]["close"] - 1.0
        self.assertAlmostEqual(last["target_return"], expected)


if __name__ == "__main__":
    unittest.main()
