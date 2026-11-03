import json
from pathlib import Path
import tempfile
import unittest

from live_validation import (
    evaluate_live_prediction,
    summarize_live_results,
    write_immutable_json,
)


class LiveValidationTests(unittest.TestCase):
    def test_prediction_file_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "2026-01-01.json"
            self.assertTrue(write_immutable_json(path, {"value": 1}))
            self.assertFalse(write_immutable_json(path, {"value": 2}))
            self.assertEqual(
                json.loads(path.read_text(encoding="utf-8")),
                {"value": 1},
            )

    def test_evaluates_next_open_to_fifth_close_with_frozen_cost(self):
        prediction = {
            "as_of": "2026-01-01",
            "horizon_sessions": 5,
            "candidates": [{"ticker": "TEST", "rank": 1, "score": 0.5}],
        }
        rows = [
            {
                "tradedate": f"2026-01-0{day}",
                "open": 100.0,
                "close": 110.0 if day == 6 else 100.0,
            }
            for day in range(1, 7)
        ]
        result = evaluate_live_prediction(
            prediction,
            {"TEST": rows},
            [],
            roundtrip_cost_bps=30.0,
        )
        self.assertIsNotNone(result)
        candidate = result["candidates"][0]
        self.assertEqual(candidate["entry_date"], "2026-01-02")
        self.assertEqual(candidate["exit_date"], "2026-01-06")
        self.assertAlmostEqual(candidate["net_return"], 0.097)

    def test_incomplete_horizon_remains_pending(self):
        prediction = {
            "as_of": "2026-01-01",
            "horizon_sessions": 5,
            "candidates": [{"ticker": "TEST"}],
        }
        rows = [
            {"tradedate": f"2026-01-0{day}", "open": 100, "close": 100}
            for day in range(2, 6)
        ]
        self.assertIsNone(
            evaluate_live_prediction(prediction, {"TEST": rows}, [], 30)
        )

    def test_live_gate_requires_enough_independent_dates(self):
        results = [
            {
                "as_of": f"2026-01-{day:02d}",
                "signals": 10,
                "horizon_sessions": 5,
                "mean_net_return": 0.01,
                "candidates": [{"net_return": 0.01}] * 10,
            }
            for day in range(1, 30)
        ]
        self.assertFalse(
            summarize_live_results(results)["live_quality_gate"]["passed"]
        )


if __name__ == "__main__":
    unittest.main()
