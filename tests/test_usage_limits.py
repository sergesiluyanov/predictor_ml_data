from pathlib import Path
import tempfile
import unittest

from usage_limits import consume_request, refund_request


class UsageLimitTests(unittest.TestCase):
    def test_enforces_persistent_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "usage.sqlite3"
            self.assertEqual(
                consume_request(database, user_id=42, limit=2),
                (True, 1),
            )
            self.assertEqual(
                consume_request(database, user_id=42, limit=2),
                (True, 0),
            )
            self.assertEqual(
                consume_request(database, user_id=42, limit=2),
                (False, 0),
            )

    def test_refunds_failed_request(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "usage.sqlite3"
            consume_request(database, user_id=42, limit=1)
            refund_request(database, user_id=42)
            self.assertEqual(
                consume_request(database, user_id=42, limit=1),
                (True, 0),
            )


if __name__ == "__main__":
    unittest.main()
