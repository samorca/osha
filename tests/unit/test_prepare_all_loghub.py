import unittest

import pandas as pd

from scripts.data.prepare_all_loghub import build_timestamp


class LoghubTimestampTests(unittest.TestCase):
    def test_partial_month_day_uses_stable_reference_year(self):
        frame = pd.DataFrame(
            {
                "Date": ["03-17"],
                "Time": ["16:13:38.811"],
            }
        )

        timestamps = build_timestamp(frame)

        self.assertEqual(
            timestamps.iloc[0],
            pd.Timestamp("2000-03-17 16:13:38.811", tz="UTC"),
        )
        self.assertEqual(str(timestamps.dtype), "datetime64[ns, UTC]")

    def test_month_day_columns_are_combined(self):
        frame = pd.DataFrame(
            {
                "Month": ["Nov"],
                "Date": ["14"],
                "Time": ["15:16:01"],
            }
        )

        timestamp = build_timestamp(frame).iloc[0]

        self.assertEqual(
            timestamp,
            pd.Timestamp("2000-11-14 15:16:01", tz="UTC"),
        )

    def test_time_only_does_not_depend_on_current_date(self):
        frame = pd.DataFrame({"Time": ["15:16:01"]})

        timestamp = build_timestamp(frame).iloc[0]

        self.assertEqual(
            timestamp,
            pd.Timestamp("2000-01-01 15:16:01", tz="UTC"),
        )

    def test_date_less_dataset_has_compatible_timestamp_dtype(self):
        frame = pd.DataFrame({"LineId": ["1"], "Content": ["message"]})

        timestamps = build_timestamp(frame)

        self.assertTrue(timestamps.isna().all())
        self.assertEqual(str(timestamps.dtype), "datetime64[ns, UTC]")

    def test_numeric_timestamp_values_are_preserved(self):
        frame = pd.DataFrame({"Time": [1_060_163_570]})

        timestamp = build_timestamp(frame).iloc[0]

        self.assertEqual(
            timestamp,
            pd.Timestamp(1_060_163_570, unit="ns", tz="UTC"),
        )


if __name__ == "__main__":
    unittest.main()
