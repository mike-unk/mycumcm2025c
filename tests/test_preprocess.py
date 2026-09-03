import unittest

import numpy as np
import pandas as pd

from mycumcm2025c.preprocess import (
    _stringify_raw,
    classify_patient_event,
    parse_aneuploidy,
    parse_date,
    parse_gestational_age,
    parse_lower_bound_count,
)


class ParseTests(unittest.TestCase):
    def test_gestational_age(self) -> None:
        self.assertEqual(parse_gestational_age("11w+6"), (11.0, 6.0, 83.0))
        self.assertEqual(parse_gestational_age("23w"), (23.0, 0.0, 161.0))
        self.assertTrue(np.isnan(parse_gestational_age("12w+7")[2]))

    def test_dates(self) -> None:
        self.assertEqual(parse_date(20230429), pd.Timestamp("2023-04-29"))
        self.assertEqual(parse_date("2023-02-25"), pd.Timestamp("2023-02-25"))
        self.assertIsNone(_stringify_raw("   "))

    def test_lower_bound_count(self) -> None:
        self.assertEqual(parse_lower_bound_count("≥3"), (3.0, 1))
        self.assertEqual(parse_lower_bound_count(2), (2.0, 0))

    def test_multilabel_target(self) -> None:
        self.assertEqual(parse_aneuploidy("T13T18"), (1, 1, 0, 1))
        self.assertEqual(parse_aneuploidy(None), (0, 0, 0, 0))


class CensoringTests(unittest.TestCase):
    @staticmethod
    def _events(flags: list[bool]) -> pd.DataFrame:
        weeks = np.arange(11, 11 + len(flags), dtype=float)
        return pd.DataFrame(
            {
                "patient_id": ["A001"] * len(flags),
                "record_count": [1] * len(flags),
                "age_years": [30] * len(flags),
                "height_cm": [160] * len(flags),
                "bmi_reported": [30] * len(flags),
                "gestational_weeks": weeks,
                "gestational_days": weeks * 7,
                "test_date": pd.date_range("2023-01-01", periods=len(flags), freq="7D"),
                "draw_number": np.arange(1, len(flags) + 1),
                "y_qualified_median": flags,
                "technical_replicate_discordant": [False] * len(flags),
            }
        )

    def test_interval_censoring(self) -> None:
        result = classify_patient_event(self._events([False, True, True]))
        self.assertEqual(result["censoring_type"], "interval")
        self.assertEqual(result["event_lower_week"], 11.0)
        self.assertEqual(result["event_upper_week"], 12.0)

    def test_left_and_right_censoring(self) -> None:
        self.assertEqual(
            classify_patient_event(self._events([True, True]))["censoring_type"], "left"
        )
        self.assertEqual(
            classify_patient_event(self._events([False, False]))["censoring_type"], "right"
        )

    def test_nonmonotonic_flag(self) -> None:
        result = classify_patient_event(self._events([False, True, False]))
        self.assertTrue(result["nonmonotonic_after_first_pass"])


if __name__ == "__main__":
    unittest.main()
