from pathlib import Path

import numpy as np
import pandas as pd

from mycumcm2025c.q2_timing import (
    choose_group_count,
    clinical_delay_risk,
    optimal_partition,
    rebuild_intervals,
)


def test_clinical_delay_risk_respects_stage_order() -> None:
    values = clinical_delay_risk(np.array([10.0, 12.0, 13.0, 27.0, 28.0]))
    assert np.all(np.diff(values) > 0)


def test_optimal_partition_finds_two_contiguous_groups() -> None:
    bmi = np.arange(20.0, 28.0)
    times = np.array([11.0, 15.0])
    risks = np.column_stack(
        [np.r_[np.zeros(4), np.full(4, 5.0)], np.r_[np.full(4, 5.0), np.zeros(4)]]
    )
    result = optimal_partition(bmi, risks, times, groups=2, min_group_size=2)
    assert result is not None
    assert result["groups"][0]["end_index"] == 4
    assert result["groups"][0]["optimal_week"] == 11.0
    assert result["groups"][1]["optimal_week"] == 15.0


def test_group_count_uses_smallest_negligible_improvement() -> None:
    partitions = {
        1: {"mean_risk": 1.0},
        2: {"mean_risk": 0.8},
        3: {"mean_risk": 0.795},
        4: {"mean_risk": 0.793},
    }
    assert choose_group_count(partitions, tolerance=0.01) == 2


def test_rebuild_at_four_percent_matches_preprocessing_contract() -> None:
    project_root = Path(__file__).resolve().parents[1]
    events = pd.read_csv(project_root / "data/processed/male_sampling_events.csv")
    expected = pd.read_csv(project_root / "data/processed/male_patient_events.csv")

    columns = [
        "patient_id",
        "bmi_first",
        "event_lower_week",
        "event_upper_week",
        "censoring_type",
    ]
    rebuilt = rebuild_intervals(events, threshold=0.04)[columns]
    expected = expected[columns].sort_values("patient_id").reset_index(drop=True)
    pd.testing.assert_frame_equal(rebuilt, expected, check_dtype=False, atol=1e-12)
