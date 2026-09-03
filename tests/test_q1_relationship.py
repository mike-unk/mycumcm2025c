import numpy as np
import pandas as pd

from mycumcm2025c.q1_relationship import aggregate_sampling_events, likelihood_ratio


def _records() -> pd.DataFrame:
    rows = []
    for event, patient, y, week, bmi in [
        ("E1", "P1", 0.04, 12.0, 28.0),
        ("E1", "P1", 0.06, 12.0, 28.0),
        ("E2", "P1", 0.08, 16.0, 29.0),
        ("E3", "P2", 0.05, 13.0, 32.0),
    ]:
        rows.append(
            {
                "patient_id": patient,
                "sampling_event_id": event,
                "y_fraction": y,
                "gestational_weeks": week,
                "bmi_analysis": bmi,
                "age_years": 30.0,
                "conception_method": "自然受孕",
                "gc_ratio": 0.4,
                "mapping_ratio": 0.8,
                "duplicate_ratio": 0.03,
                "filtered_ratio": 0.02,
                "raw_reads": 5_000_000,
            }
        )
    return pd.DataFrame(rows)


def test_aggregate_sampling_events_collapses_technical_replicates() -> None:
    events = aggregate_sampling_events(_records())
    assert len(events) == 3
    assert np.isclose(events.loc[events.sampling_event_id == "E1", "y_fraction"].iloc[0], 0.05)
    assert events["logit_y"].notna().all()


def test_likelihood_ratio_uses_parameter_difference() -> None:
    class Fit:
        def __init__(self, llf: float, df: int):
            self.llf = llf
            self.df_modelwc = df

    result = likelihood_ratio(Fit(-120.0, 4), Fit(-110.0, 7))
    assert result["chi2"] == 20.0
    assert result["df"] == 3
    assert 0 < result["p_value"] < 0.001
