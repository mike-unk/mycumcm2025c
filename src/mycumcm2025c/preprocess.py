"""Reproducible preprocessing for the 2025 CUMCM problem C dataset."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SHEETS = {
    "男胎检测数据": {"sex": "male", "expected_rows": 1082},
    "女胎检测数据": {"sex": "female", "expected_rows": 605},
}

COLUMNS = [
    "sample_id",
    "patient_id",
    "age_years",
    "height_cm",
    "weight_kg",
    "lmp_date_raw",
    "conception_method",
    "test_date_raw",
    "draw_number",
    "gestational_age_raw",
    "bmi_reported",
    "raw_reads",
    "mapping_ratio",
    "duplicate_ratio",
    "unique_mapped_reads",
    "gc_ratio",
    "z13",
    "z18",
    "z21",
    "zx",
    "zy",
    "y_fraction",
    "x_fraction",
    "gc13",
    "gc18",
    "gc21",
    "filtered_ratio",
    "aneuploidy_raw",
    "gravidity_raw",
    "parity_raw",
    "healthy_birth_raw",
]

NUMERIC_COLUMNS = [
    "sample_id",
    "age_years",
    "height_cm",
    "weight_kg",
    "draw_number",
    "bmi_reported",
    "raw_reads",
    "mapping_ratio",
    "duplicate_ratio",
    "unique_mapped_reads",
    "gc_ratio",
    "z13",
    "z18",
    "z21",
    "zx",
    "zy",
    "y_fraction",
    "x_fraction",
    "gc13",
    "gc18",
    "gc21",
    "filtered_ratio",
    "parity_raw",
]

RATIO_COLUMNS = [
    "mapping_ratio",
    "duplicate_ratio",
    "gc_ratio",
    "gc13",
    "gc18",
    "gc21",
    "filtered_ratio",
]

HEADER_CHECKS = {
    0: "序号",
    1: "孕妇代码",
    9: "检测孕周",
    10: "孕妇BMI",
    27: "染色体的非整倍体",
    30: "胎儿是否健康",
}


@dataclass(frozen=True)
class OutputPaths:
    processed_dir: Path
    report_dir: Path


def parse_date(value: Any) -> pd.Timestamp:
    if value is None or pd.isna(value):
        return pd.NaT
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return pd.Timestamp(value).normalize()
    if isinstance(value, (int, np.integer)) or (
        isinstance(value, (float, np.floating)) and float(value).is_integer()
    ):
        text = str(int(value))
        if re.fullmatch(r"\d{8}", text):
            return pd.to_datetime(text, format="%Y%m%d", errors="coerce")
        if 1 <= int(value) <= 100_000:
            return pd.Timestamp("1899-12-30") + pd.to_timedelta(int(value), unit="D")
    text = str(value).strip()
    if re.fullmatch(r"\d{8}", text):
        return pd.to_datetime(text, format="%Y%m%d", errors="coerce")
    return pd.to_datetime(text, errors="coerce")


def parse_gestational_age(value: Any) -> tuple[float, float, float]:
    if value is None or pd.isna(value):
        return np.nan, np.nan, np.nan
    match = re.fullmatch(r"\s*(\d+)\s*[wW周]\s*(?:\+\s*(\d+)\s*)?", str(value))
    if not match:
        return np.nan, np.nan, np.nan
    weeks = int(match.group(1))
    days = int(match.group(2) or 0)
    if days > 6:
        return np.nan, np.nan, np.nan
    total_days = weeks * 7 + days
    return float(weeks), float(days), float(total_days)


def parse_lower_bound_count(value: Any) -> tuple[float, int]:
    if value is None or pd.isna(value):
        return np.nan, 0
    text = str(value).strip()
    match = re.fullmatch(r"(?:≥|>=)\s*(\d+)", text)
    if match:
        return float(match.group(1)), 1
    try:
        return float(text), 0
    except ValueError:
        return np.nan, 0


def parse_aneuploidy(value: Any) -> tuple[int, int, int, int]:
    if value is None or pd.isna(value) or not str(value).strip():
        return 0, 0, 0, 0
    text = str(value).upper().replace(" ", "")
    t13 = int("13" in text)
    t18 = int("18" in text)
    t21 = int("21" in text)
    return t13, t18, t21, int(bool(t13 or t18 or t21))


def _stringify_raw(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return pd.Timestamp(value).strftime("%Y-%m-%d")
    text = str(value).strip()
    return text or None


def load_sheet(source: Path, sheet_name: str) -> pd.DataFrame:
    spec = SHEETS[sheet_name]
    raw = pd.read_excel(source, sheet_name=sheet_name, header=None, dtype=object)
    expected_shape = (spec["expected_rows"] + 1, len(COLUMNS))
    if raw.shape != expected_shape:
        raise ValueError(f"{sheet_name} shape {raw.shape}, expected {expected_shape}")
    for index, expected in HEADER_CHECKS.items():
        actual = str(raw.iat[0, index]).strip()
        if actual != expected:
            raise ValueError(
                f"{sheet_name} header mismatch at column {index + 1}: {actual!r} != {expected!r}"
            )

    frame = raw.iloc[1:].copy()
    frame.columns = COLUMNS
    frame.insert(0, "source_sheet", sheet_name)
    frame.insert(1, "fetal_sex", spec["sex"])
    frame.insert(2, "source_row", np.arange(2, len(frame) + 2))
    for column in ["lmp_date_raw", "test_date_raw", "gestational_age_raw", "gravidity_raw"]:
        frame[column] = frame[column].map(_stringify_raw)
    for column in NUMERIC_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.reset_index(drop=True)


def enrich_records(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    data["lmp_date"] = data["lmp_date_raw"].map(parse_date)
    data["test_date"] = data["test_date_raw"].map(parse_date)

    gest = data["gestational_age_raw"].map(parse_gestational_age)
    data[["gestational_weeks_int", "gestational_extra_days", "gestational_days"]] = pd.DataFrame(
        gest.tolist(), index=data.index
    )
    data["gestational_weeks"] = data["gestational_days"] / 7.0

    gravidity = data["gravidity_raw"].map(parse_lower_bound_count)
    data[["gravidity_lower_bound", "gravidity_right_censored"]] = pd.DataFrame(
        gravidity.tolist(), index=data.index
    )
    data["parity"] = pd.to_numeric(data["parity_raw"], errors="coerce")

    labels = data["aneuploidy_raw"].map(parse_aneuploidy)
    data[["target_t13", "target_t18", "target_t21", "target_any_aneuploidy"]] = pd.DataFrame(
        labels.tolist(), index=data.index
    ).astype("int8")

    height_m = data["height_cm"] / 100.0
    data["bmi_recalculated"] = data["weight_kg"] / height_m.pow(2)
    data["bmi_difference"] = data["bmi_reported"] - data["bmi_recalculated"]
    data["bmi_analysis"] = data["bmi_reported"].fillna(data["bmi_recalculated"])
    data["calendar_gestational_days"] = (data["test_date"] - data["lmp_date"]).dt.days
    data["gestational_day_difference"] = (
        data["calendar_gestational_days"] - data["gestational_days"]
    )

    data["y_qualified"] = pd.Series(pd.NA, index=data.index, dtype="Int8")
    male_with_y = data["fetal_sex"].eq("male") & data["y_fraction"].notna()
    data.loc[male_with_y, "y_qualified"] = (
        data.loc[male_with_y, "y_fraction"] >= 0.04
    ).astype("int8")

    invalid_ratio = pd.Series(False, index=data.index)
    for column in RATIO_COLUMNS:
        invalid_ratio |= data[column].notna() & ~data[column].between(0, 1)
    data["qc_ratio_invalid"] = invalid_ratio
    data["qc_gc_outside_40_60"] = data["gc_ratio"].notna() & ~data["gc_ratio"].between(0.40, 0.60)
    data["qc_read_count_conflict"] = (
        data["raw_reads"].notna()
        & data["unique_mapped_reads"].notna()
        & (data["unique_mapped_reads"] > data["raw_reads"])
    )
    data["qc_bmi_reported_missing"] = data["bmi_reported"].isna()
    data["qc_bmi_mismatch"] = data["bmi_difference"].abs() > 0.05
    data["qc_gestational_age_invalid"] = data["gestational_days"].isna()
    data["qc_outside_test_window"] = data["gestational_days"].notna() & ~data[
        "gestational_days"
    ].between(70, 175)
    data["qc_test_date_unparsed"] = data["test_date"].isna()
    data["qc_lmp_date_missing"] = data["lmp_date_raw"].isna()
    data["qc_lmp_date_unparsed"] = data["lmp_date_raw"].notna() & data["lmp_date"].isna()
    data["qc_date_gestation_mismatch"] = (
        data["gestational_day_difference"].notna()
        & (data["gestational_day_difference"].abs() > 7)
    )
    data["qc_missing_core_measurement"] = np.where(
        data["fetal_sex"].eq("male"),
        data[["y_fraction", "zy", "gc_ratio", "raw_reads"]].isna().any(axis=1),
        data[["zx", "gc_ratio", "raw_reads"]].isna().any(axis=1),
    )

    duplicate_subset = [column for column in COLUMNS if column != "sample_id"]
    data["qc_exact_duplicate_except_sample_id"] = data.duplicated(
        subset=["fetal_sex", *duplicate_subset], keep=False
    )

    event_keys = [
        "fetal_sex",
        "patient_id",
        "draw_number",
        "test_date",
        "gestational_days",
    ]
    data["sampling_event_index"] = data.groupby(event_keys, dropna=False).ngroup() + 1
    data["sampling_event_id"] = data["fetal_sex"].str[0].str.upper() + data[
        "sampling_event_index"
    ].map(lambda value: f"E{int(value):04d}")
    data["technical_replicate_count"] = data.groupby(event_keys, dropna=False)[
        "sample_id"
    ].transform("size")
    data["technical_replicate_index"] = data.groupby(event_keys, dropna=False).cumcount() + 1

    qc_columns = [column for column in data.columns if column.startswith("qc_")]
    data["qc_any_flag"] = data[qc_columns].any(axis=1)
    return data


def build_male_events(records: pd.DataFrame) -> pd.DataFrame:
    male = records.loc[records["fetal_sex"].eq("male")].copy()
    group_columns = [
        "sampling_event_id",
        "patient_id",
        "draw_number",
        "test_date",
        "gestational_days",
        "gestational_weeks",
    ]
    events = (
        male.groupby(group_columns, dropna=False, as_index=False)
        .agg(
            record_count=("sample_id", "size"),
            age_years=("age_years", "first"),
            height_cm=("height_cm", "first"),
            weight_kg=("weight_kg", "median"),
            bmi_reported=("bmi_reported", "median"),
            y_fraction_median=("y_fraction", "median"),
            y_fraction_mean=("y_fraction", "mean"),
            y_fraction_min=("y_fraction", "min"),
            y_fraction_max=("y_fraction", "max"),
            y_fraction_std=("y_fraction", "std"),
            qc_any_flag=("qc_any_flag", "max"),
        )
        .sort_values(["patient_id", "gestational_days", "test_date", "draw_number"])
        .reset_index(drop=True)
    )
    events["y_qualified_median"] = events["y_fraction_median"] >= 0.04
    events["y_qualified_any"] = events["y_fraction_max"] >= 0.04
    events["y_qualified_all"] = events["y_fraction_min"] >= 0.04
    events["technical_replicate_discordant"] = (
        events["y_qualified_any"] != events["y_qualified_all"]
    )
    return events


def classify_patient_event(group: pd.DataFrame) -> dict[str, Any]:
    ordered = group.sort_values(["gestational_days", "test_date", "draw_number"])
    qualified = ordered["y_qualified_median"].fillna(False).to_numpy(dtype=bool)
    weeks = ordered["gestational_weeks"].to_numpy(dtype=float)
    passing = np.flatnonzero(qualified)
    if len(passing) == 0:
        censoring_type = "right"
        lower_week = float(np.nanmax(weeks)) if np.isfinite(weeks).any() else np.nan
        upper_week = np.nan
        first_pass_week = np.nan
        nonmonotonic = False
    else:
        first = int(passing[0])
        first_pass_week = float(weeks[first])
        if first == 0:
            censoring_type = "left"
            lower_week = np.nan
        else:
            censoring_type = "interval"
            prior_fail_weeks = weeks[:first][~qualified[:first]]
            lower_week = float(np.nanmax(prior_fail_weeks))
        upper_week = first_pass_week
        nonmonotonic = bool((~qualified[first + 1 :]).any())

    return {
        "patient_id": ordered["patient_id"].iloc[0],
        "n_records": int(ordered["record_count"].sum()),
        "n_sampling_events": int(len(ordered)),
        "age_years": float(ordered["age_years"].iloc[0]),
        "height_cm": float(ordered["height_cm"].iloc[0]),
        "bmi_first": float(ordered["bmi_reported"].iloc[0]),
        "bmi_median": float(ordered["bmi_reported"].median()),
        "bmi_min": float(ordered["bmi_reported"].min()),
        "bmi_max": float(ordered["bmi_reported"].max()),
        "first_observed_pass_week": first_pass_week,
        "event_lower_week": lower_week,
        "event_upper_week": upper_week,
        "censoring_type": censoring_type,
        "nonmonotonic_after_first_pass": nonmonotonic,
        "technical_replicate_discordance": bool(
            ordered["technical_replicate_discordant"].any()
        ),
    }


def build_male_patient_events(events: pd.DataFrame) -> pd.DataFrame:
    rows = [classify_patient_event(group) for _, group in events.groupby("patient_id")]
    return pd.DataFrame(rows).sort_values("patient_id").reset_index(drop=True)


def build_quality_summary(records: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    checks: list[dict[str, Any]] = []

    def add(check: str, mask: pd.Series, action: str) -> None:
        affected = int(mask.fillna(False).sum())
        checks.append(
            {
                "check": check,
                "scope_records": int(len(mask)),
                "affected_records": affected,
                "affected_pct": round(100 * affected / len(mask), 4) if len(mask) else 0.0,
                "action": action,
            }
        )

    for column in [
        "qc_ratio_invalid",
        "qc_gc_outside_40_60",
        "qc_read_count_conflict",
        "qc_bmi_reported_missing",
        "qc_bmi_mismatch",
        "qc_gestational_age_invalid",
        "qc_outside_test_window",
        "qc_test_date_unparsed",
        "qc_lmp_date_missing",
        "qc_lmp_date_unparsed",
        "qc_date_gestation_mismatch",
        "qc_missing_core_measurement",
        "qc_exact_duplicate_except_sample_id",
    ]:
        add(column, records[column], "flagged; retained in canonical records")
    add(
        "technical_replicate_records",
        records["technical_replicate_count"] > 1,
        "retained; linked by sampling_event_id",
    )
    add(
        "technical_replicate_discordant_events",
        events["technical_replicate_discordant"],
        "retained for measurement-error analysis",
    )
    return pd.DataFrame(checks)


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")
    temporary.replace(path)


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def run(source: Path, outputs: OutputPaths) -> dict[str, Any]:
    if not source.is_file():
        raise FileNotFoundError(source)
    frames = [load_sheet(source, sheet_name) for sheet_name in SHEETS]
    records = enrich_records(pd.concat(frames, ignore_index=True))
    male_records = records.loc[records["fetal_sex"].eq("male")].copy()
    female_records = records.loc[records["fetal_sex"].eq("female")].copy()
    male_events = build_male_events(records)
    male_patient_events = build_male_patient_events(male_events)
    quality = build_quality_summary(records, male_events)

    missingness = (
        records.groupby("fetal_sex")
        .agg({column: lambda values: int(values.isna().sum()) for column in COLUMNS})
        .transpose()
        .reset_index(names="field")
    )
    missingness["note"] = ""
    missingness.loc[missingness["field"].isin(["zy", "y_fraction"]), "note"] = (
        "female values are structurally absent"
    )

    counts = pd.DataFrame(
        [
            ["raw male records", 1082, len(male_records), "retained; quality flags added"],
            ["raw female records", 605, len(female_records), "retained; quality flags added"],
            ["all records", 1687, len(records), "no automatic row deletion"],
            ["male sampling events", len(male_records), len(male_events), "technical replicates linked"],
            ["male patient event histories", 267, len(male_patient_events), "one row per patient"],
        ],
        columns=["stage", "before", "after", "note"],
    )

    _atomic_csv(records, outputs.processed_dir / "nipt_records.csv")
    _atomic_csv(male_records, outputs.processed_dir / "male_records.csv")
    _atomic_csv(female_records, outputs.processed_dir / "female_records.csv")
    _atomic_csv(male_events, outputs.processed_dir / "male_sampling_events.csv")
    _atomic_csv(male_patient_events, outputs.processed_dir / "male_patient_events.csv")
    _atomic_csv(quality, outputs.report_dir / "quality_flags_summary.csv")
    _atomic_csv(missingness, outputs.report_dir / "field_missingness.csv")
    _atomic_csv(counts, outputs.report_dir / "preprocessing_counts.csv")

    summary = {
        "input": str(source),
        "sheets": {
            "male": {"records": len(male_records), "patients": int(male_records["patient_id"].nunique())},
            "female": {"records": len(female_records), "patients": int(female_records["patient_id"].nunique())},
        },
        "date_range": {
            "test_min": records["test_date"].min().strftime("%Y-%m-%d"),
            "test_max": records["test_date"].max().strftime("%Y-%m-%d"),
        },
        "gestational_week_range": [
            round(float(records["gestational_weeks"].min()), 4),
            round(float(records["gestational_weeks"].max()), 4),
        ],
        "male_sampling_events": len(male_events),
        "male_event_histories": len(male_patient_events),
        "censoring_counts": male_patient_events["censoring_type"].value_counts().to_dict(),
        "nonmonotonic_patients": int(male_patient_events["nonmonotonic_after_first_pass"].sum()),
        "female_target_counts": {
            "T13": int(female_records["target_t13"].sum()),
            "T18": int(female_records["target_t18"].sum()),
            "T21": int(female_records["target_t21"].sum()),
            "any": int(female_records["target_any_aneuploidy"].sum()),
        },
        "female_healthy_birth_unique": sorted(
            female_records["healthy_birth_raw"].dropna().astype(str).unique().tolist()
        ),
        "quality_flags": {
            row["check"]: int(row["affected_records"]) for _, row in quality.iterrows()
        },
    }
    _atomic_json(summary, outputs.report_dir / "preprocessing_summary.json")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="附件.xlsx", type=Path)
    parser.add_argument("--processed-dir", default=Path("data/processed"), type=Path)
    parser.add_argument("--report-dir", default=Path("results/preprocessing"), type=Path)
    args = parser.parse_args()
    summary = run(args.input, OutputPaths(args.processed_dir, args.report_dir))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
