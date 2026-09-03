"""Question 2: BMI-only NIPT timing with interval censoring and optimal partitioning."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from patsy import build_design_matrices, dmatrix
from scipy.optimize import minimize


COLORS = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "vermillion": "#D55E00",
    "sky": "#56B4E9",
    "magenta": "#CC79A7",
    "grey": "#7A8588",
    "light_grey": "#D7DCE0",
    "ink": "#252525",
}
def configure_plot_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Noto Sans CJK SC",
                "Source Han Sans SC",
                "Microsoft YaHei",
                "SimHei",
                "SimSun",
                "DejaVu Sans",
            ],
            "axes.unicode_minus": False,
            "font.size": 9.5,
            "axes.labelsize": 10.0,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.5,
            "axes.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.constrained_layout.use": True,
            "savefig.dpi": 300,
        }
    )


@dataclass
class AFTFit:
    distribution: str
    params: np.ndarray
    design_info: object
    log_likelihood: float
    aic: float
    converged: bool


def _design(bmi: np.ndarray, design_info: object | None = None) -> tuple[np.ndarray, object]:
    frame = pd.DataFrame({"bmi": np.asarray(bmi, dtype=float)})
    if design_info is None:
        matrix = dmatrix(
            "cr(bmi, df=3, constraints='center')",
            frame,
            return_type="dataframe",
        )
        return matrix.to_numpy(), matrix.design_info
    return np.asarray(build_design_matrices([design_info], frame)[0]), design_info


def _cdf_from_scale(time: np.ndarray, scale: np.ndarray, shape: float, distribution: str) -> np.ndarray:
    time = np.maximum(np.asarray(time, dtype=float), 1e-8)
    ratio = np.maximum(time / scale, 1e-12)
    if distribution == "weibull":
        return 1 - np.exp(-np.power(ratio, shape))
    if distribution == "loglogistic":
        return 1 / (1 + np.power(ratio, -shape))
    raise ValueError(f"Unsupported distribution: {distribution}")


def fit_interval_aft(intervals: pd.DataFrame, distribution: str) -> AFTFit:
    required = {"bmi_first", "event_lower_week", "event_upper_week", "censoring_type"}
    missing = required.difference(intervals.columns)
    if missing:
        raise ValueError(f"Missing interval columns: {sorted(missing)}")
    data = intervals.dropna(subset=["bmi_first", "censoring_type"]).copy()
    x, design_info = _design(data["bmi_first"].to_numpy())
    lower = data["event_lower_week"].to_numpy(float)
    upper = data["event_upper_week"].to_numpy(float)
    censoring = data["censoring_type"].to_numpy(str)

    def objective(params: np.ndarray) -> float:
        beta = params[:-1]
        shape = math.exp(float(np.clip(params[-1], -3, 3)))
        scale = np.exp(np.clip(x @ beta, math.log(4), math.log(60)))
        probability = np.empty(len(data), dtype=float)
        left = censoring == "left"
        middle = censoring == "interval"
        right = censoring == "right"
        probability[left] = _cdf_from_scale(upper[left], scale[left], shape, distribution)
        probability[middle] = (
            _cdf_from_scale(upper[middle], scale[middle], shape, distribution)
            - _cdf_from_scale(lower[middle], scale[middle], shape, distribution)
        )
        probability[right] = 1 - _cdf_from_scale(lower[right], scale[right], shape, distribution)
        return float(-np.log(np.clip(probability, 1e-12, 1)).sum())

    initial_intercept = math.log(float(np.nanmedian(data["event_upper_week"])))
    best = None
    for shape_start in (0.0, math.log(2.0), math.log(4.0)):
        start = np.zeros(x.shape[1] + 1)
        start[0] = initial_intercept
        start[-1] = shape_start
        candidate = minimize(objective, start, method="L-BFGS-B", options={"maxiter": 3000})
        if best is None or candidate.fun < best.fun:
            best = candidate
    assert best is not None
    log_likelihood = -float(best.fun)
    return AFTFit(
        distribution=distribution,
        params=np.asarray(best.x),
        design_info=design_info,
        log_likelihood=log_likelihood,
        aic=2 * len(best.x) - 2 * log_likelihood,
        converged=bool(best.success),
    )


def aft_cdf(fit: AFTFit, bmi: np.ndarray, times: np.ndarray) -> np.ndarray:
    x, _ = _design(np.asarray(bmi), fit.design_info)
    scale = np.exp(np.clip(x @ fit.params[:-1], math.log(4), math.log(60)))[:, None]
    shape = math.exp(float(np.clip(fit.params[-1], -3, 3)))
    time_matrix = np.broadcast_to(np.asarray(times, dtype=float)[None, :], (len(x), len(times)))
    return _cdf_from_scale(time_matrix, scale, shape, fit.distribution)


def clinical_delay_risk(week: np.ndarray) -> np.ndarray:
    """Normalized risk score following the three stages stated in the problem."""
    week = np.asarray(week, dtype=float)
    early = 0.05 + 0.05 * np.clip((week - 10) / 2, 0, 1)
    middle = 0.35 + 0.35 * np.clip((week - 13) / 14, 0, 1)
    late = 0.85 + 0.15 * np.clip((week - 28) / 12, 0, 1)
    return np.where(week <= 12, early, np.where(week <= 27, middle, late))


def individual_risk_matrix(
    fit: AFTFit,
    bmi: np.ndarray,
    decision_times: np.ndarray,
    failure_cost: float = 0.5,
) -> np.ndarray:
    event_grid = np.arange(6.0, 42.0 + 1 / 7, 1 / 7)
    cdf = aft_cdf(fit, bmi, event_grid)
    masses = np.diff(np.column_stack([np.zeros(len(bmi)), cdf]), axis=1)
    tail = 1 - cdf[:, -1]
    masses[:, -1] += tail
    survival_at_decision = 1 - aft_cdf(fit, bmi, decision_times)
    risks = np.empty((len(bmi), len(decision_times)))
    for index, time in enumerate(decision_times):
        eventual_detection = np.maximum(time, event_grid)
        delay_component = masses @ clinical_delay_risk(eventual_detection)
        risks[:, index] = delay_component + failure_cost * survival_at_decision[:, index]
    return risks


def optimal_partition(
    bmi: np.ndarray,
    risk_matrix: np.ndarray,
    decision_times: np.ndarray,
    groups: int,
    min_group_size: int = 30,
) -> dict[str, object] | None:
    order = np.argsort(bmi, kind="mergesort")
    bmi_sorted = np.asarray(bmi)[order]
    risk_sorted = np.asarray(risk_matrix)[order]
    n = len(bmi_sorted)
    endpoints = [0] + [i for i in range(1, n) if bmi_sorted[i - 1] < bmi_sorted[i]] + [n]
    endpoint_set = set(endpoints)
    prefix = np.vstack([np.zeros(risk_sorted.shape[1]), np.cumsum(risk_sorted, axis=0)])

    interval_cost: dict[tuple[int, int], tuple[float, int]] = {}
    for start in endpoints[:-1]:
        for end in endpoints[1:]:
            if end - start < min_group_size:
                continue
            totals = prefix[end] - prefix[start]
            time_index = int(np.argmin(totals))
            interval_cost[(start, end)] = (float(totals[time_index]), time_index)

    dp = np.full((groups + 1, n + 1), np.inf)
    previous = np.full((groups + 1, n + 1), -1, dtype=int)
    dp[0, 0] = 0
    for group_count in range(1, groups + 1):
        for end in endpoints:
            if end < group_count * min_group_size:
                continue
            for start in endpoints:
                if start >= end or start not in endpoint_set or not np.isfinite(dp[group_count - 1, start]):
                    continue
                item = interval_cost.get((start, end))
                if item is None:
                    continue
                value = dp[group_count - 1, start] + item[0]
                if value < dp[group_count, end]:
                    dp[group_count, end] = value
                    previous[group_count, end] = start
    if not np.isfinite(dp[groups, n]):
        return None

    segments = []
    end = n
    for group_count in range(groups, 0, -1):
        start = int(previous[group_count, end])
        cost, time_index = interval_cost[(start, end)]
        segments.append((start, end, cost, time_index))
        end = start
    segments.reverse()

    output_groups = []
    for group_index, (start, end, cost, time_index) in enumerate(segments, start=1):
        lower = float(bmi_sorted[start])
        upper = float(bmi_sorted[end - 1])
        boundary_after = None
        if end < n:
            boundary_after = float((bmi_sorted[end - 1] + bmi_sorted[end]) / 2)
        output_groups.append(
            {
                "group": group_index,
                "start_index": start,
                "end_index": end,
                "n": end - start,
                "bmi_min": lower,
                "bmi_max": upper,
                "boundary_after": boundary_after,
                "optimal_week": float(decision_times[time_index]),
                "total_risk": cost,
                "mean_risk": cost / (end - start),
            }
        )
    return {
        "groups": output_groups,
        "total_risk": float(dp[groups, n]),
        "mean_risk": float(dp[groups, n] / n),
    }


def choose_group_count(partitions: dict[int, dict[str, object]], tolerance: float = 0.01) -> int:
    feasible = sorted(partitions)
    candidates = [value for value in feasible if value >= 2]
    if not candidates:
        return feasible[0]
    for groups in candidates:
        next_groups = groups + 1
        if next_groups not in partitions:
            return groups
        current = float(partitions[groups]["mean_risk"])
        following = float(partitions[next_groups]["mean_risk"])
        relative_gain = (current - following) / current
        if relative_gain < tolerance:
            return groups
    return candidates[-1]


def rebuild_intervals(events: pd.DataFrame, threshold: float) -> pd.DataFrame:
    required = {"patient_id", "gestational_weeks", "bmi_reported", "y_fraction_median"}
    missing = required.difference(events.columns)
    if missing:
        raise ValueError(f"Missing sampling-event columns: {sorted(missing)}")

    sort_columns = [
        column
        for column in ("patient_id", "gestational_days", "test_date", "draw_number")
        if column in events.columns
    ]
    if "gestational_days" not in sort_columns:
        sort_columns.insert(1, "gestational_weeks")

    rows = []
    ordered_events = events.sort_values(sort_columns, kind="stable")
    for patient_id, history in ordered_events.groupby("patient_id", sort=False):
        passed = history["y_fraction_median"].fillna(-np.inf).to_numpy(float) >= threshold
        weeks = history["gestational_weeks"].to_numpy(float)
        first_bmi = float(history["bmi_reported"].iloc[0])
        pass_indices = np.flatnonzero(passed)
        if len(pass_indices) == 0:
            censoring = "right"
            lower = float(np.nanmax(weeks))
            upper = np.nan
        elif pass_indices[0] == 0:
            censoring, lower, upper = "left", np.nan, float(weeks[0])
        else:
            index = int(pass_indices[0])
            prior_fail_weeks = weeks[:index][~passed[:index]]
            censoring = "interval"
            lower = float(np.nanmax(prior_fail_weeks))
            upper = float(weeks[index])
        rows.append(
            {
                "patient_id": patient_id,
                "bmi_first": first_bmi,
                "event_lower_week": lower,
                "event_upper_week": upper,
                "censoring_type": censoring,
            }
        )
    return pd.DataFrame(rows).sort_values("patient_id").reset_index(drop=True)


def estimate_measurement_sd(events: pd.DataFrame) -> float:
    repeat_sd = events.loc[events["record_count"] > 1, "y_fraction_std"].dropna().to_numpy(float)
    if len(repeat_sd) == 0:
        raise ValueError("No technical replicates available for measurement-error estimation")
    return float(np.sqrt(np.mean(np.square(repeat_sd))))


def turnbull_discrete(intervals: pd.DataFrame, grid: np.ndarray, max_iter: int = 5000) -> np.ndarray:
    allowed = np.zeros((len(intervals), len(grid)), dtype=bool)
    for row_index, row in enumerate(intervals.itertuples(index=False)):
        if row.censoring_type == "left":
            allowed[row_index] = grid <= row.event_upper_week
        elif row.censoring_type == "interval":
            allowed[row_index] = (grid > row.event_lower_week) & (grid <= row.event_upper_week)
        else:
            allowed[row_index] = grid > row.event_lower_week
    if np.any(allowed.sum(axis=1) == 0):
        raise ValueError("Turnbull support grid does not cover all censoring intervals")
    probability = np.full(len(grid), 1 / len(grid))
    for _ in range(max_iter):
        denominators = np.clip(allowed @ probability, 1e-15, None)
        updated = ((allowed * probability) / denominators[:, None]).mean(axis=0)
        updated /= updated.sum()
        if np.max(np.abs(updated - probability)) < 1e-10:
            probability = updated
            break
        probability = updated
    return probability


def turnbull_group_timing(
    intervals: pd.DataFrame,
    groups: list[dict[str, object]],
    decision_times: np.ndarray,
    failure_cost: float,
) -> pd.DataFrame:
    grid = np.arange(6.0, 42.0 + 1 / 7, 1 / 7)
    rows = []
    for group in groups:
        mask = intervals["bmi_first"].between(group["bmi_min"], group["bmi_max"], inclusive="both")
        subset = intervals.loc[mask]
        probability = turnbull_discrete(subset, grid)
        risks = []
        for time in decision_times:
            delay = float(probability @ clinical_delay_risk(np.maximum(time, grid)))
            failure = float(probability[grid > time].sum())
            risks.append(delay + failure_cost * failure)
        best = int(np.argmin(risks))
        rows.append(
            {
                "group": group["group"],
                "n": len(subset),
                "aft_week": group["optimal_week"],
                "turnbull_week": float(decision_times[best]),
                "turnbull_risk": float(risks[best]),
            }
        )
    return pd.DataFrame(rows)


def threshold_sensitivity(
    events: pd.DataFrame,
    thresholds: list[float],
    distribution: str,
    group_count: int,
    decision_times: np.ndarray,
    failure_cost: float,
    min_group_size: int,
) -> pd.DataFrame:
    rows = []
    for threshold in thresholds:
        intervals = rebuild_intervals(events, threshold)
        fit = fit_interval_aft(intervals, distribution)
        risks = individual_risk_matrix(fit, intervals["bmi_first"].to_numpy(), decision_times, failure_cost)
        partition = optimal_partition(
            intervals["bmi_first"].to_numpy(), risks, decision_times, group_count, min_group_size
        )
        if partition is None:
            continue
        for group in partition["groups"]:
            rows.append(
                {
                    "threshold": threshold,
                    "group": group["group"],
                    "boundary_after": group["boundary_after"],
                    "optimal_week": group["optimal_week"],
                    "mean_risk": partition["mean_risk"],
                }
            )
    return pd.DataFrame(rows)


def failure_cost_sensitivity(
    intervals: pd.DataFrame,
    fit: AFTFit,
    costs: list[float],
    group_count: int,
    decision_times: np.ndarray,
    min_group_size: int,
) -> pd.DataFrame:
    rows = []
    bmi = intervals["bmi_first"].to_numpy()
    for failure_cost in costs:
        risks = individual_risk_matrix(fit, bmi, decision_times, failure_cost)
        partition = optimal_partition(
            bmi, risks, decision_times, group_count, min_group_size
        )
        if partition is None:
            continue
        for group in partition["groups"]:
            rows.append(
                {
                    "failure_cost": failure_cost,
                    "group": group["group"],
                    "boundary_after": group["boundary_after"],
                    "optimal_week": group["optimal_week"],
                    "mean_risk": partition["mean_risk"],
                }
            )
    return pd.DataFrame(rows)


def measurement_error_simulation(
    events: pd.DataFrame,
    measurement_sd: float,
    distribution: str,
    group_count: int,
    decision_times: np.ndarray,
    failure_cost: float,
    min_group_size: int,
    simulations: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for simulation in range(simulations):
        perturbed = events.copy()
        perturbed["y_fraction_median"] = np.clip(
            perturbed["y_fraction_median"].to_numpy()
            + rng.normal(0, measurement_sd, len(perturbed)),
            1e-5,
            1 - 1e-5,
        )
        intervals = rebuild_intervals(perturbed, 0.04)
        fit = fit_interval_aft(intervals, distribution)
        risks = individual_risk_matrix(fit, intervals["bmi_first"].to_numpy(), decision_times, failure_cost)
        partition = optimal_partition(
            intervals["bmi_first"].to_numpy(), risks, decision_times, group_count, min_group_size
        )
        if partition is None:
            continue
        for group in partition["groups"]:
            rows.append(
                {
                    "simulation": simulation,
                    "group": group["group"],
                    "boundary_after": group["boundary_after"],
                    "optimal_week": group["optimal_week"],
                    "mean_risk": partition["mean_risk"],
                }
            )
    return pd.DataFrame(rows)


def create_figures(
    intervals: pd.DataFrame,
    events: pd.DataFrame,
    fit: AFTFit,
    partitions: dict[int, dict[str, object]],
    selected: dict[str, object],
    turnbull: pd.DataFrame,
    threshold_results: pd.DataFrame,
    simulation_results: pd.DataFrame,
    decision_times: np.ndarray,
    risk_matrix: np.ndarray,
    figures_dir: Path,
) -> None:
    configure_plot_style()
    figures_dir.mkdir(parents=True, exist_ok=True)

    censor_colors = {"left": COLORS["blue"], "interval": COLORS["orange"], "right": COLORS["vermillion"]}
    fig, ax = plt.subplots(figsize=(5.9, 3.8))
    categories = ["left", "interval", "right"]
    data = [intervals.loc[intervals.censoring_type == value, "bmi_first"] for value in categories]
    violin = ax.violinplot(data, positions=np.arange(3), showmedians=True, widths=0.72)
    for body, category in zip(violin["bodies"], categories):
        body.set_facecolor(censor_colors[category]); body.set_edgecolor("white"); body.set_alpha(0.72)
    ax.set_xticks(range(3), ["左删失", "区间删失", "右删失"])
    ax.set(xlabel="删失类型", ylabel="首次检测 BMI（kg/m²）")
    fig.savefig(figures_dir / "raw_q2_bmi_censor_violin.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.9, 3.8))
    sample = intervals.sort_values("bmi_first").reset_index(drop=True)
    for row_index, row in sample.iterrows():
        color = censor_colors[row.censoring_type]
        if row.censoring_type == "left":
            ax.plot([9, row.event_upper_week], [row_index, row_index], color=color, alpha=0.45)
        elif row.censoring_type == "interval":
            ax.plot([row.event_lower_week, row.event_upper_week], [row_index, row_index], color=color, alpha=0.75)
        else:
            ax.plot([row.event_lower_week, 30], [row_index, row_index], color=color, alpha=0.75)
    ax.set(xlabel="达标时间删失区间（周）", ylabel="按 BMI 排序的孕妇")
    ax.set_yticks([])
    fig.savefig(figures_dir / "raw_q2_censor_intervals.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.9, 3.8))
    repeat_sd = 100 * events.loc[events.record_count > 1, "y_fraction_std"].dropna()
    ax.hist(repeat_sd, bins=max(5, min(10, len(repeat_sd))), color=COLORS["sky"], edgecolor="white")
    ax.axvline(repeat_sd.mean(), color=COLORS["vermillion"], linestyle="--")
    ax.set(xlabel="技术重复事件内标准差（百分点）", ylabel="事件数")
    fig.savefig(figures_dir / "raw_q2_measurement_error.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.9, 3.8))
    bmi_levels = [28, 32, 36, 40]
    curve_colors = [COLORS["blue"], COLORS["orange"], COLORS["green"], COLORS["vermillion"]]
    for bmi, color in zip(bmi_levels, curve_colors):
        survival = 1 - aft_cdf(fit, np.array([bmi]), decision_times)[0]
        ax.plot(decision_times, survival, color=color, label=f"BMI={bmi}")
    ax.set(xlabel="孕周（周）", ylabel="尚未达标概率")
    ax.legend(frameon=False, ncol=2)
    fig.savefig(figures_dir / "process_q2_aft_survival.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.9, 3.8))
    group_counts = sorted(partitions)
    mean_risks = [partitions[count]["mean_risk"] for count in group_counts]
    ax.plot(group_counts, mean_risks, marker="o", color=COLORS["blue"], markerfacecolor="white")
    ax.set(xlabel="BMI 分组数", ylabel="人均最小风险")
    fig.savefig(figures_dir / "process_q2_group_tradeoff.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.9, 4.1))
    order = np.argsort(intervals["bmi_first"].to_numpy())
    display = risk_matrix[order]
    image = ax.imshow(
        display,
        aspect="auto",
        origin="lower",
        extent=[decision_times[0], decision_times[-1], 0, len(order)],
    )
    ax.set(xlabel="候选检测孕周（周）", ylabel="按 BMI 排序的孕妇")
    ax.set_yticks([])
    fig.colorbar(image, ax=ax, label="个体预期风险")
    fig.savefig(figures_dir / "process_q2_risk_heatmap.png", bbox_inches="tight")
    plt.close(fig)

    groups = selected["groups"]
    fig, ax = plt.subplots(figsize=(5.9, 3.8))
    for group, color in zip(groups, curve_colors):
        ax.hlines(group["optimal_week"], group["bmi_min"], group["bmi_max"], color=color, linewidth=5)
        ax.scatter((group["bmi_min"] + group["bmi_max"]) / 2, group["optimal_week"], color=color, s=40, zorder=3)
    ax.set(xlabel="BMI（kg/m²）", ylabel="最佳 NIPT 时点（周）")
    fig.savefig(figures_dir / "result_q2_groups_timing.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.9, 3.8))
    positions = np.arange(1, len(groups) + 1)
    ax.plot(positions, [item["optimal_week"] for item in groups], "o-", color=COLORS["blue"], label="AFT")
    ax.plot(positions, turnbull["turnbull_week"], "s--", color=COLORS["orange"], label="Turnbull")
    ax.set_xticks(positions, [f"组{value}" for value in positions])
    ax.set(xlabel="BMI 组", ylabel="最佳 NIPT 时点（周）")
    ax.legend(frameon=False)
    fig.savefig(figures_dir / "result_q2_turnbull_comparison.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.9, 3.8))
    if not simulation_results.empty:
        timing_data = [simulation_results.loc[simulation_results.group == group, "optimal_week"] for group in positions]
        box = ax.boxplot(timing_data, positions=positions, patch_artist=True, widths=0.55)
        for patch, color in zip(box["boxes"], curve_colors):
            patch.set_facecolor(color); patch.set_alpha(0.55)
    for threshold, marker in zip(sorted(threshold_results.threshold.unique()), ["v", "o", "^"]):
        subset = threshold_results[threshold_results.threshold == threshold]
        ax.scatter(subset.group, subset.optimal_week, marker=marker, s=36, label=f"阈值={100*threshold:.1f}%")
    ax.set_xticks(positions, [f"组{value}" for value in positions])
    ax.set(xlabel="BMI 组", ylabel="最佳 NIPT 时点（周）")
    ax.legend(frameon=False, ncol=3)
    fig.savefig(figures_dir / "result_q2_error_stability.png", bbox_inches="tight")
    plt.close(fig)


def run(
    patient_events_path: Path,
    sampling_events_path: Path,
    results_dir: Path,
    figures_dir: Path,
    simulations: int = 50,
    seed: int = 2025,
) -> dict[str, object]:
    intervals = pd.read_csv(patient_events_path)
    events = pd.read_csv(sampling_events_path)
    decision_times = np.arange(10.0, 25.0 + 1 / 7, 1 / 7)
    failure_cost = 2.0
    min_group_size = 30

    fits = [fit_interval_aft(intervals, name) for name in ("weibull", "loglogistic")]
    fit = min(fits, key=lambda item: item.aic)
    risks = individual_risk_matrix(
        fit, intervals["bmi_first"].to_numpy(), decision_times, failure_cost
    )
    partitions = {}
    for group_count in range(1, 7):
        partition = optimal_partition(
            intervals["bmi_first"].to_numpy(), risks, decision_times, group_count, min_group_size
        )
        if partition is not None:
            partitions[group_count] = partition
    selected_count = choose_group_count(partitions)
    selected = partitions[selected_count]
    turnbull = turnbull_group_timing(
        intervals, selected["groups"], decision_times, failure_cost
    )
    measurement_sd = estimate_measurement_sd(events)
    threshold_results = threshold_sensitivity(
        events,
        [0.038, 0.04, 0.042],
        fit.distribution,
        selected_count,
        decision_times,
        failure_cost,
        min_group_size,
    )
    failure_cost_results = failure_cost_sensitivity(
        intervals,
        fit,
        [1.0, 2.0, 3.0],
        selected_count,
        decision_times,
        min_group_size,
    )
    simulation_results = measurement_error_simulation(
        events,
        measurement_sd,
        fit.distribution,
        selected_count,
        decision_times,
        failure_cost,
        min_group_size,
        simulations,
        seed,
    )

    results_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [{"distribution": item.distribution, "log_likelihood": item.log_likelihood, "aic": item.aic, "converged": item.converged} for item in fits]
    ).to_csv(results_dir / "q2_aft_comparison.csv", index=False)
    pd.DataFrame(
        [{"group_count": count, "mean_risk": value["mean_risk"]} for count, value in partitions.items()]
    ).to_csv(results_dir / "q2_group_count_tradeoff.csv", index=False)
    pd.DataFrame(selected["groups"]).to_csv(results_dir / "q2_optimal_groups.csv", index=False)
    turnbull.to_csv(results_dir / "q2_turnbull_comparison.csv", index=False)
    threshold_results.to_csv(results_dir / "q2_threshold_sensitivity.csv", index=False)
    failure_cost_results.to_csv(results_dir / "q2_failure_cost_sensitivity.csv", index=False)
    simulation_results.to_csv(results_dir / "q2_measurement_error_simulation.csv", index=False)

    create_figures(
        intervals,
        events,
        fit,
        partitions,
        selected,
        turnbull,
        threshold_results,
        simulation_results,
        decision_times,
        risks,
        figures_dir,
    )

    summary = {
        "patients": int(len(intervals)),
        "censoring_counts": intervals["censoring_type"].value_counts().to_dict(),
        "selected_distribution": fit.distribution,
        "aft_models": {item.distribution: {"aic": item.aic, "converged": item.converged} for item in fits},
        "failure_cost": failure_cost,
        "minimum_group_size": min_group_size,
        "selected_group_count": selected_count,
        "mean_risk": selected["mean_risk"],
        "groups": selected["groups"],
        "measurement_sd": measurement_sd,
        "simulations_requested": simulations,
        "simulations_completed": int(simulation_results["simulation"].nunique()) if not simulation_results.empty else 0,
    }
    (results_dir / "q2_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--patient-events", type=Path, default=Path("data/processed/male_patient_events.csv"))
    parser.add_argument("--sampling-events", type=Path, default=Path("data/processed/male_sampling_events.csv"))
    parser.add_argument("--results-dir", type=Path, default=Path("results/q2"))
    parser.add_argument("--figures-dir", type=Path, default=Path("figures"))
    parser.add_argument("--simulations", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2025)
    args = parser.parse_args()
    summary = run(
        args.patient_events,
        args.sampling_events,
        args.results_dir,
        args.figures_dir,
        simulations=args.simulations,
        seed=args.seed,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
