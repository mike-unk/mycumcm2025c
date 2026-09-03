"""Question 2: BMI segmentation and risk-adjusted NIPT timing."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import optimize, stats


COLORS = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "red": "#D55E00",
    "sky": "#56B4E9",
    "purple": "#CC79A7",
    "grey": "#7A8588",
    "light": "#D9DEE2",
    "ink": "#252525",
}
TAU = 0.70
MIN_GROUP_SIZE = 45
FAILURE_COST = 6.0
WEEK_GRID = np.arange(11.0, 25.0 + 1e-9, 1.0 / 7.0)


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Noto Sans CJK SC",
                "Source Han Sans SC",
                "Microsoft YaHei",
                "SimHei",
                "DejaVu Sans",
            ],
            "axes.unicode_minus": False,
            "font.size": 9.5,
            "axes.labelsize": 10,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.5,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.4,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.dpi": 300,
        }
    )


def logit(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, 1e-6, 1 - 1e-6)
    return np.log(values / (1 - values))


def individual_trends(events: pd.DataFrame) -> pd.DataFrame:
    """Estimate shrinkage slopes on the logit-Y scale for every patient."""
    rows: list[dict[str, float | str | int]] = []
    pooled = stats.theilslopes(
        logit(events["y_fraction_median"].to_numpy()),
        events["gestational_weeks"].to_numpy(),
    ).slope
    for patient_id, group in events.groupby("patient_id", sort=False):
        group = group.sort_values("gestational_weeks")
        n = len(group)
        if n >= 2 and group["gestational_weeks"].nunique() >= 2:
            raw_slope = float(
                stats.theilslopes(
                    logit(group["y_fraction_median"].to_numpy()),
                    group["gestational_weeks"].to_numpy(),
                ).slope
            )
        else:
            raw_slope = float(pooled)
        weight = max(0, n - 1) / (max(0, n - 1) + 2.0)
        slope = weight * raw_slope + (1 - weight) * pooled
        rows.append(
            {
                "patient_id": patient_id,
                "n_events": n,
                "bmi": float(group["bmi_reported"].median()),
                "raw_slope": raw_slope,
                "trend_slope": slope,
            }
        )
    return pd.DataFrame(rows).sort_values(["bmi", "patient_id"]).reset_index(drop=True)


def segment_cost(prefix: np.ndarray, prefix_sq: np.ndarray, i: int, j: int) -> float:
    n = j - i
    total = prefix[j] - prefix[i]
    total_sq = prefix_sq[j] - prefix_sq[i]
    return max(0.0, float(total_sq - total * total / n))


def valid_cut_positions(sorted_trends: pd.DataFrame) -> set[int]:
    bmi = sorted_trends["bmi"].to_numpy()
    return {i for i in range(1, len(bmi)) if bmi[i - 1] < bmi[i]}


def dynamic_partition(
    trends: pd.DataFrame, k: int, min_group_size: int = MIN_GROUP_SIZE
) -> tuple[float, list[int]]:
    """Optimal ordered BMI partition for within-group trend SSE."""
    data = trends.sort_values(["bmi", "patient_id"]).reset_index(drop=True)
    values = data["trend_slope"].to_numpy()
    n = len(values)
    prefix = np.r_[0.0, np.cumsum(values)]
    prefix_sq = np.r_[0.0, np.cumsum(values * values)]
    valid = valid_cut_positions(data)
    dp = np.full((k + 1, n + 1), np.inf)
    parent = np.full((k + 1, n + 1), -1, dtype=int)
    dp[0, 0] = 0.0
    for groups in range(1, k + 1):
        low_j = groups * min_group_size
        for j in range(low_j, n + 1):
            if j < n and j not in valid:
                continue
            low_i = (groups - 1) * min_group_size
            high_i = j - min_group_size
            for i in range(low_i, high_i + 1):
                if groups > 1 and i not in valid:
                    continue
                candidate = dp[groups - 1, i] + segment_cost(prefix, prefix_sq, i, j)
                if candidate < dp[groups, j]:
                    dp[groups, j] = candidate
                    parent[groups, j] = i
    if not np.isfinite(dp[k, n]):
        raise ValueError(f"No feasible {k}-group partition")
    cuts = [n]
    j = n
    for groups in range(k, 0, -1):
        j = int(parent[groups, j])
        cuts.append(j)
    return float(dp[k, n]), sorted(cuts)


def select_partition(trends: pd.DataFrame) -> tuple[int, list[int], pd.DataFrame]:
    records = []
    solutions: dict[int, list[int]] = {}
    n = len(trends)
    for k in range(2, 6):
        cost, cuts = dynamic_partition(trends, k)
        bic = n * math.log(max(cost / n, 1e-12)) + 2 * k * math.log(n)
        records.append({"groups": k, "heterogeneity": cost, "bic": bic})
        solutions[k] = cuts
    comparison = pd.DataFrame(records)
    eligible = comparison[comparison["groups"].isin([3, 4])]
    chosen_k = int(eligible.loc[eligible["bic"].idxmin(), "groups"])
    return chosen_k, solutions[chosen_k], comparison


def assign_groups(trends: pd.DataFrame, cuts: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = trends.sort_values(["bmi", "patient_id"]).reset_index(drop=True).copy()
    data["group"] = 0
    rows = []
    for group_id, (start, stop) in enumerate(zip(cuts[:-1], cuts[1:]), start=1):
        data.loc[start : stop - 1, "group"] = group_id
        lo = float(data.loc[start:stop - 1, "bmi"].min())
        hi = float(data.loc[start:stop - 1, "bmi"].max())
        next_lo = float(data.loc[stop, "bmi"]) if stop < len(data) else np.inf
        boundary = (hi + next_lo) / 2 if np.isfinite(next_lo) else np.inf
        rows.append(
            {
                "group": group_id,
                "start_index": start,
                "stop_index": stop,
                "n": stop - start,
                "observed_bmi_min": lo,
                "observed_bmi_max": hi,
                "upper_boundary": boundary,
                "trend_mean": float(data.loc[start:stop - 1, "trend_slope"].mean()),
                "trend_sd": float(data.loc[start:stop - 1, "trend_slope"].std(ddof=1)),
            }
        )
    return data, pd.DataFrame(rows)


def _interval_nll(params: np.ndarray, frame: pd.DataFrame) -> float:
    alpha, beta, log_sigma = params
    sigma = np.exp(log_sigma)
    centered_bmi = frame["bmi"].to_numpy() - frame["bmi"].median()
    location = alpha + beta * centered_bmi
    lower = frame["event_lower_week"].to_numpy(dtype=float)
    upper = frame["event_upper_week"].to_numpy(dtype=float)
    left = np.isnan(lower)
    right = np.isnan(upper)
    interval = ~(left | right)
    likelihood = np.empty(len(frame), dtype=float)
    if left.any():
        zu = (np.log(upper[left]) - location[left]) / sigma
        likelihood[left] = stats.norm.cdf(zu)
    if right.any():
        zl = (np.log(lower[right]) - location[right]) / sigma
        likelihood[right] = stats.norm.sf(zl)
    if interval.any():
        zl = (np.log(lower[interval]) - location[interval]) / sigma
        zu = (np.log(upper[interval]) - location[interval]) / sigma
        likelihood[interval] = stats.norm.cdf(zu) - stats.norm.cdf(zl)
    return float(-np.log(np.clip(likelihood, 1e-14, 1)).sum())


def fit_censored_quantile(frame: pd.DataFrame) -> dict[str, float | bool]:
    finite = np.r_[
        frame["event_lower_week"].dropna().to_numpy(),
        frame["event_upper_week"].dropna().to_numpy(),
    ]
    initial = np.array([np.log(np.median(finite)), 0.0, np.log(0.20)])
    result = optimize.minimize(
        _interval_nll,
        initial,
        args=(frame,),
        method="L-BFGS-B",
        bounds=[(np.log(8), np.log(35)), (-0.12, 0.12), (np.log(0.03), np.log(0.8))],
        options={"maxiter": 2000, "ftol": 1e-12},
    )
    alpha, beta, log_sigma = result.x
    sigma = float(np.exp(log_sigma))
    # Left censoring makes quantiles below the first observed visit weakly
    # identified; report the earliest supported week as the operational floor.
    q70 = float(max(11.0, np.exp(alpha + sigma * stats.norm.ppf(TAU))))
    return {
        "alpha": float(alpha),
        "beta": float(beta),
        "sigma": sigma,
        "q70_week": q70,
        "nll": float(result.fun),
        "converged": bool(result.success),
    }


def survival_probability(week: np.ndarray, bmi: np.ndarray, model: dict[str, float], bmi_ref: float) -> np.ndarray:
    location = model["alpha"] + model["beta"] * (bmi - bmi_ref)
    z = (np.log(week) - location) / model["sigma"]
    return stats.norm.sf(z)


def pregnancy_penalty(week: np.ndarray) -> np.ndarray:
    """Continuous piecewise-linear loss with clinical slopes 1:3:6."""
    week = np.asarray(week, dtype=float)
    # Early (<=12), middle (13--27), and late (>27) stages have
    # incremental slopes 1:3:6.  The early term penalizes moving earlier
    # than 12 weeks, while the other terms penalize delay.
    early = np.clip(12.0 - week, 0, None)
    middle = np.clip(week - 12.0, 0, 15.0)
    late = np.clip(week - 27.0, 0, None)
    return early + 3.0 * middle + 6.0 * late


def choose_timing(frame: pd.DataFrame, model: dict[str, float]) -> tuple[float, pd.DataFrame]:
    bmi = frame["bmi"].to_numpy()
    bmi_ref = float(frame["bmi"].median())
    rows = []
    for week in WEEK_GRID:
        failure = float(survival_probability(np.full_like(bmi, week), bmi, model, bmi_ref).mean())
        delay = float(pregnancy_penalty(np.array([week]))[0])
        total = FAILURE_COST * failure + delay
        rows.append(
            {
                "week": week,
                "failure_probability": failure,
                "pregnancy_penalty": delay,
                "total_risk": total,
            }
        )
    curve = pd.DataFrame(rows)
    optimum = float(curve.loc[curve["total_risk"].idxmin(), "week"])
    return optimum, curve


def estimate_measurement_error(records: pd.DataFrame) -> tuple[float, int]:
    differences = []
    for _, group in records.groupby("sampling_event_id"):
        if len(group) >= 2:
            values = logit(group["y_fraction"].to_numpy())
            differences.extend(values - values.mean())
    if len(differences) < 2:
        raise ValueError("Insufficient technical replicates for measurement error")
    return float(np.std(differences, ddof=1)), len(differences)


def build_patient_intervals(events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for patient_id, group in events.groupby("patient_id"):
        group = group.sort_values("gestational_weeks")
        passed = group["y_fraction_median"].to_numpy() >= 0.04
        weeks = group["gestational_weeks"].to_numpy()
        if passed.any():
            first = int(np.flatnonzero(passed)[0])
            lower = np.nan if first == 0 else float(weeks[first - 1])
            upper = float(weeks[first])
            censoring = "left" if first == 0 else "interval"
        else:
            lower, upper, censoring = float(weeks[-1]), np.nan, "right"
        rows.append(
            {
                "patient_id": patient_id,
                "bmi": float(group["bmi_reported"].median()),
                "event_lower_week": lower,
                "event_upper_week": upper,
                "censoring_type": censoring,
            }
        )
    return pd.DataFrame(rows)


def perturb_events(events: pd.DataFrame, sigma_m: float, rng: np.random.Generator) -> pd.DataFrame:
    perturbed = events.copy()
    z = logit(perturbed["y_fraction_median"].to_numpy())
    perturbed["y_fraction_median"] = 1 / (1 + np.exp(-(z + rng.normal(0, sigma_m, len(z)))))
    return perturbed


def analyse_once(events: pd.DataFrame) -> dict[str, object]:
    trends = individual_trends(events)
    chosen_k, cuts, comparison = select_partition(trends)
    assigned, groups = assign_groups(trends, cuts)
    intervals = build_patient_intervals(events).merge(
        assigned[["patient_id", "group"]], on="patient_id", how="left", validate="one_to_one"
    )
    model_rows = []
    risk_curves = []
    for group_id, frame in intervals.groupby("group"):
        model = fit_censored_quantile(frame)
        optimum, curve = choose_timing(frame, model)
        model_rows.append(
            {
                "group": int(group_id),
                "bmi_reference": float(frame["bmi"].median()),
                "baseline_week_q70": model["q70_week"],
                "optimal_week": optimum,
                **model,
            }
        )
        curve["group"] = int(group_id)
        risk_curves.append(curve)
    return {
        "trends": assigned,
        "groups": groups,
        "comparison": comparison,
        "intervals": intervals,
        "models": pd.DataFrame(model_rows),
        "risk_curves": pd.concat(risk_curves, ignore_index=True),
        "chosen_k": chosen_k,
    }


def measurement_error_analysis(
    events: pd.DataFrame, sigma_m: float, baseline: dict[str, object], repetitions: int, seed: int
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    baseline_groups = baseline["trends"][["patient_id", "group"]].rename(columns={"group": "baseline_group"})
    rows = []
    for repetition in range(repetitions):
        result = analyse_once(perturb_events(events, sigma_m, rng))
        merged = result["trends"][["patient_id", "group"]].merge(baseline_groups, on="patient_id")
        agreement = float((merged["group"] == merged["baseline_group"]).mean())
        for row in result["groups"].itertuples(index=False):
            timing = result["models"].loc[result["models"]["group"] == row.group].iloc[0]
            rows.append(
                {
                    "repetition": repetition,
                    "chosen_k": result["chosen_k"],
                    "group": row.group,
                    "upper_boundary": row.upper_boundary,
                    "baseline_week_q70": timing["baseline_week_q70"],
                    "optimal_week": timing["optimal_week"],
                    "group_agreement": agreement,
                }
            )
    return pd.DataFrame(rows)


def save_figure(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def create_figures(result: dict[str, object], events: pd.DataFrame, error: pd.DataFrame, figure_dir: Path) -> None:
    configure_style()
    figure_dir.mkdir(parents=True, exist_ok=True)
    trends = result["trends"]
    groups = result["groups"]
    intervals = result["intervals"]
    models = result["models"]
    risks = result["risk_curves"]
    palette = [COLORS["blue"], COLORS["orange"], COLORS["green"], COLORS["purple"]]

    fig, ax = plt.subplots(figsize=(6.2, 3.7))
    ax.scatter(trends["bmi"], trends["trend_slope"], s=15, alpha=0.45, color=COLORS["grey"], edgecolors="none")
    for group_id, frame in trends.groupby("group"):
        order = np.argsort(frame["bmi"].to_numpy())
        x = frame["bmi"].to_numpy()[order]
        y = pd.Series(frame["trend_slope"].to_numpy()[order]).rolling(21, center=True, min_periods=7).median()
        ax.plot(x, y, color=palette[group_id - 1], label=f"组 {group_id}")
    ax.set(xlabel="BMI", ylabel="Logit(Y浓度)增长斜率")
    ax.legend(frameon=False, ncol=len(groups))
    save_figure(fig, figure_dir / "raw_q2_bmi_trend_scatter.png")

    fig, ax = plt.subplots(figsize=(6.2, 3.7))
    censor_colors = {"left": COLORS["blue"], "interval": COLORS["orange"], "right": COLORS["red"]}
    for kind, frame in intervals.groupby("censoring_type"):
        y = frame["event_upper_week"].fillna(frame["event_lower_week"])
        ax.scatter(frame["bmi"], y, s=20, alpha=0.65, color=censor_colors[kind], label={"left":"左删失", "interval":"区间删失", "right":"右删失"}[kind])
        both = frame[frame["event_lower_week"].notna() & frame["event_upper_week"].notna()]
        ax.vlines(both["bmi"], both["event_lower_week"], both["event_upper_week"], color=censor_colors[kind], alpha=0.25, lw=0.7)
    ax.set(xlabel="BMI", ylabel="首次达标孕周观测边界")
    ax.legend(frameon=False, ncol=3)
    save_figure(fig, figure_dir / "raw_q2_censoring_bmi.png")

    fig, ax = plt.subplots(figsize=(6.2, 3.7))
    margins = 100 * (events["y_fraction_median"] - 0.04)
    ax.hist(margins, bins=45, color=COLORS["sky"], edgecolor="white", linewidth=0.4)
    ax.axvline(0, color=COLORS["red"], linestyle="--", label="4% 阈值")
    ax.set(xlabel="Y染色体浓度相对4%阈值的差值（百分点）", ylabel="采样事件数")
    ax.legend(frameon=False)
    save_figure(fig, figure_dir / "raw_q2_threshold_margin.png")

    comparison = result["comparison"]
    fig, ax1 = plt.subplots(figsize=(5.9, 3.7))
    ax1.plot(comparison["groups"], comparison["heterogeneity"], marker="o", color=COLORS["blue"], label="组内异质性")
    ax1.set(xlabel="分组数量 K", ylabel="D(K)")
    ax2 = ax1.twinx()
    ax2.plot(comparison["groups"], comparison["bic"], marker="s", linestyle="--", color=COLORS["orange"], label="BIC")
    ax2.set_ylabel("BIC")
    lines = ax1.lines + ax2.lines
    ax1.legend(lines, [line.get_label() for line in lines], frameon=False)
    save_figure(fig, figure_dir / "process_q2_dp_elbow.png")

    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    for row in groups.itertuples(index=False):
        frame = trends[trends["group"] == row.group]
        ax.boxplot(
            frame["trend_slope"], positions=[row.group], widths=0.48, patch_artist=True,
            boxprops={"facecolor": palette[row.group - 1], "alpha": 0.55}, medianprops={"color": COLORS["ink"]},
            whiskerprops={"color": COLORS["grey"]}, capprops={"color": COLORS["grey"]}, flierprops={"markersize": 2},
        )
        jitter = np.linspace(-0.18, 0.18, len(frame))
        ax.scatter(row.group + jitter, np.sort(frame["trend_slope"]), s=7, alpha=0.3, color=palette[row.group - 1])
    ax.set(xlabel="BMI 分组", ylabel="收缩后增长斜率", xticks=range(1, len(groups) + 1))
    save_figure(fig, figure_dir / "process_q2_group_heterogeneity.png")

    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    for row in models.itertuples(index=False):
        frame = intervals[intervals["group"] == row.group]
        bmi_grid = np.linspace(frame["bmi"].min(), frame["bmi"].max(), 80)
        q = np.exp(row.alpha + row.beta * (bmi_grid - row.bmi_reference) + row.sigma * stats.norm.ppf(TAU))
        ax.plot(bmi_grid, q, color=palette[row.group - 1], label=f"组 {row.group}")
        ax.scatter([row.bmi_reference], [row.baseline_week_q70], marker="D", s=28, color=palette[row.group - 1])
    ax.set(xlabel="BMI", ylabel="70% 达标分位孕周")
    ax.legend(frameon=False, ncol=len(groups))
    save_figure(fig, figure_dir / "process_q2_quantile_fits.png")

    fig, ax = plt.subplots(figsize=(6.2, 3.7))
    x = np.arange(1, len(groups) + 1)
    ax.bar(x - 0.18, models["baseline_week_q70"], width=0.36, color=COLORS["sky"], label="70%达标基准")
    ax.bar(x + 0.18, models["optimal_week"], width=0.36, color=COLORS["orange"], label="风险校正时点")
    ax.set(xlabel="BMI 分组", ylabel="孕周", xticks=x, ylim=(9.5, max(20, models["baseline_week_q70"].max() + 1)))
    ax.legend(frameon=False)
    save_figure(fig, figure_dir / "result_q2_group_schedule.png")

    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    for group_id, frame in risks.groupby("group"):
        ax.plot(frame["week"], frame["total_risk"], color=palette[group_id - 1], label=f"组 {group_id}")
        optimum = models.loc[models["group"] == group_id, "optimal_week"].iloc[0]
        point = frame.iloc[(frame["week"] - optimum).abs().argmin()]
        ax.scatter(point["week"], point["total_risk"], s=30, color=palette[group_id - 1], zorder=3)
    ax.axvline(12, color=COLORS["grey"], linestyle="--", linewidth=0.9)
    ax.set(xlabel="候选检测孕周", ylabel="综合潜在风险")
    ax.legend(frameon=False, ncol=len(groups))
    save_figure(fig, figure_dir / "result_q2_risk_curves.png")

    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    summary = error.groupby("group")["optimal_week"].agg(["median", lambda x: x.quantile(0.025), lambda x: x.quantile(0.975)]).reset_index()
    summary.columns = ["group", "median", "low", "high"]
    ax.errorbar(summary["group"], summary["median"], yerr=[summary["median"] - summary["low"], summary["high"] - summary["median"]], fmt="o", capsize=4, color=COLORS["blue"])
    ax.scatter(models["group"], models["optimal_week"], marker="D", s=40, color=COLORS["orange"], label="原始数据方案")
    ax.set(xlabel="BMI 分组", ylabel="误差扰动后的推荐孕周", xticks=range(1, len(groups) + 1))
    ax.legend(frameon=False)
    save_figure(fig, figure_dir / "result_q2_error_stability.png")


def format_group_intervals(groups: pd.DataFrame) -> pd.DataFrame:
    output = groups.copy()
    lower = -np.inf
    labels = []
    for row in output.itertuples(index=False):
        upper = row.upper_boundary
        if np.isneginf(lower):
            labels.append(f"(-inf, {upper:.2f})")
        elif np.isposinf(upper):
            labels.append(f"[{lower:.2f}, +inf)")
        else:
            labels.append(f"[{lower:.2f}, {upper:.2f})")
        lower = upper
    output["bmi_interval"] = labels
    return output


def run(project_root: Path, repetitions: int = 100, seed: int = 202509) -> dict[str, object]:
    events = pd.read_csv(project_root / "data/processed/male_sampling_events.csv", encoding="utf-8-sig")
    records = pd.read_csv(project_root / "data/processed/male_records.csv", encoding="utf-8-sig")
    baseline = analyse_once(events)
    sigma_m, replicate_residuals = estimate_measurement_error(records)
    error = measurement_error_analysis(events, sigma_m, baseline, repetitions, seed)

    output_dir = project_root / "results/q2"
    output_dir.mkdir(parents=True, exist_ok=True)
    groups = format_group_intervals(baseline["groups"])
    schedule = groups.merge(baseline["models"], on="group", validate="one_to_one")
    groups.to_csv(output_dir / "q2_bmi_groups.csv", index=False)
    baseline["trends"].to_csv(output_dir / "q2_patient_trends.csv", index=False)
    baseline["comparison"].to_csv(output_dir / "q2_partition_comparison.csv", index=False)
    baseline["intervals"].to_csv(output_dir / "q2_censoring_intervals.csv", index=False)
    baseline["models"].to_csv(output_dir / "q2_quantile_models.csv", index=False)
    baseline["risk_curves"].to_csv(output_dir / "q2_risk_curves.csv", index=False)
    schedule.to_csv(output_dir / "q2_group_schedule.csv", index=False)
    error.to_csv(output_dir / "q2_measurement_error_simulation.csv", index=False)

    finite_boundaries = error[np.isfinite(error["upper_boundary"])].groupby("group")["upper_boundary"]
    boundary_summary = finite_boundaries.agg(
        median="median", low=lambda x: x.quantile(0.025), high=lambda x: x.quantile(0.975)
    ).reset_index()
    timing_summary = error.groupby("group")["optimal_week"].agg(
        median="median", low=lambda x: x.quantile(0.025), high=lambda x: x.quantile(0.975)
    ).reset_index()
    boundary_summary.to_csv(output_dir / "q2_boundary_error_summary.csv", index=False)
    timing_summary.to_csv(output_dir / "q2_timing_error_summary.csv", index=False)

    create_figures(baseline, events, error, project_root / "figures")
    summary = {
        "patients": int(events["patient_id"].nunique()),
        "chosen_groups": int(baseline["chosen_k"]),
        "minimum_group_size": MIN_GROUP_SIZE,
        "quantile": TAU,
        "failure_cost": FAILURE_COST,
        "pregnancy_risk_slopes": [1, 3, 6],
        "measurement_error_logit_sd": sigma_m,
        "technical_replicate_residuals": replicate_residuals,
        "simulation_repetitions": repetitions,
        "mean_group_agreement": float(error.groupby("repetition")["group_agreement"].first().mean()),
        "schedule": schedule.replace({np.inf: None, -np.inf: None}).to_dict(orient="records"),
    }
    (output_dir / "q2_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--seed", type=int, default=202509)
    args = parser.parse_args()
    summary = run(args.project_root.resolve(), args.repetitions, args.seed)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
