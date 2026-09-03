"""Question 1: longitudinal association model for male fetal Y fraction."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.genmod.cov_struct import Exchangeable
from statsmodels.genmod.families import Gaussian


CORE_COLUMNS = [
    "patient_id",
    "sampling_event_id",
    "y_fraction",
    "gestational_weeks",
    "bmi_analysis",
    "age_years",
    "conception_method",
    "gc_ratio",
    "mapping_ratio",
    "duplicate_ratio",
    "filtered_ratio",
    "raw_reads",
]

CONTROLS = [
    "age_years",
    "gc_ratio",
    "mapping_ratio",
    "duplicate_ratio",
    "filtered_ratio",
    "log_raw_reads",
]

CONTROL_TERMS = " + ".join([f"{name}_z" for name in CONTROLS] + ["assisted_conception"])
WEEK_SPLINE = "cr(gestational_weeks, df=4, constraints='center')"
BMI_SPLINE = "cr(bmi_analysis, df=3, constraints='center')"
BASE_FORMULA = f"logit_y ~ {CONTROL_TERMS}"
WEEK_FORMULA = BASE_FORMULA + f" + {WEEK_SPLINE}"
ADDITIVE_FORMULA = WEEK_FORMULA + f" + {BMI_SPLINE}"
FULL_FORMULA = ADDITIVE_FORMULA + f" + {WEEK_SPLINE}:bmi_analysis_z"


def aggregate_sampling_events(records: pd.DataFrame) -> pd.DataFrame:
    """Collapse technical replicates while retaining biological repeat visits."""
    missing = sorted(set(CORE_COLUMNS).difference(records.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    data = records[CORE_COLUMNS].copy()
    numeric = [column for column in CORE_COLUMNS if column not in {
        "patient_id", "sampling_event_id", "conception_method"
    }]
    aggregation = {column: "mean" for column in numeric}
    aggregation["y_fraction"] = "median"
    aggregation.update({"patient_id": "first", "conception_method": "first"})
    events = (
        data.groupby("sampling_event_id", as_index=False, sort=True)
        .agg(aggregation)
        .dropna(subset=numeric + ["patient_id"])
    )
    if not events["y_fraction"].between(0, 1, inclusive="neither").all():
        raise ValueError("y_fraction must lie strictly between 0 and 1")

    events["logit_y"] = np.log(events["y_fraction"] / (1 - events["y_fraction"]))
    events["log_raw_reads"] = np.log(events["raw_reads"])
    events["assisted_conception"] = (
        events["conception_method"].fillna("").ne("自然受孕").astype(int)
    )
    for column in ["bmi_analysis", *CONTROLS]:
        mean = events[column].mean()
        std = events[column].std(ddof=0)
        if not np.isfinite(std) or std == 0:
            events[f"{column}_z"] = 0.0
        else:
            events[f"{column}_z"] = (events[column] - mean) / std
    return events


def fit_mixed_models(events: pd.DataFrame) -> dict[str, object]:
    """Fit nested random-intercept models by maximum likelihood."""
    fitted: dict[str, object] = {}
    for name, formula in {
        "base": BASE_FORMULA,
        "week": WEEK_FORMULA,
        "additive": ADDITIVE_FORMULA,
        "full": FULL_FORMULA,
    }.items():
        model = smf.mixedlm(formula, events, groups=events["patient_id"])
        fitted[name] = model.fit(reml=False, method="lbfgs", maxiter=2000, disp=False)
    return fitted


def likelihood_ratio(reduced: object, full: object) -> dict[str, float]:
    statistic = max(0.0, 2 * (full.llf - reduced.llf))
    df = int(full.df_modelwc - reduced.df_modelwc)
    return {
        "chi2": float(statistic),
        "df": df,
        "p_value": float(stats.chi2.sf(statistic, df)),
    }


def fit_gee(events: pd.DataFrame) -> object:
    model = smf.gee(
        ADDITIVE_FORMULA,
        groups="patient_id",
        data=events,
        cov_struct=Exchangeable(),
        family=Gaussian(),
    )
    return model.fit(maxiter=200)


def _fixed_effect_frame(result: object) -> pd.DataFrame:
    conf = result.conf_int().loc[result.fe_params.index]
    frame = pd.DataFrame(
        {
            "term": result.fe_params.index,
            "estimate": result.fe_params.values,
            "std_error": result.bse_fe.values,
            "ci_low": conf.iloc[:, 0].values,
            "ci_high": conf.iloc[:, 1].values,
            "p_value": result.pvalues.loc[result.fe_params.index].values,
        }
    )
    return frame


def _gee_frame(result: object) -> pd.DataFrame:
    conf = result.conf_int()
    return pd.DataFrame(
        {
            "term": result.params.index,
            "estimate": result.params.values,
            "std_error": result.bse.values,
            "ci_low": conf.iloc[:, 0].values,
            "ci_high": conf.iloc[:, 1].values,
            "p_value": result.pvalues.values,
        }
    )


def _inverse_logit(value: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-value))


def prediction_grid(events: pd.DataFrame, result: object) -> pd.DataFrame:
    weeks = np.linspace(events["gestational_weeks"].quantile(0.01), events["gestational_weeks"].quantile(0.99), 120)
    bmi_levels = [25.0, 30.0, 35.0, 40.0]
    reference = {column: float(events[column].mean()) for column in CONTROLS}
    rows = []
    for bmi in bmi_levels:
        for week in weeks:
            row = {
                "gestational_weeks": week,
                "bmi_analysis": bmi,
                "bmi_analysis_z": (bmi - events["bmi_analysis"].mean()) / events["bmi_analysis"].std(ddof=0),
                "assisted_conception": 0,
            }
            for column in CONTROLS:
                row[column] = reference[column]
                row[f"{column}_z"] = 0.0
            rows.append(row)
    grid = pd.DataFrame(rows)
    design = result.model.data.orig_exog.design_info
    from patsy import build_design_matrices

    matrix = np.asarray(build_design_matrices([design], grid)[0])
    beta = result.fe_params.to_numpy()
    covariance = result.cov_params().loc[result.fe_params.index, result.fe_params.index].to_numpy()
    eta = matrix @ beta
    se = np.sqrt(np.einsum("ij,jk,ik->i", matrix, covariance, matrix))
    grid["predicted_y_fraction"] = _inverse_logit(eta)
    grid["ci_low"] = _inverse_logit(eta - 1.96 * se)
    grid["ci_high"] = _inverse_logit(eta + 1.96 * se)
    return grid


def key_prediction_table(grid: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for bmi in sorted(grid["bmi_analysis"].unique()):
        subset = grid[grid["bmi_analysis"] == bmi]
        for week in (12.0, 16.0, 20.0, 24.0):
            row = subset.loc[(subset["gestational_weeks"] - week).abs().idxmin()]
            rows.append(
                {
                    "gestational_week": week,
                    "bmi": bmi,
                    "prediction_percent": 100 * row["predicted_y_fraction"],
                    "ci_low_percent": 100 * row["ci_low"],
                    "ci_high_percent": 100 * row["ci_high"],
                }
            )
    return pd.DataFrame(rows)


def _setup_plot_style(project_root: Path) -> list[str]:
    skill_scripts = project_root / ".agents/skills/cumcm-step-review/scripts"
    sys.path.insert(0, str(skill_scripts))
    from plot_style import JOURNAL_PALETTES, apply_publication_style

    apply_publication_style(journal="nature", lang="zh")
    return JOURNAL_PALETTES["nature"]["main"]


def create_figures(events: pd.DataFrame, grid: pd.DataFrame, result: object, output_dir: Path, project_root: Path) -> None:
    colors = _setup_plot_style(project_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5.9, 3.8))
    hb = ax.hexbin(events["gestational_weeks"], 100 * events["y_fraction"], gridsize=28, mincnt=1, cmap="viridis")
    ax.set(xlabel="孕周（周）", ylabel="Y 染色体浓度（%）")
    fig.colorbar(hb, ax=ax, label="采样事件数")
    fig.savefig(output_dir / "raw_q1_week_hexbin.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.9, 3.8))
    hb = ax.hexbin(events["bmi_analysis"], 100 * events["y_fraction"], gridsize=28, mincnt=1, cmap="viridis")
    ax.set(xlabel="BMI（kg/m²）", ylabel="Y 染色体浓度（%）")
    fig.colorbar(hb, ax=ax, label="采样事件数")
    fig.savefig(output_dir / "raw_q1_bmi_hexbin.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    corr_columns = ["y_fraction", "gestational_weeks", "bmi_analysis", "age_years", "gc_ratio", "mapping_ratio", "duplicate_ratio", "filtered_ratio"]
    labels = ["Y浓度", "孕周", "BMI", "年龄", "GC", "比对率", "重复率", "过滤率"]
    corr = events[corr_columns].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(5.9, 4.9))
    image = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7.5)
    fig.colorbar(image, ax=ax, label="Spearman 相关系数")
    fig.savefig(output_dir / "raw_q1_correlation_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.9, 3.8))
    for color, bmi in zip(colors[:4], sorted(grid["bmi_analysis"].unique())):
        subset = grid[grid["bmi_analysis"] == bmi]
        ax.plot(subset["gestational_weeks"], 100 * subset["predicted_y_fraction"], color=color, label=f"BMI={bmi:g}")
        ax.fill_between(subset["gestational_weeks"], 100 * subset["ci_low"], 100 * subset["ci_high"], color=color, alpha=0.12)
    ax.axhline(4, color="#555555", linestyle="--", linewidth=0.9)
    ax.set(xlabel="孕周（周）", ylabel="调整后 Y 染色体浓度（%）")
    ax.legend(frameon=False, ncol=2)
    fig.savefig(output_dir / "result_q1_effect_curves.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fitted = np.asarray(result.fittedvalues)
    observed = events["logit_y"].to_numpy()
    residual = observed - fitted
    fig, ax = plt.subplots(figsize=(5.9, 3.8))
    ax.scatter(_inverse_logit(fitted) * 100, residual, s=10, alpha=0.35, color=colors[0], edgecolors="none")
    ax.axhline(0, color="#555555", linestyle="--", linewidth=0.9)
    ax.set(xlabel="拟合 Y 染色体浓度（%）", ylabel="Logit 尺度残差")
    fig.savefig(output_dir / "process_q1_residuals.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.9, 3.8))
    observed_percent = 100 * events["y_fraction"].to_numpy()
    fitted_percent = 100 * _inverse_logit(fitted)
    ax.scatter(observed_percent, fitted_percent, s=10, alpha=0.35, color=colors[0], edgecolors="none")
    limits = [min(observed_percent.min(), fitted_percent.min()), max(observed_percent.max(), fitted_percent.max())]
    ax.plot(limits, limits, color="#555555", linestyle="--", linewidth=0.9)
    ax.set(xlabel="观测 Y 染色体浓度（%）", ylabel="拟合 Y 染色体浓度（%）", xlim=limits, ylim=limits)
    fig.savefig(output_dir / "process_q1_observed_fitted.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.9, 3.8))
    stats.probplot(residual, dist="norm", plot=ax)
    ax.set(xlabel="理论分位数", ylabel="残差分位数")
    ax.get_lines()[0].set(color=colors[0], markersize=3, alpha=0.5)
    ax.get_lines()[1].set(color=colors[1], linewidth=1)
    fig.savefig(output_dir / "process_q1_qq.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    terms = _fixed_effect_frame(result)
    selected = terms[~terms["term"].str.contains(r"Intercept|cr\(", regex=True)].copy()
    selected = selected.sort_values("estimate")
    fig, ax = plt.subplots(figsize=(5.9, 3.8))
    y = np.arange(len(selected))
    ax.errorbar(selected["estimate"], y, xerr=[selected["estimate"] - selected["ci_low"], selected["ci_high"] - selected["estimate"]], fmt="o", color=colors[0], ecolor=colors[2], capsize=2)
    ax.axvline(0, color="#555555", linestyle="--", linewidth=0.9)
    ax.set_yticks(y, selected["term"].str.replace("_z", "", regex=False))
    ax.set(xlabel="Logit 尺度回归系数", ylabel="校正变量")
    fig.savefig(output_dir / "result_q1_covariate_forest.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def create_significance_figure(comparisons: dict[str, dict[str, float]], output_dir: Path, project_root: Path) -> None:
    colors = _setup_plot_style(project_root)
    labels = ["孕周", "BMI", "孕周×BMI"]
    keys = ["overall_week", "bmi_nonlinearity", "week_bmi_interaction"]
    values = [-math.log10(max(comparisons[key]["p_value"], 1e-300)) for key in keys]
    fig, ax = plt.subplots(figsize=(5.9, 3.6))
    bars = ax.barh(labels, values, color=[colors[0], colors[1], colors[4]])
    ax.axvline(-math.log10(0.05), color="#555555", linestyle="--", linewidth=0.9)
    ax.set(xlabel="-log10(P)", ylabel="联合检验项")
    for bar, key in zip(bars, keys):
        p_value = comparisons[key]["p_value"]
        text = f"P={p_value:.3g}" if p_value >= 0.001 else "P<0.001"
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2, text, va="center", fontsize=8.5)
    fig.savefig(output_dir / "result_q1_joint_significance.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def run(input_path: Path, results_dir: Path, figures_dir: Path, project_root: Path) -> dict[str, object]:
    records = pd.read_csv(input_path)
    events = aggregate_sampling_events(records)
    results_dir.mkdir(parents=True, exist_ok=True)
    events.to_csv(results_dir / "q1_sampling_events.csv", index=False)

    mixed = fit_mixed_models(events)
    main = mixed["additive"]
    gee = fit_gee(events)

    comparisons = {
        "overall_week": likelihood_ratio(mixed["base"], mixed["week"]),
        "bmi_nonlinearity": likelihood_ratio(mixed["week"], mixed["additive"]),
        "week_bmi_interaction": likelihood_ratio(mixed["additive"], mixed["full"]),
    }
    pd.DataFrame(comparisons).T.rename_axis("test").reset_index().to_csv(results_dir / "q1_likelihood_ratio_tests.csv", index=False)
    model_comparison = pd.DataFrame(
        [
            {"model": name, "aic": result.aic, "bic": result.bic, "log_likelihood": result.llf}
            for name, result in mixed.items()
        ]
    )
    model_comparison.to_csv(results_dir / "q1_model_comparison.csv", index=False)
    _fixed_effect_frame(main).to_csv(results_dir / "q1_mixed_effects_coefficients.csv", index=False)
    _gee_frame(gee).to_csv(results_dir / "q1_gee_coefficients.csv", index=False)

    grid = prediction_grid(events, main)
    grid.to_csv(results_dir / "q1_effect_curves.csv", index=False)
    key_prediction_table(grid).to_csv(results_dir / "q1_key_predictions.csv", index=False)
    create_figures(events, grid, main, figures_dir, project_root)
    create_significance_figure(comparisons, figures_dir, project_root)

    fixed_prediction = np.asarray(main.model.exog) @ main.fe_params.to_numpy()
    var_fixed = float(np.var(fixed_prediction, ddof=1))
    var_random = float(main.cov_re.iloc[0, 0])
    var_residual = float(main.scale)
    summary = {
        "records_input": int(len(records)),
        "sampling_events": int(len(events)),
        "patients": int(events["patient_id"].nunique()),
        "technical_records_collapsed": int(len(records) - len(events)),
        "mixed_model_converged": bool(main.converged),
        "gee_converged": bool(gee.converged),
        "aic": float(main.aic),
        "bic": float(main.bic),
        "random_intercept_variance": var_random,
        "residual_variance": var_residual,
        "marginal_r2": var_fixed / (var_fixed + var_random + var_residual),
        "conditional_r2": (var_fixed + var_random) / (var_fixed + var_random + var_residual),
        "likelihood_ratio_tests": comparisons,
        "gee_dependence_parameter": float(np.asarray(gee.cov_struct.dep_params).reshape(-1)[0]),
    }
    (results_dir / "q1_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/processed/male_records.csv"))
    parser.add_argument("--results-dir", type=Path, default=Path("results/q1"))
    parser.add_argument("--figures-dir", type=Path, default=Path("figures"))
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[2]
    summary = run(args.input, args.results_dir, args.figures_dir, project_root)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
