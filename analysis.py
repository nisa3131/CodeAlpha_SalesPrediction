#!/usr/bin/env python3
"""Reproduce the CodeAlpha sales regression experiment and report artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# Use writable, ignored caches in restricted cloud environments.
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, PolynomialFeatures, StandardScaler

FEATURES = ["TV", "Radio", "Newspaper"]
TARGET = "Sales"
SEED = 42


def add_tv_radio_interaction(values: np.ndarray) -> np.ndarray:
    """Append an interaction without learning from held-out observations."""
    values = np.asarray(values)
    return np.column_stack([values, values[:, 0] * values[:, 1]])


def load_data(path: Path) -> tuple[pd.DataFrame, dict]:
    raw = pd.read_csv(path)
    required = FEATURES + [TARGET]
    missing_columns = sorted(set(required) - set(raw.columns))
    if missing_columns:
        raise ValueError(f"Required columns are missing: {missing_columns}")
    unnamed = [column for column in raw if str(column).startswith("Unnamed:")]
    dropped_index_columns = []
    source_ids = pd.Series(np.arange(1, len(raw) + 1), index=raw.index)
    for column in unnamed:
        values = pd.to_numeric(raw[column], errors="coerce").to_numpy()
        sequential = np.array_equal(values, np.arange(len(raw))) or np.array_equal(
            values, np.arange(1, len(raw) + 1)
        )
        if not sequential:
            raise ValueError(f"Unnamed column {column!r} is not a verified row index.")
        dropped_index_columns.append(column)
        source_ids = raw[column].copy()
    unexpected = sorted(set(raw.columns) - set(required) - set(unnamed))
    if unexpected:
        raise ValueError(f"Inspect unexpected columns before modeling: {unexpected}")
    data = raw[required].apply(pd.to_numeric, errors="raise").copy()
    if np.isinf(data.to_numpy()).any():
        raise ValueError("Infinite input values need investigation.")
    duplicate_mask = data.duplicated()
    missing_target = data[TARGET].isna()
    audit = {
        "source_file": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "source_rows": len(raw),
        "columns": list(raw.columns),
        "missing_values": {key: int(value) for key, value in data.isna().sum().items()},
        "duplicate_observations_ignoring_index": int(duplicate_mask.sum()),
        "dropped_verified_index_columns": dropped_index_columns,
        "rows_removed_for_missing_target": int(missing_target.sum()),
        "features": FEATURES,
        "target": TARGET,
        "units": "Unverified; all values are reported in the supplied dataset's observed units.",
        "upstream_source_and_license": "Not supplied or verified; file provided by the user.",
    }
    keep = ~(duplicate_mask | missing_target)
    data = data.loc[keep].copy()
    data.insert(0, "source_row_id", source_ids.loc[keep].astype(int))
    audit["modeling_rows"] = len(data)
    if len(data) < 25:
        raise ValueError("Too few rows for the fixed split and five-fold experiment.")
    return data, audit


def make_models() -> dict[str, Pipeline]:
    def pipeline(model, transform=None):
        steps = [("impute", SimpleImputer(strategy="median"))]
        if transform is not None:
            steps.append(("features", transform))
        steps.extend([("scale", StandardScaler()), ("regressor", model)])
        return Pipeline(steps)

    return {
        "Dummy mean": pipeline(DummyRegressor(strategy="mean")),
        "Linear regression": pipeline(LinearRegression()),
        "Ridge (alpha=1)": pipeline(Ridge(alpha=1.0)),
        "TV x Radio interaction": pipeline(
            LinearRegression(), FunctionTransformer(add_tv_radio_interaction)
        ),
        "Quadratic ridge (alpha=1)": pipeline(
            Ridge(alpha=1.0), PolynomialFeatures(degree=2, include_bias=False)
        ),
        "Random forest": pipeline(
            RandomForestRegressor(
                n_estimators=300, max_depth=6, min_samples_leaf=3, random_state=SEED, n_jobs=1
            )
        ),
    }


def regression_metrics(actual, predicted) -> dict:
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(root_mean_squared_error(actual, predicted)),
        "r2": float(r2_score(actual, predicted)),
    }


def save_figure(fig, directory: Path, filename: str) -> None:
    fig.savefig(directory / filename, dpi=160, bbox_inches="tight")
    plt.close(fig)


def create_figures(data, comparison, holdout, coefficients, scenarios, charts):
    sns.set_theme(style="whitegrid", context="notebook")
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, feature in zip(axes, FEATURES):
        sns.scatterplot(data=data, x=feature, y=TARGET, alpha=0.75, s=27, ax=ax)
        ax.set_xlabel(f"{feature} (observed units)")
        ax.set_ylabel("Sales (observed units)")
    fig.suptitle("Observed advertising measurements and sales; associations only")
    fig.tight_layout()
    save_figure(fig, charts, "01_feature_sales_relationships.png")

    fig, ax = plt.subplots(figsize=(9, 4.3))
    ordered = comparison.sort_values("cv_rmse_mean", ascending=False)
    ax.barh(ordered["model"], ordered["cv_rmse_mean"], xerr=ordered["cv_rmse_std"], color="#287d9b", capsize=3)
    ax.set_xlabel("Training-only 5-fold CV RMSE (observed Sales units; lower is better)")
    ax.set_title("Candidate comparison; error bars show fold standard deviation")
    save_figure(fig, charts, "02_training_cv_comparison.png")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    sns.scatterplot(data=holdout, x="actual_sales", y="predicted_sales", ax=axes[0], s=45)
    bounds = [min(holdout.actual_sales.min(), holdout.predicted_sales.min()), max(holdout.actual_sales.max(), holdout.predicted_sales.max())]
    axes[0].plot(bounds, bounds, "--", color="gray", label="Perfect prediction")
    axes[0].set(xlabel="Observed Sales", ylabel="Predicted Sales", title="Untouched test partition")
    axes[0].legend()
    sns.scatterplot(data=holdout, x="predicted_sales", y="residual_actual_minus_predicted", ax=axes[1], s=45)
    axes[1].axhline(0, linestyle="--", color="gray")
    axes[1].set(xlabel="Predicted Sales", ylabel="Observed − predicted Sales", title="Held-out residuals")
    fig.tight_layout()
    save_figure(fig, charts, "03_heldout_predictions_and_residuals.png")

    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.barh(coefficients["feature"], coefficients["coefficient_per_training_standard_deviation"], color=["#287d9b", "#d69935", "#888888"])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Sales change associated with +1 training SD of a feature\n(holding other features fixed; linear model)")
    ax.set_title("Training linear-model coefficients: associations, not causal effects")
    save_figure(fig, charts, "04_linear_association_coefficients.png")

    fig, ax = plt.subplots(figsize=(10, 4.7))
    ax.barh(scenarios["scenario"], scenarios["predicted_sales"], color="#287d9b")
    ax.axvline(float(scenarios.iloc[0]["predicted_sales"]), linestyle="--", color="#d69935", label="Reference prediction")
    ax.set_xlabel("Conditional model prediction (observed Sales units)")
    ax.set_title("Hypothetical input scenarios within training feature ranges\nSynthetic combinations; neither causal estimates nor budget recommendations")
    ax.legend()
    save_figure(fig, charts, "05_hypothetical_scenarios.png")


def run_analysis(data_path: Path, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    tables = output_dir / "tables"
    charts = output_dir / "charts"
    tables.mkdir(exist_ok=True)
    charts.mkdir(exist_ok=True)
    data, audit = load_data(data_path)
    train, test = train_test_split(data, test_size=0.2, random_state=SEED)
    assert set(train.source_row_id).isdisjoint(test.source_row_id)
    X_train, y_train = train[FEATURES], train[TARGET]
    X_test, y_test = test[FEATURES], test[TARGET]
    models = make_models()
    cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
    rows, fold_rows = [], []
    for name, estimator in models.items():
        scores = cross_validate(
            estimator, X_train, y_train, cv=cv,
            scoring={"mae": "neg_mean_absolute_error", "rmse": "neg_root_mean_squared_error", "r2": "r2"},
            n_jobs=1, error_score="raise",
        )
        row = {"model": name}
        for metric in ("mae", "rmse", "r2"):
            values = scores[f"test_{metric}"] * (-1 if metric != "r2" else 1)
            row[f"cv_{metric}_mean"] = float(np.mean(values))
            row[f"cv_{metric}_std"] = float(np.std(values, ddof=1))
        rows.append(row)
        for fold in range(5):
            fold_rows.append({"model": name, "fold": fold + 1, "mae": -float(scores["test_mae"][fold]), "rmse": -float(scores["test_rmse"][fold]), "r2": float(scores["test_r2"][fold])})
    comparison = pd.DataFrame(rows).sort_values("cv_rmse_mean").reset_index(drop=True)
    selected_name = str(comparison.iloc[0]["model"])
    selected = clone(models[selected_name]).fit(X_train, y_train)
    baseline = clone(models["Dummy mean"]).fit(X_train, y_train)
    predictions = selected.predict(X_test)
    baseline_predictions = baseline.predict(X_test)
    heldout = test.copy()
    heldout["actual_sales"] = heldout.pop(TARGET)
    heldout["predicted_sales"] = predictions
    heldout["dummy_mean_prediction"] = baseline_predictions
    heldout["residual_actual_minus_predicted"] = y_test.to_numpy() - predictions
    heldout = heldout.sort_values("source_row_id").reset_index(drop=True)
    heldout.to_csv(output_dir / "heldout_predictions.csv", index=False)
    comparison.to_csv(tables / "cross_validation_summary.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(tables / "cross_validation_folds.csv", index=False)
    data[FEATURES + [TARGET]].describe().to_csv(tables / "descriptive_statistics.csv")
    data[FEATURES + [TARGET]].corr().to_csv(tables / "pearson_correlations.csv")

    # The interpretable reference model is fitted on training data only.
    linear = clone(models["Linear regression"]).fit(X_train, y_train)
    coefficients = pd.DataFrame({
        "feature": FEATURES,
        "training_standard_deviation": linear.named_steps["scale"].scale_,
        "coefficient_per_training_standard_deviation": linear.named_steps["regressor"].coef_,
        "coefficient_per_observed_input_unit": linear.named_steps["regressor"].coef_ / linear.named_steps["scale"].scale_,
    })
    coefficients.to_csv(tables / "training_linear_associations.csv", index=False)

    # Start from a real training row near the marginal medians, then vary one input.
    center = X_train.median()
    scale = X_train.std().replace(0, 1)
    anchor_index = (((X_train - center) / scale) ** 2).sum(axis=1).idxmin()
    anchor = X_train.loc[anchor_index].copy()
    scenario_rows = [{"scenario": "Observed training reference", **anchor.to_dict()}]
    for feature in FEATURES:
        for factor in (0.8, 1.2):
            values = anchor.copy()
            values[feature] = float(np.clip(values[feature] * factor, X_train[feature].min(), X_train[feature].max()))
            scenario_rows.append({"scenario": f"{feature} {'−' if factor < 1 else '+'}20% (range clipped)", **values.to_dict()})
    scenarios = pd.DataFrame(scenario_rows)
    scenarios["predicted_sales"] = selected.predict(scenarios[FEATURES])
    scenarios["difference_from_reference_prediction"] = scenarios.predicted_sales - scenarios.iloc[0].predicted_sales
    scenarios.to_csv(tables / "hypothetical_scenarios.csv", index=False)
    create_figures(data, comparison, heldout, coefficients, scenarios, charts)

    metrics = {
        "task": "CodeAlpha Task 4 — Sales Prediction",
        "seed": SEED,
        "data_audit": audit,
        "split": {
            "method": "Random 80/20 split; no dates are available for a chronological split.",
            "training_rows": len(train), "test_rows": len(test),
            "training_source_row_ids": sorted(train.source_row_id.tolist()),
            "test_source_row_ids": sorted(test.source_row_id.tolist()),
        },
        "selection": {
            "criterion": "Lowest mean RMSE in five-fold shuffled cross-validation on training data only.",
            "folds": 5, "selected_model": selected_name,
            "hyperparameters": "Candidate definitions are fixed in make_models(); no test-set tuning.",
            "preprocessing": "Median imputation, feature construction, and scaling are fitted inside each training fold's pipeline.",
            "cross_validation": comparison.to_dict(orient="records"),
        },
        "heldout": {
            "selected_model": regression_metrics(y_test, predictions),
            "dummy_mean_baseline": regression_metrics(y_test, baseline_predictions),
        },
        "scenario_reference_source_row_id": int(train.loc[anchor_index, "source_row_id"]),
        "limitations": [
            "No dates, customer segments, or online-platform variables are supplied.",
            "This is conditional out-of-sample regression, not an actual future time-series forecast.",
            "Observational associations do not identify causal effects, assured ROI, or an optimal advertising budget.",
            "Units and upstream licensing are unverified; avoid attaching currency or sales-volume units.",
            "Only 200 observations and one 40-row test partition; evaluation uncertainty and distribution shift remain.",
            "Scenario values stay within each training feature's marginal range, but synthetic joint combinations may lack observational support.",
        ],
    }
    # Fail rather than silently emit invalid or incomplete result artifacts.
    assert np.isfinite(predictions).all()
    assert len(heldout) == len(test)
    assert all(np.isfinite(value) for group in metrics["heldout"].values() for value in group.values())
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    selected_test = metrics["heldout"]["selected_model"]
    dummy_test = metrics["heldout"]["dummy_mean_baseline"]
    coefficient_lookup = coefficients.set_index("feature")["coefficient_per_training_standard_deviation"]
    scenario_lookup = scenarios.set_index("scenario")["predicted_sales"]
    cv_lines = "\n".join(
        f"| {row['model']} | {row['cv_mae_mean']:.3f} | {row['cv_rmse_mean']:.3f} ± {row['cv_rmse_std']:.3f} | {row['cv_r2_mean']:.3f} |"
        for row in comparison.to_dict(orient="records")
    )
    summary = f"""# Sales Prediction — Design Report

## Objective
Predict Sales from TV, Radio, and Newspaper advertising measurements using the supplied Advertising.csv. This is conditional out-of-sample regression; the data do not support a dated future sales forecast, customer segmentation, or online-platform analysis.

## Data and audit
The supplied file has {audit['source_rows']} rows and five columns. The verified sequential `Unnamed: 0` column is excluded from predictors and retained only as a source-row identifier. There are {sum(audit['missing_values'].values())} missing measurements and {audit['duplicate_observations_ignoring_index']} duplicate observations ignoring the identifier. Features and Sales are numeric. Values use **observed dataset units**: the file provides no verified currency or sales-volume unit. The task document points to [Advertising.csv on Kaggle](https://www.kaggle.com/datasets/bumba5341/advertisingcsv), while the actual input is the user's uploaded CSV. Upstream provenance and licensing were not independently verified.

Audit and source checksum: [metrics.json](metrics.json). Descriptive statistics and correlations: [tables](tables/). Correlations and scatterplots describe associations, not causal marketing effects.

## Method and reproducibility
Seed {SEED}; a fixed random 80/20 split creates {len(train)} training rows and {len(test)} held-out test rows. All six predeclared candidate pipelines use exactly the same shuffled five-fold training CV splits. Median imputation, scaling, polynomial/interaction transformations, and model fitting occur inside each fold. Candidates use fixed hyperparameters documented in `make_models()`. The lowest mean CV RMSE selects **{selected_name}** before either final held-out evaluation is computed. The held-out set is used once for the selected model and the mean-prediction baseline. Source identifiers never enter the model. No refit on all observations is performed, so reported artifacts retain the split used for evaluation.

| Candidate | CV MAE | CV RMSE ± fold SD | CV R² |
| --- | ---: | ---: | ---: |
{cv_lines}

Fold SD describes variation across these five folds, not a confidence interval. Selection across candidates can make the winning CV score optimistic; the test results are the independent check. See [fold-level results](tables/cross_validation_folds.csv).

## Held-out results
| Model | MAE | RMSE | R² |
| --- | ---: | ---: | ---: |
| {selected_name} | {selected_test['mae']:.3f} | {selected_test['rmse']:.3f} | {selected_test['r2']:.3f} |
| Mean baseline | {dummy_test['mae']:.3f} | {dummy_test['rmse']:.3f} | {dummy_test['r2']:.3f} |

MAE and RMSE use observed Sales units; lower is better. R² compares squared error with a test-mean reference; higher is better. Predictions for every held-out observation are in [heldout_predictions.csv](heldout_predictions.csv). Residual = observed minus predicted.

![Held-out predictions and residuals](charts/03_heldout_predictions_and_residuals.png)

## Marketing interpretation and hypothetical scenarios
The training-only linear reference model supplies standardized coefficients in [training_linear_associations.csv](tables/training_linear_associations.csv). A coefficient is the fitted Sales association with a one-training-standard-deviation feature increase while holding the other measured variables fixed. It describes the linear reference model, which may differ from the selected nonlinear predictor, and does not establish an intervention effect.

The fitted standardized associations are {coefficient_lookup['TV']:.3f} for TV, {coefficient_lookup['Radio']:.3f} for Radio, and {coefficient_lookup['Newspaper']:.3f} for Newspaper, in observed Sales units per training standard deviation of the corresponding measurement. TV and Radio therefore show stronger conditional linear associations than Newspaper in this training sample; a small Newspaper coefficient does not establish that its true causal effect is zero.

![Linear association coefficients](charts/04_linear_association_coefficients.png)

Scenarios start from actual training source row {metrics['scenario_reference_source_row_id']} and vary one measured channel by −20% or +20%, clipped to that channel's training range. [Scenario inputs and predictions](tables/hypothetical_scenarios.csv) are hypothetical, with no observed outcome. Each input stays within its marginal training range; joint support is not assured. The results do not estimate causal uplift, guaranteed return on investment, or an optimal budget.

At this reference, predicted Sales is {scenario_lookup['Observed training reference']:.3f}. The separate +20% input scenarios predict {scenario_lookup['TV +20% (range clipped)']:.3f} for TV, {scenario_lookup['Radio +20% (range clipped)']:.3f} for Radio, and {scenario_lookup['Newspaper +20% (range clipped)']:.3f} for Newspaper while holding the other two values fixed. These are sensitivity examples from the selected model, not estimates of what an actual spend intervention will produce.

Business next step: prioritize measured, randomized TV and Radio channel experiments, including an appropriate Newspaper comparison, to test these observational hypotheses. Collect dates, verified spend and sales units, channel costs/revenue, and relevant segment labels before changing budgets. Predefine the experiment's outcome and validation period; this dataset alone does not justify reallocating spend.

![Hypothetical scenarios](charts/05_hypothetical_scenarios.png)

## Limitations and next steps
The small dataset contains no dates, customer segments, platform details, geography, prices, or economic controls. Random held-out accuracy does not establish future-period validity or transport to other markets. Unmeasured confounding, nonlinear structure, and correlated features limit interpretation. A time-stamped dataset and rolling chronological validation are needed for a future forecast; randomized or otherwise defensibly identified interventions are needed for causal marketing decisions. Verify provenance, permission to redistribute, and the measurement units before production use. The original user-supplied CSV is included for assignment reproducibility without an invented upstream license.

## Reproduction
From the project directory, install the pinned dependencies and run `python analysis.py`. The script uses paths relative to its own file and the noninteractive Agg renderer. `python /absolute/path/to/analysis.py` therefore works from another directory too. `--data` and `--output-dir` accept optional alternate paths. The notebook calls the same pipeline and displays its saved tables and charts. Seeds and numeric settings are fixed; package versions can affect final floating-point digits.
"""
    (output_dir / "summary.md").write_text(summary, encoding="utf-8")
    return metrics


def main(argv=None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "Advertising.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    args = parser.parse_args(argv)
    metrics = run_analysis(args.data.resolve(), args.output_dir.resolve())
    print(json.dumps({"selected_model": metrics["selection"]["selected_model"], "heldout": metrics["heldout"]}, indent=2))
    return metrics


if __name__ == "__main__":
    main()
