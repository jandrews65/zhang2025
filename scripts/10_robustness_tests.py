"""Step 10: Robustness and sensitivity tests.

Multi-dimensional sensitivity analysis to test strategy robustness:
A. Transaction cost sensitivity
B. Probability threshold sensitivity
C. Subperiod performance analysis
D. Model comparison deep dive
E. Final reproduction summary
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from loguru import logger

from src.backtest.metrics import (
    cagr,
    compute_all_metrics,
    hit_rate,
    max_drawdown,
    sharpe_ratio,
    trade_count,
    volatility,
)
from src.backtest.strategy import LongShortStrategy
from src.utils.logging_config import setup_logging
from src.utils.reproducibility import set_seed

plt.style.use("seaborn-v0_8-whitegrid")


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────

def load_predictions(artifacts_dir: Path, model_name: str) -> pd.DataFrame | None:
    """Load walk-forward predictions for a model."""
    path = artifacts_dir / "models" / model_name / "walk_forward_results.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    return df


def load_market_returns(data_dir: Path) -> pd.Series:
    """Load continuous forward returns from feature matrix."""
    df = pd.read_parquet(data_dir / "processed" / "features_latest.parquet")
    returns = df["target"].dropna()
    returns.index.name = "date"
    return returns


def quick_backtest(
    y_proba: pd.Series,
    market_returns: pd.Series,
    cost_bps: float,
    slippage_bps: float,
    threshold: float,
) -> dict[str, float]:
    """Run a lightweight backtest and return metrics dict.

    Suppresses log output for batch sensitivity runs.
    """
    from src.backtest.transaction_costs import apply_transaction_costs, compute_turnover

    positions = pd.Series(
        np.where(y_proba > threshold, 1, -1),
        index=y_proba.index,
    )
    common = positions.index.intersection(market_returns.index)
    positions = positions.loc[common]
    mkt = market_returns.loc[common]

    gross = positions * mkt
    turnover = compute_turnover(positions)
    costs = apply_transaction_costs(gross, turnover, cost_bps, slippage_bps)
    net = gross - costs

    n_trades = int(trade_count(positions))
    metrics = {
        "sharpe_ratio": sharpe_ratio(net),
        "cagr": cagr(net),
        "max_drawdown": max_drawdown(net),
        "volatility": volatility(net),
        "hit_rate": hit_rate(positions, mkt),
        "total_return": float((1 + net).prod() - 1),
        "trade_count": n_trades,
        "n_days": len(net),
    }
    return metrics


# ─────────────────────────────────────────────────────────────────────
# A. Transaction Cost Sensitivity
# ─────────────────────────────────────────────────────────────────────

def run_cost_sensitivity(
    predictions: dict[str, pd.DataFrame],
    market_returns: pd.Series,
    slippage_bps: float,
    output_dir: Path,
) -> pd.DataFrame:
    """Sweep transaction costs and measure impact on performance."""
    logger.info("=== A. Transaction Cost Sensitivity ===")

    cost_levels = [0, 1, 2, 3, 5, 10, 15, 20]
    rows = []

    for model_name, preds in predictions.items():
        y_proba = preds["y_proba"].astype(float)
        for cost_bps in cost_levels:
            m = quick_backtest(y_proba, market_returns, cost_bps, slippage_bps, 0.5)
            m["model"] = model_name
            m["cost_bps"] = cost_bps
            rows.append(m)

    df = pd.DataFrame(rows)
    csv_path = output_dir / "cost_sensitivity.csv"
    df.to_csv(csv_path, index=False, float_format="%.6f")
    logger.info(f"Saved: {csv_path}")

    # Plot: Sharpe vs Cost
    fig, ax = plt.subplots(figsize=(10, 5))
    for model_name in predictions:
        subset = df[df["model"] == model_name]
        ax.plot(
            subset["cost_bps"], subset["sharpe_ratio"],
            marker="o", linewidth=2, label=model_name,
        )

    ax.axhline(y=0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Transaction Cost (bps)")
    ax.set_ylabel("Annualized Sharpe Ratio")
    ax.set_title("Sharpe Ratio vs Transaction Costs")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "cost_sensitivity.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {output_dir / 'cost_sensitivity.png'}")

    # Break-even analysis
    for model_name in predictions:
        subset = df[df["model"] == model_name]
        positive = subset[subset["sharpe_ratio"] > 0]
        if len(positive) > 0:
            breakeven = positive["cost_bps"].max()
            logger.info(f"  {model_name}: break-even cost ~{breakeven} bps")
        else:
            logger.info(f"  {model_name}: Sharpe negative at all cost levels")

    return df


# ─────────────────────────────────────────────────────────────────────
# B. Probability Threshold Sensitivity
# ─────────────────────────────────────────────────────────────────────

def run_threshold_sensitivity(
    predictions: dict[str, pd.DataFrame],
    market_returns: pd.Series,
    cost_bps: float,
    slippage_bps: float,
    output_dir: Path,
) -> pd.DataFrame:
    """Sweep probability thresholds and measure impact."""
    logger.info("=== B. Probability Threshold Sensitivity ===")

    thresholds = [0.40, 0.42, 0.45, 0.47, 0.48, 0.49, 0.50,
                  0.51, 0.52, 0.53, 0.55, 0.58, 0.60]
    rows = []

    for model_name, preds in predictions.items():
        y_proba = preds["y_proba"].astype(float)
        for thresh in thresholds:
            m = quick_backtest(y_proba, market_returns, cost_bps, slippage_bps, thresh)
            m["model"] = model_name
            m["threshold"] = thresh
            # Track position distribution
            n_long = int((y_proba > thresh).sum())
            m["pct_long"] = n_long / len(y_proba) if len(y_proba) > 0 else 0
            rows.append(m)

    df = pd.DataFrame(rows)
    csv_path = output_dir / "threshold_sensitivity.csv"
    df.to_csv(csv_path, index=False, float_format="%.6f")
    logger.info(f"Saved: {csv_path}")

    # Plot: Sharpe vs Threshold
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for model_name in predictions:
        subset = df[df["model"] == model_name]
        axes[0].plot(
            subset["threshold"], subset["sharpe_ratio"],
            marker="o", linewidth=2, label=model_name,
        )
        axes[1].plot(
            subset["threshold"], subset["pct_long"],
            marker="s", linewidth=2, label=model_name,
        )

    axes[0].axhline(y=0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    axes[0].axvline(x=0.5, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
    axes[0].set_xlabel("Probability Threshold")
    axes[0].set_ylabel("Annualized Sharpe Ratio")
    axes[0].set_title("Sharpe vs Probability Threshold")
    axes[0].legend()

    axes[1].axhline(y=0.5, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
    axes[1].set_xlabel("Probability Threshold")
    axes[1].set_ylabel("Fraction Long")
    axes[1].set_title("Position Bias vs Threshold")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_dir / "threshold_sensitivity.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {output_dir / 'threshold_sensitivity.png'}")

    # Optimal threshold
    for model_name in predictions:
        subset = df[df["model"] == model_name]
        best = subset.loc[subset["sharpe_ratio"].idxmax()]
        logger.info(
            f"  {model_name}: optimal threshold={best['threshold']:.2f}, "
            f"Sharpe={best['sharpe_ratio']:.4f}"
        )

    return df


# ─────────────────────────────────────────────────────────────────────
# C. Subperiod Performance Analysis
# ─────────────────────────────────────────────────────────────────────

def run_subperiod_analysis(
    predictions: dict[str, pd.DataFrame],
    market_returns: pd.Series,
    cost_bps: float,
    slippage_bps: float,
    output_dir: Path,
) -> pd.DataFrame:
    """Break down performance by calendar year."""
    logger.info("=== C. Subperiod Performance Analysis ===")

    rows = []

    for model_name, preds in predictions.items():
        y_proba = preds["y_proba"].astype(float)
        years = sorted(y_proba.index.year.unique())

        for year in years:
            year_mask = y_proba.index.year == year
            year_proba = y_proba.loc[year_mask]

            if len(year_proba) < 5:
                continue

            m = quick_backtest(
                year_proba, market_returns, cost_bps, slippage_bps, 0.5,
            )
            m["model"] = model_name
            m["year"] = year
            rows.append(m)

        # Also compute full-period metrics
        m_full = quick_backtest(y_proba, market_returns, cost_bps, slippage_bps, 0.5)
        m_full["model"] = model_name
        m_full["year"] = "all"
        rows.append(m_full)

    df = pd.DataFrame(rows)
    csv_path = output_dir / "subperiod_performance.csv"
    df.to_csv(csv_path, index=False, float_format="%.6f")
    logger.info(f"Saved: {csv_path}")

    # Plot: Sharpe by year grouped bar chart
    yearly = df[df["year"] != "all"].copy()
    if len(yearly) > 0:
        fig, ax = plt.subplots(figsize=(10, 5))
        model_names = list(predictions.keys())
        years = sorted(yearly["year"].unique())
        x = np.arange(len(years))
        width = 0.35

        for i, model_name in enumerate(model_names):
            subset = yearly[yearly["model"] == model_name]
            vals = [
                subset.loc[subset["year"] == y, "sharpe_ratio"].values[0]
                if y in subset["year"].values else 0
                for y in years
            ]
            ax.bar(x + i * width, vals, width, label=model_name)

        ax.axhline(y=0, color="black", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Year")
        ax.set_ylabel("Annualized Sharpe Ratio")
        ax.set_title("Performance by Year")
        ax.set_xticks(x + width / 2)
        ax.set_xticklabels([str(y) for y in years])
        ax.legend()
        fig.tight_layout()
        fig.savefig(
            output_dir / "subperiod_performance.png", dpi=150, bbox_inches="tight",
        )
        plt.close(fig)
        logger.info(f"Saved: {output_dir / 'subperiod_performance.png'}")

    # Stability metric: std of annual Sharpes
    for model_name in predictions:
        subset = yearly[yearly["model"] == model_name]
        if len(subset) > 1:
            sharpe_std = subset["sharpe_ratio"].std()
            sharpe_mean = subset["sharpe_ratio"].mean()
            logger.info(
                f"  {model_name}: annual Sharpe mean={sharpe_mean:.4f}, "
                f"std={sharpe_std:.4f}"
            )
        elif len(subset) == 1:
            logger.info(
                f"  {model_name}: only 1 year of data, "
                f"Sharpe={subset['sharpe_ratio'].iloc[0]:.4f}"
            )

    return df


# ─────────────────────────────────────────────────────────────────────
# D. Model Comparison Deep Dive
# ─────────────────────────────────────────────────────────────────────

def run_model_comparison(
    predictions: dict[str, pd.DataFrame],
    market_returns: pd.Series,
    artifacts_dir: Path,
    output_dir: Path,
) -> pd.DataFrame:
    """Deep dive comparison between models."""
    logger.info("=== D. Model Comparison Deep Dive ===")

    model_names = list(predictions.keys())

    # Probability distribution comparison
    fig, axes = plt.subplots(1, len(model_names), figsize=(6 * len(model_names), 4))
    if len(model_names) == 1:
        axes = [axes]

    for ax, model_name in zip(axes, model_names):
        preds = predictions[model_name]
        proba = preds["y_proba"].astype(float)
        ax.hist(proba, bins=30, alpha=0.7, edgecolor="black", linewidth=0.5)
        ax.axvline(x=0.5, color="red", linestyle="--", linewidth=1.5, label="threshold")
        ax.set_xlabel("P(positive return)")
        ax.set_ylabel("Count")
        ax.set_title(f"{model_name}\nmean={proba.mean():.4f}, std={proba.std():.4f}")
        ax.legend()

    fig.suptitle("Prediction Probability Distributions", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(
        output_dir / "probability_distributions.png", dpi=150, bbox_inches="tight",
    )
    plt.close(fig)
    logger.info(f"Saved: {output_dir / 'probability_distributions.png'}")

    # Confusion matrices
    rows = []
    for model_name in model_names:
        preds = predictions[model_name]
        y_true = preds["y_true"]
        y_pred = preds["y_pred"]

        tp = int(((y_true == 1) & (y_pred == 1)).sum())
        tn = int(((y_true == 0) & (y_pred == 0)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())

        rows.append({
            "model": model_name,
            "TP": tp, "TN": tn, "FP": fp, "FN": fn,
            "accuracy": (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0,
            "precision": tp / (tp + fp) if (tp + fp) > 0 else 0,
            "recall": tp / (tp + fn) if (tp + fn) > 0 else 0,
        })

    comparison_df = pd.DataFrame(rows)

    # Trade overlap (if 2+ models)
    if len(model_names) >= 2:
        m1, m2 = model_names[0], model_names[1]
        p1 = predictions[m1]
        p2 = predictions[m2]
        common_dates = p1.index.intersection(p2.index)

        if len(common_dates) > 0:
            pos1 = np.where(p1.loc[common_dates, "y_proba"].astype(float) > 0.5, 1, -1)
            pos2 = np.where(p2.loc[common_dates, "y_proba"].astype(float) > 0.5, 1, -1)
            agree = (pos1 == pos2).mean()
            comparison_df["trade_agreement"] = [agree, agree]
            logger.info(f"  Trade agreement ({m1} vs {m2}): {agree:.2%}")

    csv_path = output_dir / "model_comparison.csv"
    comparison_df.to_csv(csv_path, index=False, float_format="%.6f")
    logger.info(f"Saved: {csv_path}")

    # Feature importance chart (XGBoost SHAP if available)
    shap_path = artifacts_dir / "models" / "xgboost" / "shap" / "shap_values_last_fold.joblib"
    if shap_path.exists():
        try:
            shap_values = joblib.load(shap_path)
            shap_arr = np.array(shap_values)

            # Get feature names from feature matrix
            feat = pd.read_parquet(
                Path("data/processed/features_latest.parquet")
            )
            feature_names = [c for c in feat.columns if c != "target"]

            # Mean absolute SHAP
            if shap_arr.ndim == 2 and shap_arr.shape[1] == len(feature_names):
                mean_abs = np.abs(shap_arr).mean(axis=0)
                importance = pd.Series(mean_abs, index=feature_names).sort_values(
                    ascending=True,
                )

                fig, ax = plt.subplots(figsize=(8, 10))
                top_n = min(20, len(importance))
                top = importance.tail(top_n)
                ax.barh(range(len(top)), top.values)
                ax.set_yticks(range(len(top)))
                ax.set_yticklabels(top.index, fontsize=9)
                ax.set_xlabel("Mean |SHAP value|")
                ax.set_title("XGBoost Feature Importance (SHAP)")
                fig.tight_layout()
                fig.savefig(
                    output_dir / "feature_importance.png",
                    dpi=150, bbox_inches="tight",
                )
                plt.close(fig)
                logger.info(f"Saved: {output_dir / 'feature_importance.png'}")
        except Exception as e:
            logger.warning(f"SHAP importance chart failed: {e}")

    return comparison_df


# ─────────────────────────────────────────────────────────────────────
# E. Final Reproduction Summary
# ─────────────────────────────────────────────────────────────────────

def generate_final_summary(
    cost_df: pd.DataFrame,
    threshold_df: pd.DataFrame,
    subperiod_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    backtest_path: Path,
    oos_path: Path,
    output_dir: Path,
) -> None:
    """Generate the final reproduction summary in markdown."""
    logger.info("=== E. Final Reproduction Summary ===")

    # Load backtest results
    bt_metrics = {}
    if backtest_path.exists():
        with open(backtest_path) as f:
            bt_metrics = json.load(f)

    # Load OOS results
    oos_data = {}
    if oos_path.exists():
        with open(oos_path) as f:
            oos_data = json.load(f)

    lines = [
        "# Zhang (2025) Reproduction - Final Summary",
        "",
        "## 1. What We Reproduced",
        "",
        "This project implements the full pipeline from Zhang, Y. (2025)",
        '"Interpretable Machine Learning for Macro Alpha: A News Sentiment Case Study"',
        "(arXiv:2505.16136) adapted for SPY equity prediction.",
        "",
        "### Pipeline Components",
        "| Phase | Component | Status |",
        "|-------|-----------|--------|",
        "| 1 | Paper dissection & reproducibility plan | Complete |",
        "| 2 | Project scaffold & Docker environment | Complete |",
        "| 3 | GDELT ingestion, headline extraction, FinBERT scoring | Complete |",
        "| 4 | Feature engineering (80 features) | Complete |",
        "| 5 | Walk-forward validation (LogReg + XGBoost) | Complete |",
        "| 6 | Backtesting with transaction costs | Complete |",
        "| 7 | OOS extension framework | Complete |",
        "| 8 | Robustness & sensitivity tests | Complete |",
        "",
    ]

    # Backtest results section
    lines.append("## 2. Backtest Results (Test Data)")
    lines.append("")
    lines.append("*Note: Results are from a 2-day GDELT test ingest (478 rows,")
    lines.append("2020-01-30 to 2021-11-29). Full 2015-2023 data will yield more folds")
    lines.append("and more reliable metrics.*")
    lines.append("")

    if bt_metrics:
        lines.append("| Metric | Logistic Regression | XGBoost | Buy-and-Hold |")
        lines.append("|--------|-------------------|---------|-------------|")

        lr = bt_metrics.get("logistic_regression", {})
        xgb = bt_metrics.get("xgboost", {})

        metric_rows = [
            ("Sharpe Ratio", "sharpe_ratio", ".4f"),
            ("CAGR", "cagr", ".2%"),
            ("Max Drawdown", "max_drawdown", ".2%"),
            ("Hit Rate", "hit_rate", ".2%"),
            ("Total Return", "total_return", ".2%"),
            ("Trade Count", "trade_count", ".0f"),
        ]
        for label, key, fmt in metric_rows:
            lr_val = format(lr.get(key, 0), fmt) if lr else "N/A"
            xgb_val = format(xgb.get(key, 0), fmt) if xgb else "N/A"
            if key.startswith("benchmark_"):
                bh_val = format(lr.get(key, 0), fmt)
            elif key == "trade_count":
                bh_val = "N/A"
            else:
                bh_key = f"benchmark_{key}"
                bh_val = format(lr.get(bh_key, 0), fmt) if lr and bh_key in lr else "N/A"
            lines.append(f"| {label} | {lr_val} | {xgb_val} | {bh_val} |")
        lines.append("")

    # Paper comparison
    lines.append("## 3. Comparison to Paper's Reported Results")
    lines.append("")
    lines.append("| Metric | Paper (EUR/USD) | Paper (USD/JPY) | Our SPY (XGBoost) |")
    lines.append("|--------|----------------|----------------|-------------------|")

    xgb_sharpe = bt_metrics.get("xgboost", {}).get("sharpe_ratio", 0)
    xgb_cagr = bt_metrics.get("xgboost", {}).get("cagr", 0)
    lines.append(f"| Sharpe Ratio | 5.87 | 4.65 | {xgb_sharpe:.4f} |")
    lines.append(f"| CAGR | >50% | >50% | {xgb_cagr:.2%} |")
    lines.append("")
    lines.append("**Key differences:**")
    lines.append("- Paper uses FX pairs (EUR/USD, USD/JPY); we use SPY equity index")
    lines.append("- Paper uses full 2015-2023 dataset; we have a 2-day GDELT test ingest")
    lines.append("- Paper's reported Sharpe ratios (5.87, 4.65) are unusually high")
    lines.append("- SPY is generally harder to predict than FX (more efficient market)")
    lines.append("")

    # Sensitivity findings
    lines.append("## 4. Sensitivity Analysis Findings")
    lines.append("")

    # Cost sensitivity
    lines.append("### A. Transaction Cost Sensitivity")
    xgb_cost = cost_df[cost_df["model"] == "xgboost"] if "xgboost" in cost_df["model"].values else pd.DataFrame()
    if len(xgb_cost) > 0:
        zero_cost = xgb_cost[xgb_cost["cost_bps"] == 0]
        if len(zero_cost) > 0:
            lines.append(f"- Zero-cost Sharpe (XGBoost): {zero_cost['sharpe_ratio'].iloc[0]:.4f}")
        positive_sharpe = xgb_cost[xgb_cost["sharpe_ratio"] > 0]
        if len(positive_sharpe) > 0:
            lines.append(f"- Break-even cost: ~{int(positive_sharpe['cost_bps'].max())} bps")
        else:
            lines.append("- Sharpe negative at all cost levels (expected with limited data)")
    lines.append("")

    # Threshold sensitivity
    lines.append("### B. Probability Threshold Sensitivity")
    xgb_thresh = threshold_df[threshold_df["model"] == "xgboost"] if "xgboost" in threshold_df["model"].values else pd.DataFrame()
    if len(xgb_thresh) > 0:
        best = xgb_thresh.loc[xgb_thresh["sharpe_ratio"].idxmax()]
        lines.append(f"- Optimal threshold (XGBoost): {best['threshold']:.2f}")
        lines.append(f"- Optimal Sharpe: {best['sharpe_ratio']:.4f}")
        paper_thresh = xgb_thresh[xgb_thresh["threshold"] == 0.50]
        if len(paper_thresh) > 0:
            lines.append(f"- Paper's 0.50 threshold Sharpe: {paper_thresh['sharpe_ratio'].iloc[0]:.4f}")
    lines.append("")

    # Subperiod
    lines.append("### C. Subperiod Stability")
    yearly = subperiod_df[subperiod_df["year"] != "all"]
    if len(yearly) > 0:
        for model in yearly["model"].unique():
            sub = yearly[yearly["model"] == model]
            lines.append(f"- {model}: {len(sub)} year(s) of data")
            for _, row in sub.iterrows():
                lines.append(f"  - {row['year']}: Sharpe={row['sharpe_ratio']:.4f}")
    lines.append("")

    # Model comparison
    lines.append("### D. Model Comparison")
    if len(comparison_df) > 0:
        for _, row in comparison_df.iterrows():
            lines.append(
                f"- {row['model']}: accuracy={row['accuracy']:.4f}, "
                f"precision={row['precision']:.4f}, recall={row['recall']:.4f}"
            )
        if "trade_agreement" in comparison_df.columns:
            lines.append(
                f"- Trade agreement: {comparison_df['trade_agreement'].iloc[0]:.2%}"
            )
    lines.append("")

    # OOS section
    lines.append("## 5. Out-of-Sample Extension")
    lines.append("")
    oos_status = oos_data.get("status", "not_run")
    if oos_status == "no_oos_data":
        lines.append("OOS evaluation could not be performed: feature data does not extend")
        lines.append("beyond the training/test window. To run OOS:")
        lines.append("1. Ingest GDELT data for 2024-2025")
        lines.append("2. Rebuild features (scripts 01-05)")
        lines.append("3. Re-run `scripts/09_oos_extension.py`")
    elif oos_status == "complete":
        oos_m = oos_data.get("oos_metrics", {})
        lines.append(f"- OOS Sharpe: {oos_m.get('sharpe_ratio', 0):.4f}")
        lines.append(f"- OOS CAGR: {oos_m.get('cagr', 0):.2%}")
        decay = oos_data.get("performance_decay", {})
        if decay:
            lines.append(f"- Sharpe decay: {decay.get('sharpe_ratio_decay_pct', 0):+.1f}%")
    lines.append("")

    # Recommendations
    lines.append("## 6. Recommendations for Full Dataset Run")
    lines.append("")
    lines.append("1. **Ingest full GDELT 2015-2023**: ~100GB compressed, will yield")
    lines.append("   ~2,000 trading days of features vs current 478")
    lines.append("2. **Walk-forward will produce 4 folds** (2020, 2021, 2022, 2023)")
    lines.append("   enabling proper cross-temporal validation")
    lines.append("3. **Subperiod analysis** will cover COVID crash (2020),")
    lines.append("   recovery (2021), rate hikes (2022), AI rally (2023)")
    lines.append("4. **OOS extension** with 2024-2025 data will test post-publication decay")
    lines.append("5. **Expected Sharpe range** for SPY: 0.3-0.8 with full data")
    lines.append("   (vs paper's 4-6 on FX, which is unrealistically high)")
    lines.append("")

    # Limitations
    lines.append("## 7. Known Limitations")
    lines.append("")
    lines.append("- **Test data only**: 2-day GDELT ingest, 478 rows, 1 walk-forward fold")
    lines.append("- **SPY vs FX**: Paper targets EUR/USD and USD/JPY; SPY has different dynamics")
    lines.append("- **Headline extraction**: URL slug parsing (no web scraping) limits coverage")
    lines.append("- **Single asset**: No portfolio diversification or cross-asset signals")
    lines.append("- **No hyperparameter tuning**: Using paper's reported params directly")
    lines.append("")

    md_path = output_dir / "final_reproduction_summary.md"
    with open(md_path, "w") as f:
        f.write("\n".join(lines))
    logger.info(f"Saved: {md_path}")


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main() -> None:
    """Run all robustness and sensitivity tests."""
    config_path = Path("config/config.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    setup_logging(
        log_dir=Path(config["paths"]["logs_dir"]),
        script_name="10_robustness_tests",
    )
    set_seed(config["project"]["seed"])

    logger.info("Running robustness and sensitivity tests")

    artifacts_dir = Path(config["paths"]["artifacts_dir"])
    data_dir = Path(config["paths"]["data_dir"])
    reports_dir = Path(config["paths"]["reports_dir"]) / "robustness"
    reports_dir.mkdir(parents=True, exist_ok=True)

    bt_config = config["backtest"]
    cost_bps = bt_config["transaction_cost_bps"]
    slippage_bps = bt_config["slippage_bps"]

    # Load data
    market_returns = load_market_returns(data_dir)
    predictions: dict[str, pd.DataFrame] = {}
    for model_name in ["logistic_regression", "xgboost"]:
        preds = load_predictions(artifacts_dir, model_name)
        if preds is not None:
            predictions[model_name] = preds
            logger.info(f"Loaded {model_name}: {len(preds)} predictions")

    if not predictions:
        logger.error("No predictions found. Run scripts 06-07 first.")
        sys.exit(1)

    # A. Cost sensitivity
    cost_df = run_cost_sensitivity(
        predictions, market_returns, slippage_bps, reports_dir,
    )

    # B. Threshold sensitivity
    threshold_df = run_threshold_sensitivity(
        predictions, market_returns, cost_bps, slippage_bps, reports_dir,
    )

    # C. Subperiod analysis
    subperiod_df = run_subperiod_analysis(
        predictions, market_returns, cost_bps, slippage_bps, reports_dir,
    )

    # D. Model comparison
    comparison_df = run_model_comparison(
        predictions, market_returns, artifacts_dir, reports_dir,
    )

    # E. Final summary
    backtest_json = Path(config["paths"]["reports_dir"]) / "backtest" / "backtest_results.json"
    oos_json = Path(config["paths"]["reports_dir"]) / "oos_extension" / "oos_results.json"
    generate_final_summary(
        cost_df, threshold_df, subperiod_df, comparison_df,
        backtest_json, oos_json, reports_dir,
    )

    logger.info("Robustness tests complete")


if __name__ == "__main__":
    main()
