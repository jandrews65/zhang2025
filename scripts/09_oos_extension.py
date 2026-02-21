"""Step 9: Out-of-sample extension test.

Tests the paper's key claim: does the trained model hold up on truly
unseen data after publication? Loads a frozen model (no retraining),
applies it to any OOS data beyond the training window, and compares
performance to the in-sample backtest results.

If insufficient OOS data is available, generates a status report
documenting what would be needed for a full OOS evaluation.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
import pandas as pd
import yaml
from loguru import logger

from src.backtest.metrics import compute_all_metrics
from src.backtest.strategy import LongShortStrategy
from src.evaluation.performance_report import generate_report
from src.evaluation.visualization import plot_cumulative_returns, plot_drawdown
from src.utils.logging_config import setup_logging
from src.utils.reproducibility import set_seed


def find_last_fold(model_dir: Path) -> Path | None:
    """Find the last (highest-numbered) fold directory.

    Args:
        model_dir: Path to model artifacts (e.g. artifacts/models/xgboost/).

    Returns:
        Path to the last fold directory, or None.
    """
    fold_dirs = sorted(model_dir.glob("fold_*"))
    if not fold_dirs:
        return None
    return fold_dirs[-1]


def load_frozen_model(
    fold_dir: Path,
) -> tuple[object, object, dict]:
    """Load the frozen model, scaler, and metadata from a fold directory.

    Args:
        fold_dir: Path to fold directory containing model.joblib, scaler.joblib,
            metadata.json.

    Returns:
        Tuple of (model, scaler, metadata_dict).
    """
    model = joblib.load(fold_dir / "model.joblib")
    scaler = joblib.load(fold_dir / "scaler.joblib")
    with open(fold_dir / "metadata.json") as f:
        metadata = json.load(f)
    return model, scaler, metadata


def identify_oos_data(
    features: pd.DataFrame,
    train_end_date: str,
    oos_start_date: str,
    oos_end_date: str,
) -> pd.DataFrame:
    """Identify the OOS portion of the feature matrix.

    OOS data is anything with dates strictly after the training end date
    AND within the configured OOS window.

    Args:
        features: Full feature matrix with DatetimeIndex.
        train_end_date: Last date used in training (from fold metadata).
        oos_start_date: Configured OOS start.
        oos_end_date: Configured OOS end.

    Returns:
        DataFrame of OOS features (may be empty).
    """
    train_end = pd.Timestamp(train_end_date)
    oos_start = pd.Timestamp(oos_start_date)
    oos_end = pd.Timestamp(oos_end_date)

    # OOS = data after training ended, within the configured OOS window
    # Use the later of (train_end, oos_start) as the actual OOS boundary
    actual_start = max(train_end, oos_start)

    mask = (features.index > actual_start) & (features.index <= oos_end)
    oos = features.loc[mask]

    logger.info(
        f"OOS data identification: train_end={train_end.date()}, "
        f"oos_window=[{oos_start.date()}, {oos_end.date()}], "
        f"actual_start={actual_start.date()}, rows_found={len(oos)}"
    )
    return oos


def compute_performance_decay(
    is_metrics: dict[str, float],
    oos_metrics: dict[str, float],
) -> dict[str, float]:
    """Compute performance decay between in-sample and OOS.

    Args:
        is_metrics: In-sample (backtest) metrics.
        oos_metrics: Out-of-sample metrics.

    Returns:
        Dict with decay statistics for key metrics.
    """
    decay: dict[str, float] = {}
    compare_keys = ["sharpe_ratio", "cagr", "hit_rate", "max_drawdown"]

    for key in compare_keys:
        is_val = is_metrics.get(key, 0.0)
        oos_val = oos_metrics.get(key, 0.0)

        decay[f"{key}_is"] = is_val
        decay[f"{key}_oos"] = oos_val
        decay[f"{key}_diff"] = oos_val - is_val

        # Percentage decay (relative to absolute IS value)
        if is_val != 0:
            decay[f"{key}_decay_pct"] = (oos_val - is_val) / abs(is_val) * 100
        else:
            decay[f"{key}_decay_pct"] = 0.0

    return decay


def generate_oos_report(
    output_dir: Path,
    model_name: str,
    fold_metadata: dict,
    is_metrics: dict[str, float] | None,
    oos_metrics: dict[str, float] | None,
    decay: dict[str, float] | None,
    oos_n_days: int,
    min_oos_days: int,
    status: str,
) -> Path:
    """Generate the OOS extension report.

    Args:
        output_dir: Directory to write report files.
        model_name: Name of the model evaluated.
        fold_metadata: Training fold metadata dict.
        is_metrics: In-sample backtest metrics (may be None).
        oos_metrics: OOS metrics (may be None).
        decay: Performance decay dict (may be None).
        oos_n_days: Number of OOS trading days found.
        min_oos_days: Minimum required for evaluation.
        status: "complete", "insufficient_data", or "no_oos_data".

    Returns:
        Path to the JSON report.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report: dict = {
        "status": status,
        "model_name": model_name,
        "frozen_model_fold": fold_metadata,
        "oos_trading_days": oos_n_days,
        "min_required_days": min_oos_days,
    }

    if oos_metrics:
        report["oos_metrics"] = oos_metrics
    if is_metrics:
        report["is_metrics"] = {
            k: is_metrics[k]
            for k in ["sharpe_ratio", "cagr", "max_drawdown", "hit_rate",
                       "total_return", "n_days"]
            if k in is_metrics
        }
    if decay:
        report["performance_decay"] = decay

    # Save JSON
    json_path = output_dir / "oos_results.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"Saved OOS JSON report: {json_path}")

    # Generate text report
    txt_path = output_dir / "oos_report.txt"
    lines = [
        "=" * 70,
        "OUT-OF-SAMPLE EXTENSION REPORT",
        "Zhang (2025) Reproduction - SPY Direction Classification",
        "=" * 70,
        "",
        f"Status: {status.upper()}",
        f"Model: {model_name}",
        f"Frozen from fold: train [{fold_metadata.get('train_start', '?')}"
        f" .. {fold_metadata.get('train_end', '?')}]",
        f"OOS trading days found: {oos_n_days}",
        f"Minimum required: {min_oos_days}",
        "",
    ]

    if status == "complete" and oos_metrics and is_metrics and decay:
        lines.append("--- In-Sample (Backtest) Performance ---")
        lines.append(f"  Sharpe Ratio:   {is_metrics.get('sharpe_ratio', 0):.4f}")
        lines.append(f"  CAGR:           {is_metrics.get('cagr', 0):.4%}")
        lines.append(f"  Max Drawdown:   {is_metrics.get('max_drawdown', 0):.4%}")
        lines.append(f"  Hit Rate:       {is_metrics.get('hit_rate', 0):.4%}")
        lines.append("")

        lines.append("--- Out-of-Sample Performance ---")
        lines.append(f"  Sharpe Ratio:   {oos_metrics.get('sharpe_ratio', 0):.4f}")
        lines.append(f"  CAGR:           {oos_metrics.get('cagr', 0):.4%}")
        lines.append(f"  Max Drawdown:   {oos_metrics.get('max_drawdown', 0):.4%}")
        lines.append(f"  Hit Rate:       {oos_metrics.get('hit_rate', 0):.4%}")
        lines.append(f"  Total Return:   {oos_metrics.get('total_return', 0):.4%}")
        lines.append(f"  Trade Count:    {int(oos_metrics.get('trade_count', 0))}")
        lines.append("")

        lines.append("--- Performance Decay ---")
        lines.append(
            f"  Sharpe: {decay['sharpe_ratio_is']:.4f} -> "
            f"{decay['sharpe_ratio_oos']:.4f} "
            f"({decay['sharpe_ratio_decay_pct']:+.1f}%)"
        )
        lines.append(
            f"  CAGR:   {decay['cagr_is']:.4%} -> "
            f"{decay['cagr_oos']:.4%} "
            f"({decay['cagr_decay_pct']:+.1f}%)"
        )
        lines.append(
            f"  Hit:    {decay['hit_rate_is']:.4%} -> "
            f"{decay['hit_rate_oos']:.4%} "
            f"({decay['hit_rate_decay_pct']:+.1f}%)"
        )
        lines.append(
            f"  MaxDD:  {decay['max_drawdown_is']:.4%} -> "
            f"{decay['max_drawdown_oos']:.4%} "
            f"({decay['max_drawdown_decay_pct']:+.1f}%)"
        )
        lines.append("")

    elif status == "insufficient_data":
        lines.append(f"Found {oos_n_days} OOS days but need at least {min_oos_days}.")
        lines.append("Partial data summary logged but metrics are unreliable.")
        lines.append("")

    elif status == "no_oos_data":
        lines.append("No data available in the OOS window.")
        lines.append("")
        lines.append("To run a full OOS test:")
        lines.append("  1. Ingest GDELT data covering the OOS period")
        lines.append("  2. Run scripts 01-05 to build features for the OOS dates")
        lines.append("  3. Re-run this script")
        lines.append("")

    lines.append("=" * 70)
    lines.append("Note: The paper's claim of strong OOS performance is on FX pairs.")
    lines.append("This SPY reproduction serves as a framework for testing that claim")
    lines.append("once sufficient data is available.")
    lines.append("=" * 70)

    with open(txt_path, "w") as f:
        f.write("\n".join(lines))
    logger.info(f"Saved OOS text report: {txt_path}")

    return json_path


def main() -> None:
    """Run out-of-sample extension test."""
    config_path = Path("config/config.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    setup_logging(
        log_dir=Path(config["paths"]["logs_dir"]),
        script_name="09_oos_extension",
    )
    set_seed(config["project"]["seed"])

    oos_config = config["oos_extension"]
    bt_config = config["backtest"]
    artifacts_dir = Path(config["paths"]["artifacts_dir"])
    data_dir = Path(config["paths"]["data_dir"])
    reports_dir = Path(config["paths"]["reports_dir"]) / "oos_extension"
    model_name = oos_config.get("model_name", "xgboost")
    min_oos_days = oos_config.get("min_oos_days", 20)

    logger.info(
        f"OOS Extension: {oos_config['start_date']} to {oos_config['end_date']}, "
        f"model={model_name}"
    )

    # ── Step 1: Load frozen model from last fold ──
    model_dir = artifacts_dir / "models" / model_name
    last_fold_dir = find_last_fold(model_dir)
    if last_fold_dir is None:
        logger.error(f"No fold directories found in {model_dir}")
        sys.exit(1)

    model, scaler, fold_metadata = load_frozen_model(last_fold_dir)
    logger.info(
        f"Loaded frozen model from {last_fold_dir.name}: "
        f"train=[{fold_metadata['train_start']}..{fold_metadata['train_end']}], "
        f"original test=[{fold_metadata['test_start']}..{fold_metadata['test_end']}]"
    )
    logger.info("NO retraining will be performed — model is frozen")

    # ── Step 2: Load feature matrix and identify OOS data ──
    features_path = data_dir / "processed" / "features_latest.parquet"
    if not features_path.exists():
        logger.error(f"Features file not found: {features_path}")
        sys.exit(1)

    features = pd.read_parquet(features_path)
    logger.info(
        f"Feature matrix: {len(features)} rows, "
        f"[{features.index[0].date()}..{features.index[-1].date()}]"
    )

    # Separate target
    target_col = "target"
    y_full = features[target_col]
    X_full = features.drop(columns=[target_col])

    # Identify OOS data: anything after the last fold's test_end
    # (which is the last date the model has "seen" predictions for)
    oos_features = identify_oos_data(
        features=X_full,
        train_end_date=fold_metadata["test_end"],
        oos_start_date=oos_config["start_date"],
        oos_end_date=oos_config["end_date"],
    )

    oos_n_days = len(oos_features)

    # ── Step 3: Load in-sample backtest metrics for comparison ──
    is_metrics = None
    is_backtest_path = (
        Path(config["paths"]["reports_dir"]) / "backtest" / "backtest_results.json"
    )
    if is_backtest_path.exists():
        with open(is_backtest_path) as f:
            all_is_metrics = json.load(f)
        is_metrics = all_is_metrics.get(model_name)
        if is_metrics:
            logger.info(
                f"Loaded IS metrics: Sharpe={is_metrics['sharpe_ratio']:.4f}, "
                f"CAGR={is_metrics['cagr']:.4%}"
            )

    # ── Step 4: Handle insufficient or absent OOS data ──
    if oos_n_days == 0:
        logger.warning(
            "No OOS data available. The feature matrix does not extend "
            f"beyond {fold_metadata['test_end']}."
        )
        generate_oos_report(
            output_dir=reports_dir,
            model_name=model_name,
            fold_metadata=fold_metadata,
            is_metrics=is_metrics,
            oos_metrics=None,
            decay=None,
            oos_n_days=0,
            min_oos_days=min_oos_days,
            status="no_oos_data",
        )
        logger.info("OOS extension complete (no OOS data)")
        return

    if oos_n_days < min_oos_days:
        logger.warning(
            f"Only {oos_n_days} OOS days found (need {min_oos_days}). "
            "Running anyway but flagging as insufficient."
        )
        status = "insufficient_data"
    else:
        status = "complete"

    # ── Step 5: Apply frozen scaler and generate predictions ──
    oos_y = y_full.loc[oos_features.index].dropna()
    oos_X = oos_features.loc[oos_y.index]

    logger.info(f"OOS prediction set: {len(oos_X)} rows after dropping NaN targets")

    if len(oos_X) == 0:
        logger.warning("No valid OOS rows after dropping NaN targets")
        generate_oos_report(
            output_dir=reports_dir,
            model_name=model_name,
            fold_metadata=fold_metadata,
            is_metrics=is_metrics,
            oos_metrics=None,
            decay=None,
            oos_n_days=0,
            min_oos_days=min_oos_days,
            status="no_oos_data",
        )
        logger.info("OOS extension complete (no valid OOS rows)")
        return

    # Scale with the FROZEN scaler (fitted on training data, no re-fitting)
    oos_X_scaled = pd.DataFrame(
        scaler.transform(oos_X),
        index=oos_X.index,
        columns=oos_X.columns,
    )

    # Predict using frozen model
    y_pred = model.predict(oos_X_scaled)
    y_proba = model.predict_proba(oos_X_scaled)

    logger.info(
        f"OOS predictions: {len(y_pred)} rows, "
        f"P(up) mean={y_proba.mean():.4f}, "
        f"long_pct={(y_proba > bt_config.get('probability_threshold', 0.5)).mean():.2%}"
    )

    # ── Step 6: Run backtest on OOS predictions ──
    strategy = LongShortStrategy(
        initial_capital=bt_config["initial_capital"],
        transaction_cost_bps=bt_config["transaction_cost_bps"],
        slippage_bps=bt_config["slippage_bps"],
        probability_threshold=bt_config.get("probability_threshold", 0.5),
    )

    proba_series = pd.Series(y_proba, index=oos_X.index, name="y_proba")
    positions = strategy.generate_positions(proba_series)
    market_returns = oos_y  # continuous forward returns

    results_df = strategy.compute_strategy_returns(positions, market_returns)

    oos_metrics = compute_all_metrics(
        strategy_returns=results_df["net_return"],
        market_returns=results_df["market_return"],
        positions=results_df["position"],
        risk_free_rate=bt_config.get("risk_free_rate", 0.0),
    )

    logger.info(f"OOS Sharpe:       {oos_metrics['sharpe_ratio']:.4f}")
    logger.info(f"OOS CAGR:         {oos_metrics['cagr']:.4%}")
    logger.info(f"OOS Max Drawdown: {oos_metrics['max_drawdown']:.4%}")
    logger.info(f"OOS Hit Rate:     {oos_metrics['hit_rate']:.4%}")
    logger.info(f"OOS Total Return: {oos_metrics['total_return']:.4%}")

    # ── Step 7: Compare IS vs OOS ──
    decay = None
    if is_metrics:
        decay = compute_performance_decay(is_metrics, oos_metrics)
        logger.info("--- Performance Decay ---")
        logger.info(
            f"  Sharpe: {decay['sharpe_ratio_is']:.4f} -> "
            f"{decay['sharpe_ratio_oos']:.4f} ({decay['sharpe_ratio_decay_pct']:+.1f}%)"
        )
        logger.info(
            f"  CAGR:   {decay['cagr_is']:.4%} -> "
            f"{decay['cagr_oos']:.4%} ({decay['cagr_decay_pct']:+.1f}%)"
        )
        logger.info(
            f"  Hit:    {decay['hit_rate_is']:.4%} -> "
            f"{decay['hit_rate_oos']:.4%} ({decay['hit_rate_decay_pct']:+.1f}%)"
        )

    # ── Step 8: Save OOS predictions and generate plots ──
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Save predictions parquet
    preds_df = pd.DataFrame(
        {
            "date": oos_X.index,
            "y_true_continuous": oos_y.values,
            "y_true_binary": (oos_y > 0).astype(int).values,
            "y_pred": y_pred,
            "y_proba": y_proba,
            "position": positions.loc[oos_X.index].values,
            "net_return": results_df["net_return"].values,
        }
    )
    preds_df.to_parquet(reports_dir / "oos_predictions.parquet", index=False)

    # Save per-day results
    results_df.to_parquet(reports_dir / "oos_backtest_results.parquet")

    # Plots: OOS equity curve vs buy-and-hold
    returns_for_plot = {
        f"{model_name} (OOS)": results_df["net_return"],
        "Buy-and-Hold (SPY)": results_df["market_return"],
    }
    plot_cumulative_returns(
        returns_for_plot,
        title="OOS Extension: Cumulative Returns",
        log_scale=False,
        output_path=reports_dir / "oos_cumulative_returns.png",
    )
    plot_drawdown(
        returns_for_plot,
        title="OOS Extension: Drawdown",
        output_path=reports_dir / "oos_drawdown.png",
    )

    # ── Step 9: Generate report ──
    generate_oos_report(
        output_dir=reports_dir,
        model_name=model_name,
        fold_metadata=fold_metadata,
        is_metrics=is_metrics,
        oos_metrics=oos_metrics,
        decay=decay,
        oos_n_days=len(oos_X),
        min_oos_days=min_oos_days,
        status=status,
    )

    logger.info("OOS extension complete")


if __name__ == "__main__":
    main()
