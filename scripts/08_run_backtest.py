"""Step 8: Run backtest.

Runs the trading strategy backtest using model predictions from walk-forward
validation, computes performance metrics, generates equity curve plots,
and saves comprehensive performance reports.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import yaml
from loguru import logger

from src.backtest.metrics import compute_all_metrics
from src.backtest.strategy import LongShortStrategy
from src.evaluation.performance_report import generate_report
from src.evaluation.visualization import (
    plot_cumulative_returns,
    plot_drawdown,
    plot_rolling_sharpe,
)
from src.utils.logging_config import setup_logging
from src.utils.reproducibility import set_seed


def load_predictions(artifacts_dir: Path, model_name: str) -> pd.DataFrame | None:
    """Load walk-forward prediction results for a model.

    Args:
        artifacts_dir: Base artifacts directory.
        model_name: Model subdirectory name.

    Returns:
        DataFrame with date, y_true, y_pred, y_proba, fold columns,
        or None if not found.
    """
    path = artifacts_dir / "models" / model_name / "walk_forward_results.parquet"
    if not path.exists():
        logger.warning(f"Predictions not found: {path}")
        return None

    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    logger.info(f"Loaded {model_name} predictions: {len(df)} rows, folds={df['fold'].nunique()}")
    return df


def load_market_returns(data_dir: Path) -> pd.Series:
    """Load the continuous forward returns (target) from the feature matrix.

    Args:
        data_dir: Base data directory.

    Returns:
        Series of market returns indexed by date.
    """
    features_path = data_dir / "processed" / "features_latest.parquet"
    df = pd.read_parquet(features_path)
    returns = df["target"].dropna()
    returns.index.name = "date"
    logger.info(f"Loaded market returns: {len(returns)} rows")
    return returns


def run_backtest_for_model(
    model_name: str,
    predictions: pd.DataFrame,
    market_returns: pd.Series,
    strategy: LongShortStrategy,
    risk_free_rate: float,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Run backtest for a single model's predictions.

    Args:
        model_name: Name for logging.
        predictions: Walk-forward prediction DataFrame.
        market_returns: Full market return series.
        strategy: Strategy instance.
        risk_free_rate: Annual risk-free rate.

    Returns:
        Tuple of (metrics dict, strategy results DataFrame).
    """
    logger.info(f"=== Backtesting {model_name} ===")

    # Build probability series indexed by date
    y_proba = predictions["y_proba"].astype(float)

    # Generate positions from predicted probabilities
    positions = strategy.generate_positions(y_proba)

    # Align market returns with prediction dates (OOS only)
    common_dates = positions.index.intersection(market_returns.index)
    if len(common_dates) == 0:
        logger.error(f"No overlapping dates between predictions and market returns")
        return {}, pd.DataFrame()

    aligned_returns = market_returns.loc[common_dates]
    positions = positions.loc[common_dates]

    # Compute strategy returns with transaction costs
    results_df = strategy.compute_strategy_returns(positions, aligned_returns)

    # Compute all metrics
    metrics = compute_all_metrics(
        strategy_returns=results_df["net_return"],
        market_returns=results_df["market_return"],
        positions=results_df["position"],
        risk_free_rate=risk_free_rate,
    )

    # Log key metrics
    logger.info(f"  Sharpe Ratio:   {metrics['sharpe_ratio']:.4f}")
    logger.info(f"  CAGR:           {metrics['cagr']:.4%}")
    logger.info(f"  Max Drawdown:   {metrics['max_drawdown']:.4%}")
    logger.info(f"  Hit Rate:       {metrics['hit_rate']:.4%}")
    logger.info(f"  Total Return:   {metrics['total_return']:.4%}")
    logger.info(f"  Trade Count:    {int(metrics['trade_count'])}")

    return metrics, results_df


def main() -> None:
    """Run backtest pipeline."""
    config_path = Path("config/config.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    setup_logging(
        log_dir=Path(config["paths"]["logs_dir"]),
        script_name="08_run_backtest",
    )
    set_seed(config["project"]["seed"])

    logger.info("Running backtest")

    # Config
    bt_config = config["backtest"]
    artifacts_dir = Path(config["paths"]["artifacts_dir"])
    reports_dir = Path(config["paths"]["reports_dir"]) / "backtest"
    data_dir = Path(config["paths"]["data_dir"])
    risk_free_rate = bt_config.get("risk_free_rate", 0.0)

    # Initialize strategy
    strategy = LongShortStrategy(
        initial_capital=bt_config["initial_capital"],
        transaction_cost_bps=bt_config["transaction_cost_bps"],
        slippage_bps=bt_config["slippage_bps"],
        probability_threshold=bt_config.get("probability_threshold", 0.5),
    )

    # Load market returns
    market_returns = load_market_returns(data_dir)

    # Models to backtest
    model_names = ["logistic_regression", "xgboost"]
    all_metrics: dict[str, dict[str, float]] = {}
    all_results: dict[str, pd.DataFrame] = {}

    for model_name in model_names:
        predictions = load_predictions(artifacts_dir, model_name)
        if predictions is None:
            continue

        metrics, results_df = run_backtest_for_model(
            model_name=model_name,
            predictions=predictions,
            market_returns=market_returns,
            strategy=strategy,
            risk_free_rate=risk_free_rate,
        )

        if metrics:
            all_metrics[model_name] = metrics
            all_results[model_name] = results_df

            # Save per-model results
            model_reports = reports_dir / model_name
            model_reports.mkdir(parents=True, exist_ok=True)
            results_df.to_parquet(model_reports / "backtest_results.parquet")

    if not all_metrics:
        logger.error("No models produced backtest results")
        sys.exit(1)

    # Generate plots
    logger.info("Generating visualizations")
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Build returns dict for plotting (strategies + buy-and-hold)
    returns_for_plot: dict[str, pd.Series] = {}
    for model_name, results_df in all_results.items():
        returns_for_plot[model_name] = results_df["net_return"]

    # Add buy-and-hold (using common date range from first model)
    first_results = next(iter(all_results.values()))
    bh_dates = first_results.index
    returns_for_plot["Buy-and-Hold (SPY)"] = market_returns.loc[
        market_returns.index.isin(bh_dates)
    ]

    # Cumulative returns
    plot_cumulative_returns(
        returns_for_plot,
        title="Cumulative Returns: Strategy vs Buy-and-Hold",
        log_scale=False,
        output_path=reports_dir / "cumulative_returns.png",
    )

    # Drawdown
    plot_drawdown(
        returns_for_plot,
        title="Drawdown Comparison",
        output_path=reports_dir / "drawdown.png",
    )

    # Rolling Sharpe (may return None if insufficient data)
    plot_rolling_sharpe(
        returns_for_plot,
        window=63,
        output_path=reports_dir / "rolling_sharpe.png",
    )

    # Generate report
    report_path = generate_report(
        metrics_by_model=all_metrics,
        output_dir=reports_dir,
    )

    logger.info(f"Report saved to: {report_path}")
    logger.info("Backtest complete")


if __name__ == "__main__":
    main()
