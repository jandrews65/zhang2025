"""Performance report generation.

Generates summary reports comparing model performance
across strategies and periods.
"""

import json
from pathlib import Path

import pandas as pd
from loguru import logger


def generate_summary_table(
    metrics_by_model: dict[str, dict[str, float]],
) -> pd.DataFrame:
    """Generate a summary table of metrics across models.

    Args:
        metrics_by_model: Dict mapping model names to metric dicts.

    Returns:
        DataFrame with models as rows and metrics as columns.
    """
    df = pd.DataFrame(metrics_by_model).T
    # Reorder columns for readability
    preferred_order = [
        "sharpe_ratio", "cagr", "max_drawdown", "volatility",
        "hit_rate", "win_loss_ratio", "calmar_ratio", "trade_count",
        "total_return", "n_days",
        "benchmark_sharpe", "benchmark_cagr",
        "benchmark_max_drawdown", "benchmark_total_return",
    ]
    cols = [c for c in preferred_order if c in df.columns]
    cols += [c for c in df.columns if c not in cols]
    return df[cols]


def generate_report(
    metrics_by_model: dict[str, dict[str, float]],
    output_dir: Path,
) -> Path:
    """Generate a full performance report as JSON and text.

    Args:
        metrics_by_model: Dict mapping model names to metric dicts.
        output_dir: Directory to save the report.

    Returns:
        Path to the generated report JSON file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save JSON
    json_path = output_dir / "backtest_results.json"
    with open(json_path, "w") as f:
        json.dump(metrics_by_model, f, indent=2, default=str)
    logger.info(f"Saved JSON report: {json_path}")

    # Save summary table as CSV
    summary = generate_summary_table(metrics_by_model)
    csv_path = output_dir / "backtest_summary.csv"
    summary.to_csv(csv_path, float_format="%.6f")
    logger.info(f"Saved CSV summary: {csv_path}")

    # Generate text report
    txt_path = output_dir / "backtest_report.txt"
    lines = [
        "=" * 70,
        "BACKTEST PERFORMANCE REPORT",
        "Zhang (2025) Reproduction - SPY Direction Classification",
        "=" * 70,
        "",
    ]

    for model_name, metrics in metrics_by_model.items():
        lines.append(f"--- {model_name} ---")
        lines.append(f"  Sharpe Ratio:     {metrics.get('sharpe_ratio', 0):.4f}")
        lines.append(f"  CAGR:             {metrics.get('cagr', 0):.4%}")
        lines.append(f"  Max Drawdown:     {metrics.get('max_drawdown', 0):.4%}")
        lines.append(f"  Volatility:       {metrics.get('volatility', 0):.4%}")
        lines.append(f"  Hit Rate:         {metrics.get('hit_rate', 0):.4%}")
        lines.append(f"  Win/Loss Ratio:   {metrics.get('win_loss_ratio', 0):.4f}")
        lines.append(f"  Calmar Ratio:     {metrics.get('calmar_ratio', 0):.4f}")
        lines.append(f"  Trade Count:      {int(metrics.get('trade_count', 0))}")
        lines.append(f"  Total Return:     {metrics.get('total_return', 0):.4%}")
        lines.append(f"  Trading Days:     {int(metrics.get('n_days', 0))}")
        lines.append("")

    # Buy-and-hold benchmark (use first model's benchmark metrics)
    first_metrics = next(iter(metrics_by_model.values()))
    lines.append("--- Buy-and-Hold (SPY) ---")
    lines.append(f"  Sharpe Ratio:     {first_metrics.get('benchmark_sharpe', 0):.4f}")
    lines.append(f"  CAGR:             {first_metrics.get('benchmark_cagr', 0):.4%}")
    lines.append(f"  Max Drawdown:     {first_metrics.get('benchmark_max_drawdown', 0):.4%}")
    lines.append(f"  Total Return:     {first_metrics.get('benchmark_total_return', 0):.4%}")
    lines.append("")

    lines.append("=" * 70)
    lines.append("Note: Paper reports EUR/USD Sharpe ~5.87, USD/JPY ~4.65.")
    lines.append("This reproduction uses SPY with limited test data (2-day GDELT ingest).")
    lines.append("Results will improve with full 2015-2023 data ingestion.")
    lines.append("=" * 70)

    report_text = "\n".join(lines)
    with open(txt_path, "w") as f:
        f.write(report_text)
    logger.info(f"Saved text report: {txt_path}")

    return json_path
