"""Performance evaluation and visualization modules."""

from src.evaluation.performance_report import generate_report, generate_summary_table
from src.evaluation.visualization import (
    plot_cumulative_returns,
    plot_drawdown,
    plot_rolling_sharpe,
    plot_shap_summary,
)

__all__ = [
    "generate_report",
    "generate_summary_table",
    "plot_cumulative_returns",
    "plot_drawdown",
    "plot_rolling_sharpe",
    "plot_shap_summary",
]
