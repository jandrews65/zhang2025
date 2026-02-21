"""Visualization utilities for results analysis.

Creates plots for cumulative returns, drawdowns,
rolling Sharpe, and model comparison charts.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for Docker
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger

from src.backtest.metrics import TRADING_DAYS_PER_YEAR, drawdown_series

# Style
plt.style.use("seaborn-v0_8-whitegrid")


def plot_cumulative_returns(
    returns_dict: dict[str, pd.Series],
    title: str = "Cumulative Returns",
    log_scale: bool = True,
    output_path: Path | None = None,
) -> plt.Figure:
    """Plot cumulative returns for multiple strategies.

    Args:
        returns_dict: Dict mapping strategy name to daily return series.
        title: Plot title.
        log_scale: Use logarithmic y-axis.
        output_path: Optional path to save the figure.

    Returns:
        Matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    for name, returns in returns_dict.items():
        cumulative = (1 + returns).cumprod()
        ax.plot(cumulative.index, cumulative.values, label=name, linewidth=1.5)

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return")
    if log_scale:
        ax.set_yscale("log")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved: {output_path}")

    plt.close(fig)
    return fig


def plot_drawdown(
    returns_dict: dict[str, pd.Series],
    title: str = "Drawdown",
    output_path: Path | None = None,
) -> plt.Figure:
    """Plot drawdown chart for multiple strategies.

    Args:
        returns_dict: Dict mapping strategy name to daily return series.
        title: Plot title.
        output_path: Optional path to save the figure.

    Returns:
        Matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=(12, 4))

    for name, returns in returns_dict.items():
        dd = drawdown_series(returns)
        ax.fill_between(dd.index, dd.values, 0, alpha=0.3, label=name)
        ax.plot(dd.index, dd.values, linewidth=0.8)

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved: {output_path}")

    plt.close(fig)
    return fig


def plot_rolling_sharpe(
    returns_dict: dict[str, pd.Series],
    window: int = 63,
    title: str = "Rolling Sharpe Ratio (63-day)",
    output_path: Path | None = None,
) -> plt.Figure | None:
    """Plot rolling Sharpe ratio for multiple strategies.

    Args:
        returns_dict: Dict mapping strategy name to daily return series.
        window: Rolling window in trading days.
        title: Plot title.
        output_path: Optional path to save the figure.

    Returns:
        Matplotlib Figure object, or None if insufficient data.
    """
    # Need at least window + some buffer
    min_len = min(len(r) for r in returns_dict.values())
    if min_len < window + 10:
        logger.warning(
            f"Not enough data for rolling Sharpe (need {window + 10}, have {min_len})"
        )
        return None

    fig, ax = plt.subplots(figsize=(12, 4))

    for name, returns in returns_dict.items():
        rolling_mean = returns.rolling(window).mean()
        rolling_std = returns.rolling(window).std()
        rolling_sr = (rolling_mean / rolling_std) * np.sqrt(TRADING_DAYS_PER_YEAR)
        ax.plot(rolling_sr.index, rolling_sr.values, label=name, linewidth=1.2)

    ax.axhline(y=0, color="black", linestyle="--", linewidth=0.8)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("Sharpe Ratio")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved: {output_path}")

    plt.close(fig)
    return fig


def plot_shap_summary(
    shap_values,
    feature_names: list[str],
    max_display: int = 20,
    output_path: Path | None = None,
) -> plt.Figure:
    """Plot SHAP summary chart.

    Args:
        shap_values: SHAP values array.
        feature_names: List of feature names.
        max_display: Maximum features to display.
        output_path: Optional path to save the figure.

    Returns:
        Matplotlib Figure object.
    """
    import shap

    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(
        shap_values,
        feature_names=feature_names,
        max_display=max_display,
        show=False,
    )

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved: {output_path}")

    fig = plt.gcf()
    plt.close(fig)
    return fig
