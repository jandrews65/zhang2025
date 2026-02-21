"""Performance metrics for strategy evaluation.

Computes Sharpe ratio, maximum drawdown, hit rate, and other
standard portfolio performance metrics.
"""

import numpy as np
import pandas as pd
from loguru import logger

TRADING_DAYS_PER_YEAR = 252


def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    annualize: bool = True,
) -> float:
    """Compute the Sharpe ratio.

    Args:
        returns: Series of daily strategy returns.
        risk_free_rate: Annual risk-free rate.
        annualize: Whether to annualize the ratio.

    Returns:
        Sharpe ratio (annualized if requested).
    """
    if len(returns) < 2 or returns.std() == 0:
        return 0.0

    daily_rf = risk_free_rate / TRADING_DAYS_PER_YEAR
    excess = returns - daily_rf
    sr = excess.mean() / excess.std()

    if annualize:
        sr *= np.sqrt(TRADING_DAYS_PER_YEAR)

    return float(sr)


def cagr(returns: pd.Series) -> float:
    """Compute compound annual growth rate.

    Args:
        returns: Series of daily strategy returns.

    Returns:
        CAGR as a decimal (e.g. 0.15 = 15%).
    """
    if len(returns) < 2:
        return 0.0

    cumulative = (1 + returns).prod()
    n_years = len(returns) / TRADING_DAYS_PER_YEAR

    if n_years <= 0 or cumulative <= 0:
        return 0.0

    return float(cumulative ** (1 / n_years) - 1)


def max_drawdown(returns: pd.Series) -> float:
    """Compute maximum drawdown from cumulative returns.

    Args:
        returns: Series of daily strategy returns.

    Returns:
        Maximum drawdown as a negative fraction (e.g. -0.15 = 15% drawdown).
    """
    if len(returns) < 2:
        return 0.0

    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdowns = cumulative / running_max - 1
    return float(drawdowns.min())


def drawdown_series(returns: pd.Series) -> pd.Series:
    """Compute the full drawdown time series.

    Args:
        returns: Series of daily strategy returns.

    Returns:
        Series of drawdown values (non-positive).
    """
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    return cumulative / running_max - 1


def volatility(returns: pd.Series, annualize: bool = True) -> float:
    """Compute return volatility.

    Args:
        returns: Series of daily strategy returns.
        annualize: Whether to annualize.

    Returns:
        Standard deviation of returns (annualized if requested).
    """
    if len(returns) < 2:
        return 0.0

    vol = returns.std()
    if annualize:
        vol *= np.sqrt(TRADING_DAYS_PER_YEAR)
    return float(vol)


def hit_rate(positions: pd.Series, market_returns: pd.Series) -> float:
    """Compute directional accuracy (hit rate).

    A "hit" is when position sign matches the actual return sign.

    Args:
        positions: Position signals (+1 or -1).
        market_returns: Actual market returns.

    Returns:
        Fraction of correct directional predictions.
    """
    if len(positions) == 0:
        return 0.0

    correct = (positions * market_returns) > 0
    return float(correct.mean())


def win_loss_ratio(strategy_returns: pd.Series) -> float:
    """Compute average winning return / average losing return.

    Args:
        strategy_returns: Series of strategy returns.

    Returns:
        Win/loss ratio. Returns inf if no losses, 0 if no wins.
    """
    wins = strategy_returns[strategy_returns > 0]
    losses = strategy_returns[strategy_returns < 0]

    if len(losses) == 0:
        return float("inf") if len(wins) > 0 else 0.0
    if len(wins) == 0:
        return 0.0

    return float(abs(wins.mean() / losses.mean()))


def calmar_ratio(returns: pd.Series) -> float:
    """Compute Calmar ratio: CAGR / abs(max_drawdown).

    Args:
        returns: Series of daily strategy returns.

    Returns:
        Calmar ratio. Returns 0 if max_drawdown is 0.
    """
    c = cagr(returns)
    mdd = max_drawdown(returns)

    if mdd == 0:
        return 0.0

    return float(c / abs(mdd))


def trade_count(positions: pd.Series) -> int:
    """Count the number of position changes.

    Args:
        positions: Series of position signals.

    Returns:
        Number of position changes (excluding initial entry).
    """
    changes = positions.diff().abs()
    # Exclude first row (initial entry)
    return int((changes.iloc[1:] > 0).sum())


def compute_all_metrics(
    strategy_returns: pd.Series,
    market_returns: pd.Series,
    positions: pd.Series,
    risk_free_rate: float = 0.0,
) -> dict[str, float]:
    """Compute full suite of performance metrics.

    Args:
        strategy_returns: Net strategy return series.
        market_returns: Actual market returns (for buy-and-hold comparison).
        positions: Position signals.
        risk_free_rate: Annual risk-free rate.

    Returns:
        Dictionary of metric name to value.
    """
    metrics = {
        "sharpe_ratio": sharpe_ratio(strategy_returns, risk_free_rate),
        "cagr": cagr(strategy_returns),
        "max_drawdown": max_drawdown(strategy_returns),
        "volatility": volatility(strategy_returns),
        "hit_rate": hit_rate(positions, market_returns),
        "win_loss_ratio": win_loss_ratio(strategy_returns),
        "calmar_ratio": calmar_ratio(strategy_returns),
        "trade_count": trade_count(positions),
        "total_return": float((1 + strategy_returns).prod() - 1),
        "n_days": len(strategy_returns),
        # Buy-and-hold comparison
        "benchmark_sharpe": sharpe_ratio(market_returns, risk_free_rate),
        "benchmark_cagr": cagr(market_returns),
        "benchmark_max_drawdown": max_drawdown(market_returns),
        "benchmark_total_return": float((1 + market_returns).prod() - 1),
    }

    return metrics
