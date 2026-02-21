"""Transaction cost modeling.

Models realistic trading costs including commissions and slippage.
Costs are incurred only when position changes.
"""

import numpy as np
import pandas as pd
from loguru import logger


def compute_turnover(positions: pd.Series) -> pd.Series:
    """Compute daily portfolio turnover.

    Turnover is the absolute change in position. For a +1/-1 strategy,
    a flip from long to short has turnover=2, no change has turnover=0.

    The first day has turnover equal to the absolute initial position
    (entering the market).

    Args:
        positions: Series of position signals (+1 or -1).

    Returns:
        Series of absolute position changes.
    """
    changes = positions.diff().abs()
    # First day: cost of entering the position
    changes.iloc[0] = abs(positions.iloc[0])
    return changes


def apply_transaction_costs(
    gross_returns: pd.Series,
    turnover: pd.Series,
    cost_bps: float,
    slippage_bps: float,
) -> pd.Series:
    """Compute transaction costs from turnover.

    Cost = turnover * (cost_bps + slippage_bps) / 10_000.
    Costs are always non-negative.

    Args:
        gross_returns: Strategy returns before costs (unused, kept for interface).
        turnover: Portfolio turnover series.
        cost_bps: Transaction cost in basis points.
        slippage_bps: Slippage in basis points.

    Returns:
        Series of costs to subtract from gross returns.
    """
    total_bps = cost_bps + slippage_bps
    costs = turnover * (total_bps / 10_000)

    n_trades = (turnover > 0).sum()
    total_cost = costs.sum()
    logger.debug(
        f"Transaction costs: {n_trades} position changes, "
        f"total_bps={total_bps}, total_cost={total_cost:.6f}"
    )

    return costs
