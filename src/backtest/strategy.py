"""Trading strategy based on model predictions.

Converts predicted probabilities into portfolio positions
and computes strategy returns with transaction costs.
"""

import numpy as np
import pandas as pd
from loguru import logger

from src.backtest.transaction_costs import apply_transaction_costs, compute_turnover


class LongShortStrategy:
    """Long/short strategy based on predicted probability of positive return.

    Signals: Long (+1) if P(up) > threshold, Short (-1) otherwise.
    Always 100% allocated. Daily rebalancing.
    """

    def __init__(
        self,
        initial_capital: float = 1_000_000,
        transaction_cost_bps: float = 3.0,
        slippage_bps: float = 2.0,
        probability_threshold: float = 0.5,
    ) -> None:
        """Initialize the strategy.

        Args:
            initial_capital: Starting capital in dollars.
            transaction_cost_bps: Transaction costs in basis points.
            slippage_bps: Slippage in basis points.
            probability_threshold: Threshold for long signal.
        """
        self.initial_capital = initial_capital
        self.transaction_cost_bps = transaction_cost_bps
        self.slippage_bps = slippage_bps
        self.probability_threshold = probability_threshold

    def generate_positions(self, y_proba: pd.Series) -> pd.Series:
        """Convert predicted probabilities to positions (+1 long, -1 short).

        Args:
            y_proba: Series of predicted P(positive return).

        Returns:
            Series of position signals (+1 or -1).
        """
        positions = pd.Series(
            np.where(y_proba > self.probability_threshold, 1, -1),
            index=y_proba.index,
            name="position",
        )
        n_long = (positions == 1).sum()
        n_short = (positions == -1).sum()
        logger.info(f"Positions: {n_long} long, {n_short} short out of {len(positions)}")
        return positions

    def compute_strategy_returns(
        self,
        positions: pd.Series,
        market_returns: pd.Series,
    ) -> pd.DataFrame:
        """Compute strategy returns accounting for transaction costs.

        Strategy return[t] = position[t] * market_return[t].
        Costs are incurred only when position changes.

        Args:
            positions: Position signals (+1 or -1), indexed by date.
            market_returns: Actual market returns, indexed by date.

        Returns:
            DataFrame with columns: position, market_return, gross_return,
            turnover, cost, net_return, cumulative_gross, cumulative_net, equity.
        """
        # Align indices
        common_idx = positions.index.intersection(market_returns.index)
        positions = positions.loc[common_idx]
        market_returns = market_returns.loc[common_idx]

        # Gross return: position * market return
        gross_returns = positions * market_returns

        # Turnover and costs
        turnover = compute_turnover(positions)
        costs = apply_transaction_costs(
            gross_returns,
            turnover,
            self.transaction_cost_bps,
            self.slippage_bps,
        )
        net_returns = gross_returns - costs

        # Cumulative returns
        cumulative_gross = (1 + gross_returns).cumprod()
        cumulative_net = (1 + net_returns).cumprod()
        equity = self.initial_capital * cumulative_net

        result = pd.DataFrame(
            {
                "position": positions,
                "market_return": market_returns,
                "gross_return": gross_returns,
                "turnover": turnover,
                "cost": costs,
                "net_return": net_returns,
                "cumulative_gross": cumulative_gross,
                "cumulative_net": cumulative_net,
                "equity": equity,
            },
            index=common_idx,
        )

        n_trades = (turnover > 0).sum()
        total_cost = costs.sum()
        logger.info(
            f"Strategy: {len(result)} days, {n_trades} trades, "
            f"total cost={total_cost:.6f}, "
            f"gross cum={cumulative_gross.iloc[-1]:.4f}, "
            f"net cum={cumulative_net.iloc[-1]:.4f}"
        )

        return result
