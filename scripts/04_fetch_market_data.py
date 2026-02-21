"""Step 4: Fetch market data.

Downloads SPY price data and computes forward returns.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml
from loguru import logger

from src.data.market_data import compute_returns, fetch_market_data, save_market_data
from src.utils.logging_config import setup_logging
from src.utils.reproducibility import set_seed


def main() -> None:
    """Run market data fetch pipeline."""
    config_path = Path("config/config.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    setup_logging(
        log_dir=Path(config["paths"]["logs_dir"]),
        script_name="04_fetch_market_data",
    )
    set_seed(config["project"]["seed"])

    mkt_config = config["market_data"]
    data_dir = Path(config["paths"]["data_dir"])

    logger.info(f"Fetching market data for {mkt_config['ticker']}")
    prices = fetch_market_data(
        ticker=mkt_config["ticker"],
        start_date=mkt_config["start_date"],
        end_date=mkt_config["end_date"],
        source=mkt_config["source"],
    )

    logger.info("Computing forward returns")
    returns = compute_returns(prices, horizon=config["features"]["prediction_horizon"])
    prices["forward_return"] = returns

    output_path = data_dir / "market" / f"{mkt_config['ticker']}.parquet"
    save_market_data(prices, output_path)
    logger.info("Market data saved")


if __name__ == "__main__":
    main()
