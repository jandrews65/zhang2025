"""Step 5: Build feature matrix.

Combines sentiment indices with market features,
applies rolling windows, and aligns with forward returns.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import yaml
from loguru import logger

from src.data.data_alignment import align_features_and_target, validate_no_lookahead
from src.features.feature_engineering import (
    build_feature_matrix,
    log_feature_statistics,
    save_features,
    save_features_versioned,
)
from src.features.sentiment_indices import compute_daily_sentiment, compute_rolling_sentiment
from src.utils.logging_config import setup_logging
from src.utils.reproducibility import set_seed


def main() -> None:
    """Run feature engineering pipeline."""
    config_path = Path("config/config.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    setup_logging(
        log_dir=Path(config["paths"]["logs_dir"]),
        script_name="05_build_features",
    )
    set_seed(config["project"]["seed"])

    data_dir = Path(config["paths"]["data_dir"])
    artifacts_dir = Path(config["paths"]["artifacts_dir"])
    features_cfg = config["features"]
    rolling_windows = features_cfg["rolling_windows"]
    prediction_horizon = features_cfg["prediction_horizon"]

    # New config keys with backward-compatible defaults
    sentiment_lag_days = features_cfg.get("sentiment_lag_days")
    sentiment_lag_columns = features_cfg.get("sentiment_lag_columns")
    acceleration_windows = features_cfg.get("acceleration_windows")
    market_vol_windows = features_cfg.get("market_vol_windows")
    versioned_output = features_cfg.get("versioned_output", False)

    # Load scored sentiment data
    logger.info("Loading sentiment data")
    sentiment_dir = data_dir / "sentiment"
    sentiment_files = sorted(sentiment_dir.rglob("*.parquet"))
    if not sentiment_files:
        raise FileNotFoundError(f"No sentiment files found in {sentiment_dir}")
    scored_headlines = pd.concat(
        [pd.read_parquet(f) for f in sentiment_files],
        ignore_index=True,
    )
    logger.info(f"Loaded {len(scored_headlines)} scored headlines")

    # Aggregate to daily sentiment
    daily_sentiment = compute_daily_sentiment(scored_headlines)

    # Compute rolling features
    rolling_sentiment = compute_rolling_sentiment(daily_sentiment, rolling_windows)

    # Load market data
    logger.info("Loading market data")
    ticker = config["market_data"]["ticker"]
    market_path = data_dir / "market" / f"{ticker}.parquet"
    market_df = pd.read_parquet(market_path)
    market_df.index = pd.to_datetime(market_df.index)

    # Build combined feature matrix (with new optional features)
    features = build_feature_matrix(
        rolling_sentiment,
        market_df,
        rolling_windows,
        sentiment_lag_days=sentiment_lag_days,
        sentiment_lag_columns=sentiment_lag_columns,
        acceleration_windows=acceleration_windows,
        market_vol_windows=market_vol_windows,
    )

    # Align with forward returns
    forward_returns = market_df["forward_return"].dropna()
    aligned = align_features_and_target(features, forward_returns, prediction_horizon)

    # Validate no lookahead
    validate_no_lookahead(aligned)

    # Log feature statistics and save to artifacts
    stats = log_feature_statistics(aligned)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    stats_path = artifacts_dir / "feature_stats.parquet"
    stats.to_parquet(stats_path, engine="pyarrow", compression="snappy")
    logger.info(f"Saved feature statistics to {stats_path}")

    # Save standard output (backward compatible)
    output_path = data_dir / "features" / "aligned_features.parquet"
    save_features(aligned, output_path)

    # Save versioned output if enabled
    if versioned_output:
        processed_dir = data_dir / "processed"
        save_features_versioned(aligned, processed_dir, features_cfg)

    logger.info("Feature engineering complete")


if __name__ == "__main__":
    main()
