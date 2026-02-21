"""Feature engineering pipeline.

Combines sentiment indices with market-derived features
into the final feature matrix for model training.
"""

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from loguru import logger


def add_lagged_sentiment_features(
    df: pd.DataFrame,
    columns: list[str],
    lags: list[int],
) -> pd.DataFrame:
    """Add lagged sentiment features using shift (no lookahead).

    Args:
        df: DataFrame containing sentiment columns.
        columns: Column names to lag.
        lags: Number of days to lag (must be >= 1).

    Returns:
        DataFrame with lagged features appended.
    """
    result = df.copy()
    added = 0
    for col in columns:
        if col not in result.columns:
            logger.warning(f"Column '{col}' not found for lagging — skipping")
            continue
        for lag in lags:
            result[f"{col}_lag{lag}"] = result[col].shift(lag)
            added += 1
    logger.info(f"Lagged sentiment: added {added} features (lags={lags})")
    return result


def add_acceleration_features(
    df: pd.DataFrame,
    base_columns: list[str],
    windows: list[int],
) -> pd.DataFrame:
    """Add acceleration (first-difference) of rolling moving averages.

    For each base column and window, computes diff(1) of the MA column,
    capturing the rate-of-change of the moving average.

    Args:
        df: DataFrame containing `{col}_ma{window}` columns.
        base_columns: Base column names (e.g., 'mean_sentiment').
        windows: MA window sizes to compute acceleration for.

    Returns:
        DataFrame with acceleration features appended.
    """
    result = df.copy()
    added = 0
    for col in base_columns:
        for w in windows:
            ma_col = f"{col}_ma{w}"
            if ma_col not in result.columns:
                logger.warning(f"Column '{ma_col}' not found for acceleration — skipping")
                continue
            result[f"{col}_ma{w}_accel"] = result[ma_col].diff(1)
            added += 1
    logger.info(f"Acceleration features: added {added} features (windows={windows})")
    return result


def log_feature_statistics(
    df: pd.DataFrame,
    target_col: str = "target",
) -> pd.DataFrame:
    """Log feature statistics and correlations with target.

    Args:
        df: Aligned DataFrame with features and target column.
        target_col: Name of the target column.

    Returns:
        DataFrame of feature statistics (describe + correlation).
    """
    feature_cols = [c for c in df.columns if c != target_col]
    stats = df[feature_cols].describe().T

    if target_col in df.columns:
        corr = df[feature_cols].corrwith(df[target_col])
        stats["corr_with_target"] = corr

        # Log top and bottom correlations
        sorted_corr = corr.dropna().sort_values()
        n_show = min(10, len(sorted_corr))

        logger.info(f"Feature statistics: {len(feature_cols)} features")
        logger.info(f"Top {n_show} positive correlations with target:")
        for name, val in sorted_corr.tail(n_show).items():
            logger.info(f"  {name}: {val:.4f}")
        logger.info(f"Top {n_show} negative correlations with target:")
        for name, val in sorted_corr.head(n_show).items():
            logger.info(f"  {name}: {val:.4f}")
    else:
        logger.warning(f"Target column '{target_col}' not found — skipping correlations")

    return stats


def compute_config_hash(config_dict: dict) -> str:
    """Compute an 8-character SHA-256 hash of a config dictionary.

    Args:
        config_dict: Configuration dictionary to hash.

    Returns:
        First 8 characters of the hex digest.
    """
    serialized = json.dumps(config_dict, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode()).hexdigest()[:8]


def save_features_versioned(
    features: pd.DataFrame,
    output_dir: Path,
    config_dict: dict,
) -> Path:
    """Save features with versioned filename and JSON metadata sidecar.

    Creates:
      - `features_v{hash}_{timestamp}.parquet`
      - `features_v{hash}_{timestamp}.json` (metadata)
      - `features_latest.parquet` (copy for convenience)

    Args:
        features: Feature matrix DataFrame.
        output_dir: Directory for versioned output.
        config_dict: Config dict used for hashing and metadata.

    Returns:
        Path to the versioned parquet file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config_hash = compute_config_hash(config_dict)
    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    base_name = f"features_v{config_hash}_{timestamp}"

    parquet_path = output_dir / f"{base_name}.parquet"
    features.to_parquet(parquet_path, engine="pyarrow", compression="snappy")

    metadata = {
        "config_hash": config_hash,
        "timestamp_utc": timestamp,
        "n_rows": len(features),
        "n_columns": len(features.columns),
        "columns": list(features.columns),
        "date_range": {
            "start": str(features.index.min().date()),
            "end": str(features.index.max().date()),
        },
        "config": config_dict,
    }
    meta_path = output_dir / f"{base_name}.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    latest_path = output_dir / "features_latest.parquet"
    shutil.copy2(parquet_path, latest_path)

    size_mb = parquet_path.stat().st_size / 1e6
    logger.info(
        f"Saved versioned features to {parquet_path} "
        f"({size_mb:.2f} MB, {len(features.columns)} cols) + metadata JSON"
    )
    return parquet_path


def build_feature_matrix(
    sentiment_df: pd.DataFrame,
    market_df: pd.DataFrame,
    rolling_windows: list[int],
    *,
    sentiment_lag_days: list[int] | None = None,
    sentiment_lag_columns: list[str] | None = None,
    acceleration_windows: list[int] | None = None,
    market_vol_windows: list[int] | None = None,
) -> pd.DataFrame:
    """Build the complete feature matrix.

    Merges daily sentiment features with market-derived features
    (lagged returns, rolling volatility).

    Args:
        sentiment_df: Daily sentiment features indexed by date.
        market_df: Market data with 'Adj Close' and 'forward_return', indexed by date.
        rolling_windows: Windows for rolling market features.
        sentiment_lag_days: Lags for lagged sentiment features (e.g., [1, 2, 3, 5]).
        sentiment_lag_columns: Columns to apply lagged sentiment to.
        acceleration_windows: MA windows to compute acceleration for.
        market_vol_windows: Windows for rolling volatility (default [21, 63]).

    Returns:
        Feature matrix DataFrame indexed by date.
    """
    # Add lagged sentiment features if configured
    if sentiment_lag_days and sentiment_lag_columns:
        sentiment_df = add_lagged_sentiment_features(
            sentiment_df, sentiment_lag_columns, sentiment_lag_days,
        )

    # Add acceleration features if configured
    if acceleration_windows:
        accel_cols = ["mean_sentiment", "positive_ratio", "negative_ratio", "mean_avg_tone"]
        sentiment_df = add_acceleration_features(
            sentiment_df, accel_cols, acceleration_windows,
        )

    # Compute lagged returns from market data
    adj_close = market_df["Adj Close"]
    daily_return = adj_close.pct_change()

    market_features = pd.DataFrame(index=market_df.index)
    for lag in [1, 5, 10]:
        market_features[f"lagged_return_{lag}"] = daily_return.shift(lag)

    # Rolling volatility — use configured windows or default
    vol_windows = market_vol_windows if market_vol_windows else [21, 63]
    for window in vol_windows:
        market_features[f"rolling_vol_{window}"] = (
            daily_return.rolling(window=window, min_periods=1).std()
        )

    # Rolling return momentum
    for window in [5, 21]:
        market_features[f"momentum_{window}"] = (
            adj_close / adj_close.shift(window) - 1.0
        )

    # Merge sentiment and market features on date (inner join)
    combined = sentiment_df.join(market_features, how="inner")

    # Drop rows with NaN from lagged features
    n_before = len(combined)
    combined = combined.dropna()
    n_dropped = n_before - len(combined)

    if n_dropped > 0:
        logger.info(f"Dropped {n_dropped} rows with NaN (from lagged features)")

    if combined.empty:
        logger.warning("Feature matrix is empty after dropping NaN — not enough overlapping data")
        return combined

    logger.info(
        f"Feature matrix: {len(combined)} rows, {len(combined.columns)} features, "
        f"date range {combined.index[0].date()} to {combined.index[-1].date()}"
    )
    return combined


def save_features(features: pd.DataFrame, output_path: Path) -> None:
    """Save feature matrix to Parquet.

    Args:
        features: Feature matrix DataFrame.
        output_path: Output file path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(output_path, engine="pyarrow", compression="snappy")
    size_mb = output_path.stat().st_size / 1e6
    logger.info(f"Saved features to {output_path} ({size_mb:.2f} MB, {len(features.columns)} cols)")
