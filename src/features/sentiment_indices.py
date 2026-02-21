"""Sentiment index construction from scored headlines.

Aggregates headline-level sentiment scores into daily sentiment indices
and computes rolling window features.
"""

import pandas as pd
from loguru import logger


def compute_daily_sentiment(
    scored_headlines: pd.DataFrame,
    date_column: str = "date",
) -> pd.DataFrame:
    """Aggregate headline scores into daily sentiment measures.

    Computes per-day: mean/std sentiment, positive/negative/neutral ratios,
    headline count, and mean GDELT AvgTone.

    Args:
        scored_headlines: DataFrame with sentiment scores per headline.
        date_column: Name of the date column.

    Returns:
        DataFrame with daily aggregated sentiment indices, indexed by date.
    """
    df = scored_headlines.copy()
    df[date_column] = pd.to_datetime(df[date_column])

    grouped = df.groupby(date_column)

    daily = pd.DataFrame({
        "mean_sentiment": grouped["sentiment_score"].mean(),
        "std_sentiment": grouped["sentiment_score"].std().fillna(0),
        "median_sentiment": grouped["sentiment_score"].median(),
        "positive_ratio": grouped["positive"].apply(
            lambda x: (x > 0.5).mean() if x.notna().any() else 0.0
        ),
        "negative_ratio": grouped["negative"].apply(
            lambda x: (x > 0.5).mean() if x.notna().any() else 0.0
        ),
        "neutral_ratio": grouped["neutral"].apply(
            lambda x: (x > 0.5).mean() if x.notna().any() else 0.0
        ),
        "num_headlines": grouped.size(),
        "mean_avg_tone": grouped["avg_tone"].mean(),
    })

    daily.index.name = "date"
    logger.info(
        f"Daily sentiment: {len(daily)} days, "
        f"mean sentiment {daily['mean_sentiment'].mean():.4f}"
    )
    return daily


def compute_rolling_sentiment(
    daily_sentiment: pd.DataFrame,
    windows: list[int],
) -> pd.DataFrame:
    """Compute rolling window sentiment features.

    For each window, computes rolling mean and std of key sentiment metrics.
    Uses min_periods=1 to handle the start of the series.

    Args:
        daily_sentiment: DataFrame of daily sentiment values.
        windows: List of rolling window sizes in days.

    Returns:
        DataFrame with original + rolling sentiment features.
    """
    result = daily_sentiment.copy()

    base_cols = ["mean_sentiment", "positive_ratio", "negative_ratio", "mean_avg_tone"]

    for window in windows:
        for col in base_cols:
            if col not in result.columns:
                continue
            result[f"{col}_ma{window}"] = (
                result[col].rolling(window=window, min_periods=1).mean()
            )
            result[f"{col}_std{window}"] = (
                result[col].rolling(window=window, min_periods=1).std().fillna(0)
            )

    n_features = len(result.columns) - len(daily_sentiment.columns)
    logger.info(f"Rolling features: added {n_features} features for windows {windows}")
    return result
