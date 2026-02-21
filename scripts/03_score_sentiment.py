"""Step 3: Score headlines with FinBERT.

Loads extracted headlines and computes sentiment scores
using ProsusAI/finbert with GPU-optimized batching.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml
from loguru import logger

from src.features.sentiment_scoring import score_all_headlines
from src.utils.logging_config import setup_logging
from src.utils.reproducibility import set_seed


def main() -> None:
    """Run sentiment scoring pipeline."""
    config_path = Path("config/config.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    setup_logging(
        log_dir=Path(config["paths"]["logs_dir"]),
        script_name="03_score_sentiment",
    )
    set_seed(config["project"]["seed"])

    data_dir = Path(config["paths"]["data_dir"])
    input_path = data_dir / "headlines"
    output_path = data_dir / "sentiment"

    logger.info("Starting sentiment scoring")
    score_all_headlines(input_path, output_path, config)
    logger.info("Sentiment scoring complete")


if __name__ == "__main__":
    main()
