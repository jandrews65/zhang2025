"""Step 2: Extract headlines from GDELT source URLs.

Reads GDELT event Parquet files and extracts news headlines
from source article URLs with caching and retry logic.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml
from loguru import logger

from src.data.headline_extractor import extract_headlines_from_parquet
from src.utils.logging_config import setup_logging
from src.utils.reproducibility import set_seed


def main() -> None:
    """Run headline extraction pipeline."""
    config_path = Path("config/config.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    setup_logging(
        log_dir=Path(config["paths"]["logs_dir"]),
        script_name="02_extract_headlines",
    )
    set_seed(config["project"]["seed"])

    data_dir = Path(config["paths"]["data_dir"])
    input_dir = data_dir / "gdelt"
    output_dir = data_dir / "headlines"

    logger.info("Starting headline extraction")
    extract_headlines_from_parquet(input_dir, output_dir, config)
    logger.info("Headline extraction complete")


if __name__ == "__main__":
    main()
