"""Step 1: Ingest GDELT v2 event data.

Downloads GDELT v2 files for the configured date range,
processes them via streaming, and saves as partitioned Parquet.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml
from loguru import logger

from src.data.gdelt_ingestion import ingest_gdelt
from src.utils.logging_config import setup_logging
from src.utils.reproducibility import log_environment, set_seed


def main() -> None:
    """Run GDELT ingestion pipeline."""
    config_path = Path("config/config.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    setup_logging(
        log_dir=Path(config["paths"]["logs_dir"]),
        script_name="01_ingest_gdelt",
    )
    set_seed(config["project"]["seed"])
    log_environment()

    logger.info("Starting GDELT ingestion")
    ingest_gdelt(config)
    logger.info("GDELT ingestion complete")


if __name__ == "__main__":
    main()
