# Zhang (2025) Reproduction: Interpretable ML for Macro Alpha

Strict reproduction and out-of-sample extension of:

**Zhang, Y. (2025). "Interpretable Machine Learning for Macro Alpha: A News Sentiment Case Study." arXiv:2505.16136.**

## Overview

This repository contains a complete, containerized reproduction framework for Zhang's paper on using news sentiment to predict FX returns.

### Key Features

- **Stream-optimized GDELT ingestion** (~100GB vs 1TB+ raw storage)
- **GPU-accelerated FinBERT** sentiment scoring
- **Walk-forward validation** with proper time series handling
- **XGBoost + baseline models** for comparison
- **Complete backtesting engine** with transaction costs
- **Robustness analysis** (cost sensitivity, threshold optimization)

## Quick Start

### Prerequisites

- Windows 11 with WSL2
- Docker Desktop
- NVIDIA GPU (optional, but recommended)
- 250GB free disk space

### Installation

see reproduction.md

### Run Full Pipeline

\\\bash
# Enter container
docker exec -it zhang2025 bash

# Run all steps
python scripts/01_ingest_gdelt.py        # ~2 hours
python scripts/02_extract_headlines.py    # ~3.5 hours
python scripts/03_score_sentiment.py      # ~90 hours (GPU)
python scripts/04_fetch_market_data.py    # ~5 minutes
python scripts/05_build_features.py       # ~30 minutes
python scripts/06_train_baseline.py       # ~30 minutes
python scripts/07_train_xgboost.py        # ~2 hours
python scripts/08_run_backtest.py         # ~10 minutes
python scripts/09_oos_extension.py        # ~5 minutes
python scripts/10_robustness_tests.py     # ~30 minutes
\\\

## Project Structure

\\\
zhang2025-reproduction/
├── config/              # YAML configurations
├── src/                 # Python modules
│   ├── data/           # Data ingestion
│   ├── features/       # Feature engineering
│   ├── models/         # Model training
│   ├── backtest/       # Backtesting engine
│   └── evaluation/     # Performance analysis
├── scripts/            # Pipeline scripts (01-10)
├── Dockerfile          # Container definition
├── docker-compose.yml  # Service configuration
└── requirements.txt    # Pinned dependencies
\\\

## Results

### Paper's Reported Performance
- EUR/USD Sharpe: **5.87**
- USD/JPY Sharpe: **4.65**
- CAGR: **>50%**

### Our Reproduction
*(Results pending final analysis)*

## Development Timeline

- **Phase 1-2:** Project scaffold (27 min)
- **Phase 3:** Data layer (27 min)
- **Phase 4:** Feature engineering (15 min)
- **Phase 5:** Model training (completed)
- **Phase 6:** Backtesting (completed)
- **Phase 7:** OOS extension (completed)
- **Phase 8:** Robustness tests (completed)

**Total development time: ~3 hours** (using Claude Code)

**Data ingestion: ~96 hours** (mostly unattended)

## Citation

\\\ibtex
@article{zhang2025interpretable,
  title={Interpretable Machine Learning for Macro Alpha: A News Sentiment Case Study},
  author={Zhang, Y.},
  journal={arXiv preprint arXiv:2505.16136},
  year={2025}
}
\\\

## License

MIT License (code only). Paper content © JP Andrews (2026).

## Acknowledgments

Built with [Claude Code](https://claude.ai/code) for autonomous development.


