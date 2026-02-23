# Zhang (2025) Reproduction

Strict reproduction and out-of-sample extension of:

> Zhang, Y. (2025). "Interpretable Machine Learning for Macro Alpha: A News Sentiment Case Study." arXiv:2505.16136.

## Setup

### Prerequisites

- Docker Desktop with WSL2 backend
- NVIDIA GPU drivers (optional, for GPU acceleration)
- NVIDIA Container Toolkit (optional, for GPU in Docker)

### Build

```bash
docker compose build
```

### Configuration

Copy the example environment file and edit as needed:

```bash
cp .env.example .env
```

Configuration files are in `config/`:
- `config.yaml` - main pipeline settings
- `hyperparameters.yaml` - model hyperparameters

## Usage

### Run individual pipeline steps (CPU)

```bash
make run-cpu SCRIPT=scripts/01_ingest_gdelt.py
```

### Run individual pipeline steps (GPU)

```bash
make run-gpu SCRIPT=scripts/03_score_sentiment.py
```

### Run full pipeline

```bash
make pipeline
```

## Pipeline Steps

| Step | Script | Description |
|------|--------|-------------|
| 1 | `01_ingest_gdelt.py` | Download and process GDELT v2 event data |
| 2 | `02_extract_headlines.py` | Extract headlines from source URLs |
| 3 | `03_score_sentiment.py` | Score headlines with FinBERT |
| 4 | `04_fetch_market_data.py` | Fetch SPY price data |
| 5 | `05_build_features.py` | Build sentiment feature matrix |
| 6 | `06_train_baseline.py` | Train baseline models |
| 7 | `07_train_xgboost.py` | Train XGBoost with SHAP |
| 8 | `08_run_backtest.py` | Run trading strategy backtest |
| 9 | `09_oos_extension.py` | Out-of-sample extension |
| 10 | `10_robustness_tests.py` | Robustness and sensitivity tests |

## Project Structure

```
zhang2025-reproduction/
├── config/          # YAML configuration files
├── src/             # Source modules
│   ├── data/        # Data ingestion and alignment
│   ├── features/    # Sentiment scoring and feature engineering
│   ├── models/      # Baseline, XGBoost, walk-forward
│   ├── backtest/    # Strategy and performance metrics
│   ├── evaluation/  # Reports and visualization
│   └── utils/       # Logging and reproducibility
├── scripts/         # Pipeline execution scripts (01-10)
├── notebooks/       # Exploration and analysis notebooks
├── data/            # Raw and processed data (git-ignored)
├── artifacts/       # Trained models and outputs (git-ignored)
├── reports/         # Generated reports (git-ignored)
└── logs/            # Pipeline logs (git-ignored)
```
# zhang2025
