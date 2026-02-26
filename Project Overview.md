# Zhang (2025) Reproduction Project

## Project Overview
Strict reproduction and out-of-sample extension of:
Zhang, Y. (2025). "Interpretable Machine Learning for Macro Alpha: A News Sentiment Case Study." arXiv:2505.16136.

## System Configuration
- **OS:** Windows 11 with WSL2
- **GPU:** NVIDIA GTX 1660 (6GB VRAM)
- **Storage:** 678GB available on C:
- **Docker:** Docker Desktop with WSL2 backend
- **IDE:** Visual Studio Code

## Critical Implementation Rules

### Storage Optimization
- Use stream processing for GDELT (target: 100GB, not 1TB+)
- Partition Parquet files by year/month
- Never store uncompressed GDELT CSV files
- Delete compressed files after processing

### Anti-Lookahead Rules
- Features from day t predict return from t to t+1
- No data shuffling in time series
- Fit scalers only on training data
- No threshold tuning on test data

### Code Quality Standards
- All file paths must use Path() from pathlib
- All scripts must have proper logging with loguru
- All data operations must use pandas/numpy
- All config must come from YAML files
- Type hints required for all functions
- Docstrings required for all classes/functions

## Implementation Phases

### Phase 1: ✅ COMPLETE
Paper dissection and reproducibility plan documented.

### Phase 2: Project Scaffold
Create complete folder structure with Docker configuration.

**Deliverables:**
- docker-compose.yml
- Dockerfile  
- requirements.txt (pinned versions)
- config/config.yaml
- config/hyperparameters.yaml
- src/ module structure with all __init__.py files
- scripts/ for pipeline execution (01-10)
- Makefile for workflow automation
- README.md with setup instructions
- .vscode/tasks.json for VS Code integration
- .gitignore
- .env.example

**Docker Requirements:**
- Base image: nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04
- Python 3.11
- All dependencies from requirements.txt with pinned versions
- Volumes for data/, artifacts/, reports/, logs/
- Working directory: /app
- Support for GPU (optional, fallback to CPU)

**Directory Structure:**
```
zhang2025-reproduction/
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── Makefile
├── README.md
├── CLAUDE.md
├── .env.example
├── .gitignore
├── .vscode/
│   └── tasks.json
├── config/
│   ├── config.yaml
│   └── hyperparameters.yaml
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── gdelt_ingestion.py
│   │   ├── headline_extractor.py
│   │   ├── market_data.py
│   │   └── data_alignment.py
│   ├── features/
│   │   ├── __init__.py
│   │   ├── sentiment_scoring.py
│   │   ├── sentiment_indices.py
│   │   └── feature_engineering.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── baseline.py
│   │   ├── xgboost_model.py
│   │   └── walk_forward.py
│   ├── backtest/
│   │   ├── __init__.py
│   │   ├── strategy.py
│   │   ├── metrics.py
│   │   └── transaction_costs.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── performance_report.py
│   │   └── visualization.py
│   └── utils/
│       ├── __init__.py
│       ├── logging_config.py
│       └── reproducibility.py
├── scripts/
│   ├── 01_ingest_gdelt.py
│   ├── 02_extract_headlines.py
│   ├── 03_score_sentiment.py
│   ├── 04_fetch_market_data.py
│   ├── 05_build_features.py
│   ├── 06_train_baseline.py
│   ├── 07_train_xgboost.py
│   ├── 08_run_backtest.py
│   ├── 09_oos_extension.py
│   └── 10_robustness_tests.py
├── notebooks/
│   ├── exploration.ipynb
│   └── results_analysis.ipynb
├── data/ (Docker volume)
├── artifacts/ (Docker volume)
├── reports/ (Docker volume)
└── logs/ (Docker volume)
```

### Phase 3: Data Layer
Implement optimized data ingestion pipeline.

**Critical for Phase 3:**
- GDELT: Stream processing with Parquet partitioning by year/month
- Headlines: Caching with retry logic, exponential backoff
- FinBERT: Batch size 16 for GTX 1660, handle OOM gracefully
- Alignment: Ensure sentiment(t) → return(t+1), no lookahead

## Current Status
- **Phase 1:** ✅ Complete
- **Phase 2:** ✅ Complete
- **Phase 3:** ✅ Complete
- **Phase 4:** ✅ Complete
- **Phase 5:** ✅ Complete (Backtesting)
- **Phase 6:** ✅ Complete (OOS Extension)
- **Phase 7:** ✅ Complete (Robustness & Sensitivity Tests)

### Phase 3 Implementation Notes
- **GDELT Ingestion**: Stream processing with ThreadPoolExecutor(10), checkpoint/resume, Parquet partitioned by year/month. 96% success rate on test data.
- **Headline Extraction**: URL slug parsing (no web scraping). 88.7% extraction rate. Fallback to GDELT AvgTone for unextractable URLs.
- **Sentiment Scoring**: ProsusAI/finbert with adaptive batch sizing (halve on OOM, grow 25% on success). AvgTone normalized to [-1,1] as fallback.
- **Market Data**: yfinance with direct Yahoo Finance v8 API fallback (handles rate limiting and Docker incompatibility).
- **Data Alignment**: Inner join on date index, forward returns as target, anti-lookahead validation (monotonic index, no duplicates, no NaN gaps).
- **Feature Matrix**: 55 features (sentiment rolling windows + lagged returns + rolling volatility + momentum).

### Phase 4 Implementation Notes
- **Walk-Forward Validation**: Date-based expanding windows with auto-adaptation to available data. 1 fold with test data (train=2020, test=2021).
- **Baseline Model**: LogisticRegression (sklearn) with balanced class weights. Accuracy=46.84%.
- **XGBoost**: binary:logistic objective, early stopping with chronological 20% validation split. Accuracy=51.05%.
- **GPU Detection**: Uses torch.cuda.is_available() (not xgb probe which silently falls back).

### Phase 5 Implementation Notes
- **Strategy**: LongShortStrategy with threshold-based signals (+1 if P(up) > 0.5, -1 otherwise).
- **Transaction Costs**: Turnover-based model with 3 bps cost + 2 bps slippage.
- **Results (test data)**: XGBoost Sharpe=-0.29, LogReg Sharpe=-1.87 (limited data).

### Phase 6 Implementation Notes
- **Frozen Model**: Loads model.joblib + scaler.joblib from last fold, no retraining.
- **Current Status**: no_oos_data (test data ends 2021-11-29, OOS starts 2024-01-01).

### Phase 7 Implementation Notes
- **Cost Sensitivity**: XGBoost zero-cost Sharpe=0.10, break-even at ~0 bps.
- **Threshold Sensitivity**: Optimal threshold=0.55 (Sharpe=1.14) vs paper's 0.50 (Sharpe=-0.29).
- **Model Comparison**: Trade agreement between LogReg and XGBoost=62.87%.
- **Final Summary**: Comprehensive markdown report in reports/robustness/final_reproduction_summary.md.

## Next Action
All phases complete. To improve results, ingest full GDELT 2015-2023 data (~100GB) and re-run the full pipeline for proper multi-fold validation.
