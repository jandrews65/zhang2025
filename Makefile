.PHONY: build run-gpu run-cpu pipeline clean help

DOCKER_IMAGE = zhang2025-reproduction:latest
GPU_SERVICE = zhang2025
CPU_SERVICE = zhang2025-cpu

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

build: ## Build Docker image
	docker compose build

run-gpu: ## Run a script with GPU (usage: make run-gpu SCRIPT=scripts/01_ingest_gdelt.py)
	docker compose run --rm $(GPU_SERVICE) $(SCRIPT)

run-cpu: ## Run a script on CPU only (usage: make run-cpu SCRIPT=scripts/01_ingest_gdelt.py)
	docker compose run --rm --profile cpu $(CPU_SERVICE) $(SCRIPT)

pipeline: ## Run full pipeline (CPU)
	docker compose run --rm --profile cpu $(CPU_SERVICE) scripts/01_ingest_gdelt.py
	docker compose run --rm --profile cpu $(CPU_SERVICE) scripts/02_extract_headlines.py
	docker compose run --rm --profile cpu $(CPU_SERVICE) scripts/03_score_sentiment.py
	docker compose run --rm --profile cpu $(CPU_SERVICE) scripts/04_fetch_market_data.py
	docker compose run --rm --profile cpu $(CPU_SERVICE) scripts/05_build_features.py
	docker compose run --rm --profile cpu $(CPU_SERVICE) scripts/06_train_baseline.py
	docker compose run --rm --profile cpu $(CPU_SERVICE) scripts/07_train_xgboost.py
	docker compose run --rm --profile cpu $(CPU_SERVICE) scripts/08_run_backtest.py
	docker compose run --rm --profile cpu $(CPU_SERVICE) scripts/09_oos_extension.py
	docker compose run --rm --profile cpu $(CPU_SERVICE) scripts/10_robustness_tests.py

clean: ## Remove containers and dangling images
	docker compose down --remove-orphans
	docker image prune -f
