"""Step 7: Train XGBoost model.

Trains the XGBoost classifier using walk-forward validation
with SHAP-based interpretation.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import pandas as pd
import yaml
from loguru import logger

from src.models.walk_forward import WalkForwardValidator
from src.models.xgboost_model import XGBoostReturnPredictor
from src.utils.logging_config import setup_logging
from src.utils.reproducibility import set_seed


def main() -> None:
    """Run XGBoost model training."""
    config_path = Path("config/config.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    hp_path = Path("config/hyperparameters.yaml")
    with open(hp_path) as f:
        hyperparams = yaml.safe_load(f)

    setup_logging(
        log_dir=Path(config["paths"]["logs_dir"]),
        script_name="07_train_xgboost",
    )
    set_seed(config["project"]["seed"])

    # Load features
    features_path = Path(config["paths"]["data_dir"]) / "processed" / "features_latest.parquet"
    if not features_path.exists():
        logger.error(f"Features file not found: {features_path}")
        sys.exit(1)

    df = pd.read_parquet(features_path)
    logger.info(f"Loaded features: {df.shape[0]} rows, {df.shape[1]} columns")
    logger.info(f"Date range: {df.index[0]} to {df.index[-1]}")

    # Separate features and target
    target_col = "target"
    if target_col not in df.columns:
        logger.error(f"Target column '{target_col}' not found in features")
        sys.exit(1)

    y = df[target_col]
    X = df.drop(columns=[target_col])

    # Drop rows with NaN target
    valid_mask = y.notna()
    X = X.loc[valid_mask]
    y = y.loc[valid_mask]
    logger.info(f"After dropping NaN targets: {len(X)} rows")

    # Initialize walk-forward validator
    wf_config = config["walk_forward"]
    validator = WalkForwardValidator(
        initial_train_end=wf_config["initial_train_end"],
        test_fold_ends=wf_config["test_fold_ends"],
        expanding=wf_config["expanding"],
        task_type=wf_config["task_type"],
        classification_threshold=wf_config["classification_threshold"],
    )

    # XGBoost uses hyperparams dict, wrapped in a single-key dict for the constructor
    artifacts_dir = Path(config["paths"]["artifacts_dir"]) / "models" / "xgboost"
    xgb_params = dict(hyperparams["xgboost"])

    logger.info("Running walk-forward validation with XGBoost")
    results = validator.run(
        model_class=XGBoostReturnPredictor,
        model_params={"hyperparams": xgb_params},
        X=X,
        y=y,
        artifacts_dir=artifacts_dir,
    )

    if not results:
        logger.error("No results from walk-forward validation")
        sys.exit(1)

    # Compute SHAP on last fold (non-fatal if it fails)
    try:
        last_fold = results[-1]
        last_fold_dir = artifacts_dir / f"fold_{last_fold['fold']}"
        last_model = joblib.load(last_fold_dir / "model.joblib")

        # Re-load and scale the test data for the last fold
        test_dates = last_fold["dates"]
        X_last_test = X.loc[test_dates]
        last_scaler = joblib.load(last_fold_dir / "scaler.joblib")
        X_last_test_scaled = pd.DataFrame(
            last_scaler.transform(X_last_test),
            index=X_last_test.index,
            columns=X_last_test.columns,
        )

        shap_values = last_model.get_shap_values(X_last_test_scaled)

        shap_dir = artifacts_dir / "shap"
        shap_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(shap_values, shap_dir / "shap_values_last_fold.joblib")
        logger.info(f"SHAP values saved to {shap_dir}")
    except Exception as e:
        logger.warning(f"SHAP computation failed (non-fatal): {e}")

    # Save consolidated predictions
    all_preds = []
    for r in results:
        fold_df = pd.DataFrame(
            {
                "date": r["dates"],
                "y_true": r["y_true"].values,
                "y_pred": r["y_pred"],
                "y_proba": r["y_proba"] if r["y_proba"] is not None else float("nan"),
                "fold": r["fold"],
            }
        )
        all_preds.append(fold_df)

    preds_df = pd.concat(all_preds, ignore_index=True)
    preds_path = artifacts_dir / "walk_forward_results.parquet"
    preds_df.to_parquet(preds_path, index=False)
    logger.info(f"Saved consolidated predictions: {preds_path} ({len(preds_df)} rows)")

    # Feature importance from last fold
    try:
        importance = last_model.get_feature_importance("gain")
        logger.info("=== Top 10 Features (gain) ===")
        for feat, val in importance.head(10).items():
            logger.info(f"  {feat}: {val:.4f}")
    except Exception as e:
        logger.warning(f"Feature importance extraction failed: {e}")

    # Summary
    logger.info("=== XGBoost Training Summary ===")
    for r in results:
        m = r["metrics"]
        logger.info(
            f"  Fold {r['fold']}: "
            f"acc={m['accuracy']:.4f}, prec={m['precision']:.4f}, "
            f"rec={m['recall']:.4f}, auc={m.get('roc_auc', 'N/A')}"
        )

    logger.info("XGBoost training complete")


if __name__ == "__main__":
    main()
