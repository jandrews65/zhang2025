"""XGBoost model for return/direction prediction.

Implements the main XGBoost model from Zhang (2025)
with SHAP-based interpretation and GPU fallback.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from loguru import logger


class XGBoostReturnPredictor:
    """XGBoost model for predicting equity returns/direction from sentiment features."""

    def __init__(self, hyperparams: dict) -> None:
        """Initialize XGBoost predictor.

        Auto-detects classifier vs regressor from the objective parameter.
        Falls back from GPU to CPU if CUDA is unavailable.

        Args:
            hyperparams: Dictionary of XGBoost hyperparameters.
        """
        self.hyperparams = dict(hyperparams)
        self._is_classifier = "logistic" in self.hyperparams.get("objective", "")
        self._val_fraction = self.hyperparams.pop("val_fraction", 0.2)
        self._early_stopping = self.hyperparams.pop("early_stopping_rounds", 50)

        # Resolve device (GPU fallback)
        self._resolve_device()

        self.model = None

    def _resolve_device(self) -> None:
        """Auto-detect GPU availability and adjust tree_method/device."""
        requested_device = self.hyperparams.pop("device", "cpu")

        if requested_device == "cuda":
            gpu_available = False
            try:
                import torch
                gpu_available = torch.cuda.is_available()
            except ImportError:
                pass

            if gpu_available:
                self.hyperparams["device"] = "cuda"
                logger.info("XGBoost: using CUDA GPU")
            else:
                self.hyperparams.pop("device", None)
                logger.info("XGBoost: CUDA unavailable, falling back to CPU")
        else:
            self.hyperparams.pop("device", None)

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
        **kwargs: object,
    ) -> "XGBoostReturnPredictor":
        """Train the XGBoost model.

        If no explicit validation set is provided, carves the last val_fraction
        of training data chronologically for early stopping.

        Args:
            X_train: Training features.
            y_train: Training target.
            X_val: Validation features for early stopping.
            y_val: Validation target for early stopping.
            **kwargs: Extra arguments (ignored).

        Returns:
            Self.
        """
        import xgboost as xgb

        # Carve validation from training if not provided (chronological split)
        if X_val is None or y_val is None:
            n = len(X_train)
            val_size = max(int(n * self._val_fraction), 1)
            split_idx = n - val_size

            X_val = X_train.iloc[split_idx:]
            y_val = y_train.iloc[split_idx:]
            X_train = X_train.iloc[:split_idx]
            y_train = y_train.iloc[:split_idx]

            logger.info(
                f"Carved validation set: train={len(X_train)}, val={len(X_val)}"
            )

        # Build params for xgb model
        params = dict(self.hyperparams)
        n_estimators = params.pop("n_estimators", 500)
        random_state = params.pop("random_state", 42)

        if self._is_classifier:
            self.model = xgb.XGBClassifier(
                n_estimators=n_estimators,
                random_state=random_state,
                early_stopping_rounds=self._early_stopping,
                **params,
            )
        else:
            self.model = xgb.XGBRegressor(
                n_estimators=n_estimators,
                random_state=random_state,
                early_stopping_rounds=self._early_stopping,
                **params,
            )

        self.model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        best_iteration = getattr(self.model, "best_iteration", n_estimators)
        logger.info(
            f"XGBoost fitted: best_iteration={best_iteration}, "
            f"n_features={X_train.shape[1]}"
        )

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions (binary labels for classifier, values for regressor).

        Args:
            X: Feature matrix.

        Returns:
            Array of predictions.
        """
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probability of positive class (classifier only).

        Args:
            X: Feature matrix.

        Returns:
            Array of probabilities for positive class.
        """
        if self._is_classifier:
            return self.model.predict_proba(X)[:, 1]
        # For regressor, clip predictions as pseudo-probabilities
        return np.clip(self.model.predict(X), 0.0, 1.0)

    def get_feature_importance(
        self, importance_type: str = "gain"
    ) -> pd.Series:
        """Get feature importance from the booster.

        Args:
            importance_type: Type of importance ("gain", "weight", "cover").

        Returns:
            Named Series of feature importances.
        """
        booster = self.model.get_booster()
        scores = booster.get_score(importance_type=importance_type)
        return pd.Series(scores).sort_values(ascending=False)

    def get_shap_values(self, X: pd.DataFrame) -> np.ndarray:
        """Compute SHAP values for interpretability.

        Args:
            X: Feature matrix.

        Returns:
            SHAP values array (n_samples x n_features).
        """
        import shap

        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X)
        logger.info(f"SHAP values computed: shape={np.array(shap_values).shape}")
        return shap_values

    def save(self, path: Path) -> None:
        """Save model and hyperparameters to disk.

        Args:
            path: Directory path to save the model.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.model, path / "model.joblib")
        with open(path / "hyperparams.json", "w") as f:
            json.dump(self.hyperparams, f, indent=2)

        logger.info(f"XGBoost model saved to {path}")

    @classmethod
    def load(cls, path: Path) -> "XGBoostReturnPredictor":
        """Load model from disk.

        Args:
            path: Directory path to load the model from.

        Returns:
            Loaded model instance.
        """
        path = Path(path)

        with open(path / "hyperparams.json") as f:
            hyperparams = json.load(f)

        instance = cls.__new__(cls)
        instance.hyperparams = hyperparams
        instance._is_classifier = "logistic" in hyperparams.get("objective", "")
        instance._val_fraction = 0.2
        instance._early_stopping = 50
        instance.model = joblib.load(path / "model.joblib")

        logger.info(f"XGBoost model loaded from {path}")
        return instance
