"""Walk-forward validation framework.

Implements date-based expanding-window time-series cross-validation
to prevent lookahead bias in model evaluation.
"""

import json
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler


class WalkForwardValidator:
    """Date-based expanding-window walk-forward validator."""

    def __init__(
        self,
        initial_train_end: str,
        test_fold_ends: list[str],
        expanding: bool = True,
        task_type: str = "classification",
        classification_threshold: float = 0.0,
    ) -> None:
        """Initialize walk-forward validator.

        Args:
            initial_train_end: Last date (inclusive) for the initial training period.
            test_fold_ends: Ordered list of test fold end dates.
            expanding: If True, training window expands; else not supported yet.
            task_type: "classification" or "regression".
            classification_threshold: Return threshold for binarizing target.
        """
        self.initial_train_end = pd.Timestamp(initial_train_end)
        self.test_fold_ends = [pd.Timestamp(d) for d in test_fold_ends]
        self.expanding = expanding
        self.task_type = task_type
        self.classification_threshold = classification_threshold

    def split(
        self, X: pd.DataFrame
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Generate train/test index splits from date boundaries.

        Auto-skips folds with no data. Falls back to 70/30 split
        if no valid folds can be constructed.

        Args:
            X: Feature matrix with DatetimeIndex (must be sorted).

        Returns:
            List of (train_indices, test_indices) as integer position arrays.
        """
        idx = X.index
        folds: list[tuple[np.ndarray, np.ndarray]] = []

        # Build fold boundaries: each fold's test period runs from
        # the previous fold end (exclusive) to this fold end (inclusive).
        boundaries = [self.initial_train_end] + self.test_fold_ends

        for i in range(1, len(boundaries)):
            test_start = boundaries[i - 1]  # train ends here (inclusive)
            test_end = boundaries[i]

            train_mask = idx <= test_start
            test_mask = (idx > test_start) & (idx <= test_end)

            train_idx = np.where(train_mask)[0]
            test_idx = np.where(test_mask)[0]

            if len(train_idx) == 0 or len(test_idx) == 0:
                logger.warning(
                    f"Skipping fold {i}: train={len(train_idx)}, "
                    f"test={len(test_idx)} (test_start={test_start.date()}, "
                    f"test_end={test_end.date()})"
                )
                continue

            folds.append((train_idx, test_idx))
            logger.info(
                f"Fold {len(folds)-1}: train={len(train_idx)} "
                f"[{idx[train_idx[0]].date()}..{idx[train_idx[-1]].date()}], "
                f"test={len(test_idx)} "
                f"[{idx[test_idx[0]].date()}..{idx[test_idx[-1]].date()}]"
            )

        # Fallback: 70/30 split if no valid folds
        if len(folds) == 0:
            n = len(X)
            split_point = int(n * 0.7)
            if split_point > 0 and split_point < n:
                train_idx = np.arange(split_point)
                test_idx = np.arange(split_point, n)
                folds.append((train_idx, test_idx))
                logger.warning(
                    f"No date-based folds possible. "
                    f"Falling back to 70/30 split: train={split_point}, test={n - split_point}"
                )
            else:
                logger.error("Not enough data for any split")

        return folds

    def run(
        self,
        model_class: type,
        model_params: dict,
        X: pd.DataFrame,
        y: pd.Series,
        artifacts_dir: Path,
        fit_kwargs_fn: Callable[..., dict] | None = None,
    ) -> list[dict[str, Any]]:
        """Run walk-forward validation.

        Per fold: binarize target (if classification), fit scaler on train only,
        instantiate and fit model, predict, compute metrics, save artifacts.

        Args:
            model_class: Model class to instantiate per fold.
            model_params: Parameters for model __init__.
            X: Full feature matrix with DatetimeIndex.
            y: Full target series (continuous returns).
            artifacts_dir: Directory to save per-fold artifacts.
            fit_kwargs_fn: Optional callable(X_train, y_train, X_test, y_test)
                returning extra kwargs for model.fit().

        Returns:
            List of result dicts per fold with predictions and metrics.
        """
        folds = self.split(X)
        if not folds:
            logger.error("No folds generated — cannot run walk-forward")
            return []

        artifacts_dir.mkdir(parents=True, exist_ok=True)
        all_results: list[dict[str, Any]] = []

        for fold_i, (train_idx, test_idx) in enumerate(folds):
            logger.info(f"=== Fold {fold_i} ===")
            fold_dir = artifacts_dir / f"fold_{fold_i}"
            fold_dir.mkdir(parents=True, exist_ok=True)

            # Split data
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train_raw, y_test_raw = y.iloc[train_idx], y.iloc[test_idx]

            # Binarize target for classification
            if self.task_type == "classification":
                y_train = (y_train_raw > self.classification_threshold).astype(int)
                y_test = (y_test_raw > self.classification_threshold).astype(int)
                logger.info(
                    f"  Target binarized: train pos_rate={y_train.mean():.3f}, "
                    f"test pos_rate={y_test.mean():.3f}"
                )
            else:
                y_train, y_test = y_train_raw, y_test_raw

            # Fit scaler on training data only (anti-lookahead)
            scaler = StandardScaler()
            X_train_scaled = pd.DataFrame(
                scaler.fit_transform(X_train),
                index=X_train.index,
                columns=X_train.columns,
            )
            X_test_scaled = pd.DataFrame(
                scaler.transform(X_test),
                index=X_test.index,
                columns=X_test.columns,
            )

            # Instantiate model
            model = model_class(**model_params)

            # Build fit kwargs
            fit_kwargs: dict[str, Any] = {}
            if fit_kwargs_fn is not None:
                fit_kwargs = fit_kwargs_fn(
                    X_train_scaled, y_train, X_test_scaled, y_test
                )

            # Fit
            model.fit(X_train_scaled, y_train, **fit_kwargs)

            # Predict
            y_pred = model.predict(X_test_scaled)
            y_proba = (
                model.predict_proba(X_test_scaled)
                if hasattr(model, "predict_proba")
                else None
            )

            # Compute metrics
            metrics = {}
            if self.task_type == "classification":
                metrics = self._compute_classification_metrics(
                    y_test, y_pred, y_proba
                )
                logger.info(
                    f"  Metrics: acc={metrics['accuracy']:.4f}, "
                    f"prec={metrics['precision']:.4f}, "
                    f"rec={metrics['recall']:.4f}, "
                    f"auc={metrics.get('roc_auc', 'N/A')}"
                )

            # Save artifacts
            joblib.dump(model, fold_dir / "model.joblib")
            joblib.dump(scaler, fold_dir / "scaler.joblib")

            metadata = {
                "fold": fold_i,
                "train_start": str(X_train.index[0].date()),
                "train_end": str(X_train.index[-1].date()),
                "test_start": str(X_test.index[0].date()),
                "test_end": str(X_test.index[-1].date()),
                "train_size": len(X_train),
                "test_size": len(X_test),
                "metrics": metrics,
            }
            with open(fold_dir / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)

            # Collect results
            result = {
                "fold": fold_i,
                "dates": X_test.index,
                "y_true": y_test,
                "y_pred": y_pred,
                "y_proba": y_proba,
                "metrics": metrics,
                "metadata": metadata,
            }
            all_results.append(result)

        return all_results

    @staticmethod
    def _compute_classification_metrics(
        y_true: pd.Series,
        y_pred: np.ndarray,
        y_proba: np.ndarray | None,
    ) -> dict[str, float]:
        """Compute classification metrics.

        Args:
            y_true: True binary labels.
            y_pred: Predicted binary labels.
            y_proba: Predicted probabilities for positive class.

        Returns:
            Dict with accuracy, precision, recall, roc_auc.
        """
        metrics: dict[str, float] = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        }

        if y_proba is not None and len(np.unique(y_true)) > 1:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        else:
            metrics["roc_auc"] = float("nan")

        return metrics
