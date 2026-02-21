"""Baseline models: historical mean, OLS, and logistic regression.

Simple benchmarks for comparison with XGBoost.
"""

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.linear_model import LinearRegression, LogisticRegression


class HistoricalMeanModel:
    """Predicts the expanding historical mean return."""

    def __init__(self, window: int | None = None) -> None:
        """Initialize the historical mean model.

        Args:
            window: Rolling window size. None for expanding window.
        """
        self.window = window
        self._mean: float = 0.0

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs: object) -> "HistoricalMeanModel":
        """Fit on training returns.

        Args:
            X: Training features (ignored).
            y: Training target series.
            **kwargs: Ignored.

        Returns:
            Self.
        """
        self._mean = float(y.mean())
        logger.info(f"HistoricalMean fitted: mean={self._mean:.6f}")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict using the historical mean.

        Args:
            X: Feature matrix (used only for length).

        Returns:
            Array of predicted values.
        """
        return np.full(len(X), self._mean)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return mean as probability (clipped to [0, 1]).

        Args:
            X: Feature matrix.

        Returns:
            Array of probabilities.
        """
        prob = np.clip(self._mean, 0.0, 1.0)
        return np.full(len(X), prob)


class LinearModel:
    """OLS linear regression baseline."""

    def __init__(self, fit_intercept: bool = True, **kwargs: object) -> None:
        """Initialize linear model.

        Args:
            fit_intercept: Whether to fit an intercept.
            **kwargs: Ignored.
        """
        self.model = LinearRegression(fit_intercept=fit_intercept)

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs: object) -> "LinearModel":
        """Fit OLS on training data.

        Args:
            X: Training features.
            y: Training target.
            **kwargs: Ignored.

        Returns:
            Self.
        """
        self.model.fit(X, y)
        logger.info(
            f"LinearModel fitted: {len(self.model.coef_)} coefficients, "
            f"intercept={self.model.intercept_:.6f}"
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict returns.

        Args:
            X: Feature matrix.

        Returns:
            Array of predicted returns.
        """
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return predictions clipped to [0, 1] as pseudo-probabilities.

        Args:
            X: Feature matrix.

        Returns:
            Array of pseudo-probabilities.
        """
        return np.clip(self.model.predict(X), 0.0, 1.0)

    def get_coefficients(self) -> pd.Series:
        """Get model coefficients.

        Returns:
            Series of coefficients.
        """
        return pd.Series(self.model.coef_)


class LogisticRegressionModel:
    """Logistic regression classifier for direction prediction."""

    def __init__(
        self,
        C: float = 1.0,
        penalty: str = "l2",
        class_weight: str | None = "balanced",
        solver: str = "lbfgs",
        max_iter: int = 1000,
    ) -> None:
        """Initialize logistic regression model.

        Args:
            C: Inverse regularization strength.
            penalty: Regularization type.
            class_weight: Class weighting strategy.
            solver: Optimization algorithm.
            max_iter: Maximum iterations.
        """
        self.model = LogisticRegression(
            C=C,
            penalty=penalty,
            class_weight=class_weight,
            solver=solver,
            max_iter=max_iter,
        )

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs: object) -> "LogisticRegressionModel":
        """Fit logistic regression on training data.

        Args:
            X: Training features.
            y: Training binary target.
            **kwargs: Ignored.

        Returns:
            Self.
        """
        self.model.fit(X, y)
        logger.info(
            f"LogisticRegression fitted: {X.shape[1]} features, "
            f"classes={self.model.classes_.tolist()}"
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict binary labels.

        Args:
            X: Feature matrix.

        Returns:
            Array of predicted labels (0 or 1).
        """
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probability of positive class.

        Args:
            X: Feature matrix.

        Returns:
            Array of probabilities for positive class.
        """
        return self.model.predict_proba(X)[:, 1]

    def get_coefficients(self) -> pd.Series:
        """Get model coefficients.

        Returns:
            Series of coefficients.
        """
        return pd.Series(self.model.coef_[0])
