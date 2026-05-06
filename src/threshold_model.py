"""Threshold-aware model wrappers for binary congestion prediction."""

from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.features import FEATURE_COLUMNS, prepare_model_features


class ThresholdedRandomForestClassifier(RandomForestClassifier):
    """RandomForestClassifier with tuned probability thresholding."""

    default_prediction_threshold = 0.5
    feature_columns = FEATURE_COLUMNS

    def _as_feature_frame(self, X) -> pd.DataFrame:
        """Return model features as a DataFrame with stable column order."""
        if isinstance(X, pd.DataFrame):
            return X[self.feature_columns].astype(float).copy()
        return pd.DataFrame(X, columns=self.feature_columns).astype(float)

    def _prepare_features(self, X) -> pd.DataFrame:
        """Apply the same feature transform during training and prediction."""
        return prepare_model_features(self._as_feature_frame(X))

    def fit(self, X, y, sample_weight=None):
        """Fit on the same normalized feature space used by predictions."""
        return super().fit(self._prepare_features(X), y, sample_weight=sample_weight)

    def predict_proba(self, X):
        """Predict probabilities from transformed, production-scale features."""
        return super().predict_proba(self._prepare_features(X))

    def predict(self, X):
        """Predict labels from class-1 probabilities instead of sklearn's implicit threshold."""
        positive_probs = self.predict_proba(X)[:, 1]
        threshold = getattr(self, "prediction_threshold_", self.default_prediction_threshold)
        return (positive_probs > threshold).astype(int)

    def set_prediction_threshold(self, threshold: float) -> None:
        """Persist the selected class-1 threshold on the trained estimator."""
        self.prediction_threshold_ = float(threshold)


class ThresholdedLogisticRegressionClassifier(LogisticRegression):
    """Balanced logistic classifier with production feature scaling and tuned threshold."""

    default_prediction_threshold = 0.5
    feature_columns = FEATURE_COLUMNS

    def _as_feature_frame(self, X) -> pd.DataFrame:
        """Return model features as a DataFrame with stable column order."""
        if isinstance(X, pd.DataFrame):
            return X[self.feature_columns].astype(float).copy()
        return pd.DataFrame(X, columns=self.feature_columns).astype(float)

    def _prepare_features(self, X) -> pd.DataFrame:
        """Apply the same feature transform during training and prediction."""
        return prepare_model_features(self._as_feature_frame(X))

    def fit(self, X, y, sample_weight=None):
        """Fit on monotonic risk features and expose dashboard-compatible importances."""
        fitted = super().fit(self._prepare_features(X), y, sample_weight=sample_weight)
        importance = abs(self.coef_[0])
        total = importance.sum()
        self.feature_importances_ = importance / total if total else importance
        self.feature_names_in_ = pd.Index(self.feature_columns).to_numpy()
        return fitted

    def predict_proba(self, X):
        """Predict probabilities from the same transformed features used in training."""
        return super().predict_proba(self._prepare_features(X))

    def predict(self, X):
        """Predict labels from class-1 probabilities using the tuned threshold."""
        positive_probs = self.predict_proba(X)[:, 1]
        threshold = getattr(self, "prediction_threshold_", self.default_prediction_threshold)
        return (positive_probs > threshold).astype(int)

    def set_prediction_threshold(self, threshold: float) -> None:
        """Persist the selected class-1 threshold on the trained estimator."""
        self.prediction_threshold_ = float(threshold)
