"""Mutual Information Feature Selector for Quantum Machine Learning.

Selects top k most informative features using non-linear mutual information
and scales selected features to [-pi, pi] for quantum gate rotations.
Preserves original clinical feature names for explainability.
"""

from typing import Optional, List
import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin


class MISelector(BaseEstimator, TransformerMixin):
    """Mutual Information feature selector for quantum input dimensionality reduction."""

    def __init__(
        self,
        k: int = 4,
        scale_to_pi: bool = True,
        random_state: int = 42,
    ):
        self.k = k
        self.scale_to_pi = scale_to_pi
        self.random_state = random_state

        self.scores_ = None
        self.selected_indices_ = None
        self.selected_feature_names_: List[str] = []
        self.scaler_ = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> "MISelector":
        """Computes mutual information between features and target labels."""
        n_features = X.shape[1]
        actual_k = min(self.k, n_features)

        self.scores_ = mutual_info_classif(X, y, random_state=self.random_state)
        # Select indices of top k scores
        self.selected_indices_ = np.argsort(self.scores_)[::-1][:actual_k]

        if feature_names is not None and len(feature_names) == n_features:
            self.selected_feature_names_ = [feature_names[i] for i in self.selected_indices_]
        else:
            self.selected_feature_names_ = [f"Feature_{i}" for i in self.selected_indices_]

        if self.scale_to_pi:
            self.scaler_ = MinMaxScaler(feature_range=(-np.pi, np.pi))
            X_selected = X[:, self.selected_indices_]
            self.scaler_.fit(X_selected)

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Extracts top k features from input matrix X."""
        if self.selected_indices_ is None:
            raise RuntimeError("MISelector must be fitted before transforming.")

        X_selected = X[:, self.selected_indices_]
        if self.scale_to_pi and self.scaler_ is not None:
            X_selected = self.scaler_.transform(X_selected)

        return X_selected

    def fit_transform(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> np.ndarray:
        return self.fit(X, y, feature_names=feature_names).transform(X)

    def get_ranking_summary(self) -> List[dict]:
        """Returns sorted list of features with their mutual information scores."""
        summary = []
        for rank, idx in enumerate(self.selected_indices_):
            summary.append({
                "rank": rank + 1,
                "feature_index": int(idx),
                "feature_name": self.selected_feature_names_[rank]
                if rank < len(self.selected_feature_names_) else f"Col_{idx}",
                "mi_score": float(self.scores_[idx]),
            })
        return summary
