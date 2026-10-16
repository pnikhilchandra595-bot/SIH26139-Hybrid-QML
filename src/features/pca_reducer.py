"""Principal Component Analysis (PCA) Dimensionality Reduction for Quantum Circuits.

Compresses high-dimensional biomedical features down to N in [4, 8] qubits
and scales the orthogonal components to [-pi, pi] for quantum angle rotation embeddings.
"""

from typing import Optional, Tuple
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin


class PCAReducer(BaseEstimator, TransformerMixin):
    """PCA Dimensionality Reducer tailored for NISQ Quantum Feature Maps."""

    def __init__(
        self,
        n_components: int = 4,
        scale_to_pi: bool = True,
        random_state: int = 42,
    ):
        self.n_components = n_components
        self.scale_to_pi = scale_to_pi
        self.random_state = random_state

        self.pca_ = None
        self.scaler_ = None
        self.explained_variance_ratio_ = None
        self.cumulative_variance_ = None

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "PCAReducer":
        """Fits PCA projection on training feature matrix."""
        n_features = X.shape[1]
        actual_components = min(self.n_components, n_features)

        self.pca_ = PCA(n_components=actual_components, random_state=self.random_state)
        X_pca = self.pca_.fit_transform(X)

        self.explained_variance_ratio_ = self.pca_.explained_variance_ratio_
        self.cumulative_variance_ = float(np.sum(self.explained_variance_ratio_))

        if self.scale_to_pi:
            # Scale PCA components to [-pi, pi] for periodic quantum angle rotations
            self.scaler_ = MinMaxScaler(feature_range=(-np.pi, np.pi))
            self.scaler_.fit(X_pca)

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Projects feature matrix X onto principal quantum components."""
        if self.pca_ is None:
            raise RuntimeError("PCAReducer must be fitted before transforming data.")

        X_pca = self.pca_.transform(X)
        if self.scale_to_pi and self.scaler_ is not None:
            X_pca = self.scaler_.transform(X_pca)

        return X_pca

    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """Fits and transforms input feature matrix X."""
        return self.fit(X, y).transform(X)

    def get_variance_summary(self) -> dict:
        """Returns explained variance details."""
        return {
            "n_components": self.n_components,
            "explained_variance_ratio": self.explained_variance_ratio_.tolist()
            if self.explained_variance_ratio_ is not None else [],
            "total_explained_variance": self.cumulative_variance_,
        }
