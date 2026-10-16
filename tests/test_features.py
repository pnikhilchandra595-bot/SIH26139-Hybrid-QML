"""Unit tests for Feature Reduction and Selection Modules (T2.1, T2.2)."""

import pytest
import numpy as np

from src.features.pca_reducer import PCAReducer
from src.features.mi_selector import MISelector


class TestPCAReducer:
    """Tests for PCA reduction module."""

    @pytest.fixture
    def synthetic_data(self):
        rng = np.random.default_rng(42)
        X = rng.normal(loc=0.0, scale=1.0, size=(40, 10))
        return X

    def test_pca_shape_and_range(self, synthetic_data):
        X = synthetic_data
        reducer = PCAReducer(n_components=4, scale_to_pi=True, random_state=42)
        X_reduced = reducer.fit_transform(X)

        assert X_reduced.shape == (40, 4)
        # Check angle scale [-pi, pi]
        assert np.all(X_reduced >= -np.pi - 1e-6)
        assert np.all(X_reduced <= np.pi + 1e-6)

        summary = reducer.get_variance_summary()
        assert summary["n_components"] == 4
        assert 0.0 < summary["total_explained_variance"] <= 1.0


class TestMISelector:
    """Tests for Mutual Information feature selector."""

    @pytest.fixture
    def synthetic_data(self):
        rng = np.random.default_rng(42)
        X = rng.normal(loc=0.0, scale=1.0, size=(50, 8))
        # Label correlated primarily with feature 0 and 1
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        names = [f"BioMarker_{i}" for i in range(8)]
        return X, y, names

    def test_mi_selection_and_ranking(self, synthetic_data):
        X, y, names = synthetic_data
        selector = MISelector(k=3, scale_to_pi=True, random_state=42)
        X_selected = selector.fit_transform(X, y, feature_names=names)

        assert X_selected.shape == (50, 3)
        # Check angle scale [-pi, pi]
        assert np.all(X_selected >= -np.pi - 1e-6)
        assert np.all(X_selected <= np.pi + 1e-6)

        ranking = selector.get_ranking_summary()
        assert len(ranking) == 3
        assert "feature_name" in ranking[0]
        assert "mi_score" in ranking[0]
