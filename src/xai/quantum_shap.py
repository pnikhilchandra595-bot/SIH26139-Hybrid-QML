"""Quantum Explainability Module (Quantum KernelSHAP & Permutation Importance).

Computes local and global feature attributions for quantum models (QSVC, VQC, HQNN)
with computational mitigations for quantum simulator latency and clinical reporting.
"""

from typing import Optional, List, Dict, Any, Union
import numpy as np
import logging
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score, accuracy_score

try:
    import shap
except ImportError:
    shap = None

logger = logging.getLogger(__name__)


class QuantumKernelSHAPExplainer:
    """KernelSHAP explainer specialized for quantum models.

    Mitigates exponential circuit evaluations by k-means clustering the background
    reference dataset down to a small, representative set of medoids (default: 15-25 samples).
    """

    def __init__(
        self,
        model: Any,
        background_data: np.ndarray,
        n_background_samples: int = 20,
        random_state: int = 42,
    ):
        self.model = model
        self.n_background_samples = min(n_background_samples, len(background_data))
        self.random_state = random_state

        # Summarize background data to avoid excessive quantum simulator calls
        self.reference_data = self._summarize_background(background_data)
        self._explainer = None

    def _summarize_background(self, data: np.ndarray) -> np.ndarray:
        """Compresses background training set using K-Means centroids."""
        if len(data) <= self.n_background_samples:
            return np.copy(data)

        kmeans = KMeans(
            n_clusters=self.n_background_samples,
            random_state=self.random_state,
            n_init=10,
        )
        kmeans.fit(data)
        logger.info(
            f"Compressed background dataset from {len(data)} to {self.n_background_samples} "
            f"centroids to accelerate quantum SHAP evaluations."
        )
        return kmeans.cluster_centers_

    def _init_explainer(self) -> None:
        if shap is None:
            logger.warning("SHAP library is not installed. Using local linear surrogate approximation.")
            return

        # Wrapper function returning class-1 probabilities
        def predict_fn(x_matrix: np.ndarray) -> np.ndarray:
            probs = self.model.predict_proba(x_matrix)
            return probs[:, 1]

        self._explainer = shap.KernelExplainer(predict_fn, self.reference_data)

    def explain_instance(
        self,
        sample: np.ndarray,
        nsamples: int = 50,
    ) -> np.ndarray:
        """Computes local SHAP attribution values for a single patient sample.

        Args:
            sample: 1D array of feature values for single patient.
            nsamples: Number of background coalition evaluations.

        Returns:
            1D array of SHAP importance values matching feature length.
        """
        if sample.ndim == 1:
            sample = sample.reshape(1, -1)

        if self._explainer is None:
            self._init_explainer()

        if self._explainer is not None:
            shap_values = self._explainer.shap_values(sample, nsamples=nsamples, l1_reg="auto")
            return shap_values[0] if isinstance(shap_values, list) else shap_values

        # Lightweight fallback surrogate if shap is not installed
        base_prob = float(self.model.predict_proba(sample)[0, 1])
        ref_mean = np.mean(self.reference_data, axis=0)
        perturb = 0.05
        attributions = []
        for i in range(sample.shape[1]):
            perturbed_sample = np.copy(sample)
            perturbed_sample[0, i] += perturb
            new_prob = float(self.model.predict_proba(perturbed_sample)[0, 1])
            attributions.append((new_prob - base_prob) / perturb * (sample[0, i] - ref_mean[i]))
        return np.array(attributions)

    def explain_dataset(
        self,
        X_test: np.ndarray,
        nsamples: int = 50,
    ) -> np.ndarray:
        """Computes SHAP matrix across multiple patient samples."""
        if self._explainer is None:
            self._init_explainer()

        if self._explainer is not None:
            logger.info(f"Computing Quantum SHAP values for {len(X_test)} test samples...")
            shap_values = self._explainer.shap_values(X_test, nsamples=nsamples, l1_reg="auto")
            return np.array(shap_values)

        return np.array([self.explain_instance(x, nsamples=nsamples) for x in X_test])

    def generate_patient_attribution_report(
        self,
        sample: np.ndarray,
        feature_names: Optional[List[str]] = None,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """Generates a structured clinical risk attribution report for a single patient."""
        if sample.ndim == 1:
            sample = sample.flatten()
        n_features = len(sample)

        if feature_names is None or len(feature_names) != n_features:
            feature_names = [f"Feature_{i}" for i in range(n_features)]

        shap_vals = self.explain_instance(sample).flatten()
        prob_malignant = float(self.model.predict_proba(sample.reshape(1, -1))[0, 1])

        ranked_indices = np.argsort(np.abs(shap_vals))[::-1][:top_k]
        top_factors = []
        for idx in ranked_indices:
            top_factors.append({
                "feature": feature_names[idx],
                "feature_value": float(sample[idx]),
                "shap_attribution": float(shap_vals[idx]),
                "impact": "INCREASES_RISK" if shap_vals[idx] > 0 else "PROTECTIVE",
            })

        return {
            "prediction_probability": prob_malignant,
            "prediction_label": "MALIGNANT / DISEASE" if prob_malignant >= 0.5 else "BENIGN / HEALTHY",
            "top_clinical_factors": top_factors,
            "all_attributions": {feature_names[i]: float(shap_vals[i]) for i in range(n_features)},
        }


class QuantumPermutationImportance:
    """Computes Permutation Feature Importance for quantum models.

    Measures performance degradation when each individual feature is randomly permuted.
    Serves as an ultra-fast, model-agnostic feature importance metric.
    """

    def __init__(
        self,
        model: Any,
        scoring: str = "roc_auc",
        n_repeats: int = 5,
        random_state: int = 42,
    ):
        self.model = model
        self.scoring = scoring
        self.n_repeats = n_repeats
        self.random_state = random_state

    def _score(self, y_true: np.ndarray, y_pred_prob: np.ndarray) -> float:
        if self.scoring == "roc_auc":
            return roc_auc_score(y_true, y_pred_prob)
        elif self.scoring == "accuracy":
            return accuracy_score(y_true, (y_pred_prob >= 0.5).astype(int))
        else:
            raise ValueError(f"Unsupported scoring: {self.scoring}")

    def compute_importance(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, float]]:
        """Calculates mean and standard deviation of importance for each feature.

        Returns:
            Dict mapping feature name/index to {'mean_drop': float, 'std_drop': float}.
        """
        rng = np.random.default_rng(self.random_state)
        baseline_prob = self.model.predict_proba(X)[:, 1]
        baseline_score = self._score(y, baseline_prob)

        n_features = X.shape[1]
        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(n_features)]

        results = {}

        for j in range(n_features):
            drops = []
            for _ in range(self.n_repeats):
                X_permuted = np.copy(X)
                X_permuted[:, j] = rng.permutation(X_permuted[:, j])
                perm_prob = self.model.predict_proba(X_permuted)[:, 1]
                perm_score = self._score(y, perm_prob)
                drops.append(baseline_score - perm_score)

            results[feature_names[j]] = {
                "mean_drop": float(np.mean(drops)),
                "std_drop": float(np.std(drops)),
            }

        return results
