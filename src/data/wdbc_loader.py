"""Wisconsin Diagnostic Breast Cancer (WDBC) Dataset Loader.

Loads the 30 continuous clinical features derived from digitized images
of fine needle aspirate (FNA) of breast masses.
"""

from typing import Tuple, List
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler


def load_wdbc(scale: bool = True) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Loads and standardizes the WDBC dataset.

    Args:
        scale: If True, applies standard z-score normalization (mean=0, std=1).

    Returns:
        X: Feature matrix of shape (569, 30).
        y: Binary target labels (1 = Malignant, 0 = Benign).
        feature_names: List of 30 feature names.
    """
    raw_data = load_breast_cancer()
    X = raw_data.data.astype(np.float64)
    y = raw_data.target.astype(np.int64)
    feature_names = list(raw_data.feature_names)

    # In sklearn's breast cancer dataset, target 0 is malignant and 1 is benign.
    # To follow standard clinical convention (1 = disease/malignant, 0 = healthy/benign),
    # we flip the labels:
    y = 1 - y

    if scale:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    return X, y, feature_names
