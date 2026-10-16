"""Stratified Cross-Validation Splitter.

Generates reproducible train and validation fold indices with preserved class ratios.
"""

from typing import List, Tuple
import numpy as np
from sklearn.model_selection import StratifiedKFold


def get_stratified_folds(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    seed: int = 42,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Generates stratified cross-validation train/validation index splits.

    Args:
        X: Feature matrix.
        y: Binary target labels.
        n_splits: Number of folds (default: 5).
        seed: Random seed for reproducibility.

    Returns:
        List of tuples (train_indices, val_indices).
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []
    for train_idx, val_idx in skf.split(X, y):
        folds.append((train_idx, val_idx))
    return folds
