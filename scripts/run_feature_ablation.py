"""Feature Reduction Ablation Study Script (Deliverable D3).

Evaluates classification performance and information retention across
different dimensionalities k in [2, 4, 6, 8, 10, 15, 30] to determine
optimal qubit allocation for quantum circuits.
"""

import os
import argparse
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, recall_score, f1_score

from src.data.wdbc_loader import load_wdbc
from src.data.heart_loader import load_heart_disease
from src.data.splitter import get_stratified_folds
from src.features.pca_reducer import PCAReducer
from src.features.mi_selector import MISelector


def run_ablation(dataset_name: str = "wdbc") -> pd.DataFrame:
    """Executes dimensionality ablation across k dimensions."""
    if dataset_name.lower() == "wdbc":
        X, y, feature_names = load_wdbc(scale=True)
        k_values = [2, 4, 6, 8, 10, 15, 30]
    else:
        X, y, feature_names = load_heart_disease(scale=True)
        k_values = [2, 4, 6, 8, 10, 13]

    folds = get_stratified_folds(X, y, n_splits=5, seed=42)
    results = []

    print(f"\n=======================================================")
    print(f" Running Feature Ablation Study for: {dataset_name.upper()}")
    print(f" Samples: {X.shape[0]}, Original Features: {X.shape[1]}")
    print(f"=======================================================")

    for k in k_values:
        pca_accs, pca_recalls, pca_f1s = [], [], []
        mi_accs, mi_recalls, mi_f1s = [], [], []
        pca_variances = []

        for train_idx, val_idx in folds:
            X_tr, y_tr = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]

            # 1. Evaluate PCA reduction
            pca = PCAReducer(n_components=k, scale_to_pi=True, random_state=42)
            X_tr_pca = pca.fit_transform(X_tr)
            X_val_pca = pca.transform(X_val)
            pca_variances.append(pca.cumulative_variance_)

            clf_pca = SVC(kernel="rbf", C=1.0, random_state=42)
            clf_pca.fit(X_tr_pca, y_tr)
            preds_pca = clf_pca.predict(X_val_pca)

            pca_accs.append(accuracy_score(y_val, preds_pca))
            pca_recalls.append(recall_score(y_val, preds_pca, zero_division=0))
            pca_f1s.append(f1_score(y_val, preds_pca, zero_division=0))

            # 2. Evaluate Mutual Information selection
            mi = MISelector(k=k, scale_to_pi=True, random_state=42)
            X_tr_mi = mi.fit_transform(X_tr, y_tr)
            X_val_mi = mi.transform(X_val)

            clf_mi = SVC(kernel="rbf", C=1.0, random_state=42)
            clf_mi.fit(X_tr_mi, y_tr)
            preds_mi = clf_mi.predict(X_val_mi)

            mi_accs.append(accuracy_score(y_val, preds_mi))
            mi_recalls.append(recall_score(y_val, preds_mi, zero_division=0))
            mi_f1s.append(f1_score(y_val, preds_mi, zero_division=0))

        row = {
            "dataset": dataset_name,
            "k_qubits": k,
            "pca_variance_retained": float(np.mean(pca_variances)),
            "pca_accuracy_mean": float(np.mean(pca_accs)),
            "pca_accuracy_std": float(np.std(pca_accs)),
            "pca_sensitivity_mean": float(np.mean(pca_recalls)),
            "mi_accuracy_mean": float(np.mean(mi_accs)),
            "mi_accuracy_std": float(np.std(mi_accs)),
            "mi_sensitivity_mean": float(np.mean(mi_recalls)),
        }
        results.append(row)

        print(
            f"k = {k:2d} Qubits | PCA Var: {row['pca_variance_retained']*100:5.1f}% | "
            f"PCA Acc: {row['pca_accuracy_mean']*100:5.2f}% | "
            f"MI Acc: {row['mi_accuracy_mean']*100:5.2f}%"
        )

    df_results = pd.DataFrame(results)

    os.makedirs("reports/tables", exist_ok=True)
    out_path = f"reports/tables/feature_ablation_{dataset_name}.csv"
    df_results.to_csv(out_path, index=False)
    print(f"\n[Done] Ablation results saved to: {out_path}\n")
    return df_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="wdbc", choices=["wdbc", "heart"])
    args = parser.parse_args()
    run_ablation(args.dataset)
