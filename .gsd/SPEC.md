# SPEC.md — Hybrid Quantum Machine Learning Platform for Early Disease Detection

> **Status**: `FINALIZED`
>
> ⚠️ **Planning Lock**: LIFTED. All requirements, constraints, and architecture finalized.

---

## 1. Vision
To design and build a clinically interpretable, modular, simulator-first **Hybrid Quantum-Classical Machine Learning Platform** that detects early-stage diseases (Breast Cancer and Heart Disease) by exploiting quantum Hilbert space embeddings, quantum kernels, and variational parameterized circuits alongside classical baseline models.

---

## 2. Core Objectives & Scope
1. **Biomedical Data Pipelines (D2):** Automated ingestion, cleaning, standard scaling, and stratified splitting for Wisconsin Diagnostic Breast Cancer (WDBC) and UCI Heart Disease.
2. **Qubit-Constrained Feature Engineering (D3):** Dimensionality reduction (PCA and Mutual Information) compressing features to $N \in [4, 8]$ qubits to operate within near-term quantum simulator budgets.
3. **Quantum-Enhanced Classifiers (D4) [Member B Primary Ownership]:**
   - **QSVC (Quantum Support Vector Classifier):** High-dimensional Hilbert space mapping via PennyLane with $ZZ$-FeatureMap kernels and fidelity-based Gram matrix computation.
   - **VQC (Variational Quantum Classifier):** Parameterized quantum circuit using Angle/Amplitude embeddings, Strongly Entangling Layers, and analytical parameter-shift gradient optimization.
   - **HQNN (Hybrid Quantum Neural Network):** End-to-end PyTorch hybrid model coupling classical feedforward layers with a PennyLane `TorchLayer` variational quantum bottleneck.
4. **Classical Baseline Benchmarks (D5):** Linear SVM, Random Forest, XGBoost, and classical MLP trained on identical splits and metrics.
5. **Universal CLI & Backend Switching (D6):** Unified dispatcher allowing seamless switching between PennyLane `default.qubit`, Qiskit `aer_simulator`, and optional cloud QPUs.
6. **Empirical Benchmarking Suite (D7):** Automated calculation of Accuracy, Sensitivity (Recall), Specificity, F1, ROC-AUC, and wall-clock execution latency.
7. **Explainability Module (D8):** Local and global feature attribution via KernelSHAP and permutation importance across both classical and quantum architectures.
8. **Interactive Web Dashboard (D9):** Streamlit GUI for dataset inspection, feature reduction tuning, live model training, comparator leaderboard, and SHAP patient explanations.

---

## 3. Non-Goals (Explicitly Out of Scope)
- No physical QPU access requirement as a blocker (simulator-first, $N \le 8$ qubits).
- No direct raw image training on quantum circuits without prior feature reduction.
- No commercial EHR integration or clinical regulatory certification (HIPAA/FDA).

---

## 4. Technical Constraints
- **Hardware/Simulators:** PennyLane (`default.qubit`), Qiskit (`aer_simulator`), PyTorch autograd bridge.
- **Compute Ceiling:** $N \le 8$ qubits per circuit to guarantee laptop/Colab execution under 60 seconds per epoch.
- **Reproducibility:** Seed pinning across NumPy, PyTorch, and PennyLane RNG.

---

## 5. Success Criteria
- [x] Full architectural specification and mathematical formulation completed.
- [ ] Quantum models train with monotonic loss reduction and achieve competitive or superior sensitivity on minority clinical classes.
- [ ] Automated testing suite achieves $>85\%$ test coverage across quantum circuit primitives.
- [ ] End-to-end user journey (data upload $\rightarrow$ feature reduction $\rightarrow$ quantum model training $\rightarrow$ SHAP explanation) functional inside Streamlit web UI.

---

*Last updated: 2026-10-16 — Phase 1 & 2 Transition*
