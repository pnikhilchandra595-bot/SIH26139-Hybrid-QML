# ROADMAP.md — Project Roadmap & Phase Decomposition

## Milestones Overview

### Milestone 1: Classical Baseline & Scaffolding (Phases 1)
- Data loaders for WDBC and UCI Heart Disease.
- Preprocessing, stratified 5-fold cross-validation splitters.
- Classical baselines (SVM, Random Forest, XGBoost, MLP).

### Milestone 2: Quantum Core Development (Phase 2) — [Member B Primary Ownership]
- Dimensionality reduction to fit qubit limits (PCA, Mutual Information).
- Quantum backend and device abstraction manager (`default.qubit`, `qasm_simulator`).
- Quantum Feature Maps (ZZFeatureMap, AngleEmbedding).
- Quantum Support Vector Classifier (QSVC) via quantum kernel Gram matrix.
- Variational Quantum Classifier (VQC) with Strongly Entangling Layers.
- Hybrid Quantum Neural Network (HQNN) combining PyTorch Linear layers and PennyLane `TorchLayer`.
- Quantum circuit numerical unit testing suite.

### Milestone 3: Integration, Benchmarking & Explainability (Phase 3)
- Unified CLI runner (`python cli.py train --model ...`).
- Comprehensive evaluation suite (Acc, Sens, Spec, F1, ROC-AUC).
- KernelSHAP and Permutation explainability for hybrid models.
- Hardening Sprint 1 (memory profiling, statevector caching).

### Milestone 4: Web UI & Platform Delivery (Phase 4)
- Multi-page Streamlit application.
- Interactive data explorer, feature dimension slider, and model studio.
- Interactive SHAP waterfall and clinical risk gauge.
- Docker containerization & GitHub Actions CI.

### Milestone 5: Hardening & Final Submission (Phase 5)
- Edge-case testing, IEEE technical report, presentation deck, demo video, and final repository tag.

---

## Current Active Phase: Phase 2 (Quantum Core Development)
**Lead:** Member B (Quantum ML Engineer)  
**Backup:** Member A / Member D  
**Goal:** Deliver rock-solid, production-grade quantum circuit primitives, feature maps, QSVC, VQC, and Hybrid QNN.
