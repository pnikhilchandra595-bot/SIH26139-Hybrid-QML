# 🧬 Hybrid Quantum Machine Learning Platform for Early Disease Detection (HQML-EDD)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PennyLane](https://img.shields.io/badge/PennyLane-0.38+-purple.svg)](https://pennylane.ai/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-ee4c2c.svg)](https://pytorch.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.4+-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](demo_walkthrough.ipynb)

A modular, simulator-first **Hybrid Quantum-Classical Machine Learning Platform** engineered for early and accurate diagnosis of critical diseases (Breast Cancer, Cardiovascular Disease, and Diabetic Retinopathy). 

By coupling classical pre-processing and dimensionality reduction with quantum Hilbert-space embeddings, fidelity kernels, and variational parameterized circuits, HQML-EDD achieves competitive diagnostic accuracy with improved clinical sensitivity on critical minority classes under near-term NISQ constraints ($N \le 8$ qubits).

---

## 🔬 Core Capabilities

* **Multi-Modal Biomedical Ingestion:**
  * **Oncology:** Wisconsin Diagnostic Breast Cancer (WDBC, 30 FNA continuous features).
  * **Cardiology:** UCI Cleveland Heart Disease (13 hemodynamic & ECG features).
  * **Ophthalmology:** Clinical Retinal Fundus Photos (11 biomarkers extracted from Green-channel luminance, vessel density, and exudate contrast).
* **Quantum Qubit Compression:** Automated PCA and Mutual Information modules reducing high-dimensional inputs to $k \in [4, 8]$ features scaled to $[-\pi, \pi]$ for periodic rotation gates.
* **Three Quantum Architectures:**
  1. **QSVC (Quantum Kernel SVM):** Evaluates state fidelity $K(x_i, x_j) = |\langle \Phi(x_i) | \Phi(x_j) \rangle|^2$ using entangling $ZZ$-FeatureMaps.
  2. **VQC (Variational Quantum Classifier):** Parameterized quantum circuit using `StronglyEntanglingLayers` optimized via analytical parameter-shift gradient descent and Adam.
  3. **HQNN (Hybrid Quantum Neural Network):** End-to-end PyTorch network linking classical feedforward layers with a PennyLane `TorchLayer` quantum bottleneck.
* **Rigorous Classical Benchmarking:** Direct 5-fold cross-validation against Linear SVM, RBF SVM, Random Forest, XGBoost, and a PyTorch MLP.
* **Clinical Explainability (XAI):** Quantum KernelSHAP with K-Means background clustering ($k=20$) and Permutation Feature Importance for single-patient risk factor attribution.
* **Interactive Web Platform:** Multi-page Streamlit dashboard featuring live data exploration, qubit tuning, model training, and circuit visualization.

---

## 🏗️ Architecture Overview

```
Biomedical Data (WDBC / Heart / Fundus)
               │
               ▼
   [ Classical Feature Preprocessing ]
   (StandardScaler + PCA / Mutual Information → 4–8 Qubits in [-π, π])
               │
               ▼
   [ Quantum-Classical Engine Tier ]
   ┌────────────────────────────────────────────────────────┐
   │ • QSVC (Fidelity Transition Kernel: ZZ-FeatureMap)     │
   │ • VQC (Variational Ansatz + Parameter-Shift Adam)      │
   │ • HQNN (PyTorch nn.Linear + PennyLane TorchLayer)      │
   │ • Classical Baselines (SVM, Random Forest, XGB, MLP)   │
   └───────────────────────────┬────────────────────────────┘
                               │
                               ▼
   [ Evaluation & Clinical Explainability (XAI) ]
   (5-Fold CV Benchmarks + Quantum KernelSHAP Patient Risk Attribution)
                               │
                               ▼
   [ Streamlit GUI Console & Visualizer ]
```

---

## ⚡ Quickstart

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Run Test Suite
```bash
pytest tests/ -v
```

### 3. Launch Interactive Dashboard
```bash
py -3.13 -m streamlit run app/main.py
```
Open `http://localhost:8501` in your browser.
