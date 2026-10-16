---
updated: 2026-10-16T19:22:45+05:30
---

# Project State

## Current Position

**Milestone:** Clinical Dataset Ingestion & Quantum Model Integration  
**Active Role:** Member B (Quantum ML) & Member A (Data Pipelines)  
**Status:** verified (Fundus Retinopathy Ingestion Complete)  
**Plan:** Integrated Real Clinical Fundus Photos (Diabetic Retinopathy) into Quantum Pipeline  

## New Dataset Ingestion (Task D2 Upgrade)
- [x] **Sample Fundus Photos Ingested** (`data/Sample_Fundus_Photos/`)
  - 22 real human eye fundus clinical photographs covering Level 0 Normal, Level 2 Moderate NPDR, Level 3 Severe NPDR, and Level 4 Proliferative Diabetic Retinopathy.
- [x] **Biomedical Image Feature Extractor** (`src/data/fundus_loader.py`)
  - Extracts 11 clinical biomarkers per eye image:
    1. Green Channel Luminance & Variance (vital for microaneurysms)
    2. Exudate Brightness (95th percentile luminance)
    3. Hemorrhage Darkness (5th percentile intraretinal markers)
    4. Red Vascularity & Perfusion
    5. Blue Channel Attenuation
    6. Red-to-Green Perfusion Ratio
    7. Optical Disc-to-Background Contrast Ratio
    8. Retinal Vessel Spatial Gradient Density
  - Automatically parses ground-truth binary classification: `0 = Healthy/Normal`, `1 = Diabetic Retinopathy`.
- [x] **End-to-End Hybrid Quantum Flow Verified** (`tests/test_fundus_loader.py`)
  - Image features $\rightarrow$ 4-qubit `PCAReducer` ($[-\pi, \pi]$) $\rightarrow$ `QSVC` with $ZZ$-FeatureMap kernel $\rightarrow$ Predictions.

## Active Decisions
| Decision | Choice | Made | Affects |
|---|---|---|---|
| Image Preprocessing | Classical Feature Extraction $\rightarrow$ Quantum Reduction | 2026-10-16 | Fundus Image QML Pipeline |
| Target Classes | Binary (0 = Normal, 1 = Retinopathy) | 2026-10-16 | Clinical screening priority |
| Quantum Embedding | 4 Qubits with ZZFeatureMap | 2026-10-16 | QSVC Gram matrix execution speed |

## Blockers
None.
