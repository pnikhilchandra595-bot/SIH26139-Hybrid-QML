"""Data Ingestion and Preprocessing Module for Biomedical Datasets."""

from src.data.wdbc_loader import load_wdbc
from src.data.heart_loader import load_heart_disease
from src.data.splitter import get_stratified_folds
from src.data.fundus_loader import load_fundus_dataset, FUNDUS_FEATURE_NAMES

__all__ = [
    "load_wdbc",
    "load_heart_disease",
    "get_stratified_folds",
    "load_fundus_dataset",
    "FUNDUS_FEATURE_NAMES",
]
