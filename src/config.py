"""
config.py - Centralized configuration for Project 3

All constants, random seeds, sample sizes, and configuration flags
are defined here to ensure consistency across all modules.
"""

import os
from pathlib import Path

# ============================================================================
# Project Paths
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = PROJECT_ROOT / "figures"
REPORTS_DIR = PROJECT_ROOT / "reports"
CACHE_DIR = DATA_DIR / "cache"

# Create directories if they don't exist
for dir_path in [DATA_DIR, OUTPUT_DIR, FIGURES_DIR, REPORTS_DIR, CACHE_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Set to True to ignore any existing cache and recompute everything
# (useful when the data or feature engineering changes)
DISABLE_CACHE = False

# ============================================================================
# Data Source
# ============================================================================

# GitHub raw URL for the cleaned dataset
DATA_URL = "https://raw.githubusercontent.com/AHK-0214/DataMining_Practice3/refs/heads/main/bank_cleaned.xlsx"
DATA_FILE = DATA_DIR / "bank_cleaned.xlsx"

# ============================================================================
# Randomness & Reproducibility
# ============================================================================

RANDOM_STATE = 42
RNG = None  # Will be initialized in utils

# ============================================================================
# Memory & Performance
# ============================================================================

USE_FLOAT64 = True  # Numerical stability for PCA / covariance / distances
ENABLE_GC = True    # Garbage collection after heavy operations

# ============================================================================
# Tiered Sample Sizes
# ============================================================================

# FAST algorithms: runtime is already very low -> can afford larger samples
SAMPLE_KERNEL = 6000              # Kernel K-Means (O(n^2) kernel matrix)
SAMPLE_DENSITY_HDBSCAN = 10000    # HDBSCAN
SAMPLE_GMM = 15000                # Gaussian Mixture Models
SAMPLE_CLUSTER = 8000             # K-Medoids, K-Median, Fuzzy C-Means

# MEDIUM algorithms: moderate increase
SAMPLE_AGGLOMERATIVE = 7000
SAMPLE_MEDIUM_KMEANS = 20000      # KMeans variants for speed

# SLOW algorithms: keep reasonable
SAMPLE_OPTICS = 4000              # OPTICS (separate sample)

# Supporting samples
SAMPLE_DENDROGRAM = 1000
SAMPLE_VISUALIZATION = 4000
SAMPLE_HOPKINS = 3000
SAMPLE_SILHOUETTE = 6000

# Small-vs-full dataset comparison
SAMPLE_SMALL_DATASET = 4500

# ============================================================================
# Column Names
# ============================================================================

NUM_COLS = ['age', 'balance', 'day', 'duration', 'campaign']
CAT_COLS = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'month']
TARGET_COL = 'y'

# Engineered features
ENGINEERED_NUM_COLS = ['balance_per_age', 'campaign_intensity']
ENGINEERED_FLAG_COLS = ['is_high_balance', 'is_long_call']

# ============================================================================
# Clustering Configuration
# ============================================================================

# Range for optimal K determination
K_RANGE = range(2, 11)

# Default clustering hyperparameters
DBSCAN_DEFAULT_EPS_PERCENTILE = 50  # Percentile for eps selection
DBSCAN_DEFAULT_MIN_SAMPLES = 5
OPTICS_DEFAULT_MIN_SAMPLES = 10
HDBSCAN_DEFAULT_MIN_CLUSTER_SIZE = 30

# ============================================================================
# Classification Configuration
# ============================================================================

TEST_SIZE = 0.2
CV_FOLDS = 5

# Grid search parameter grids
LOGISTIC_REGRESSION_PARAMS = {'clf__C': [0.01, 0.1, 1, 10]}
DECISION_TREE_PARAMS = {
    'clf__max_depth': [3, 5, 7, None],
    'clf__min_samples_split': [2, 10, 20]
}
RANDOM_FOREST_PARAMS = {
    'clf__n_estimators': [100, 200],
    'clf__max_depth': [5, 10, None]
}

# XGBoost parameters
XGBOOST_PARAMS = {
    'n_estimators': 300,
    'learning_rate': 0.05,
    'max_depth': 6,
    'tree_method': 'hist',
    'eval_metric': 'logloss',
    'early_stopping_rounds': 50
}

# ============================================================================
# Noise Experiments
# ============================================================================

NOISE_LEVELS = [0.05, 0.15]
NOISE_SCALE = 3.0
NOISE_REPEATS = 2

# ============================================================================
# Nonlinear Shape Detection
# ============================================================================

TWO_MOONS_SAMPLES = 1500
TWO_MOONS_NOISE = 0.07
TWO_MOONS_BANDWIDTH_QUANTILE = 0.2

# ============================================================================
# Plotting Configuration
# ============================================================================

PLOT_STYLE = 'whitegrid'
FIGURE_DPI = 100
COLORS = {
    'primary': '#4C72B0',
    'secondary': '#DD8452',
    'success': '#55A868',
    'danger': '#C44E52',
    'warning': '#ECD872',
    'info': '#64B5F6'
}

# ============================================================================
# Hardware Configuration
# ============================================================================

try:
    import torch
    GPU_AVAILABLE = torch.cuda.is_available()
    DEVICE = 'cuda' if GPU_AVAILABLE else 'cpu'
except ImportError:
    torch = None
    GPU_AVAILABLE = False
    DEVICE = 'cpu'

XGB_DEVICE = DEVICE if DEVICE == 'cuda' else 'cpu'

# ============================================================================
# Logging
# ============================================================================

VERBOSE = True

# ============================================================================
# Summary
# ============================================================================

def print_config():
    """Print current configuration summary."""
    print("=" * 70)
    print("CONFIGURATION SUMMARY")
    print("=" * 70)
    print(f"RANDOM_STATE: {RANDOM_STATE}")
    print(f"USE_FLOAT64: {USE_FLOAT64}")
    print(f"ENABLE_GC: {ENABLE_GC}")
    print(f"GPU available: {GPU_AVAILABLE} | Device: {DEVICE}")
    print(f"XGBoost device: {XGB_DEVICE}")
    print("\nSample sizes:")
    sample_sizes = {
        'SAMPLE_KERNEL': SAMPLE_KERNEL,
        'SAMPLE_DENSITY_HDBSCAN': SAMPLE_DENSITY_HDBSCAN,
        'SAMPLE_GMM': SAMPLE_GMM,
        'SAMPLE_CLUSTER': SAMPLE_CLUSTER,
        'SAMPLE_AGGLOMERATIVE': SAMPLE_AGGLOMERATIVE,
        'SAMPLE_OPTICS': SAMPLE_OPTICS,
        'SAMPLE_SMALL_DATASET': SAMPLE_SMALL_DATASET,
    }
    for name, val in sample_sizes.items():
        print(f"  {name:24s} = {val:,}")
    print("=" * 70)


if __name__ == "__main__":
    print_config()