
# Project 3 — Clustering & Classification

## Overview

This project implements a complete machine learning pipeline for unsupervised and supervised learning experiments. The objective is to explore multiple clustering algorithms, evaluate their performance using several internal and robustness metrics, analyze the discovered clusters, and compare various classification models built upon the processed dataset.

The project emphasizes:

* Modular and maintainable code
* Reproducible experiments
* Fair algorithm comparison
* Comprehensive evaluation
* Clear visualization
* Scalable architecture suitable for large datasets

The implementation is organized into independent modules so that each stage of the workflow can be developed, tested, and improved separately.

---

# Project Objectives

## Clustering

* Prepare and preprocess the dataset
* Evaluate clustering tendency
* Determine the optimal number of clusters
* Compare multiple clustering algorithms
* Select the best-performing algorithm using holistic evaluation
* Profile discovered clusters
* Study the effect of feature removal

## Classification

* Prepare classification datasets
* Train multiple baseline classifiers
* Train ensemble methods
* Evaluate using cross-validation
* Analyze overfitting and underfitting
* Compare small and full datasets
* Produce practical recommendations

---

# Project Architecture

```
Project_3/
├── notebooks/
│   └── Project_3_Clustering_Classification.ipynb
│
├── src/
│   ├── data_preparation.py
│   ├── clustering_tendency.py
│   ├── optimal_k.py
│   ├── clustering_algorithms.py
│   ├── clustering_evaluation.py
│   ├── cluster_profiling.py
│   ├── classification_prep.py
│   ├── base_classifiers.py
│   ├── neural_network.py
│   ├── ensemble_methods.py
│   ├── classification_eval.py
│   ├── classification_analysis.py
│   ├── final_summary.py
│   ├── visualization.py
│   └── utils.py
│
├── docs/
├── data/
├── outputs/
├── figures/
├── reports/
├── requirements.txt
└── README.md
```

---

# Workflow

```
Data Loading
      ↓
Data Cleaning
      ↓
Feature Selection
      ↓
PCA
      ↓
Tiered Dataset Preparation
      ↓
Hopkins Test
      ↓
Optimal K
      ↓
Clustering Algorithms
      ↓
Algorithm Comparison
      ↓
Best Model Selection
      ↓
Cluster Profiling
      ↓
Feature Removal
      ↓
Classification
      ↓
Model Comparison
      ↓
Final Summary
```

---

# Clustering Algorithms

## Centroid-Based

* K-Means
* Bisecting K-Means
* K-Medoids
* K-Median
* Kernel K-Means
* Fuzzy C-Means

## Density-Based

* DBSCAN
* OPTICS
* HDBSCAN

## Hierarchical

* Agglomerative (Ward)
* Agglomerative (Single)
* Agglomerative (Complete)
* Dendrogram Analysis

## Probabilistic

* Gaussian Mixture Models (GMM)

---

# Classification Algorithms

## Base Models

* Logistic Regression
* Decision Tree
* Naive Bayes
* Feedforward Neural Network (PyTorch)

## Ensemble Models

### Bagging

* Random Forest
* Manual Bagging

### Boosting

* XGBoost
* Manual Boosting

---

# Evaluation Strategy

## Clustering Metrics

### Internal Metrics

* Silhouette Score
* Davies–Bouldin Index
* Dunn Index

### Practical Metric

* Runtime

### Robustness

* Noise Resistance (ARI)

### Shape Detection

* Two-Moons ARI

---

## Classification Metrics

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC
* Cross-Validation Mean
* Cross-Validation Standard Deviation
* Lift@10%

---

# Technologies

## Programming Language

* Python 3.11+

## Core Libraries

* NumPy
* Pandas
* SciPy
* Scikit-learn
* PyTorch
* XGBoost
* Matplotlib
* Plotly
* Seaborn (optional)
* Yellowbrick
* HDBSCAN
* Scikit-Fuzzy

---

# Hardware Support

The project supports:

* CPU execution
* GPU acceleration using CUDA (PyTorch) when available

The implementation automatically falls back to CPU if CUDA is unavailable.

---

# Reproducibility

To ensure reproducible experiments:

* Fixed random seeds
* Consistent preprocessing
* Shared utility functions
* Centralized configuration
* Modular pipeline

---

# Outputs

The project generates:

* Clean datasets
* PCA datasets
* Cluster assignments
* Evaluation tables
* Feature importance reports
* Confusion matrices
* ROC curves
* Learning curves
* Validation curves
* Comparison tables
* Cluster profiling reports
* Figures
* Final summary tables

---

# Running the Project

## 1. Clone the repository

```bash
git clone <repository-url>
cd Project_3
```

## 2. Install dependencies

```bash
pip install -r requirements.txt
```

## 3. Launch Jupyter

```bash
jupyter notebook
```

Open:

```
notebooks/Project_3_Clustering_Classification.ipynb
```

Execute the notebook from top to bottom.

---

# Coding Principles

* Modular design
* Single responsibility per module
* Reusable functions
* Consistent naming conventions
* Clear documentation
* Reproducible experiments
* Minimal duplicated code

---

# Documentation

Additional project documentation is available in the `docs/` directory:

* `PROJECT_MEMORY.md` — Project architecture, assumptions, and conventions
* `TODO.md` — Development roadmap and remaining improvements
* `DECISIONS.md` — Key design decisions and their rationale

---

# Project Status

Current Phase:

**Phase 2 — Documentation**

The architecture has been finalized and frozen. Documentation is being prepared before implementation begins.

---

# License

This project is intended for academic and educational purposes.

