
# PROJECT_MEMORY.md

## Purpose

This document serves as the permanent memory of the project.

It contains:

* Frozen architecture
* Design decisions
* Naming conventions
* Library selections
* Assumptions
* Evaluation methodology
* Development guidelines

This file should be provided to future AI assistants or developers before making modifications to the project.

---

# Project Information

## Project Name

Project 3 — Clustering & Classification

## Current Status

Architecture Frozen

Phase 2 — Documentation

Implementation Not Started

---

# Project Goals

The project has two major objectives:

## Part 1 — Clustering

Perform a comprehensive comparison of clustering algorithms and identify the best model through a holistic evaluation framework.

## Part 2 — Classification

Build and compare multiple classification models using both traditional machine learning and neural network approaches.

---

# Frozen Architecture

## Part A — Setup & Data Preparation

### A.1 Imports & Configuration

Responsibilities:

* Import libraries
* Configure random seeds
* Configure plotting
* Configure hardware

---

### A.2 Memory Optimization

Responsibilities:

* Reduce memory usage
* Optimize dataframe dtypes
* Monitor memory footprint

Decision:

Use float64 throughout the project for numerical consistency and stability.

---

### A.3 Data Loading & Cleaning

Responsibilities:

* Load dataset
* Remove duplicates
* Handle missing values
* Fix data inconsistencies

---

### A.4 Feature Selection

Responsibilities:

* Remove irrelevant variables
* Remove highly redundant variables
* Prepare final feature set

Important:

Feature Engineering was intentionally removed from the architecture.

---

### A.5 PCA

Responsibilities:

* Fit PCA once on the full dataset
* Generate PCA representations
* Support visualization

---

### A.6 Tiered Dataset Preparation

Responsibilities:

Create multiple dataset sizes:

* Small
* Medium
* Large
* Full

Purpose:

Support scalability experiments and computational comparisons.

---

# Part B — Clustering Experiments

## B.1 Clustering Tendency

Method:

Hopkins Statistic

Purpose:

Determine whether meaningful clusters exist before clustering.

---

## B.2 Optimal K Determination

Methods:

### Elbow Method

Purpose:

Visualization only.

### Silhouette Score

Purpose:

Primary method for selecting K.

Important:

Run on the full dataset.

Not Used:

* Calinski-Harabasz
* Davies-Bouldin

for K selection.

---

## B.3 Clustering Algorithms

### Centroid-Based

* K-Means
* Bisecting K-Means
* K-Medoids
* K-Median
* Kernel K-Means
* Fuzzy C-Means

### Density-Based

* DBSCAN
* OPTICS
* HDBSCAN

### Hierarchical

* Agglomerative Ward
* Agglomerative Single
* Agglomerative Complete
* Dendrogram

### Probabilistic

* Gaussian Mixture Model

---

## B.4 Comprehensive Algorithm Comparison

Evaluation Categories

### Internal Quality

Metrics:

* Silhouette
* Davies-Bouldin
* Dunn

---

### Practicality

Metric:

* Runtime

---

### Robustness

Metric:

* Noise ARI

Multiple noise levels must be tested.

---

### Nonlinear Detection

Metric:

* Two-Moons ARI

Purpose:

Measure ability to detect non-convex clusters.

---

### Final Selection

All algorithms are ranked holistically.

The highest-ranked algorithm becomes the project's best clustering model.

---

## B.5 Cluster Profiling

Performed only on the best clustering model.

Required outputs:

### Numerical Analysis

* Means
* Medians
* Distributions

### Categorical Analysis

* Category frequencies
* Percentages

### Feature Importance

All clusters must be displayed.

### PCA Visualization

Visualize cluster separation.

Important:

Past profiling issues involving missing clusters must not occur.

All clusters must be shown.

---

## B.6 Feature Removal

Purpose:

Evaluate the impact of removing features using the best clustering algorithm.

---

# Part C — Classification Experiments

## C.1 Target Selection & Train/Test Split

Responsibilities:

* Select target variable
* Create train/test split
* Maintain class distribution

Method:

Stratified split when appropriate.

---

## C.2 Feature Preprocessing

Responsibilities:

* Scaling
* Encoding
* Transformation

---

## C.3 Base Classifiers

Cross-validation is the primary evaluation approach.

Models:

### Logistic Regression

### Decision Tree

### Naive Bayes

### Neural Network

Framework:

PyTorch

Hardware:

CUDA GPU when available

CPU fallback otherwise

Architecture:

Feedforward network

2–3 hidden layers

Activation:

ReLU

Output:

Sigmoid

Optimizer:

Adam

Loss:

Binary Cross-Entropy

---

## C.4 Ensemble Methods

### Bagging

* Random Forest
* Manual Bagging

### Boosting

* XGBoost
* Manual Boosting

Additional Experiment:

Manual Boosting Depth Analysis

---

## C.5 Results Comparison

Compare all classifiers in one consolidated table.

Must include:

* Base models
* Ensemble models

Purpose:

Provide a fair comparison across all methods.

---

## C.6 Overfitting & Underfitting Analysis

Required:

### Learning Curves

### Validation Curves

### Interpretation

Must explain observed behavior rather than only showing plots.

---

## C.7 Pre-Call Feature Analysis

Purpose:

Investigate feature importance and predictive power before classification.

---

## C.8 Small vs Full Dataset Comparison

Purpose:

Analyze scalability and performance changes across dataset sizes.

---

# Part D — Summary & Conclusions

## D.1 Final Comparison Tables

Required:

### Clustering Ranking Table

### Classification Ranking Table

---

## D.2 Extrinsic Evaluation

Purpose:

Compare clustering structure against classification behavior.

---

## D.3 Final Discussion

Must include:

* Practical recommendations
* Limitations
* Future work

---

# Evaluation Standards

## Clustering

### Primary Ranking Inputs

* Silhouette
* Davies-Bouldin
* Dunn
* Runtime
* Noise ARI
* Two-Moons ARI

---

## Classification

### Primary Metric

F1 Score

Reason:

Most appropriate for imbalanced targets.

---

### Secondary Metrics

* Accuracy
* Precision
* Recall
* ROC-AUC
* Lift@10%

---

### Generalization Metrics

* Cross-validation mean
* Cross-validation standard deviation

---

# Libraries

## Data Processing

* NumPy
* Pandas

## Scientific Computing

* SciPy

## Machine Learning

* Scikit-Learn

## Deep Learning

* PyTorch

## Gradient Boosting

* XGBoost

## Visualization

* Matplotlib
* Plotly

## Clustering Extensions

* HDBSCAN
* Scikit-Fuzzy

## Utilities

* Joblib
* TQDM

---

# Naming Conventions

## Files

snake_case

Example:

```text
clustering_evaluation.py
classification_analysis.py
```

---

## Functions

snake_case

Example:

```text
calculate_hopkins()
evaluate_classifier()
```

---

## Classes

PascalCase

Example:

```text
ClusterEvaluator
NeuralNetworkTrainer
```

---

## Constants

UPPER_CASE

Example:

```text
RANDOM_STATE
DEFAULT_FOLDS
```

---

# Assumptions

1. Dataset size may exceed memory-friendly limits.

2. Some clustering algorithms may not scale to full data.

3. GPU may not always be available.

4. PCA is used primarily for visualization and dimensionality reduction.

5. Cross-validation is preferred over a single train/test result.

6. Reproducibility is a project requirement.

7. Float64 is the standard numeric precision throughout the project.

---

# Development Rules

1. Keep modules independent.

2. Avoid duplicate code.

3. Use utility functions whenever possible.

4. Preserve reproducibility.

5. Document major changes.

6. Update DECISIONS.md when changing methodology.

7. Update TODO.md when adding improvements.

8. Do not modify the frozen architecture without explicit approval.

---

# Last Architecture Freeze

Date:

2026-07-11

Status:

Approved and Frozen

Implementation Phase:

Not Started

