
# DECISIONS.md

## Purpose

This document records all important technical and architectural decisions made during the development of Project 3.

Each decision contains:

* The chosen approach
* The reason behind the choice
* The alternatives considered
* The impact on the project

This file acts as the project's decision history and should be updated whenever a major methodology or architecture change occurs.

---

# Decision 1 — Project Modular Architecture

## Decision

The project will use a modular Python architecture instead of placing all code inside a single notebook.

## Reason

A large machine learning project becomes difficult to maintain when all logic is written directly inside a notebook.

A modular structure provides:

* Easier debugging
* Code reuse
* Better testing
* Clear separation of responsibilities
* Easier collaboration

## Implementation

All major components are separated into individual modules inside:

```
src/
```

---

# Decision 2 — Notebook as Experiment Controller Only

## Decision

The notebook will not contain the main implementation logic.

The notebook will only:

* Import modules
* Execute experiments
* Display results
* Generate final analysis

## Reason

The notebook should represent the research workflow rather than become a large code file.

---

# Decision 3 — Remove Feature Engineering Section

## Decision

Feature Engineering was removed from the architecture.

## Reason

The project requirements focus on:

* Feature selection
* PCA
* Clustering
* Classification

Additional feature engineering could introduce unnecessary complexity and bias.

## Impact

The pipeline becomes:

```
Cleaning
   ↓
Feature Selection
   ↓
PCA
```

---

# Decision 4 — Float64 as Numerical Standard

## Decision

All numerical processing will use float64.

## Reason

Although float32 reduces memory usage, float64 provides:

* Higher numerical precision
* More stable calculations
* Better consistency between algorithms

## Impact

All preprocessing, PCA, clustering, and evaluation calculations should maintain float64 precision.

---

# Decision 5 — PCA Fitted Once on Full Dataset

## Decision

PCA will be fitted once using the complete dataset.

## Reason

Using different PCA transformations for different experiments can make comparisons inconsistent.

A single transformation ensures:

* Comparable representations
* Consistent visualization
* Reproducibility

---

# Decision 6 — Tiered Dataset Strategy

## Decision

Multiple dataset sizes will be created.

## Datasets

* Small
* Medium
* Large
* Full

## Reason

Some clustering algorithms have poor scalability.

Tiered datasets allow:

* Runtime comparison
* Scalability analysis
* Resource management

---

# Decision 7 — Hopkins Test Before Clustering

## Decision

The clustering pipeline begins with Hopkins Statistic.

## Reason

Not every dataset naturally contains clusters.

Hopkins provides evidence that clustering structures may exist.

---

# Decision 8 — Silhouette as Primary Optimal K Metric

## Decision

Silhouette Score is the primary method for selecting the number of clusters.

## Reason

Silhouette provides:

* Cluster compactness measurement
* Cluster separation measurement
* Easy interpretation

## Alternatives Rejected

### Calinski-Harabasz

Rejected because it was not necessary for the project's selection methodology.

### Davies-Bouldin

Rejected for K selection because it is reserved for algorithm comparison.

---

# Decision 9 — Elbow Method as Visualization Only

## Decision

The Elbow Method will be included but not used as the final K selection method.

## Reason

The elbow point can be subjective.

Its purpose is:

* Visual interpretation
* Supporting evidence

Final selection relies on Silhouette Score.

---

# Decision 10 — Algorithm Comparison After All Clustering Experiments

## Decision

All clustering algorithms will be compared together after execution.

## Reason

Selecting a model too early can introduce bias.

The final model must be selected using:

* Quality
* Robustness
* Runtime
* Shape detection

---

# Decision 11 — Holistic Clustering Ranking

## Decision

The best clustering algorithm will be selected using multiple metrics.

## Metrics

Internal:

* Silhouette
* Davies-Bouldin
* Dunn

Practical:

* Runtime

Robustness:

* Noise ARI

Nonlinear capability:

* Two-Moons ARI

## Reason

No single metric can represent all clustering characteristics.

---

# Decision 12 — Cluster Profiling Only on Best Model

## Decision

Cluster profiling will only be performed after selecting the best clustering algorithm.

## Reason

Profiling every algorithm creates unnecessary analysis and confusion.

The goal is to understand the final chosen clustering solution.

---

# Decision 13 — Show All Clusters During Profiling

## Decision

Cluster profiling must display every generated cluster.

## Reason

Previous profiling approaches could hide clusters due to filtering or visualization limitations.

All clusters are required for correct interpretation.

---

# Decision 14 — Feature Removal Uses Best Clustering Algorithm

## Decision

Feature removal experiments will use the best clustering model selected in B.4.

## Reason

Comparing feature removal on weak algorithms does not provide meaningful conclusions.

---

# Decision 15 — Classification Evaluation Uses Cross Validation

## Decision

Cross-validation is the primary evaluation method.

## Reason

A single train/test split can produce unstable results.

Cross-validation provides:

* Better generalization estimation
* Variance measurement
* Fair model comparison

---

# Decision 16 — Neural Network as Base Classifier

## Decision

The neural network belongs in the base classifier section.

## Reason

Neural networks are standalone predictive models, not ensemble methods.

---

# Decision 17 — PyTorch for Neural Network

## Decision

The neural network implementation uses PyTorch.

## Reason

PyTorch provides:

* GPU acceleration
* Flexible architecture design
* Better control over training

---

# Decision 18 — GPU Support with CPU Fallback

## Decision

The neural network automatically uses CUDA when available.

## Reason

The project should run on different hardware environments.

Behavior:

```
CUDA available
      ↓
GPU training

CUDA unavailable
      ↓
CPU training
```

---

# Decision 19 — Random Forest and XGBoost as Ensemble Methods

## Decision

Random Forest and XGBoost are categorized as ensemble methods.

## Reason

They combine multiple learners:

Random Forest:

* Bagging

XGBoost:

* Boosting

They are not simple base classifiers.

---

# Decision 20 — Compare All Classifiers Together

## Decision

The final classification comparison happens after ensemble experiments.

## Reason

The comparison must include:

* Base classifiers
* Bagging models
* Boosting models

A complete ranking requires all models.

---

# Decision 21 — Classification Primary Metric

## Decision

F1 Score is the primary classification metric.

## Reason

The target problem may contain class imbalance.

F1 balances:

* Precision
* Recall

---

# Decision 22 — Learning and Validation Curves Required

## Decision

Overfitting analysis must include learning curves and validation curves.

## Reason

Performance numbers alone cannot explain model behavior.

Curves reveal:

* Underfitting
* Overfitting
* Generalization issues

---

# Decision 23 — Documentation Before Implementation

## Decision

Documentation must be completed before writing implementation code.

## Reason

A clear architecture reduces:

* Rework
* Inconsistent decisions
* Duplicate development

---

# Decision 24 — Architecture Freeze

## Decision

The project architecture is frozen before implementation.

## Date

2026-07-11

## Reason

Implementation should follow a stable design.

Changes after this point require updating:

* PROJECT_MEMORY.md
* TODO.md
* DECISIONS.md

---

# Change Management Rule

Any future modification must answer:

1. What changed?
2. Why was it changed?
3. What files are affected?
4. Does PROJECT_MEMORY need updating?
5. Does TODO need updating?
6. Does this create a new decision entry?

---

# Current Status

Architecture:

✅ Frozen

Documentation:

✅ Completed

Implementation:

□ Not Started

