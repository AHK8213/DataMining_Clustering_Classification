"""
classification_eval.py - Classification evaluation utilities

Provides:
- Comprehensive metrics computation
- Cross-validation with multiple metrics
- ROC and Precision-Recall curves
- Confusion matrix visualization
- Lift analysis
- Model comparison
"""

import warnings
from typing import Dict, Any, List, Tuple, Optional, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    average_precision_score,
    classification_report
)

from src.config import RANDOM_STATE, CV_FOLDS, VERBOSE
from src.utils import timer

warnings.filterwarnings("ignore")


# ============================================================================
# Metrics Computation
# ============================================================================

def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Compute comprehensive classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities (for AUC)
        verbose: Print results
    
    Returns:
        Dictionary with metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
    }
    
    if y_proba is not None:
        metrics['auc_roc'] = roc_auc_score(y_true, y_proba)
        metrics['average_precision'] = average_precision_score(y_true, y_proba)
    
    if verbose:
        print("\nClassification Metrics:")
        print(f"  Accuracy: {metrics['accuracy']:.3f}")
        print(f"  Precision: {metrics['precision']:.3f}")
        print(f"  Recall: {metrics['recall']:.3f}")
        print(f"  F1 Score: {metrics['f1']:.3f}")
        if 'auc_roc' in metrics:
            print(f"  AUC-ROC: {metrics['auc_roc']:.3f}")
        if 'average_precision' in metrics:
            print(f"  Average Precision: {metrics['average_precision']:.3f}")
    
    return metrics


def get_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: List[str] = ['no', 'yes'],
    verbose: bool = VERBOSE
) -> str:
    """
    Get classification report as string.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        target_names: Names of target classes
        verbose: Print report
    
    Returns:
        Classification report string
    """
    report = classification_report(
        y_true, y_pred,
        target_names=target_names,
        digits=3
    )
    
    if verbose:
        print("\nClassification Report:")
        print(report)
    
    return report


# ============================================================================
# Cross-Validation
# ============================================================================

def cross_validate_classifier(
    model,
    X: np.ndarray,
    y: np.ndarray,
    cv_folds: int = CV_FOLDS,
    random_state: int = RANDOM_STATE,
    scoring: List[str] = None,
    verbose: bool = VERBOSE
) -> Dict[str, Dict[str, float]]:
    """
    Cross-validate a classifier with multiple metrics.
    
    Args:
        model: Scikit-learn compatible classifier
        X: Features
        y: Targets
        cv_folds: Number of CV folds
        random_state: Random seed
        scoring: List of scoring metrics
        verbose: Print progress
    
    Returns:
        Dictionary with mean and std for each metric
    """
    if scoring is None:
        scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    
    skf = StratifiedKFold(
        n_splits=cv_folds,
        shuffle=True,
        random_state=random_state
    )
    
    if verbose:
        print(f"Running {cv_folds}-fold cross-validation...")
    
    with timer("Cross-validation", verbose=verbose):
        cv_results = cross_validate(
            model, X, y,
            cv=skf,
            scoring=scoring,
            n_jobs=-1
        )
    
    results = {}
    for metric in scoring:
        scores = cv_results[f'test_{metric}']
        results[metric] = {
            'mean': scores.mean(),
            'std': scores.std()
        }
        
        if verbose:
            print(f"  {metric}: {scores.mean():.3f} ± {scores.std():.3f}")
    
    # Add timing
    results['cv_time'] = {
        'mean': cv_results['fit_time'].mean(),
        'std': cv_results['fit_time'].std()
    }
    
    return results


def cross_validate_multiple_classifiers(
    models: Dict[str, Any],
    X: np.ndarray,
    y: np.ndarray,
    cv_folds: int = CV_FOLDS,
    random_state: int = RANDOM_STATE,
    verbose: bool = VERBOSE
) -> pd.DataFrame:
    """
    Cross-validate multiple classifiers and return comparison.
    
    Args:
        models: Dictionary mapping names to classifiers
        X: Features
        y: Targets
        cv_folds: Number of CV folds
        random_state: Random seed
        verbose: Print progress
    
    Returns:
        DataFrame with CV results for each model
    """
    results = []
    
    for name, model in models.items():
        if verbose:
            print(f"\n{'='*50}")
            print(f"Cross-validating: {name}")
            print('='*50)
        
        cv_results = cross_validate_classifier(
            model, X, y,
            cv_folds=cv_folds,
            random_state=random_state,
            verbose=verbose
        )
        
        row = {'model': name}
        for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
            if metric in cv_results:
                row[f'{metric}_mean'] = cv_results[metric]['mean']
                row[f'{metric}_std'] = cv_results[metric]['std']
        
        results.append(row)
    
    df = pd.DataFrame(results)
    
    # Sort by F1
    if 'f1_mean' in df.columns:
        df = df.sort_values('f1_mean', ascending=False)
    
    return df


# ============================================================================
# Lift Analysis
# ============================================================================

def lift_at_k(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    k: float = 0.1
) -> float:
    """
    Calculate lift at k% of highest predicted probabilities.
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        k: Proportion of top predictions to consider
    
    Returns:
        Lift value (ratio of positive rate in top k to overall positive rate)
    """
    n = len(y_true)
    n_top = int(n * k)
    
    # Sort by probability descending
    idx_sorted = np.argsort(y_proba)[::-1]
    top_k_idx = idx_sorted[:n_top]
    
    top_k_rate = y_true.iloc[top_k_idx].mean() if hasattr(y_true, 'iloc') else y_true[top_k_idx].mean()
    overall_rate = y_true.mean()
    
    return top_k_rate / overall_rate if overall_rate > 0 else 0


def cumulative_gain_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    steps: int = 100,
    title: str = "Cumulative Gain Curve"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate cumulative gain curve points.
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        steps: Number of steps
    
    Returns:
        Tuple of (x_points, y_points)
    """
    n = len(y_true)
    idx_sorted = np.argsort(y_proba)[::-1]
    y_sorted = y_true.iloc[idx_sorted] if hasattr(y_true, 'iloc') else y_true[idx_sorted]
    
    x_points = np.linspace(0, 1, steps + 1)
    y_points = np.zeros(steps + 1)
    
    for i, x in enumerate(x_points):
        n_top = int(n * x)
        if n_top > 0:
            y_points[i] = y_sorted[:n_top].sum() / y_sorted.sum()
    
    return x_points, y_points


# ============================================================================
# Visualization
# ============================================================================

def plot_roc_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    label: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6),
    ax: Optional[plt.Axes] = None
) -> plt.Figure:
    """
    Plot ROC curve.
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        label: Label for the curve
        figsize: Figure size
        ax: Matplotlib axes
    
    Returns:
        Matplotlib figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    
    label_text = f"{label} (AUC={auc:.3f})" if label else f"AUC={auc:.3f}"
    ax.plot(fpr, tpr, label=label_text, linewidth=2)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.4, label='Random (AUC=0.5)')
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve", fontsize=14)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    return fig


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    label: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6),
    ax: Optional[plt.Axes] = None
) -> plt.Figure:
    """
    Plot Precision-Recall curve.
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        label: Label for the curve
        figsize: Figure size
        ax: Matplotlib axes
    
    Returns:
        Matplotlib figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)
    
    label_text = f"{label} (AP={ap:.3f})" if label else f"AP={ap:.3f}"
    ax.plot(recall, precision, label=label_text, linewidth=2)
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curve", fontsize=14)
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)
    
    return fig


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: List[str] = ['no', 'yes'],
    figsize: Tuple[int, int] = (6, 5),
    ax: Optional[plt.Axes] = None
) -> plt.Figure:
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: Label names
        figsize: Figure size
        ax: Matplotlib axes
    
    Returns:
        Matplotlib figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    ConfusionMatrixDisplay(
        confusion_matrix(y_true, y_pred),
        display_labels=labels
    ).plot(ax=ax, colorbar=False)
    
    ax.set_title("Confusion Matrix", fontsize=14)
    
    return fig


def plot_cumulative_gain(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    label: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6),
    ax: Optional[plt.Axes] = None
) -> plt.Figure:
    """
    Plot cumulative gain curve.
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        label: Label for the curve
        figsize: Figure size
        ax: Matplotlib axes
    
    Returns:
        Matplotlib figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    x_points, y_points = cumulative_gain_curve(y_true, y_proba)
    
    label_text = label if label else "Model"
    ax.plot(x_points, y_points, label=label_text, linewidth=2)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.4, label='Random')
    ax.set_xlabel("Percentage of Population", fontsize=12)
    ax.set_ylabel("Percentage of Positives Captured", fontsize=12)
    ax.set_title("Cumulative Gain Curve", fontsize=14)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    return fig


# ============================================================================
# Model Comparison
# ============================================================================

def compare_classifiers(
    results_dict: Dict[str, Dict[str, float]],
    sort_by: str = 'f1'
) -> pd.DataFrame:
    """
    Compare multiple classifiers.
    
    Args:
        results_dict: Dictionary mapping model names to metric dictionaries
        sort_by: Metric to sort by
    
    Returns:
        DataFrame with comparison
    """
    df = pd.DataFrame(results_dict).T
    
    if sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=False)
    
    return df


def plot_model_comparison(
    results_df: pd.DataFrame,
    metrics: List[str] = ['f1', 'auc_roc', 'accuracy', 'precision', 'recall'],
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Plot comparison of multiple models.
    
    Args:
        results_df: DataFrame from compare_classifiers
        metrics: Metrics to plot
        figsize: Figure size
    
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Filter available metrics
    available_metrics = [m for m in metrics if m in results_df.columns]
    
    # Plot
    results_df[available_metrics].plot(
        kind='bar',
        ax=ax,
        width=0.8,
        rot=20
    )
    
    ax.set_title("Model Comparison", fontsize=14)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_ylim(0, 1)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig


# ============================================================================
# Testing
# ============================================================================

if __name__ == "__main__":
    print("Testing classification_eval.py...")
    
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    
    # Generate test data
    X, y = make_classification(
        n_samples=200, n_features=10, n_informative=8,
        n_redundant=2, n_classes=2, random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Test metrics
    print("\n1. Testing metrics:")
    metrics = compute_classification_metrics(y_test, y_pred, y_proba, verbose=True)
    
    # Test cross-validation
    print("\n2. Testing cross-validation:")
    cv_results = cross_validate_classifier(model, X, y, cv_folds=3, verbose=True)
    
    # Test lift
    print("\n3. Testing lift analysis:")
    y_series = pd.Series(y_test)
    lift = lift_at_k(y_series, y_proba, k=0.1)
    print(f"  Lift at 10%: {lift:.2f}")
    
    # Test plots
    print("\n4. Testing plots:")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    plot_roc_curve(y_test, y_proba, label="Logistic Regression", ax=axes[0, 0])
    plot_precision_recall_curve(y_test, y_proba, label="Logistic Regression", ax=axes[0, 1])
    plot_confusion_matrix(y_test, y_pred, ax=axes[1, 0])
    plot_cumulative_gain(y_series, y_proba, label="Logistic Regression", ax=axes[1, 1])
    
    plt.tight_layout()
    plt.show()
    
    print("\nAll tests passed!")