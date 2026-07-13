"""
ensemble_methods.py - Ensemble learning implementations

Provides:
- Manual Bagging
- Manual AdaBoost with configurable base learner depth
- Random Forest (wrapper for scikit-learn)
- XGBoost (wrapper for xgboost library)
- Boosting depth experiments
"""

import time
import warnings
from typing import Tuple, Optional, Dict, Any, List, Callable

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier

from src.config import RANDOM_STATE, XGB_DEVICE, VERBOSE
from src.utils import get_rng

warnings.filterwarnings("ignore")


# ============================================================================
# Manual Bagging
# ============================================================================

def manual_bagging(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    n_estimators: int = 15,
    base_learner_factories: List[Callable] = None,
    random_state: int = RANDOM_STATE,
    verbose: bool = VERBOSE
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Manual bagging implementation with multiple base learners.
    
    Args:
        X_train: Training features
        y_train: Training targets
        X_test: Test features
        n_estimators: Number of base learners
        base_learner_factories: List of functions returning base learners
        random_state: Random seed
        verbose: Print progress
    
    Returns:
        Tuple of (predicted probabilities, predicted classes)
    """
    import xgboost as xgb
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    
    rng = get_rng(random_state)
    n = X_train.shape[0]
    
    # Default base learners if not provided
    if base_learner_factories is None:
        # Calculate scale_pos_weight for XGBoost
        n_neg = (y_train == 0).sum()
        n_pos = (y_train == 1).sum()
        scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
        
        base_learner_factories = [
            lambda: xgb.XGBClassifier(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=6,
                scale_pos_weight=scale_pos_weight,
                tree_method='hist',
                device=XGB_DEVICE,
                eval_metric='logloss',
                random_state=RANDOM_STATE
            ),
            lambda: LogisticRegression(
                max_iter=1000,
                class_weight='balanced',
                random_state=RANDOM_STATE
            ),
            lambda: RandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                class_weight='balanced',
                random_state=RANDOM_STATE
            )
        ]
    
    if verbose:
        print(f"Manual Bagging: Training {n_estimators} base learners...")
    
    proba_sum = np.zeros(X_test.shape[0])
    
    for i in range(n_estimators):
        # Bootstrap sample
        idx = rng.choice(n, n, replace=True)
        X_boot = X_train[idx]
        y_boot = y_train[idx]
        
        # Train base learner
        model = base_learner_factories[i % len(base_learner_factories)]()
        model.fit(X_boot, y_boot)
        
        # Predict probabilities
        proba_sum += model.predict_proba(X_test)[:, 1]
        
        if verbose and (i + 1) % 5 == 0:
            print(f"  Trained {i + 1}/{n_estimators} learners")
    
    proba_avg = proba_sum / n_estimators
    pred = (proba_avg >= 0.5).astype(int)
    
    return proba_avg, pred


# ============================================================================
# Manual AdaBoost
# ============================================================================

def manual_adaboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    n_iterations: int = 40,
    base_max_depth: int = 1,
    random_state: int = RANDOM_STATE,
    verbose: bool = VERBOSE
) -> Tuple[np.ndarray, np.ndarray, List, List]:
    """
    Manual AdaBoost implementation with decision stumps.
    
    Args:
        X_train: Training features
        y_train: Training targets
        X_test: Test features
        n_iterations: Number of boosting iterations
        base_max_depth: Max depth of base learners
        random_state: Random seed
        verbose: Print progress
    
    Returns:
        Tuple of (predicted probabilities, predicted classes, models, alphas)
    """
    n = X_train.shape[0]
    
    # Convert y to -1/1 for AdaBoost
    y_signed = np.where(y_train == 1, 1, -1)
    
    # Initialize weights
    w = np.ones(n) / n
    
    models = []
    alphas = []
    
    if verbose:
        print(f"Manual AdaBoost: Training {n_iterations} iterations with depth={base_max_depth}")
    
    for t in range(n_iterations):
        # Train weak learner (decision stump by default)
        stump = DecisionTreeClassifier(
            max_depth=base_max_depth,
            random_state=random_state + t
        )
        stump.fit(X_train, y_train, sample_weight=w)
        
        # Predict
        pred_signed = np.where(stump.predict(X_train) == 1, 1, -1)
        
        # Calculate error
        incorrect = (pred_signed != y_signed).astype(float)
        error = np.clip(
            np.sum(w * incorrect) / np.sum(w),
            1e-10,
            1 - 1e-10
        )
        
        # Calculate alpha
        alpha = 0.5 * np.log((1 - error) / error)
        
        # Update weights
        w = w * np.exp(-alpha * y_signed * pred_signed)
        w = w / w.sum()
        
        models.append(stump)
        alphas.append(alpha)
        
        if verbose and (t + 1) % 10 == 0:
            print(f"  Iteration {t + 1}/{n_iterations}: error={error:.4f}, alpha={alpha:.4f}")
    
    # Predict on test set
    test_scores = np.zeros(X_test.shape[0])
    for model, alpha in zip(models, alphas):
        pred_signed = np.where(model.predict(X_test) == 1, 1, -1)
        test_scores += alpha * pred_signed
    
    # Convert scores to probabilities
    test_proba = 1 / (1 + np.exp(-test_scores))
    test_pred = (test_scores > 0).astype(int)
    
    return test_proba, test_pred, models, alphas


# ============================================================================
# Boosting Depth Experiments
# ============================================================================

def run_boosting_depth_experiment(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    depths: List[int] = [1, 2, 3, 5, 7],
    n_iterations: int = 40,
    random_state: int = RANDOM_STATE,
    verbose: bool = VERBOSE
) -> pd.DataFrame:
    """
    Run AdaBoost experiments with different base learner depths.
    
    Args:
        X_train: Training features
        y_train: Training targets
        X_test: Test features
        y_test: Test targets
        depths: List of depths to try
        n_iterations: Number of boosting iterations
        random_state: Random seed
        verbose: Print progress
    
    Returns:
        DataFrame with results for each depth
    """
    from sklearn.metrics import f1_score, roc_auc_score
    
    results = []
    
    for depth in depths:
        if verbose:
            print(f"\nTesting depth={depth}...")
        
        t0 = time.time()
        
        proba, pred, models, alphas = manual_adaboost(
            X_train, y_train, X_test,
            n_iterations=n_iterations,
            base_max_depth=depth,
            random_state=random_state,
            verbose=False
        )
        
        elapsed = time.time() - t0
        
        results.append({
            'base_learner': f'Depth {depth}' if depth > 1 else 'Decision Stump (depth=1)',
            'depth': depth,
            'f1': f1_score(y_test, pred),
            'auc_roc': roc_auc_score(y_test, proba),
            'train_time_s': elapsed,
            'n_learners': len(models),
        })
        
        if verbose:
            print(f"  F1: {results[-1]['f1']:.3f}, AUC: {results[-1]['auc_roc']:.3f}, "
                  f"time: {elapsed:.1f}s")
    
    return pd.DataFrame(results)


# ============================================================================
# Convenience Functions
# ============================================================================

def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    n_estimators: int = 100,
    max_depth: int = 10,
    random_state: int = RANDOM_STATE,
    verbose: bool = VERBOSE
) -> Tuple[Any, np.ndarray, np.ndarray]:
    """
    Train a Random Forest classifier.
    
    Args:
        X_train: Training features
        y_train: Training targets
        X_test: Test features
        n_estimators: Number of trees
        max_depth: Maximum depth
        random_state: Random seed
        verbose: Print progress
    
    Returns:
        Tuple of (model, predictions, probabilities)
    """
    from sklearn.ensemble import RandomForestClassifier
    
    if verbose:
        print(f"Random Forest: Training {n_estimators} trees...")
    
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight='balanced',
        random_state=random_state,
        n_jobs=-1
    )
    
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0
    
    if verbose:
        print(f"Random Forest: Trained in {train_time:.2f}s")
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    return model, y_pred, y_proba


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    random_state: int = RANDOM_STATE,
    verbose: bool = VERBOSE,
    **kwargs
) -> Tuple[Any, np.ndarray, np.ndarray]:
    """
    Train an XGBoost classifier with early stopping.
    
    Args:
        X_train: Training features
        y_train: Training targets
        X_val: Validation features
        y_val: Validation targets
        X_test: Test features
        random_state: Random seed
        verbose: Print progress
        **kwargs: Additional XGBoost parameters
    
    Returns:
        Tuple of (model, predictions, probabilities)
    """
    import xgboost as xgb
    
    # Calculate scale_pos_weight for imbalance
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
    
    if verbose:
        print(f"XGBoost: scale_pos_weight = {scale_pos_weight:.2f}")
        print(f"XGBoost: Device = {XGB_DEVICE}")
    
    model = xgb.XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        scale_pos_weight=scale_pos_weight,
        tree_method='hist',
        device=XGB_DEVICE,
        eval_metric='logloss',
        early_stopping_rounds=50,
        random_state=random_state,
        **kwargs
    )
    
    t0 = time.time()
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    train_time = time.time() - t0
    
    if verbose:
        print(f"XGBoost: Trained in {train_time:.2f}s")
        print(f"XGBoost: Best iteration: {model.best_iteration}")
        print(f"XGBoost: Best score: {model.best_score:.4f}")
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    return model, y_pred, y_proba


# ============================================================================
# Testing
# ============================================================================

if __name__ == "__main__":
    print("Testing ensemble_methods.py...")
    
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import f1_score, roc_auc_score
    
    # Generate test data
    X, y = make_classification(
        n_samples=300, n_features=10, n_informative=8,
        n_redundant=2, n_classes=2, random_state=42
    )
    X = X.astype(np.float64)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
    )
    
    # Test Manual Bagging
    print("\n1. Testing Manual Bagging:")
    proba_bag, pred_bag = manual_bagging(
        X_train, y_train, X_test,
        n_estimators=10,
        verbose=True
    )
    print(f"  F1: {f1_score(y_test, pred_bag):.3f}")
    print(f"  AUC: {roc_auc_score(y_test, proba_bag):.3f}")
    
    # Test Manual AdaBoost
    print("\n2. Testing Manual AdaBoost:")
    proba_boost, pred_boost, models, alphas = manual_adaboost(
        X_train, y_train, X_test,
        n_iterations=20,
        base_max_depth=1,
        verbose=True
    )
    print(f"  F1: {f1_score(y_test, pred_boost):.3f}")
    print(f"  AUC: {roc_auc_score(y_test, proba_boost):.3f}")
    
    # Test Boosting Depth Experiment
    print("\n3. Testing Boosting Depth Experiment:")
    df_results = run_boosting_depth_experiment(
        X_train, y_train, X_test, y_test,
        depths=[1, 2, 3],
        n_iterations=10,
        verbose=True
    )
    print(df_results)
    
    print("\nAll tests passed!")