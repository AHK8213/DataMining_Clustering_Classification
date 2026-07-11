"""
base_classifiers.py - Base classification models

Provides:
- Logistic Regression with GridSearch
- Decision Tree with GridSearch
- Naive Bayes
- Random Forest with GridSearch
- XGBoost with early stopping
- Unified interface for training and evaluation
"""

import time
import warnings
from typing import Dict, Any, Tuple, Optional, List

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

import xgboost as xgb

from src.config import (
    RANDOM_STATE,
    CV_FOLDS,
    LOGISTIC_REGRESSION_PARAMS,
    DECISION_TREE_PARAMS,
    RANDOM_FOREST_PARAMS,
    XGBOOST_PARAMS,
    XGB_DEVICE,
    VERBOSE,
    NUM_COLS,
    CAT_COLS,
)
from src.classification_prep import get_preprocessor_for_classification
from src.utils import ensure_float64, timer

warnings.filterwarnings("ignore")


# ============================================================================
# Base Classifier Wrapper
# ============================================================================

class BaseClassifier:
    """
    Wrapper for base classifiers with GridSearch support.
    
    Usage:
        classifier = BaseClassifier('Logistic Regression', params_grid)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
    """
    
    def __init__(
        self,
        name: str,
        model,
        param_grid: Optional[Dict] = None,
        random_state: int = RANDOM_STATE,
        cv_folds: int = CV_FOLDS,
        verbose: bool = VERBOSE
    ):
        """
        Initialize classifier.
        
        Args:
            name: Classifier name
            model: Scikit-learn model instance
            param_grid: Parameter grid for GridSearch
            random_state: Random seed
            cv_folds: Number of CV folds
            verbose: Print progress
        """
        self.name = name
        self.model = model
        self.param_grid = param_grid
        self.random_state = random_state
        self.cv_folds = cv_folds
        self.verbose = verbose
        
        self.pipeline = None
        self.best_params_ = None
        self.train_time_ = None
        self.is_fitted = False
    
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        **fit_params
    ) -> 'BaseClassifier':
        """
        Fit the classifier with GridSearch if parameters are provided.
        
        Args:
            X_train: Training features
            y_train: Training targets
            **fit_params: Additional fit parameters
        
        Returns:
            Self
        """
        t0 = time.time()
        
        if self.param_grid:
            # Use GridSearch
            if self.verbose:
                print(f"{self.name}: Running GridSearch with {self.cv_folds}-fold CV...")
            
            gs = GridSearchCV(
                self.model,
                self.param_grid,
                scoring='f1',
                cv=StratifiedKFold(self.cv_folds, shuffle=True, random_state=self.random_state),
                n_jobs=-1,
                verbose=0
            )
            gs.fit(X_train, y_train, **fit_params)
            
            self.pipeline = gs.best_estimator_
            self.best_params_ = gs.best_params_
            
            if self.verbose:
                print(f"{self.name}: Best params: {gs.best_params_}")
                print(f"{self.name}: Best CV F1: {gs.best_score_:.3f}")
        else:
            # Direct fit
            self.pipeline = self.model
            self.pipeline.fit(X_train, y_train, **fit_params)
        
        self.train_time_ = time.time() - t0
        self.is_fitted = True
        
        if self.verbose:
            print(f"{self.name}: Training completed in {self.train_time_:.2f}s")
        
        return self
    
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Predict labels."""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet. Call fit() first.")
        return self.pipeline.predict(X_test)
    
    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        """Predict probabilities."""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet. Call fit() first.")
        return self.pipeline.predict_proba(X_test)
    
    def get_params(self) -> Dict:
        """Get best parameters if GridSearch was used."""
        return self.best_params_ or {}
    
    def get_pipeline(self):
        """Get the fitted pipeline."""
        return self.pipeline


# ============================================================================
# Preprocessing Pipeline Builders
# ============================================================================

def build_classification_pipeline(
    classifier,
    num_cols: List[str] = NUM_COLS,
    cat_cols: List[str] = CAT_COLS,
    use_float64: bool = True
) -> Pipeline:
    """
    Build a full classification pipeline with preprocessing.
    
    Args:
        classifier: Scikit-learn classifier instance
        num_cols: Numeric columns
        cat_cols: Categorical columns
        use_float64: Use float64
    
    Returns:
        Pipeline with preprocessing and classifier
    """
    preprocessor = get_preprocessor_for_classification(num_cols, cat_cols, use_float64)
    
    pipeline = Pipeline([
        ('pre', preprocessor),
        ('clf', classifier)
    ])
    
    return pipeline


# ============================================================================
# Specific Classifier Functions
# ============================================================================

def create_logistic_regression(
    preprocessor: Optional[ColumnTransformer] = None,
    random_state: int = RANDOM_STATE,
    use_gridsearch: bool = True,
    **kwargs
) -> BaseClassifier:
    """
    Create a Logistic Regression classifier.
    
    Args:
        preprocessor: Optional preprocessor
        random_state: Random seed
        use_gridsearch: Use GridSearch for C parameter
        **kwargs: Additional parameters
    
    Returns:
        BaseClassifier instance
    """
    model = LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        random_state=random_state,
        **kwargs
    )
    
    param_grid = LOGISTIC_REGRESSION_PARAMS if use_gridsearch else None
    
    return BaseClassifier(
        name='Logistic Regression',
        model=model,
        param_grid=param_grid,
        random_state=random_state
    )


def create_decision_tree(
    preprocessor: Optional[ColumnTransformer] = None,
    random_state: int = RANDOM_STATE,
    use_gridsearch: bool = True,
    **kwargs
) -> BaseClassifier:
    """
    Create a Decision Tree classifier.
    
    Args:
        preprocessor: Optional preprocessor
        random_state: Random seed
        use_gridsearch: Use GridSearch for hyperparameters
        **kwargs: Additional parameters
    
    Returns:
        BaseClassifier instance
    """
    model = DecisionTreeClassifier(
        random_state=random_state,
        class_weight='balanced',
        **kwargs
    )
    
    param_grid = DECISION_TREE_PARAMS if use_gridsearch else None
    
    return BaseClassifier(
        name='Decision Tree',
        model=model,
        param_grid=param_grid,
        random_state=random_state
    )


def create_naive_bayes(
    preprocessor: Optional[ColumnTransformer] = None,
    random_state: int = RANDOM_STATE,
    **kwargs
) -> BaseClassifier:
    """
    Create a Naive Bayes classifier.
    
    Args:
        preprocessor: Optional preprocessor
        random_state: Random seed
        **kwargs: Additional parameters
    
    Returns:
        BaseClassifier instance
    """
    model = GaussianNB(**kwargs)
    
    return BaseClassifier(
        name='Naive Bayes',
        model=model,
        param_grid=None,  # No GridSearch for Naive Bayes
        random_state=random_state
    )


def create_random_forest(
    preprocessor: Optional[ColumnTransformer] = None,
    random_state: int = RANDOM_STATE,
    use_gridsearch: bool = True,
    **kwargs
) -> BaseClassifier:
    """
    Create a Random Forest classifier.
    
    Args:
        preprocessor: Optional preprocessor
        random_state: Random seed
        use_gridsearch: Use GridSearch for hyperparameters
        **kwargs: Additional parameters
    
    Returns:
        BaseClassifier instance
    """
    model = RandomForestClassifier(
        random_state=random_state,
        class_weight='balanced',
        n_jobs=-1,
        **kwargs
    )
    
    param_grid = RANDOM_FOREST_PARAMS if use_gridsearch else None
    
    return BaseClassifier(
        name='Random Forest',
        model=model,
        param_grid=param_grid,
        random_state=random_state
    )


# ============================================================================
# XGBoost Classifier
# ============================================================================

def create_xgboost_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    random_state: int = RANDOM_STATE,
    verbose: bool = VERBOSE,
    **kwargs
) -> Tuple[xgb.XGBClassifier, Dict[str, Any]]:
    """
    Create and train an XGBoost classifier with early stopping.
    
    Args:
        X_train: Training features
        y_train: Training targets
        X_val: Validation features
        y_val: Validation targets
        random_state: Random seed
        verbose: Print progress
        **kwargs: Additional XGBoost parameters
    
    Returns:
        Tuple of (trained model, training info)
    """
    # Calculate scale_pos_weight for imbalance
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
    
    if verbose:
        print(f"XGBoost: scale_pos_weight = {scale_pos_weight:.2f}")
    
    # Merge with default parameters
    params = XGBOOST_PARAMS.copy()
    params.update(kwargs)
    params['scale_pos_weight'] = scale_pos_weight
    params['device'] = XGB_DEVICE
    params['random_state'] = random_state
    
    if verbose:
        print(f"XGBoost: Training on device '{XGB_DEVICE}'")
    
    # Create model
    model = xgb.XGBClassifier(**params)
    
    # Train with early stopping
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
    
    info = {
        'train_time': train_time,
        'best_iteration': model.best_iteration,
        'best_score': model.best_score,
        'scale_pos_weight': scale_pos_weight,
        'params': params
    }
    
    return model, info


def create_xgboost_pipeline(
    num_cols: List[str] = NUM_COLS,
    cat_cols: List[str] = CAT_COLS,
    random_state: int = RANDOM_STATE,
    **kwargs
) -> Pipeline:
    """
    Create a pipeline with preprocessing and XGBoost.
    
    Args:
        num_cols: Numeric columns
        cat_cols: Categorical columns
        random_state: Random seed
        **kwargs: Additional XGBoost parameters
    
    Returns:
        Pipeline with preprocessing and XGBoost
    """
    preprocessor = get_preprocessor_for_classification(num_cols, cat_cols)
    
    params = XGBOOST_PARAMS.copy()
    params.update(kwargs)
    params['random_state'] = random_state
    params['device'] = XGB_DEVICE
    
    xgb_model = xgb.XGBClassifier(**params)
    
    return Pipeline([
        ('pre', preprocessor),
        ('clf', xgb_model)
    ])


# ============================================================================
# Convenience Functions
# ============================================================================

def create_all_base_classifiers(
    use_gridsearch: bool = True,
    random_state: int = RANDOM_STATE
) -> Dict[str, BaseClassifier]:
    """
    Create all base classifiers.
    
    Args:
        use_gridsearch: Use GridSearch for applicable classifiers
        random_state: Random seed
    
    Returns:
        Dictionary mapping classifier names to instances
    """
    classifiers = {
        'Logistic Regression': create_logistic_regression(
            random_state=random_state,
            use_gridsearch=use_gridsearch
        ),
        'Decision Tree': create_decision_tree(
            random_state=random_state,
            use_gridsearch=use_gridsearch
        ),
        'Naive Bayes': create_naive_bayes(
            random_state=random_state
        ),
        'Random Forest': create_random_forest(
            random_state=random_state,
            use_gridsearch=use_gridsearch
        )
    }
    
    return classifiers


def train_and_evaluate_classifier(
    classifier: BaseClassifier,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    verbose: bool = VERBOSE
) -> Dict[str, Any]:
    """
    Train a classifier and return evaluation metrics.
    
    Args:
        classifier: BaseClassifier instance
        X_train: Training features
        y_train: Training targets
        X_test: Test features
        y_test: Test targets
        verbose: Print progress
    
    Returns:
        Dictionary with metrics
    """
    from sklearn.metrics import (
        f1_score, roc_auc_score, accuracy_score,
        precision_score, recall_score
    )
    
    # Train
    classifier.fit(X_train, y_train)
    
    # Predict
    y_pred = classifier.predict(X_test)
    y_proba = classifier.predict_proba(X_test)[:, 1] if hasattr(classifier, 'predict_proba') else None
    
    # Metrics
    metrics = {
        'model': classifier.name,
        'f1': f1_score(y_test, y_pred),
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'train_time_s': classifier.train_time_,
    }
    
    if y_proba is not None:
        metrics['auc_roc'] = roc_auc_score(y_test, y_proba)
    
    if classifier.best_params_:
        metrics['best_params'] = str(classifier.best_params_)
    
    if verbose:
        print(f"\n{classifier.name} Results:")
        print(f"  F1: {metrics['f1']:.3f}")
        print(f"  AUC-ROC: {metrics.get('auc_roc', np.nan):.3f}")
        print(f"  Accuracy: {metrics['accuracy']:.3f}")
        print(f"  Precision: {metrics['precision']:.3f}")
        print(f"  Recall: {metrics['recall']:.3f}")
        print(f"  Training time: {metrics['train_time_s']:.2f}s")
    
    return metrics


# ============================================================================
# Testing
# ============================================================================

if __name__ == "__main__":
    print("Testing base_classifiers.py...")
    
    from sklearn.datasets import make_classification
    
    # Generate test data
    X, y = make_classification(
        n_samples=500, n_features=10, n_informative=8,
        n_redundant=2, n_classes=2, random_state=42
    )
    X = X.astype(np.float64)
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Test each classifier
    print("\n1. Testing Logistic Regression:")
    lr = create_logistic_regression(use_gridsearch=False)
    metrics = train_and_evaluate_classifier(lr, X_train, y_train, X_test, y_test)
    
    print("\n2. Testing Decision Tree:")
    dt = create_decision_tree(use_gridsearch=False)
    metrics = train_and_evaluate_classifier(dt, X_train, y_train, X_test, y_test)
    
    print("\n3. Testing Random Forest:")
    rf = create_random_forest(use_gridsearch=False)
    metrics = train_and_evaluate_classifier(rf, X_train, y_train, X_test, y_test)
    
    print("\nAll tests passed!")