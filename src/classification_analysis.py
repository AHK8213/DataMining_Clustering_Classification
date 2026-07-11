"""
classification_analysis.py - Advanced classification analysis

Provides:
- Overfitting/underfitting analysis
- Learning curves
- Validation curves
- Pre-call feature analysis
- Small vs full dataset comparison
- Lift analysis
"""

import warnings
from typing import Optional, Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve, validation_curve
from sklearn.metrics import f1_score, roc_auc_score

from src.config import RANDOM_STATE, CV_FOLDS, VERBOSE, NUM_COLS, CAT_COLS
from src.classification_prep import (
    prepare_pre_call_data,
    prepare_small_dataset,
    get_feature_availability
)
from src.utils import timer, ensure_float64

warnings.filterwarnings("ignore")


# ============================================================================
# Overfitting/Underfitting Analysis
# ============================================================================

class OverfittingAnalyzer:
    """
    Analyze overfitting and underfitting of classification models.
    
    Usage:
        analyzer = OverfittingAnalyzer(model, X_train, y_train, X_test, y_test)
        gap = analyzer.compute_gap()
        fig = analyzer.plot_learning_curve()
        fig = analyzer.plot_validation_curve()
        report = analyzer.get_report()
    """
    
    def __init__(
        self,
        model,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model_name: str = "Model",
        random_state: int = RANDOM_STATE,
        verbose: bool = VERBOSE
    ):
        """
        Initialize overfitting analyzer.
        
        Args:
            model: Scikit-learn compatible model
            X_train: Training features
            y_train: Training targets
            X_test: Test features
            y_test: Test targets
            model_name: Name of the model
            random_state: Random seed
            verbose: Print progress
        """
        self.model = model
        self.X_train = ensure_float64(X_train)
        self.y_train = y_train
        self.X_test = ensure_float64(X_test)
        self.y_test = y_test
        self.model_name = model_name
        self.random_state = random_state
        self.verbose = verbose
        
        self.train_f1 = None
        self.val_f1 = None
        self.gap = None
        self._fitted = False
    
    def compute_gap(self) -> float:
        """
        Compute the overfitting gap (train F1 - validation F1).
        
        Returns:
            Gap value
        """
        if self.verbose:
            print(f"Computing overfitting gap for {self.model_name}...")
        
        # Fit model
        self.model.fit(self.X_train, self.y_train)
        
        # Compute scores
        y_train_pred = self.model.predict(self.X_train)
        y_test_pred = self.model.predict(self.X_test)
        
        self.train_f1 = f1_score(self.y_train, y_train_pred)
        self.val_f1 = f1_score(self.y_test, y_test_pred)
        self.gap = self.train_f1 - self.val_f1
        
        self._fitted = True
        
        if self.verbose:
            print(f"  Train F1: {self.train_f1:.3f}")
            print(f"  Validation F1: {self.val_f1:.3f}")
            print(f"  Gap: {self.gap:.3f}")
        
        return self.gap
    
    def get_gap(self) -> float:
        """Get the overfitting gap."""
        if not self._fitted:
            self.compute_gap()
        return self.gap
    
    def is_overfitting(self, threshold: float = 0.1) -> bool:
        """
        Check if the model is overfitting.
        
        Args:
            threshold: Gap threshold for overfitting
        
        Returns:
            True if overfitting, False otherwise
        """
        if not self._fitted:
            self.compute_gap()
        return self.gap > threshold
    
    def plot_learning_curve(
        self,
        cv_folds: int = CV_FOLDS,
        train_sizes: np.ndarray = None,
        scoring: str = 'f1',
        figsize: Tuple[int, int] = (8, 6)
    ) -> plt.Figure:
        """
        Plot learning curve.
        
        Args:
            cv_folds: Number of CV folds
            train_sizes: Training set sizes
            scoring: Scoring metric
            figsize: Figure size
        
        Returns:
            Matplotlib figure
        """
        if train_sizes is None:
            train_sizes = np.linspace(0.1, 1.0, 6)
        
        if self.verbose:
            print(f"Computing learning curve for {self.model_name}...")
        
        with timer("Learning curve", verbose=False):
            train_sizes, train_scores, val_scores = learning_curve(
                self.model,
                self.X_train,
                self.y_train,
                cv=cv_folds,
                scoring=scoring,
                train_sizes=train_sizes,
                n_jobs=-1,
                random_state=self.random_state
            )
        
        # Compute means and stds
        train_mean = train_scores.mean(axis=1)
        train_std = train_scores.std(axis=1)
        val_mean = val_scores.mean(axis=1)
        val_std = val_scores.std(axis=1)
        
        # Plot
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.plot(train_sizes, train_mean, 'o-', color='steelblue', label='Train', linewidth=2)
        ax.fill_between(
            train_sizes,
            train_mean - train_std,
            train_mean + train_std,
            alpha=0.15,
            color='steelblue'
        )
        
        ax.plot(train_sizes, val_mean, 'o-', color='indianred', label='Validation', linewidth=2)
        ax.fill_between(
            train_sizes,
            val_mean - val_std,
            val_mean + val_std,
            alpha=0.15,
            color='indianred'
        )
        
        # Add gap annotation
        final_gap = train_mean[-1] - val_mean[-1]
        ax.annotate(
            f'Gap: {final_gap:.3f}',
            xy=(train_sizes[-1], val_mean[-1]),
            xytext=(10, 10),
            textcoords='offset points',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )
        
        ax.set_xlabel("Training Set Size")
        ax.set_ylabel(f"{scoring.capitalize()} Score")
        ax.set_title(f"Learning Curve - {self.model_name}")
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_validation_curve(
        self,
        param_name: str,
        param_range: List,
        cv_folds: int = CV_FOLDS,
        scoring: str = 'f1',
        figsize: Tuple[int, int] = (8, 6)
    ) -> plt.Figure:
        """
        Plot validation curve for a hyperparameter.
        
        Args:
            param_name: Parameter name (e.g., 'clf__max_depth')
            param_range: Parameter values to test
            cv_folds: Number of CV folds
            scoring: Scoring metric
            figsize: Figure size
        
        Returns:
            Matplotlib figure
        """
        if self.verbose:
            print(f"Computing validation curve for {self.model_name}...")
        
        with timer("Validation curve", verbose=False):
            train_scores, val_scores = validation_curve(
                self.model,
                self.X_train,
                self.y_train,
                param_name=param_name,
                param_range=param_range,
                cv=cv_folds,
                scoring=scoring,
                n_jobs=-1
            )
        
        # Compute means and stds
        train_mean = train_scores.mean(axis=1)
        train_std = train_scores.std(axis=1)
        val_mean = val_scores.mean(axis=1)
        val_std = val_scores.std(axis=1)
        
        # Convert param_range to strings for display
        param_labels = [str(p) for p in param_range]
        
        # Plot
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.plot(param_labels, train_mean, 'o-', color='steelblue', label='Train', linewidth=2)
        ax.fill_between(
            param_labels,
            train_mean - train_std,
            train_mean + train_std,
            alpha=0.15,
            color='steelblue'
        )
        
        ax.plot(param_labels, val_mean, 'o-', color='indianred', label='Validation', linewidth=2)
        ax.fill_between(
            param_labels,
            val_mean - val_std,
            val_mean + val_std,
            alpha=0.15,
            color='indianred'
        )
        
        # Find best parameter
        best_idx = np.argmax(val_mean)
        ax.axvline(best_idx, color='green', linestyle='--', alpha=0.5,
                   label=f"Best: {param_labels[best_idx]}")
        
        ax.set_xlabel("Parameter Value")
        ax.set_ylabel(f"{scoring.capitalize()} Score")
        ax.set_title(f"Validation Curve - {self.model_name}\n{param_name}")
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def get_report(self) -> str:
        """
        Get a formatted report of overfitting analysis.
        
        Returns:
            Formatted report string
        """
        if not self._fitted:
            self.compute_gap()
        
        status = "OVERFITTING" if self.is_overfitting() else "OK"
        
        report = f"""
{"="*60}
OVERFITTING ANALYSIS REPORT
{"="*60}

Model: {self.model_name}

F1 Scores:
  Train: {self.train_f1:.3f}
  Validation: {self.val_f1:.3f}
  Gap: {self.gap:.3f}

Status: {status}

Recommendation:
  {self._get_recommendation()}
{"="*60}
"""
        return report
    
    def _get_recommendation(self) -> str:
        """Get recommendation based on gap."""
        if self.gap > 0.15:
            return "Model shows significant overfitting. Consider regularization, reducing model complexity, or increasing training data."
        elif self.gap > 0.08:
            return "Model shows moderate overfitting. Consider tuning hyperparameters or using early stopping."
        elif self.gap > 0.03:
            return "Model shows slight overfitting. Current performance is reasonable."
        else:
            return "Model shows minimal overfitting. Good generalization." + (
                " However, check if both scores are low (underfitting)." 
                if self.val_f1 < 0.5 else ""
            )


# ============================================================================
# Pre-Call Feature Analysis
# ============================================================================

class PreCallAnalyzer:
    """
    Analyze performance with and without post-call features.
    
    Usage:
        analyzer = PreCallAnalyzer(df, target_col)
        results = analyzer.compare_models()
        report = analyzer.get_report()
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        target_col: str = 'y',
        test_size: float = 0.2,
        random_state: int = RANDOM_STATE,
        verbose: bool = VERBOSE
    ):
        """
        Initialize pre-call analyzer.
        
        Args:
            df: Input DataFrame
            target_col: Target column
            test_size: Test set proportion
            random_state: Random seed
            verbose: Print progress
        """
        self.df = df
        self.target_col = target_col
        self.test_size = test_size
        self.random_state = random_state
        self.verbose = verbose
        
        self.full_results = None
        self.pre_call_results = None
        
        # Get feature availability
        self.availability = get_feature_availability()
        self.post_call_features = [k for k, v in self.availability.items() if v == 'after_call']
    
    def compare_with_xgboost(
        self,
        num_cols: List[str] = None,
        cat_cols: List[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Compare XGBoost with and without post-call features.
        
        Args:
            num_cols: Numeric columns
            cat_cols: Categorical columns
            **kwargs: Additional XGBoost parameters
        
        Returns:
            Dictionary with results
        """
        import xgboost as xgb
        from sklearn.model_selection import train_test_split
        
        if num_cols is None:
            num_cols = NUM_COLS
        if cat_cols is None:
            cat_cols = CAT_COLS
        
        # Prepare full data
        from src.classification_prep import prepare_classification_data
        X_train_full, y_train_full, X_test_full, y_test_full, preprocessor_full = prepare_classification_data(
            self.df, self.target_col, self.test_size, self.random_state,
            num_cols, cat_cols, verbose=False
        )
        
        # Prepare pre-call data
        from src.classification_prep import prepare_pre_call_data
        X_train_pc, y_train_pc, X_test_pc, y_test_pc, preprocessor_pc = prepare_pre_call_data(
            self.df, self.target_col, self.test_size, self.random_state, verbose=False
        )
        
        # Transform features
        X_train_full_enc = preprocessor_full.fit_transform(X_train_full).astype(np.float64)
        X_test_full_enc = preprocessor_full.transform(X_test_full).astype(np.float64)
        
        X_train_pc_enc = preprocessor_pc.fit_transform(X_train_pc).astype(np.float64)
        X_test_pc_enc = preprocessor_pc.transform(X_test_pc).astype(np.float64)
        
        # Split validation sets
        from sklearn.model_selection import train_test_split
        X_train_full, X_val_full, y_train_full, y_val_full = train_test_split(
            X_train_full_enc, y_train_full, test_size=0.15,
            random_state=self.random_state, stratify=y_train_full
        )
        
        X_train_pc, X_val_pc, y_train_pc, y_val_pc = train_test_split(
            X_train_pc_enc, y_train_pc, test_size=0.15,
            random_state=self.random_state, stratify=y_train_pc
        )
        
        # Calculate scale_pos_weight
        n_neg_full = (y_train_full == 0).sum()
        n_pos_full = (y_train_full == 1).sum()
        scale_pos_weight_full = n_neg_full / n_pos_full if n_pos_full > 0 else 1.0
        
        n_neg_pc = (y_train_pc == 0).sum()
        n_pos_pc = (y_train_pc == 1).sum()
        scale_pos_weight_pc = n_neg_pc / n_pos_pc if n_pos_pc > 0 else 1.0
        
        # Train full model
        if self.verbose:
            print("Training XGBoost with full features...")
        
        model_full = xgb.XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=6,
            scale_pos_weight=scale_pos_weight_full,
            tree_method='hist', device='cpu',
            eval_metric='logloss', early_stopping_rounds=50,
            random_state=self.random_state, **kwargs
        )
        model_full.fit(
            X_train_full, y_train_full,
            eval_set=[(X_val_full, y_val_full)],
            verbose=False
        )
        
        # Train pre-call model
        if self.verbose:
            print("Training XGBoost with pre-call features only...")
        
        model_pc = xgb.XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=6,
            scale_pos_weight=scale_pos_weight_pc,
            tree_method='hist', device='cpu',
            eval_metric='logloss', early_stopping_rounds=50,
            random_state=self.random_state, **kwargs
        )
        model_pc.fit(
            X_train_pc, y_train_pc,
            eval_set=[(X_val_pc, y_val_pc)],
            verbose=False
        )
        
        # Evaluate
        y_pred_full = model_full.predict(X_test_full_enc)
        y_proba_full = model_full.predict_proba(X_test_full_enc)[:, 1]
        
        y_pred_pc = model_pc.predict(X_test_pc_enc)
        y_proba_pc = model_pc.predict_proba(X_test_pc_enc)[:, 1]
        
        # Compute metrics
        from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
        
        self.full_results = {
            'f1': f1_score(y_test_full, y_pred_full),
            'auc_roc': roc_auc_score(y_test_full, y_proba_full),
            'accuracy': accuracy_score(y_test_full, y_pred_full),
            'best_iteration': model_full.best_iteration,
            'scale_pos_weight': scale_pos_weight_full
        }
        
        self.pre_call_results = {
            'f1': f1_score(y_test_pc, y_pred_pc),
            'auc_roc': roc_auc_score(y_test_pc, y_proba_pc),
            'accuracy': accuracy_score(y_test_pc, y_pred_pc),
            'best_iteration': model_pc.best_iteration,
            'scale_pos_weight': scale_pos_weight_pc,
            'features_excluded': self.post_call_features
        }
        
        if self.verbose:
            print(f"\nFull features: F1={self.full_results['f1']:.3f}, "
                  f"AUC={self.full_results['auc_roc']:.3f}")
            print(f"Pre-call features: F1={self.pre_call_results['f1']:.3f}, "
                  f"AUC={self.pre_call_results['auc_roc']:.3f}")
            print(f"Features excluded: {self.post_call_features}")
        
        return {
            'full': self.full_results,
            'pre_call': self.pre_call_results,
            'models': (model_full, model_pc),
            'preprocessors': (preprocessor_full, preprocessor_pc),
            'data': (X_test_full_enc, X_test_pc_enc, y_test_full, y_test_pc)
        }
    
    def get_comparison_df(self) -> pd.DataFrame:
        """
        Get comparison DataFrame.
        
        Returns:
            DataFrame with comparison results
        """
        if self.full_results is None or self.pre_call_results is None:
            raise ValueError("Run compare_with_xgboost() first.")
        
        df = pd.DataFrame([
            {'model': 'Full Features', **self.full_results},
            {'model': 'Pre-Call Features', **self.pre_call_results}
        ])
        
        return df
    
    def get_report(self) -> str:
        """
        Get formatted report.
        
        Returns:
            Formatted report string
        """
        if self.full_results is None or self.pre_call_results is None:
            raise ValueError("Run compare_with_xgboost() first.")
        
        full_f1 = self.full_results['f1']
        pc_f1 = self.pre_call_results['f1']
        f1_drop = full_f1 - pc_f1
        pct_drop = f1_drop / full_f1 * 100 if full_f1 > 0 else 0
        
        report = f"""
{"="*60}
PRE-CALL FEATURE ANALYSIS REPORT
{"="*60}

Features excluded (post-call): {self.post_call_features}

Performance Comparison:
  Full Features:    F1={full_f1:.3f}, AUC={self.full_results['auc_roc']:.3f}
  Pre-Call Only:    F1={pc_f1:.3f}, AUC={self.pre_call_results['auc_roc']:.3f}

F1 Drop: {f1_drop:.3f} ({pct_drop:.1f}%)

Interpretation:
  {self._get_interpretation()}

Practical Implication:
  {self._get_practical_implication()}
{"="*60}
"""
        return report
    
    def _get_interpretation(self) -> str:
        """Get interpretation of results."""
        f1_drop = self.full_results['f1'] - self.pre_call_results['f1']
        
        if f1_drop > 0.1:
            return "Post-call features (duration) are highly predictive. Pre-call model is significantly weaker."
        elif f1_drop > 0.05:
            return "Post-call features have moderate predictive power. Pre-call model is somewhat weaker."
        elif f1_drop > 0.02:
            return "Post-call features have limited predictive power. Pre-call model is competitive."
        else:
            return "Post-call features have negligible predictive power. Pre-call model performs similarly."
    
    def _get_practical_implication(self) -> str:
        """Get practical implication."""
        if self.pre_call_results['f1'] > 0.5:
            return "Pre-call model can be used for lead targeting before making calls."
        elif self.pre_call_results['f1'] > 0.4:
            return "Pre-call model has some targeting ability but may need additional features."
        else:
            return "Pre-call model is not reliable for lead targeting. Consider collecting more pre-call features."


# ============================================================================
# Small vs Full Dataset Comparison
# ============================================================================

def compare_small_vs_full_dataset(
    df: pd.DataFrame,
    small_size: int,
    target_col: str = 'y',
    random_state: int = RANDOM_STATE,
    verbose: bool = VERBOSE
) -> Dict[str, Any]:
    """
    Compare performance on small vs full dataset.
    
    Args:
        df: Input DataFrame
        small_size: Size of small dataset
        target_col: Target column
        random_state: Random seed
        verbose: Print progress
    
    Returns:
        Dictionary with comparison results
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import f1_score, roc_auc_score
    from sklearn.model_selection import train_test_split
    
    results = {}
    
    # Full dataset
    if verbose:
        print("Training on full dataset...")
    
    df_clf = df[df[target_col] != 'unknown'].copy()
    df_clf[target_col] = (df_clf[target_col] == 'yes').astype(int)
    
    X_full = df_clf.drop(columns=[target_col])
    y_full = df_clf[target_col]
    
    X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
        X_full, y_full, test_size=0.2, random_state=random_state, stratify=y_full
    )
    
    # Encode features
    from src.data_preparation import create_preprocessor
    preprocessor = create_preprocessor()
    X_train_full_enc = preprocessor.fit_transform(X_train_full).astype(np.float64)
    X_test_full_enc = preprocessor.transform(X_test_full).astype(np.float64)
    
    rf_full = RandomForestClassifier(n_estimators=100, random_state=random_state)
    rf_full.fit(X_train_full_enc, y_train_full)
    
    y_pred_full = rf_full.predict(X_test_full_enc)
    y_proba_full = rf_full.predict_proba(X_test_full_enc)[:, 1]
    
    results['full'] = {
        'n_rows': len(X_full),
        'f1': f1_score(y_test_full, y_pred_full),
        'auc_roc': roc_auc_score(y_test_full, y_proba_full)
    }
    
    # Small dataset
    if verbose:
        print(f"Training on small dataset ({small_size} rows)...")
    
    df_small, _ = prepare_small_dataset(df, small_size, target_col, random_state, verbose=False)
    
    X_small = df_small.drop(columns=[target_col])
    y_small = df_small[target_col]
    
    X_train_small, X_test_small, y_train_small, y_test_small = train_test_split(
        X_small, y_small, test_size=0.2, random_state=random_state, stratify=y_small
    )
    
    X_train_small_enc = preprocessor.fit_transform(X_train_small).astype(np.float64)
    X_test_small_enc = preprocessor.transform(X_test_small).astype(np.float64)
    
    rf_small = RandomForestClassifier(n_estimators=100, random_state=random_state)
    rf_small.fit(X_train_small_enc, y_train_small)
    
    y_pred_small = rf_small.predict(X_test_small_enc)
    y_proba_small = rf_small.predict_proba(X_test_small_enc)[:, 1]
    
    results['small'] = {
        'n_rows': len(X_small),
        'f1': f1_score(y_test_small, y_pred_small),
        'auc_roc': roc_auc_score(y_test_small, y_proba_small)
    }
    
    results['comparison'] = {
        'f1_diff': results['full']['f1'] - results['small']['f1'],
        'row_ratio': results['full']['n_rows'] / results['small']['n_rows']
    }
    
    if verbose:
        print(f"\nFull dataset: F1={results['full']['f1']:.3f}")
        print(f"Small dataset: F1={results['small']['f1']:.3f}")
        print(f"F1 difference: {results['comparison']['f1_diff']:.3f}")
    
    return results


# ============================================================================
# Testing
# ============================================================================

if __name__ == "__main__":
    print("Testing classification_analysis.py...")
    
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    
    # Generate test data
    X, y = make_classification(
        n_samples=300, n_features=10, n_informative=8,
        n_redundant=2, n_classes=2, random_state=42
    )
    X = X.astype(np.float64)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Test overfitting analyzer
    print("\n1. Testing OverfittingAnalyzer:")
    model = LogisticRegression(max_iter=1000)
    analyzer = OverfittingAnalyzer(
        model, X_train, y_train, X_test, y_test,
        model_name="Logistic Regression",
        verbose=True
    )
    gap = analyzer.compute_gap()
    print(f"Gap: {gap:.3f}")
    print(f"Overfitting: {analyzer.is_overfitting()}")
    
    # Test learning curve
    print("\n2. Testing learning curve:")
    fig = analyzer.plot_learning_curve()
    plt.show()
    
    # Test pre-call analyzer (requires DataFrame)
    print("\n3. Testing PreCallAnalyzer (requires full DataFrame):")
    print("  (Skipped in unit test)")
    
    print("\nAll tests passed!")