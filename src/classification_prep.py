"""
classification_prep.py - Data preparation for classification

Handles:
- Train/test split with stratification
- Preprocessing pipelines for classification
- Feature availability analysis (pre-call vs post-call)
- Small vs full dataset preparation
"""

import warnings
from typing import Tuple, Optional, Dict, List, Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

from src.config import (
    RANDOM_STATE,
    TEST_SIZE,
    NUM_COLS,
    CAT_COLS,
    TARGET_COL,
    USE_FLOAT64,
    VERBOSE,
)
from src.utils import ensure_float64, get_sample
from src.data_preparation import create_preprocessor

warnings.filterwarnings("ignore")


# ============================================================================
# Classification Data Preparation
# ============================================================================

def prepare_classification_data(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
    num_cols: List[str] = None,
    cat_cols: List[str] = None,
    verbose: bool = VERBOSE
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, ColumnTransformer]:
    """
    Prepare data for classification with train/test split.
    
    Args:
        df: Input DataFrame
        target_col: Target column name
        test_size: Test set proportion
        random_state: Random seed
        num_cols: Numeric columns (default: NUM_COLS)
        cat_cols: Categorical columns (default: CAT_COLS)
        verbose: Print progress
    
    Returns:
        Tuple of (X_train, y_train, X_test, y_test, preprocessor)
    """
    if num_cols is None:
        num_cols = NUM_COLS
    if cat_cols is None:
        cat_cols = CAT_COLS
    
    # Filter unknown target
    df_clf = df[df[target_col] != 'unknown'].copy()
    df_clf[target_col] = (df_clf[target_col] == 'yes').astype(int)
    
    if verbose:
        print(f"Classification data shape: {df_clf.shape}")
        print(f"Class distribution:")
        print(df_clf[target_col].value_counts())
        print(f"Class proportions:")
        print(df_clf[target_col].value_counts(normalize=True))
    
    # Split features and target
    X = df_clf.drop(columns=[target_col])
    y = df_clf[target_col]
    
    # Train-test split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    
    if verbose:
        print(f"Train: {X_train.shape[0]:,}, Test: {X_test.shape[0]:,}")
        print(f"Train class distribution:\n{y_train.value_counts(normalize=True)}")
        print(f"Test class distribution:\n{y_test.value_counts(normalize=True)}")
    
    # Create preprocessor
    preprocessor = create_preprocessor(num_cols, cat_cols)
    
    return X_train, y_train, X_test, y_test, preprocessor


def prepare_classification_data_with_engineered(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
    verbose: bool = VERBOSE
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, ColumnTransformer]:
    """
    Prepare data for classification with engineered features.
    
    Args:
        df: Input DataFrame (already with engineered features)
        target_col: Target column name
        test_size: Test set proportion
        random_state: Random seed
        verbose: Print progress
    
    Returns:
        Tuple of (X_train, y_train, X_test, y_test, preprocessor)
    """
    from src.data_preparation import create_preprocessor_with_engineered
    
    # Get all numeric columns including engineered
    all_num_cols = [c for c in df.columns if c not in CAT_COLS + [target_col] 
                    and df[c].dtype in ['float64', 'int64', 'int32', 'float32']]
    all_cat_cols = [c for c in CAT_COLS if c in df.columns]
    
    # Filter unknown target
    df_clf = df[df[target_col] != 'unknown'].copy()
    df_clf[target_col] = (df_clf[target_col] == 'yes').astype(int)
    
    # Split
    X = df_clf.drop(columns=[target_col])
    y = df_clf[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    
    # Create preprocessor with engineered features
    preprocessor = create_preprocessor_with_engineered(
        num_cols=all_num_cols,
        cat_cols=all_cat_cols
    )
    
    if verbose:
        print(f"Train: {X_train.shape[0]:,}, Test: {X_test.shape[0]:,}")
        print(f"Features: {len(all_num_cols)} numeric + {len(all_cat_cols)} categorical + engineered")
    
    return X_train, y_train, X_test, y_test, preprocessor


# ============================================================================
# Feature Availability (Pre-Call vs Post-Call)
# ============================================================================

def get_feature_availability() -> Dict[str, str]:
    """
    Get feature availability mapping for pre-call vs post-call analysis.
    
    Returns:
        Dictionary mapping feature name to 'before_call' or 'after_call'
    """
    availability = {
        'age': 'before_call',
        'job': 'before_call',
        'marital': 'before_call',
        'education': 'before_call',
        'default': 'before_call',
        'balance': 'before_call',
        'housing': 'before_call',
        'loan': 'before_call',
        'day': 'before_call',
        'month': 'before_call',
        'campaign': 'before_call',
        'duration': 'after_call',
    }
    return availability


def get_pre_call_features(
    num_cols: List[str] = None,
    cat_cols: List[str] = None
) -> Tuple[List[str], List[str]]:
    """
    Get pre-call features only (exclude duration).
    
    Args:
        num_cols: List of numeric columns
        cat_cols: List of categorical columns
    
    Returns:
        Tuple of (pre_call_num, pre_call_cat)
    """
    if num_cols is None:
        num_cols = NUM_COLS
    if cat_cols is None:
        cat_cols = CAT_COLS
    
    availability = get_feature_availability()
    
    pre_call_num = [c for c in num_cols if availability.get(c) == 'before_call']
    pre_call_cat = [c for c in cat_cols if availability.get(c) == 'before_call']
    
    return pre_call_num, pre_call_cat


def prepare_pre_call_data(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
    verbose: bool = VERBOSE
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, ColumnTransformer]:
    """
    Prepare data using only pre-call features (excludes duration).
    
    Args:
        df: Input DataFrame
        target_col: Target column name
        test_size: Test set proportion
        random_state: Random seed
        verbose: Print progress
    
    Returns:
        Tuple of (X_train, y_train, X_test, y_test, preprocessor)
    """
    pre_call_num, pre_call_cat = get_pre_call_features()
    
    if verbose:
        print(f"Using pre-call features only: {pre_call_num + pre_call_cat}")
        print(f"Excluding: duration (post-call feature)")
    
    return prepare_classification_data(
        df=df,
        target_col=target_col,
        test_size=test_size,
        random_state=random_state,
        num_cols=pre_call_num,
        cat_cols=pre_call_cat,
        verbose=verbose
    )


# ============================================================================
# Small Dataset Preparation
# ============================================================================

def prepare_small_dataset(
    df: pd.DataFrame,
    n_samples: int,
    target_col: str = TARGET_COL,
    random_state: int = RANDOM_STATE,
    verbose: bool = VERBOSE
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Create a smaller version of the dataset for comparison experiments.
    
    Args:
        df: Input DataFrame
        n_samples: Number of samples to keep
        target_col: Target column name
        random_state: Random seed
        verbose: Print progress
    
    Returns:
        Tuple of (small_df, indices)
    """
    # Filter unknown target for sampling
    df_valid = df[df[target_col] != 'unknown'].copy()
    
    # Stratified sampling to maintain class distribution
    from sklearn.model_selection import StratifiedShuffleSplit
    
    splitter = StratifiedShuffleSplit(
        n_splits=1,
        train_size=min(n_samples / len(df_valid), 1.0),
        random_state=random_state
    )
    
    idx_train, _ = next(splitter.split(df_valid, df_valid[target_col]))
    small_df = df_valid.iloc[idx_train].reset_index(drop=True)
    
    if verbose:
        print(f"Small dataset: {len(small_df):,} rows "
              f"({len(small_df)/len(df_valid)*100:.1f}% of full)")
        print(f"Class distribution:\n{small_df[target_col].value_counts(normalize=True)}")
    
    return small_df, idx_train


# ============================================================================
# Preprocessing Transformers
# ============================================================================

def get_preprocessor_for_classification(
    num_cols: List[str] = None,
    cat_cols: List[str] = None,
    use_float64: bool = USE_FLOAT64
) -> ColumnTransformer:
    """
    Get preprocessor for classification.
    
    Args:
        num_cols: Numeric columns
        cat_cols: Categorical columns
        use_float64: Use float64
    
    Returns:
        ColumnTransformer
    """
    return create_preprocessor(num_cols, cat_cols)


def transform_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    preprocessor: ColumnTransformer
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Transform features using the preprocessor.
    
    Args:
        X_train: Training features
        X_test: Test features
        preprocessor: Fitted preprocessor
    
    Returns:
        Tuple of (X_train_transformed, X_test_transformed)
    """
    X_train_transformed = preprocessor.fit_transform(X_train).astype(np.float64)
    X_test_transformed = preprocessor.transform(X_test).astype(np.float64)
    
    return X_train_transformed, X_test_transformed


# ============================================================================
# Testing
# ============================================================================

if __name__ == "__main__":
    print("Testing classification_prep.py...")
    
    from src.data_preparation import load_data, clean_data, engineer_features
    
    # Load data
    df = load_data(verbose=False)
    df = clean_data(df, verbose=False)
    df = engineer_features(df, verbose=False)
    
    # Test basic preparation
    print("\n1. Testing basic classification preparation:")
    X_train, y_train, X_test, y_test, preprocessor = prepare_classification_data(
        df, verbose=True
    )
    
    # Test pre-call preparation
    print("\n2. Testing pre-call preparation:")
    X_train_pc, y_train_pc, X_test_pc, y_test_pc, preprocessor_pc = prepare_pre_call_data(
        df, verbose=True
    )
    
    # Test small dataset
    print("\n3. Testing small dataset preparation:")
    df_small, idx = prepare_small_dataset(df, n_samples=5000, verbose=True)
    
    print("\nAll tests passed!")