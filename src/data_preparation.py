"""
data_preparation.py - Data loading, cleaning, and preprocessing for Project 3

Handles:
- Data loading from GitHub or local file
- Data cleaning and validation
- Feature engineering
- Feature selection and importance analysis
- PCA transformation
- Preprocessing pipelines for clustering and classification
"""

import warnings
from typing import Tuple, Optional, Dict, Any, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif

# Import configuration and utilities
from src.config import (
    DATA_URL,
    DATA_FILE,
    RANDOM_STATE,
    NUM_COLS,
    CAT_COLS,
    TARGET_COL,
    ENGINEERED_NUM_COLS,
    ENGINEERED_FLAG_COLS,
    USE_FLOAT64,
    VERBOSE,
)
from src.utils import (
    reduce_memory,
    cleanup,
    get_rng,
    timer,
    ensure_float64,
)

warnings.filterwarnings("ignore")


# ============================================================================
# Data Loading
# ============================================================================

def load_data(
    url: str = DATA_URL,
    local_path: Optional[str] = None,
    verbose: bool = VERBOSE
) -> pd.DataFrame:
    """
    Load the bank marketing dataset from GitHub or local file.
    
    Args:
        url: GitHub raw URL for the dataset
        local_path: Local file path (optional)
        verbose: Print progress
    
    Returns:
        Loaded DataFrame
    """
    try:
        if local_path and DATA_FILE.exists():
            if verbose:
                print(f"Loading dataset from local file: {DATA_FILE}")
            df = pd.read_excel(DATA_FILE)
        else:
            if verbose:
                print(f"Loading dataset from URL: {url}")
            df = pd.read_excel(url)
            
            # Save local copy for future use
            DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
            df.to_excel(DATA_FILE, index=False)
            if verbose:
                print(f"Saved local copy to: {DATA_FILE}")
    
    except Exception as e:
        print(f"Error loading data: {e}")
        raise
    
    if verbose:
        print(f"Loaded data shape: {df.shape}")
    
    return df


# ============================================================================
# Data Cleaning
# ============================================================================

def clean_data(
    df: pd.DataFrame,
    verbose: bool = VERBOSE
) -> pd.DataFrame:
    """
    Clean the dataset by:
    - Dropping constant columns
    - Removing duplicates
    - Fixing numerical ranges
    - Validating categorical values
    
    Args:
        df: Input DataFrame
        verbose: Print progress
    
    Returns:
        Cleaned DataFrame
    """
    df_clean = df.copy()
    n_original = len(df_clean)
    
    # 1. Drop constant columns
    const_cols = [c for c in df_clean.columns if df_clean[c].nunique(dropna=True) <= 1]
    if const_cols and verbose:
        print(f"Dropping constant columns: {const_cols}")
    df_clean = df_clean.drop(columns=const_cols)
    
    # 2. Remove duplicates
    n_dupes = df_clean.duplicated().sum()
    if n_dupes > 0 and verbose:
        print(f"Duplicated rows found: {n_dupes}")
    df_clean = df_clean.drop_duplicates().reset_index(drop=True)
    
    # 3. Fix numerical ranges
    # Day: 1-31
    df_clean.loc[(df_clean['day'] < 1) | (df_clean['day'] > 31), 'day'] = np.nan
    df_clean['day'] = df_clean['day'].fillna(df_clean['day'].median())
    
    # Age: 18-90
    df_clean.loc[(df_clean['age'] < 18) | (df_clean['age'] > 90), 'age'] = np.nan
    df_clean['age'] = df_clean['age'].fillna(df_clean['age'].median())
    
    # Duration: non-negative
    df_clean.loc[df_clean['duration'] < 0, 'duration'] = np.nan
    df_clean['duration'] = df_clean['duration'].fillna(df_clean['duration'].median())
    
    # Campaign: >= 1
    df_clean.loc[df_clean['campaign'] < 1, 'campaign'] = np.nan
    df_clean['campaign'] = df_clean['campaign'].fillna(df_clean['campaign'].median())
    
    # 4. Validate categorical values
    cat_cols = [c for c in CAT_COLS + [TARGET_COL] if c in df_clean.columns]
    for c in cat_cols:
        if c in df_clean.columns:
            df_clean[c] = df_clean[c].astype(str).str.strip().str.lower().astype('category')
    
    n_cleaned = len(df_clean)
    if verbose:
        print(f"Cleaned shape: {df_clean.shape}")
        print(f"Rows removed: {n_original - n_cleaned}")
        print(f"Missing values after cleaning: {df_clean.isna().sum().sum()}")
    
    return df_clean


# ============================================================================
# Feature Engineering
# ============================================================================

def engineer_features(
    df: pd.DataFrame,
    verbose: bool = VERBOSE
) -> pd.DataFrame:
    """
    Engineer new features:
    - balance_per_age = balance / age
    - campaign_intensity = campaign / day
    - is_high_balance = balance > median balance
    - is_long_call = duration > median duration
    - Merge rare categories into 'other'
    
    Args:
        df: Input DataFrame
        verbose: Print progress
    
    Returns:
        DataFrame with engineered features
    """
    df_eng = df.copy()
    
    # 1. Derived features
    df_eng['balance_per_age'] = df_eng['balance'] / df_eng['age'].replace(0, np.nan)
    df_eng['balance_per_age'] = df_eng['balance_per_age'].fillna(0)
    
    df_eng['campaign_intensity'] = df_eng['campaign'] / df_eng['day'].replace(0, np.nan)
    df_eng['campaign_intensity'] = df_eng['campaign_intensity'].fillna(0)
    
    # 2. Binary flags
    df_eng['is_high_balance'] = (df_eng['balance'] > df_eng['balance'].median()).astype('int8')
    df_eng['is_long_call'] = (df_eng['duration'] > df_eng['duration'].median()).astype('int8')
    
    # 3. Merge rare categories in categorical columns
    cat_cols = [c for c in CAT_COLS if c in df_eng.columns]
    for c in cat_cols:
        freq = df_eng[c].value_counts(normalize=True)
        rare_levels = freq[freq < 0.005].index.tolist()
        if rare_levels:
            df_eng[c] = df_eng[c].astype(str)
            df_eng.loc[df_eng[c].isin(rare_levels), c] = 'other'
            df_eng[c] = df_eng[c].astype('category')
            if verbose:
                print(f"{c}: merged rare levels {rare_levels} -> 'other'")
    
    if verbose:
        print(f"\nEngineered features added: {ENGINEERED_NUM_COLS + ENGINEERED_FLAG_COLS}")
    
    return df_eng


def remove_highly_correlated_features(
    df: pd.DataFrame,
    num_cols: List[str],
    threshold: float = 0.8,
    verbose: bool = VERBOSE
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Remove highly correlated features (> threshold).
    
    Args:
        df: Input DataFrame
        num_cols: List of numeric columns to check
        threshold: Correlation threshold
        verbose: Print progress
    
    Returns:
        Tuple of (DataFrame with removed columns, list of removed columns)
    """
    corr = df[num_cols].corr()
    high_corr_pairs = []
    
    for i, c1 in enumerate(corr.columns):
        for c2 in corr.columns[i+1:]:
            if abs(corr.loc[c1, c2]) > threshold:
                high_corr_pairs.append((c1, c2, corr.loc[c1, c2]))
    
    if high_corr_pairs and verbose:
        print(f"Highly correlated feature pairs (> {threshold}):")
        for c1, c2, val in high_corr_pairs:
            print(f"  {c1} - {c2}: {val:.3f}")
    
    # Drop the second column of each pair (keeping the first)
    cols_to_drop = [c2 for _, c2, _ in high_corr_pairs]
    if cols_to_drop:
        df = df.drop(columns=list(set(cols_to_drop)))
        if verbose:
            print(f"Dropped columns: {list(set(cols_to_drop))}")
    
    return df, cols_to_drop


# ============================================================================
# Feature Selection
# ============================================================================

def compute_feature_importance(
    df: pd.DataFrame,
    preprocessor: ColumnTransformer,
    target_col: str = TARGET_COL,
    n_estimators: int = 200,
    random_state: int = RANDOM_STATE,
    verbose: bool = VERBOSE
) -> Tuple[pd.Series, pd.Series]:
    """
    Compute feature importance using Random Forest and Mutual Information.
    
    Args:
        df: Input DataFrame
        preprocessor: Preprocessing pipeline
        target_col: Target column name
        n_estimators: Number of trees for Random Forest
        random_state: Random seed
        verbose: Print progress
    
    Returns:
        Tuple of (RF importances, MI scores)
    """
    # Prepare data
    df_clf = df[df[target_col] != 'unknown'].copy()
    df_clf[target_col] = (df_clf[target_col] == 'yes').astype(int)
    
    X = preprocessor.fit_transform(df_clf.drop(columns=[target_col])).astype(np.float64)
    y = df_clf[target_col].values
    
    # Random Forest importance
    if verbose:
        print("Computing Random Forest feature importance...")
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        class_weight='balanced',
        n_jobs=-1
    )
    rf.fit(X, y)
    rf_importances = pd.Series(
        rf.feature_importances_,
        index=preprocessor.get_feature_names_out()
    ).sort_values(ascending=False)
    
    # Mutual Information
    if verbose:
        print("Computing Mutual Information scores...")
    mi_scores = mutual_info_classif(X, y, random_state=random_state)
    mi_importances = pd.Series(
        mi_scores,
        index=preprocessor.get_feature_names_out()
    ).sort_values(ascending=False)
    
    return rf_importances, mi_importances


def get_combined_feature_ranking(
    rf_importances: pd.Series,
    mi_importances: pd.Series
) -> pd.DataFrame:
    """
    Combine RF importance and MI scores into a single ranking.
    
    Args:
        rf_importances: Random Forest importances
        mi_importances: Mutual Information scores
    
    Returns:
        DataFrame with combined ranking
    """
    rf_norm = rf_importances / rf_importances.sum()
    mi_norm = mi_importances / mi_importances.sum()
    
    ranking = pd.DataFrame({
        'rf_importance': rf_norm,
        'mutual_info': mi_norm
    }).fillna(0)
    
    ranking['combined_score'] = ranking.mean(axis=1)
    ranking = ranking.sort_values('combined_score', ascending=False)
    
    return ranking


# ============================================================================
# Preprocessing Pipelines
# ============================================================================

def create_preprocessor(
    num_cols: List[str] = NUM_COLS,
    cat_cols: List[str] = CAT_COLS,
    use_float64: bool = USE_FLOAT64
) -> ColumnTransformer:
    """
    Create a preprocessing pipeline for the dataset.
    
    Args:
        num_cols: List of numeric columns
        cat_cols: List of categorical columns
        use_float64: Use float64 for numeric data
    
    Returns:
        ColumnTransformer for preprocessing
    """
    dtype = np.float64 if use_float64 else np.float32
    
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
    ])
    
    return preprocessor


def create_preprocessor_with_engineered(
    num_cols: List[str] = None,
    cat_cols: List[str] = None,
    flag_cols: List[str] = None,
    use_float64: bool = USE_FLOAT64
) -> ColumnTransformer:
    """
    Create a preprocessing pipeline including engineered features.
    
    Args:
        num_cols: List of numeric columns (including engineered)
        cat_cols: List of categorical columns
        flag_cols: List of binary flag columns
        use_float64: Use float64 for numeric data
    
    Returns:
        ColumnTransformer for preprocessing
    """
    if num_cols is None:
        num_cols = NUM_COLS + ENGINEERED_NUM_COLS
    if cat_cols is None:
        cat_cols = CAT_COLS
    if flag_cols is None:
        flag_cols = ENGINEERED_FLAG_COLS
    
    dtype = np.float64 if use_float64 else np.float32
    
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols),
        ('flag', 'passthrough', flag_cols)
    ])
    
    return preprocessor


# ============================================================================
# PCA
# ============================================================================

def fit_pca(
    X: np.ndarray,
    n_components: int = 2,
    random_state: int = RANDOM_STATE,
    verbose: bool = VERBOSE
) -> PCA:
    """
    Fit PCA on the dataset.
    
    Args:
        X: Input data
        n_components: Number of PCA components
        random_state: Random seed
        verbose: Print progress
    
    Returns:
        Fitted PCA transformer
    """
    if verbose:
        print(f"Fitting PCA with {n_components} components on {X.shape[0]:,} rows...")
    
    pca = PCA(n_components=n_components, random_state=random_state)
    pca.fit(X)
    
    if verbose:
        explained = pca.explained_variance_ratio_
        total = explained.sum()
        print(f"Explained variance: PC1={explained[0]:.2%}, PC2={explained[1]:.2%}")
        print(f"Total explained variance: {total:.2%}")
    
    return pca


def plot_pca_variance(
    pca: PCA,
    figsize: Tuple[int, int] = (8, 5)
) -> plt.Figure:
    """
    Plot the cumulative explained variance for PCA components.
    
    Args:
        pca: Fitted PCA transformer
        figsize: Figure size
    
    Returns:
        Matplotlib figure
    """
    cum_var = np.cumsum(pca.explained_variance_ratio_)
    n_components = len(cum_var)
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(range(1, n_components + 1), cum_var, marker='o', linewidth=2)
    ax.axhline(0.9, color='red', linestyle='--', alpha=0.6, label='90% variance')
    ax.set_xlabel("Number of components")
    ax.set_ylabel("Cumulative explained variance")
    ax.set_title("PCA — Cumulative explained variance")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig


# ============================================================================
# Complete Pipeline
# ============================================================================

def prepare_data_for_clustering(
    url: str = DATA_URL,
    num_cols: List[str] = NUM_COLS,
    cat_cols: List[str] = CAT_COLS,
    verbose: bool = VERBOSE
) -> Tuple[pd.DataFrame, np.ndarray, ColumnTransformer]:
    """
    Complete data preparation pipeline for clustering.
    
    Args:
        url: Data URL
        num_cols: Numeric columns
        cat_cols: Categorical columns
        verbose: Print progress
    
    Returns:
        Tuple of (cleaned DataFrame, feature matrix, preprocessor)
    """
    with timer("Data preparation", verbose=verbose):
        # 1. Load data
        df = load_data(url, verbose=verbose)
        
        # 2. Clean data
        df = clean_data(df, verbose=verbose)
        
        # 3. Feature engineering
        df = engineer_features(df, verbose=verbose)
        
        # 4. Create preprocessor
        preprocessor = create_preprocessor(num_cols, cat_cols)
        
        # 5. Transform data
        X_cluster_df = df.drop(columns=[TARGET_COL])
        X = preprocessor.fit_transform(X_cluster_df).astype(np.float64)
        
        if verbose:
            print(f"Feature matrix shape: {X.shape}")
            print(f"Feature matrix dtype: {X.dtype}")
            print(f"Memory usage: {X.nbytes / 1024**2:.2f} MB")
    
    return df, X, preprocessor


def prepare_data_for_classification(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    num_cols: List[str] = NUM_COLS,
    cat_cols: List[str] = CAT_COLS,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
    verbose: bool = VERBOSE
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, ColumnTransformer]:
    """
    Prepare data for classification with train/test split.
    
    Args:
        df: Input DataFrame
        target_col: Target column name
        num_cols: Numeric columns
        cat_cols: Categorical columns
        test_size: Test set proportion
        random_state: Random seed
        verbose: Print progress
    
    Returns:
        Tuple of (X_train, y_train, X_test, y_test, preprocessor)
    """
    from sklearn.model_selection import train_test_split
    
    # Filter unknown target
    df_clf = df[df[target_col] != 'unknown'].copy()
    df_clf[target_col] = (df_clf[target_col] == 'yes').astype(int)
    
    if verbose:
        print(f"Classification data shape: {df_clf.shape}")
        print(f"Class distribution:\n{df_clf[target_col].value_counts()}")
        print(f"Class proportions:\n{df_clf[target_col].value_counts(normalize=True)}")
    
    # Split features and target
    X = df_clf.drop(columns=[target_col])
    y = df_clf[target_col]
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    
    if verbose:
        print(f"Train: {X_train.shape[0]:,}, Test: {X_test.shape[0]:,}")
    
    # Create preprocessor
    preprocessor = create_preprocessor(num_cols, cat_cols)
    
    return X_train, y_train, X_test, y_test, preprocessor


# ============================================================================
# Visualization Helpers
# ============================================================================

def plot_distributions(
    df: pd.DataFrame,
    num_cols: List[str],
    figsize: Tuple[int, int] = (15, 8)
) -> plt.Figure:
    """
    Plot distributions of numeric columns.
    
    Args:
        df: Input DataFrame
        num_cols: List of numeric columns
        figsize: Figure size
    
    Returns:
        Matplotlib figure
    """
    n_cols = min(3, len(num_cols))
    n_rows = int(np.ceil(len(num_cols) / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = np.array(axes).flatten()
    
    for ax, col in zip(axes, num_cols):
        sns.histplot(df[col], kde=True, ax=ax)
        ax.set_title(col)
        ax.set_xlabel('')
    
    # Hide unused subplots
    for ax in axes[len(num_cols):]:
        ax.axis('off')
    
    plt.tight_layout()
    return fig


def plot_target_distribution(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    figsize: Tuple[int, int] = (6, 4)
) -> plt.Figure:
    """
    Plot the distribution of the target variable.
    
    Args:
        df: Input DataFrame
        target_col: Target column name
        figsize: Figure size
    
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Filter unknown
    df_plot = df[df[target_col] != 'unknown']
    
    colors = ['#4C72B0', '#DD8452']
    df_plot[target_col].value_counts().plot(kind='bar', ax=ax, color=colors)
    ax.set_title(f"Target '{target_col}' distribution")
    ax.set_xlabel('')
    ax.set_ylabel('Count')
    
    # Add percentage labels
    total = len(df_plot)
    for i, v in enumerate(df_plot[target_col].value_counts().values):
        ax.text(i, v + total*0.01, f'{v/total:.1%}', ha='center')
    
    return fig


# ============================================================================
# Testing
# ============================================================================

if __name__ == "__main__":
    print("Testing data_preparation.py...")
    
    # Test loading
    df = load_data(verbose=True)
    print(f"Loaded {len(df)} rows")
    
    # Test cleaning
    df_clean = clean_data(df, verbose=True)
    
    # Test feature engineering
    df_eng = engineer_features(df_clean, verbose=True)
    
    # Test preprocessing
    preprocessor = create_preprocessor()
    X = preprocessor.fit_transform(df_eng.drop(columns=[TARGET_COL]))
    print(f"Preprocessed shape: {X.shape}")
    
    # Test PCA
    pca = fit_pca(X, verbose=True)
    X_pca = pca.transform(X)
    print(f"PCA shape: {X_pca.shape}")
    
    print("\nAll tests passed!")