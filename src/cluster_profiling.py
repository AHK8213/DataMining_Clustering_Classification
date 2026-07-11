"""
cluster_profiling.py - Cluster profiling and characterization

Provides:
- Numerical feature profiling per cluster
- Categorical feature profiling per cluster
- Feature importance within clusters
- PCA visualization colored by clusters
- Cluster summary reports
"""

import warnings
from typing import Optional, Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

from src.config import RANDOM_STATE, NUM_COLS, CAT_COLS, VERBOSE
from src.data_preparation import create_preprocessor
from src.utils import ensure_float64

warnings.filterwarnings("ignore")


# ============================================================================
# Cluster Profiler
# ============================================================================

class ClusterProfiler:
    """
    Profile and characterize clusters.
    
    Usage:
        profiler = ClusterProfiler(df, labels, feature_columns)
        numerical_profile = profiler.profile_numerical()
        categorical_profile = profiler.profile_categorical()
        importance = profiler.feature_importance()
        fig = profiler.plot_pca(X_pca)
        report = profiler.get_report()
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        labels: np.ndarray,
        feature_columns: List[str],
        num_cols: List[str] = None,
        cat_cols: List[str] = None,
        random_state: int = RANDOM_STATE,
        verbose: bool = VERBOSE
    ):
        """
        Initialize cluster profiler.
        
        Args:
            df: DataFrame with features
            labels: Cluster labels
            feature_columns: Columns to profile
            num_cols: Numeric columns
            cat_cols: Categorical columns
            random_state: Random seed
            verbose: Print progress
        """
        self.df = df.copy()
        self.labels = np.asarray(labels)
        self.feature_columns = feature_columns
        self.num_cols = num_cols or [c for c in feature_columns if c in NUM_COLS]
        self.cat_cols = cat_cols or [c for c in feature_columns if c in CAT_COLS]
        self.random_state = random_state
        self.verbose = verbose
        
        # Add labels to DataFrame
        self.df['cluster'] = self.labels
        
        # Filter out noise (-1) if present
        self.has_noise = -1 in self.labels
        if self.has_noise:
            if verbose:
                print(f"Found {len(self.df[self.df['cluster'] == -1]):,} noise points. "
                      f"Profiling will exclude noise.")
            self.df_clean = self.df[self.df['cluster'] != -1].copy()
            self.labels_clean = self.labels[self.labels != -1]
            self.clusters = sorted([c for c in set(self.labels) if c != -1])
        else:
            self.df_clean = self.df.copy()
            self.labels_clean = self.labels
            self.clusters = sorted(set(self.labels))
        
        self.n_clusters = len(self.clusters)
        
        if verbose:
            print(f"Profiling {self.n_clusters} clusters with {len(self.df_clean):,} points")
    
    def profile_numerical(self) -> pd.DataFrame:
        """
        Profile numerical features per cluster.
        
        Returns:
            DataFrame with numerical profiles
        """
        if not self.num_cols:
            return pd.DataFrame()
        
        # Group by cluster and aggregate
        agg_dict = {col: ['mean', 'std', 'median', 'min', 'max'] for col in self.num_cols}
        profile = self.df_clean.groupby('cluster')[self.num_cols].agg(agg_dict)
        
        # Flatten column names
        profile.columns = [f'{col}_{agg}' for col, agg in profile.columns]
        
        return profile
    
    def profile_categorical(self) -> Dict[str, pd.DataFrame]:
        """
        Profile categorical features per cluster.
        
        Returns:
            Dictionary mapping categorical column to profile DataFrame
        """
        profiles = {}
        
        for col in self.cat_cols:
            if col not in self.df_clean.columns:
                continue
            
            # Get value counts per cluster
            props = self.df_clean.groupby('cluster')[col].value_counts(normalize=True)
            props = props.unstack().fillna(0)
            
            # Get dominant category
            dominant = props.idxmax(axis=1)
            dominant_share = props.max(axis=1)
            
            summary = pd.DataFrame({
                'dominant_category': dominant,
                'share': dominant_share
            })
            
            profiles[col] = {
                'summary': summary,
                'full': props
            }
        
        return profiles
    
    def profile_mixed(self) -> Dict[str, pd.DataFrame]:
        """
        Profile both numerical and categorical features.
        
        Returns:
            Dictionary with numerical and categorical profiles
        """
        return {
            'numerical': self.profile_numerical(),
            'categorical': self.profile_categorical()
        }
    
    def feature_importance(
        self,
        n_estimators: int = 100,
        max_depth: int = 6
    ) -> Dict[int, pd.Series]:
        """
        Compute feature importance for each cluster (one-vs-rest).
        
        Args:
            n_estimators: Number of trees for Random Forest
            max_depth: Max depth of trees
        
        Returns:
            Dictionary mapping cluster to feature importances
        """
        if self.verbose:
            print("Computing feature importance per cluster...")
        
        # Prepare data
        X = self.df_clean[self.feature_columns]
        y = self.labels_clean
        
        # Encode categorical features if needed
        cat_cols_present = [c for c in self.cat_cols if c in X.columns]
        num_cols_present = [c for c in self.num_cols if c in X.columns]
        
        if cat_cols_present:
            preprocessor = create_preprocessor(num_cols_present, cat_cols_present)
            X_encoded = preprocessor.fit_transform(X).astype(np.float64)
            feature_names = preprocessor.get_feature_names_out()
        else:
            X_encoded = X.values.astype(np.float64)
            feature_names = X.columns
        
        cluster_importance = {}
        
        for cluster in self.clusters:
            # Create binary target
            y_ovr = (y == cluster).astype(int)
            
            # Skip if too few samples
            if y_ovr.sum() < 5 or y_ovr.sum() == len(y_ovr):
                continue
            
            # Train Random Forest
            rf = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=self.random_state,
                class_weight='balanced',
                n_jobs=-1
            )
            rf.fit(X_encoded, y_ovr)
            
            # Get feature importances
            importance = pd.Series(
                rf.feature_importances_,
                index=feature_names
            ).sort_values(ascending=False)
            
            cluster_importance[cluster] = importance
        
        return cluster_importance
    
    def plot_numerical(
        self,
        cols: List[str] = None,
        figsize: Tuple[int, int] = None,
        n_cols: int = 3
    ) -> plt.Figure:
        """
        Plot numerical feature distributions per cluster.
        
        Args:
            cols: Columns to plot (default: all num_cols)
            figsize: Figure size
            n_cols: Number of columns in subplot grid
        
        Returns:
            Matplotlib figure
        """
        if cols is None:
            cols = self.num_cols[:6]  # Limit to 6 for readability
        
        if not cols:
            print("No numerical columns to plot.")
            return None
        
        if figsize is None:
            figsize = (4 * n_cols, 4 * len(cols) // n_cols + 1)
        
        n_rows = int(np.ceil(len(cols) / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = np.array(axes).flatten()
        
        for ax, col in zip(axes, cols):
            if col in self.df_clean.columns:
                # Boxplot by cluster
                sns.boxplot(
                    data=self.df_clean,
                    x='cluster',
                    y=col,
                    ax=ax,
                    palette='Set3'
                )
                ax.set_title(col)
                ax.set_xlabel('Cluster')
        
        # Hide unused subplots
        for ax in axes[len(cols):]:
            ax.axis('off')
        
        plt.tight_layout()
        return fig
    
    def plot_categorical(
        self,
        cols: List[str] = None,
        figsize: Tuple[int, int] = None,
        n_cols: int = 3
    ) -> plt.Figure:
        """
        Plot categorical feature distributions per cluster.
        
        Args:
            cols: Columns to plot (default: all cat_cols)
            figsize: Figure size
            n_cols: Number of columns in subplot grid
        
        Returns:
            Matplotlib figure
        """
        if cols is None:
            cols = self.cat_cols
        
        if not cols:
            print("No categorical columns to plot.")
            return None
        
        if figsize is None:
            figsize = (4 * n_cols, 4 * len(cols) // n_cols + 1)
        
        n_rows = int(np.ceil(len(cols) / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = np.array(axes).flatten()
        
        for ax, col in zip(axes, cols):
            if col in self.df_clean.columns:
                # Cross-tabulation as heatmap
                crosstab = pd.crosstab(
                    self.df_clean['cluster'],
                    self.df_clean[col],
                    normalize='index'
                )
                sns.heatmap(crosstab, ax=ax, annot=True, fmt='.2f', cmap='Blues',
                            cbar_kws={'label': 'Proportion'})
                ax.set_title(f'Cluster distribution - {col}')
                ax.set_xlabel('')
                ax.set_ylabel('Cluster')
        
        # Hide unused subplots
        for ax in axes[len(cols):]:
            ax.axis('off')
        
        plt.tight_layout()
        return fig
    
    def plot_pca(
        self,
        X_pca: np.ndarray,
        figsize: Tuple[int, int] = (8, 6),
        alpha: float = 0.7,
        s: float = 12
    ) -> plt.Figure:
        """
        Plot PCA projection colored by clusters.
        
        Args:
            X_pca: PCA-transformed data
            figsize: Figure size
            alpha: Point transparency
            s: Point size
        
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Filter to match clean data
        if self.has_noise:
            mask = self.labels != -1
            X_pca_clean = X_pca[mask]
            labels_clean = self.labels[mask]
        else:
            X_pca_clean = X_pca
            labels_clean = self.labels
        
        scatter = ax.scatter(
            X_pca_clean[:, 0],
            X_pca_clean[:, 1],
            c=labels_clean,
            cmap='tab10',
            alpha=alpha,
            s=s
        )
        
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title(f"PCA Projection Colored by Clusters ({self.n_clusters} clusters)")
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Cluster')
        
        # Add noise points if present
        if self.has_noise:
            noise_mask = self.labels == -1
            if noise_mask.any():
                ax.scatter(
                    X_pca[noise_mask, 0],
                    X_pca[noise_mask, 1],
                    c='gray',
                    alpha=0.3,
                    s=s,
                    label='Noise'
                )
                ax.legend()
        
        plt.tight_layout()
        return fig
    
    def get_summary_table(self) -> pd.DataFrame:
        """
        Get summary table of cluster sizes and basic stats.
        
        Returns:
            DataFrame with cluster summary
        """
        summary = self.df_clean.groupby('cluster').size().reset_index(name='count')
        summary['percentage'] = summary['count'] / len(self.df_clean) * 100
        
        # Add numerical means
        for col in self.num_cols:
            if col in self.df_clean.columns:
                means = self.df_clean.groupby('cluster')[col].mean()
                summary[f'{col}_mean'] = summary['cluster'].map(means)
        
        return summary
    
    def get_report(self) -> str:
        """
        Get a formatted report of cluster profiling.
        
        Returns:
            Formatted report string
        """
        summary = self.get_summary_table()
        numerical = self.profile_numerical()
        
        report = f"""
{"="*60}
CLUSTER PROFILING REPORT
{"="*60}

Number of Clusters: {self.n_clusters}
Total Points: {len(self.df_clean):,}
Noise Points: {len(self.df) - len(self.df_clean):,} ({(len(self.df) - len(self.df_clean)) / len(self.df) * 100:.1f}%)

Cluster Summary:
{summary.to_string(index=False)}

Numerical Profiles:
{numerical.to_string() if not numerical.empty else "No numerical columns"}

Categorical Profiles:
{", ".join(self.cat_cols) if self.cat_cols else "No categorical columns"}
{"="*60}
"""
        return report


# ============================================================================
# Convenience Functions
# ============================================================================

def profile_clusters(
    df: pd.DataFrame,
    labels: np.ndarray,
    feature_columns: List[str],
    num_cols: List[str] = None,
    cat_cols: List[str] = None,
    random_state: int = RANDOM_STATE,
    verbose: bool = VERBOSE
) -> ClusterProfiler:
    """
    Convenience function to profile clusters.
    
    Args:
        df: DataFrame with features
        labels: Cluster labels
        feature_columns: Columns to profile
        num_cols: Numeric columns
        cat_cols: Categorical columns
        random_state: Random seed
        verbose: Print progress
    
    Returns:
        ClusterProfiler instance
    """
    profiler = ClusterProfiler(
        df, labels, feature_columns,
        num_cols, cat_cols,
        random_state, verbose
    )
    return profiler


# ============================================================================
# Testing
# ============================================================================

if __name__ == "__main__":
    print("Testing cluster_profiling.py...")
    
    from sklearn.datasets import make_blobs
    from sklearn.cluster import KMeans
    
    # Generate test data
    X, y_true = make_blobs(n_samples=300, centers=3, random_state=42)
    X = X.astype(np.float64)
    
    # Create DataFrame
    df = pd.DataFrame(X, columns=['feat_1', 'feat_2'])
    df['category'] = np.random.choice(['A', 'B', 'C'], size=300)
    
    # Cluster
    km = KMeans(n_clusters=3, random_state=42)
    labels = km.fit_predict(X)
    
    # Profile
    print("\n1. Testing ClusterProfiler:")
    profiler = ClusterProfiler(
        df, labels,
        feature_columns=['feat_1', 'feat_2', 'category'],
        num_cols=['feat_1', 'feat_2'],
        cat_cols=['category'],
        verbose=True
    )
    
    # Numerical profile
    print("\n2. Numerical profile:")
    print(profiler.profile_numerical())
    
    # Categorical profile
    print("\n3. Categorical profile:")
    cat_profiles = profiler.profile_categorical()
    for col, data in cat_profiles.items():
        print(f"\n{col}:")
        print(data['summary'])
    
    # Feature importance
    print("\n4. Feature importance:")
    importance = profiler.feature_importance(n_estimators=50)
    for cluster, imp in importance.items():
        print(f"\nCluster {cluster}:")
        print(imp.head(3))
    
    # Summary
    print("\n5. Summary:")
    print(profiler.get_summary_table())
    
    # Report
    print("\n6. Report:")
    print(profiler.get_report())
    
    print("\nAll tests passed!")