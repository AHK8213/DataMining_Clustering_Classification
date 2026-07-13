"""
clustering_evaluation.py - Evaluation and comparison of clustering results

Provides:
- Computing metrics for clustering results
- Comparing multiple algorithms
- Holistic model selection
- Optimal K determination (Elbow, Silhouette, etc.)
- Visualization of comparisons
"""

import warnings
from typing import Dict, List, Tuple, Optional, Any, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

from src.config import RANDOM_STATE, K_RANGE, VERBOSE
from src.utils import (
    compute_clustering_metrics,
    dunn_index,
    get_sample,
    cleanup,
    timer,
    ensure_float64,
)
from src.clustering_algorithms import ClusteringRunner

warnings.filterwarnings("ignore")


# ============================================================================
# Optimal K Determination
# ============================================================================

class OptimalKDeterminer:
    """
    Determine optimal number of clusters using multiple criteria.
    
    Supports:
    - Elbow method (SSE)
    - Silhouette score
    - Calinski-Harabasz score
    - Davies-Bouldin score
    """
    
    def __init__(
        self,
        X: np.ndarray,
        k_range: range = K_RANGE,
        random_state: int = RANDOM_STATE,
        verbose: bool = VERBOSE,
        sample_size: Optional[int] = None
    ):
        """
        Initialize optimal K determiner.
        
        Args:
            X: Data points
            k_range: Range of K values to test
            random_state: Random seed
            verbose: Print progress
            sample_size: Sample size for silhouette (optional)
        """
        self.X = ensure_float64(X)
        self.k_range = list(k_range)
        self.random_state = random_state
        self.verbose = verbose
        self.sample_size = sample_size or min(6000, X.shape[0])
        
        self.results = {
            'k_values': self.k_range,
            'sse': [],
            'silhouette': [],
            'calinski_harabasz': [],
            'davies_bouldin': []
        }
    
    def compute_all(self) -> Dict[str, list]:
        """
        Compute all metrics for K range.
        
        Returns:
            Dictionary with all metrics
        """
        if self.verbose:
            print(f"Computing metrics for K={self.k_range[0]}..{self.k_range[-1]}...")
        
        # Get sample for silhouette (full data too slow)
        X_sample, sample_idx = get_sample(self.X, self.sample_size, self.random_state)
        
        for k in self.k_range:
            # Use MiniBatchKMeans for speed
            km = MiniBatchKMeans(
                n_clusters=k,
                n_init=10,
                random_state=self.random_state,
                batch_size=min(1000, self.X.shape[0] // 10),
                max_iter=100
            )
            km.fit(self.X)
            
            # SSE (inertia)
            self.results['sse'].append(km.inertia_)
            
            # Silhouette (on sample)
            labels_sample = km.predict(X_sample)
            sil = silhouette_score(X_sample, labels_sample)
            self.results['silhouette'].append(sil)
            
            # Calinski-Harabasz and Davies-Bouldin (on full data, these are fast)
            labels_full = km.predict(self.X)
            ch = calinski_harabasz_score(self.X, labels_full)
            db = davies_bouldin_score(self.X, labels_full)
            self.results['calinski_harabasz'].append(ch)
            self.results['davies_bouldin'].append(db)
            
            if self.verbose:
                print(f"  K={k}: SSE={km.inertia_:.0f}, Sil={sil:.3f}, "
                      f"CH={ch:.0f}, DB={db:.3f}")
        
        return self.results
    
    def get_best_k(self) -> Dict[str, int]:
        """
        Get optimal K by each criterion.
        
        Returns:
            Dictionary mapping criterion to best K
        """
        k_values = self.k_range
        sil_scores = self.results['silhouette']
        ch_scores = self.results['calinski_harabasz']
        db_scores = self.results['davies_bouldin']
        
        best = {
            'silhouette': k_values[np.argmax(sil_scores)],
            'calinski_harabasz': k_values[np.argmax(ch_scores)],
            'davies_bouldin': k_values[np.argmin(db_scores)]
        }
        
        if self.verbose:
            print(f"\nOptimal K by different criteria:")
            print(f"  Silhouette: {best['silhouette']}")
            print(f"  Calinski-Harabasz: {best['calinski_harabasz']}")
            print(f"  Davies-Bouldin: {best['davies_bouldin']}")
        
        return best
    
    def get_majority_k(self) -> int:
        """
        Get optimal K by majority vote.
        
        Returns:
            Most common optimal K among criteria
        """
        best = self.get_best_k()
        votes = {}
        for v in best.values():
            votes[v] = votes.get(v, 0) + 1
        return max(votes, key=votes.get)
    
    def plot(self, figsize: Tuple[int, int] = (12, 10)) -> plt.Figure:
        """
        Plot all optimal K metrics.
        
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        k_values = self.k_range
        
        # Elbow plot
        axes[0, 0].plot(k_values, self.results['sse'], marker='o', linewidth=2)
        axes[0, 0].set_title("Elbow Method (SSE)")
        axes[0, 0].set_xlabel("K")
        axes[0, 0].set_ylabel("SSE")
        axes[0, 0].grid(True, alpha=0.3)
        
        # Silhouette
        axes[0, 1].plot(k_values, self.results['silhouette'], marker='o', 
                        color='orange', linewidth=2)
        axes[0, 1].set_title("Silhouette Score (higher = better)")
        axes[0, 1].set_xlabel("K")
        axes[0, 1].set_ylabel("Silhouette")
        axes[0, 1].grid(True, alpha=0.3)
        
        # Calinski-Harabasz
        axes[1, 0].plot(k_values, self.results['calinski_harabasz'], 
                        marker='o', color='green', linewidth=2)
        axes[1, 0].set_title("Calinski-Harabasz (higher = better)")
        axes[1, 0].set_xlabel("K")
        axes[1, 0].set_ylabel("CH Score")
        axes[1, 0].grid(True, alpha=0.3)
        
        # Davies-Bouldin
        axes[1, 1].plot(k_values, self.results['davies_bouldin'], 
                        marker='o', color='red', linewidth=2)
        axes[1, 1].set_title("Davies-Bouldin (lower = better)")
        axes[1, 1].set_xlabel("K")
        axes[1, 1].set_ylabel("DB Score")
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig


# ============================================================================
# Algorithm Comparison
# ============================================================================

class ClusteringComparator:
    """
    Compare multiple clustering algorithms using various metrics.
    
    Supports:
    - Internal metrics (Silhouette, Davies-Bouldin, Dunn, Calinski-Harabasz)
    - Runtime comparison
    - Holistic ranking
    """
    
    def __init__(
        self,
        X: np.ndarray,
        labels_dict: Dict[str, np.ndarray],
        runtimes: Dict[str, float] = None,
        verbose: bool = VERBOSE
    ):
        """
        Initialize comparator.
        
        Args:
            X: Data points
            labels_dict: Dictionary mapping algorithm names to labels
            runtimes: Dictionary mapping algorithm names to runtimes
            verbose: Print progress
        """
        self.X = ensure_float64(X)
        self.labels_dict = labels_dict
        self.runtimes = runtimes or {}
        self.verbose = verbose
        self.metrics_df = None
    
    def compute_metrics_all(self) -> pd.DataFrame:
        """
        Compute all metrics for all algorithms.
        
        Returns:
            DataFrame with metrics for each algorithm
        """
        results = {}
        
        for name, labels in self.labels_dict.items():
            if self.verbose:
                print(f"Computing metrics for: {name}")
            
            metrics = compute_clustering_metrics(self.X, labels)
            
            # Add runtime
            metrics['runtime_s'] = self.runtimes.get(name, np.nan)
            
            results[name] = metrics
        
        self.metrics_df = pd.DataFrame(results).T
        
        # Sort by silhouette (default)
        if 'silhouette' in self.metrics_df.columns:
            self.metrics_df = self.metrics_df.sort_values(
                'silhouette', ascending=False
            )
        
        return self.metrics_df
    
    def get_comparison_df(self) -> pd.DataFrame:
        """Get the comparison DataFrame."""
        if self.metrics_df is None:
            self.compute_metrics_all()
        return self.metrics_df
    
    def holistic_best_model(
        self,
        min_clusters: int = 2,
        max_noise_pct: float = 50.0,
        primary_metric: str = 'silhouette'
    ) -> str:
        """
        Select the best model holistically.
        
        Args:
            min_clusters: Minimum number of clusters (excluding noise)
            max_noise_pct: Maximum allowed noise percentage
            primary_metric: Primary metric for selection
        
        Returns:
            Name of best model
        """
        df = self.get_comparison_df()
        
        # Filter degenerate models
        eligible = df[
            (df['n_clusters'] >= min_clusters) &
            (df['noise_pct'] <= max_noise_pct)
        ]
        
        if primary_metric not in df.columns:
            raise ValueError(f"Metric '{primary_metric}' not found")
        
        if eligible.empty:
            # Fallback to all models with the primary metric
            return df[primary_metric].idxmax()
        
        return eligible[primary_metric].idxmax()
    
    def rank_algorithms(
        self,
        metrics: List[str] = None,
        ascending: Dict[str, bool] = None
    ) -> pd.DataFrame:
        """
        Rank algorithms by multiple metrics.
        
        Args:
            metrics: List of metrics to include in ranking
            ascending: Dict mapping metric to ascending/descending
        
        Returns:
            DataFrame with ranks and overall score
        """
        df = self.get_comparison_df().copy()
        
        if metrics is None:
            metrics = ['silhouette', 'davies_bouldin', 'dunn', 'runtime_s']
        
        if ascending is None:
            ascending = {
                'silhouette': False,  # higher is better
                'davies_bouldin': True,  # lower is better
                'dunn': False,  # higher is better
                'calinski_harabasz': False,  # higher is better
                'runtime_s': True  # lower is better
            }
        
        # Compute ranks
        rank_df = pd.DataFrame(index=df.index)
        
        for metric in metrics:
            if metric not in df.columns:
                continue
            
            asc = ascending.get(metric, False)
            rank_df[f'{metric}_rank'] = df[metric].rank(ascending=asc)
        
        # Overall rank score (average of ranks)
        rank_cols = [c for c in rank_df.columns if c.endswith('_rank')]
        if rank_cols:
            rank_df['overall_rank_score'] = rank_df[rank_cols].mean(axis=1)
            df['overall_rank_score'] = rank_df['overall_rank_score']
        
        # Sort by overall rank
        df = df.sort_values('overall_rank_score', ascending=True)
        
        return df
    
    def plot_comparison(
        self,
        metrics: List[str] = None,
        figsize: Tuple[int, int] = (20, 5)
    ) -> plt.Figure:
        """
        Plot comparison of algorithms.
        
        Args:
            metrics: List of metrics to plot
            figsize: Figure size
        
        Returns:
            Matplotlib figure
        """
        df = self.get_comparison_df()
        
        if metrics is None:
            metrics = ['silhouette', 'davies_bouldin', 'dunn', 'runtime_s']
        
        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
        
        colors = {
            'silhouette': 'seagreen',
            'davies_bouldin': 'indianred',
            'dunn': 'steelblue',
            'runtime_s': 'goldenrod',
            'calinski_harabasz': 'purple'
        }
        
        for ax, metric in zip(axes, metrics):
            if metric in df.columns:
                # Sort by value for better visualization
                sorted_df = df.sort_values(metric, 
                                          ascending=metric in ['davies_bouldin', 'runtime_s'])
                sorted_df[metric].plot(kind='barh', ax=ax, 
                                      color=colors.get(metric, 'gray'))
                
                ax.set_title(f"{metric}\n({'lower=better' if metric in ['davies_bouldin', 'runtime_s'] else 'higher=better'})")
                ax.set_xlabel('')
        
        plt.tight_layout()
        return fig


# ============================================================================
# Convenience Functions
# ============================================================================

def determine_optimal_k(
    X: np.ndarray,
    k_range: range = K_RANGE,
    random_state: int = RANDOM_STATE,
    verbose: bool = VERBOSE
) -> Tuple[int, Dict[str, int], 'OptimalKDeterminer']:
    """
    Convenience function to determine optimal K.
    
    Args:
        X: Data points
        k_range: Range of K values
        random_state: Random seed
        verbose: Print progress
    
    Returns:
        Tuple of (majority_k, best_by_criteria, determiner)
    """
    determiner = OptimalKDeterminer(X, k_range, random_state, verbose)
    determiner.compute_all()
    best_k = determiner.get_majority_k()
    best_by_criteria = determiner.get_best_k()
    
    return best_k, best_by_criteria, determiner


def compare_clustering_results(
    X: np.ndarray,
    labels_dict: Dict[str, np.ndarray],
    runtimes: Dict[str, float] = None,
    verbose: bool = VERBOSE
) -> Tuple[pd.DataFrame, 'ClusteringComparator']:
    """
    Convenience function to compare clustering results.
    
    Args:
        X: Data points
        labels_dict: Dictionary mapping algorithm names to labels
        runtimes: Dictionary mapping algorithm names to runtimes
        verbose: Print progress
    
    Returns:
        Tuple of (comparison_df, comparator)
    """
    comparator = ClusteringComparator(X, labels_dict, runtimes, verbose)
    df = comparator.compute_metrics_all()
    return df, comparator


def evaluate_clustering(
    X: np.ndarray,
    labels: np.ndarray,
    algorithm_name: str = "Unknown",
    verbose: bool = VERBOSE
) -> Dict[str, Any]:
    """
    Evaluate a single clustering result.
    
    Args:
        X: Data points
        labels: Cluster labels
        algorithm_name: Name of the algorithm
        verbose: Print progress
    
    Returns:
        Dictionary with metrics
    """
    metrics = compute_clustering_metrics(X, labels)
    
    if verbose:
        print(f"\nEvaluation for {algorithm_name}:")
        print(f"  Number of clusters: {metrics['n_clusters']}")
        print(f"  Noise percentage: {metrics['noise_pct']:.1f}%")
        print(f"  Silhouette: {metrics.get('silhouette', np.nan):.3f}")
        print(f"  Davies-Bouldin: {metrics.get('davies_bouldin', np.nan):.3f}")
        print(f"  Dunn Index: {metrics.get('dunn', np.nan):.3f}")
        print(f"  Calinski-Harabasz: {metrics.get('calinski_harabasz', np.nan):.0f}")
    
    return metrics


# ============================================================================
# Testing
# ============================================================================

if __name__ == "__main__":
    print("Testing clustering_evaluation.py...")
    
    from sklearn.datasets import make_blobs
    
    # Generate test data
    X, y = make_blobs(n_samples=500, centers=3, random_state=42)
    X = X.astype(np.float64)
    
    # Test optimal K determination
    print("\n--- Optimal K Determination ---")
    best_k, best_by_criteria, determiner = determine_optimal_k(
        X, k_range=range(2, 8), verbose=True
    )
    print(f"Best K (majority): {best_k}")
    
    # Test with real clustering
    print("\n--- Clustering Comparison ---")
    from src.clustering_algorithms import ClusteringRunner
    
    runner = ClusteringRunner(X, k=3, verbose=False)
    labels = runner.run_all()
    
    # Compare
    df, comparator = compare_clustering_results(X, labels, runner.runtimes)
    print("\nComparison results:")
    print(df[['n_clusters', 'noise_pct', 'silhouette', 'runtime_s']].head())
    
    # Holistic best
    best = comparator.holistic_best_model()
    print(f"\nHolistically best model: {best}")
    
    # Ranking
    ranked = comparator.rank_algorithms()
    print(f"\nRanked algorithms:")
    print(ranked[['silhouette', 'davies_bouldin', 'runtime_s', 'overall_rank_score']].head())
    
    print("\nAll tests passed!")