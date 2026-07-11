"""
optimal_k.py - Optimal K determination wrapper

Provides:
- High-level interface for determining optimal number of clusters
- Integration with clustering_evaluation
- Visualization of optimal K metrics
- Support for multiple criteria
"""

import warnings
from typing import Optional, Tuple, Dict, List, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.config import RANDOM_STATE, K_RANGE, VERBOSE
from src.clustering_evaluation import OptimalKDeterminer
from src.utils import get_sample

warnings.filterwarnings("ignore")


# ============================================================================
# Optimal K Wrapper
# ============================================================================

class OptimalKAnalyzer:
    """
    High-level interface for optimal K determination.
    
    Usage:
        analyzer = OptimalKAnalyzer(X)
        best_k = analyzer.get_best_k()
        fig = analyzer.plot()
        df = analyzer.get_summary()
    """
    
    def __init__(
        self,
        X: np.ndarray,
        k_range: range = K_RANGE,
        random_state: int = RANDOM_STATE,
        sample_size: Optional[int] = None,
        verbose: bool = VERBOSE
    ):
        """
        Initialize optimal K analyzer.
        
        Args:
            X: Data points
            k_range: Range of K values to test
            random_state: Random seed
            sample_size: Sample size for silhouette (optional)
            verbose: Print progress
        """
        self.X = X
        self.k_range = k_range
        self.random_state = random_state
        self.verbose = verbose
        self.sample_size = sample_size
        
        self.determiner = None
        self.best_k = None
        self.best_by_criteria = None
        self.results = None
        self._computed = False
    
    def compute(self) -> Dict[str, Any]:
        """
        Compute all metrics for optimal K determination.
        
        Returns:
            Dictionary with results
        """
        if self.verbose:
            print(f"Determining optimal K for K={self.k_range[0]}..{self.k_range[-1]}...")
        
        # Initialize determiner
        self.determiner = OptimalKDeterminer(
            self.X,
            k_range=self.k_range,
            random_state=self.random_state,
            verbose=self.verbose,
            sample_size=self.sample_size
        )
        
        # Compute metrics
        self.results = self.determiner.compute_all()
        
        # Get best K
        self.best_by_criteria = self.determiner.get_best_k()
        self.best_k = self.determiner.get_majority_k()
        
        self._computed = True
        
        if self.verbose:
            print(f"\nOptimal K (majority vote): {self.best_k}")
            print(f"  Silhouette: {self.best_by_criteria['silhouette']}")
            print(f"  Calinski-Harabasz: {self.best_by_criteria['calinski_harabasz']}")
            print(f"  Davies-Bouldin: {self.best_by_criteria['davies_bouldin']}")
        
        return {
            'best_k': self.best_k,
            'best_by_criteria': self.best_by_criteria,
            'results': self.results
        }
    
    def get_best_k(self) -> int:
        """
        Get the optimal number of clusters.
        
        Returns:
            Optimal K value
        """
        if not self._computed:
            self.compute()
        return self.best_k
    
    def get_best_by_criteria(self) -> Dict[str, int]:
        """
        Get optimal K by each criterion.
        
        Returns:
            Dictionary mapping criterion to best K
        """
        if not self._computed:
            self.compute()
        return self.best_by_criteria
    
    def get_results(self) -> Dict[str, list]:
        """
        Get all metric results.
        
        Returns:
            Dictionary with metric values for each K
        """
        if not self._computed:
            self.compute()
        return self.results
    
    def get_summary(self) -> pd.DataFrame:
        """
        Get a summary DataFrame of results.
        
        Returns:
            DataFrame with all metrics
        """
        if not self._computed:
            self.compute()
        
        df = pd.DataFrame({
            'K': self.results['k_values'],
            'SSE': self.results['sse'],
            'Silhouette': self.results['silhouette'],
            'Calinski_Harabasz': self.results['calinski_harabasz'],
            'Davies_Bouldin': self.results['davies_bouldin']
        })
        
        # Highlight best K
        df['is_best_k'] = df['K'] == self.best_k
        
        return df
    
    def plot(
        self,
        figsize: Tuple[int, int] = (12, 10)
    ) -> plt.Figure:
        """
        Plot optimal K metrics.
        
        Args:
            figsize: Figure size
        
        Returns:
            Matplotlib figure
        """
        if not self._computed:
            self.compute()
        
        return self.determiner.plot(figsize=figsize)
    
    def plot_elbow(
        self,
        figsize: Tuple[int, int] = (8, 5)
    ) -> plt.Figure:
        """
        Plot only the elbow method.
        
        Args:
            figsize: Figure size
        
        Returns:
            Matplotlib figure
        """
        if not self._computed:
            self.compute()
        
        fig, ax = plt.subplots(figsize=figsize)
        
        k_values = self.results['k_values']
        sse = self.results['sse']
        
        ax.plot(k_values, sse, marker='o', linewidth=2)
        ax.axvline(self.best_k, color='red', linestyle='--', alpha=0.5,
                   label=f'Optimal K = {self.best_k}')
        ax.set_xlabel("Number of Clusters (K)")
        ax.set_ylabel("SSE (Within-Cluster Sum of Squares)")
        ax.set_title("Elbow Method for Optimal K")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_silhouette(
        self,
        figsize: Tuple[int, int] = (8, 5)
    ) -> plt.Figure:
        """
        Plot only the silhouette scores.
        
        Args:
            figsize: Figure size
        
        Returns:
            Matplotlib figure
        """
        if not self._computed:
            self.compute()
        
        fig, ax = plt.subplots(figsize=figsize)
        
        k_values = self.results['k_values']
        silhouette = self.results['silhouette']
        
        ax.plot(k_values, silhouette, marker='o', color='orange', linewidth=2)
        ax.axvline(self.best_k, color='red', linestyle='--', alpha=0.5,
                   label=f'Optimal K = {self.best_k}')
        ax.set_xlabel("Number of Clusters (K)")
        ax.set_ylabel("Silhouette Score")
        ax.set_title("Silhouette Score for Optimal K")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig


# ============================================================================
# Convenience Functions
# ============================================================================

def determine_optimal_k_simple(
    X: np.ndarray,
    k_range: range = K_RANGE,
    random_state: int = RANDOM_STATE,
    verbose: bool = VERBOSE
) -> int:
    """
    Simple function to get optimal K.
    
    Args:
        X: Data points
        k_range: Range of K values
        random_state: Random seed
        verbose: Print progress
    
    Returns:
        Optimal K value
    """
    analyzer = OptimalKAnalyzer(X, k_range, random_state, verbose=verbose)
    return analyzer.get_best_k()


def get_optimal_k_report(
    X: np.ndarray,
    k_range: range = K_RANGE,
    random_state: int = RANDOM_STATE,
    verbose: bool = VERBOSE
) -> str:
    """
    Get a formatted report for optimal K determination.
    
    Args:
        X: Data points
        k_range: Range of K values
        random_state: Random seed
        verbose: Print progress
    
    Returns:
        Formatted report string
    """
    analyzer = OptimalKAnalyzer(X, k_range, random_state, verbose=verbose)
    analyzer.compute()
    
    df = analyzer.get_summary()
    best_by = analyzer.get_best_by_criteria()
    
    report = f"""
{"="*60}
OPTIMAL K DETERMINATION REPORT
{"="*60}

K Range: {k_range.start} to {k_range.stop - 1}

Optimal K (majority vote): {analyzer.best_k}

Best K by criterion:
  Silhouette:          {best_by['silhouette']}
  Calinski-Harabasz:   {best_by['calinski_harabasz']}
  Davies-Bouldin:      {best_by['davies_bouldin']}

Summary of metrics:
{df.to_string(index=False)}

Recommendation: Use K = {analyzer.best_k} for clustering.
{"="*60}
"""
    return report


# ============================================================================
# Testing
# ============================================================================

if __name__ == "__main__":
    print("Testing optimal_k.py...")
    
    from sklearn.datasets import make_blobs
    
    # Generate test data
    X, y = make_blobs(n_samples=500, centers=3, random_state=42)
    X = X.astype(np.float64)
    
    # Test analyzer
    print("\n1. Testing OptimalKAnalyzer:")
    analyzer = OptimalKAnalyzer(X, k_range=range(2, 8), verbose=True)
    best_k = analyzer.get_best_k()
    print(f"Best K: {best_k}")
    
    # Test summary
    print("\n2. Testing summary:")
    df = analyzer.get_summary()
    print(df)
    
    # Test report
    print("\n3. Testing report:")
    report = get_optimal_k_report(X, k_range=range(2, 8))
    print(report)
    
    # Test plots
    print("\n4. Testing plots:")
    fig1 = analyzer.plot_elbow()
    fig2 = analyzer.plot_silhouette()
    plt.show()
    
    print("\nAll tests passed!")