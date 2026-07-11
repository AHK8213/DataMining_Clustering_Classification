"""
clustering_tendency.py - Clustering tendency assessment

Provides:
- Hopkins statistic computation
- Interpretation of Hopkins results
- Visualization of clustering tendency
"""

import warnings
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

from src.config import RANDOM_STATE, SAMPLE_HOPKINS, VERBOSE
from src.utils import hopkins_statistic, interpret_hopkins, get_sample

warnings.filterwarnings("ignore")


# ============================================================================
# Clustering Tendency Analysis
# ============================================================================

class ClusteringTendency:
    """
    Assess clustering tendency of a dataset.
    
    Usage:
        tendency = ClusteringTendency(X)
        H = tendency.compute_hopkins()
        tendency.interpret()
        fig = tendency.plot()
    """
    
    def __init__(
        self,
        X: np.ndarray,
        n_samples: int = SAMPLE_HOPKINS,
        random_state: int = RANDOM_STATE,
        verbose: bool = VERBOSE
    ):
        """
        Initialize clustering tendency analysis.
        
        Args:
            X: Data points (n_samples, n_features)
            n_samples: Number of points to sample for Hopkins
            random_state: Random seed
            verbose: Print progress
        """
        self.X = X
        self.n_samples = min(n_samples, X.shape[0] - 1)
        self.random_state = random_state
        self.verbose = verbose
        
        self.hopkins_value = None
        self.interpretation = None
        
        # Sample data for faster computation
        self.X_sample, self.idx = get_sample(
            X, min(n_samples * 2, X.shape[0]), random_state
        )
    
    def compute_hopkins(self) -> float:
        """
        Compute the Hopkins statistic.
        
        Returns:
            Hopkins statistic value
        """
        if self.verbose:
            print(f"Computing Hopkins statistic with {self.n_samples} samples...")
        
        self.hopkins_value = hopkins_statistic(
            self.X,
            n_samples=self.n_samples,
            random_state=self.random_state
        )
        
        self.interpretation = interpret_hopkins(self.hopkins_value)
        
        if self.verbose:
            print(f"Hopkins statistic: {self.hopkins_value:.3f}")
            print(f"Interpretation: {self.interpretation}")
        
        return self.hopkins_value
    
    def interpret(self) -> str:
        """
        Get interpretation of the Hopkins statistic.
        
        Returns:
            Interpretation string
        """
        if self.hopkins_value is None:
            self.compute_hopkins()
        return self.interpretation
    
    def is_clusterable(self, threshold: float = 0.6) -> bool:
        """
        Determine if the data is clusterable.
        
        Args:
            threshold: Threshold for clusterability (default: 0.6)
        
        Returns:
            True if clusterable, False otherwise
        """
        if self.hopkins_value is None:
            self.compute_hopkins()
        return self.hopkins_value > threshold
    
    def plot(
        self,
        figsize: Tuple[int, int] = (12, 5),
        show_random: bool = True
    ) -> plt.Figure:
        """
        Visualize clustering tendency.
        
        Args:
            figsize: Figure size
            show_random: Show random data comparison
        
        Returns:
            Matplotlib figure
        """
        if self.hopkins_value is None:
            self.compute_hopkins()
        
        fig, axes = plt.subplots(1, 2 if show_random else 1, figsize=figsize)
        if not show_random:
            axes = [axes]
        
        # Plot 1: Hopkins statistic value
        ax = axes[0]
        colors = ['red', 'orange', 'green'] if self.hopkins_value > 0.7 else ['orange', 'red']
        color = 'green' if self.hopkins_value > 0.7 else 'orange' if self.hopkins_value > 0.5 else 'red'
        
        ax.bar(['Hopkins Statistic'], [self.hopkins_value], color=color, alpha=0.7)
        ax.axhline(0.5, color='black', linestyle='--', alpha=0.5, label='Random (0.5)')
        ax.axhline(0.7, color='green', linestyle='--', alpha=0.5, label='Strong (0.7)')
        ax.set_ylim(0, 1)
        ax.set_ylabel('Hopkins Statistic')
        ax.set_title(f"Clustering Tendency\n{self.interpretation}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Comparison with random data (optional)
        if show_random and len(axes) > 1:
            ax = axes[1]
            
            # Generate random data with same shape
            rng = np.random.RandomState(self.random_state)
            mins, maxs = self.X.min(axis=0), self.X.max(axis=0)
            X_random = rng.uniform(mins, maxs, size=(min(1000, self.X.shape[0]), self.X.shape[1]))
            
            # Compute Hopkins for random data
            H_random = hopkins_statistic(
                X_random,
                n_samples=min(self.n_samples, X_random.shape[0] - 1),
                random_state=self.random_state
            )
            
            ax.bar(['Actual Data', 'Random Data'], [self.hopkins_value, H_random],
                   color=['steelblue', 'gray'], alpha=0.7)
            ax.axhline(0.5, color='black', linestyle='--', alpha=0.5, label='Random (0.5)')
            ax.set_ylim(0, 1)
            ax.set_ylabel('Hopkins Statistic')
            ax.set_title("Comparison with Random Data")
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def get_report(self) -> str:
        """
        Get a formatted report of clustering tendency.
        
        Returns:
            Formatted report string
        """
        if self.hopkins_value is None:
            self.compute_hopkins()
        
        report = f"""
{"="*60}
CLUSTERING TENDENCY REPORT
{"="*60}

Dataset:
  Samples: {self.X.shape[0]:,}
  Features: {self.X.shape[1]}

Hopkins Statistic: {self.hopkins_value:.4f}

Interpretation: {self.interpretation}

Clusterability: {'YES' if self.is_clusterable() else 'NO'}

Recommendation:
  {self._get_recommendation()}
{"="*60}
"""
        return report
    
    def _get_recommendation(self) -> str:
        """Get recommendation based on Hopkins statistic."""
        if self.hopkins_value > 0.7:
            return "Data is strongly clusterable. Proceed with clustering analysis."
        elif self.hopkins_value > 0.6:
            return "Data has moderate clustering structure. Clustering may be useful."
        elif self.hopkins_value > 0.5:
            return "Data has weak clustering tendency. Clustering results may be limited."
        else:
            return "Data appears random or uniformly distributed. Clustering may not be meaningful."


# ============================================================================
# Convenience Functions
# ============================================================================

def assess_clustering_tendency(
    X: np.ndarray,
    n_samples: int = SAMPLE_HOPKINS,
    random_state: int = RANDOM_STATE,
    verbose: bool = VERBOSE
) -> Tuple[float, str, ClusteringTendency]:
    """
    Convenience function to assess clustering tendency.
    
    Args:
        X: Data points
        n_samples: Number of samples for Hopkins
        random_state: Random seed
        verbose: Print progress
    
    Returns:
        Tuple of (Hopkins value, interpretation, tendency object)
    """
    tendency = ClusteringTendency(X, n_samples, random_state, verbose)
    H = tendency.compute_hopkins()
    interpretation = tendency.interpret()
    return H, interpretation, tendency


# ============================================================================
# Testing
# ============================================================================

if __name__ == "__main__":
    print("Testing clustering_tendency.py...")
    
    from sklearn.datasets import make_blobs, make_circles
    
    # Test with blobs (clusterable)
    print("\n1. Testing with blobs:")
    X_blobs, _ = make_blobs(n_samples=500, centers=3, random_state=42)
    X_blobs = X_blobs.astype(np.float64)
    
    tendency_blobs = ClusteringTendency(X_blobs, verbose=True)
    H_blobs = tendency_blobs.compute_hopkins()
    print(tendency_blobs.get_report())
    
    # Test with random data
    print("\n2. Testing with random data:")
    X_random = np.random.randn(500, 2).astype(np.float64)
    
    tendency_random = ClusteringTendency(X_random, verbose=True)
    H_random = tendency_random.compute_hopkins()
    print(tendency_random.get_report())
    
    # Test with circles (non-linear)
    print("\n3. Testing with circles:")
    X_circles, _ = make_circles(n_samples=500, noise=0.05, random_state=42)
    X_circles = X_circles.astype(np.float64)
    
    tendency_circles = ClusteringTendency(X_circles, verbose=True)
    H_circles = tendency_circles.compute_hopkins()
    print(tendency_circles.get_report())
    
    # Plot
    fig = tendency_blobs.plot()
    plt.show()
    
    print("\nAll tests passed!")