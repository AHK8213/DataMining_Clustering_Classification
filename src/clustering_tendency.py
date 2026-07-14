"""
clustering_tendency.py - Clustering tendency assessment

Provides:
- Hopkins statistic computation
- Spatial histogram KL/JS divergence (data vs. random) for clustering tendency
- Comprehensive visualization of clustering tendency
- Interpretation of results
"""

import warnings
from typing import Optional, Tuple, Dict, Any, List, Union

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import RANDOM_STATE, SAMPLE_HOPKINS, VERBOSE
from src.utils import hopkins_statistic, interpret_hopkins, get_sample

warnings.filterwarnings("ignore")


# ============================================================================
# KL Divergence Utilities
# ============================================================================

def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-10) -> float:
    """
    Compute KL Divergence D(P || Q) = sum(P * log(P/Q)).
    
    Args:
        p: First probability distribution (true/reference)
        q: Second probability distribution (approximation)
        eps: Small epsilon to avoid log(0)
    
    Returns:
        KL divergence value (non-negative, 0 if P == Q)
    """
    p = np.asarray(p).flatten()
    q = np.asarray(q).flatten()
    
    # Normalize to ensure probability distributions
    p = p / (p.sum() + eps)
    q = q / (q.sum() + eps)
    
    # Clip to avoid log(0)
    p = np.clip(p, eps, 1)
    q = np.clip(q, eps, 1)
    
    return np.sum(p * np.log(p / q))


def symmetric_kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-10) -> float:
    """
    Compute symmetric KL Divergence: (KL(P||Q) + KL(Q||P)) / 2.
    
    Args:
        p: First probability distribution
        q: Second probability distribution
        eps: Small epsilon to avoid log(0)
    
    Returns:
        Symmetric KL divergence value
    """
    return (kl_divergence(p, q, eps) + kl_divergence(q, p, eps)) / 2


def js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-10) -> float:
    """
    Compute Jensen-Shannon Divergence: symmetric and bounded [0, log(2)].
    
    Args:
        p: First probability distribution
        q: Second probability distribution
        eps: Small epsilon to avoid log(0)
    
    Returns:
        Jensen-Shannon divergence value
    """
    p = np.asarray(p).flatten()
    q = np.asarray(q).flatten()
    
    p = p / (p.sum() + eps)
    q = q / (q.sum() + eps)
    
    m = (p + q) / 2
    
    return (kl_divergence(p, m, eps) + kl_divergence(q, m, eps)) / 2


# ============================================================================
# Spatial Histogram Approach (KL-based Clustering Tendency)
# ============================================================================

def _reduce_dims_for_histogram(
    X: np.ndarray,
    max_dims: int = 3,
    random_state: int = RANDOM_STATE
):
    """
    PCA-reduce data for joint histogram binning when dimensionality is too
    high. A full joint histogram has n_bins**d cells, so beyond a handful
    of dimensions it becomes mostly-empty and statistically meaningless.

    Args:
        X: Data points (n_samples, n_features)
        max_dims: Max number of dimensions to keep. If X already has
            <= max_dims features, X is returned unchanged (no PCA).
        random_state: Random seed for PCA

    Returns:
        (X_reduced, pca_or_None)
    """
    if X.shape[1] <= max_dims:
        return X, None
    from sklearn.decomposition import PCA
    pca = PCA(n_components=max_dims, random_state=random_state)
    X_reduced = pca.fit_transform(X)
    return X_reduced, pca


def spatial_histogram_kl(
    X: np.ndarray,
    n_bins: int = 5,
    max_dims: int = 3,
    random_state: int = RANDOM_STATE,
    eps: float = 1e-10
) -> Dict[str, Any]:
    """
    Assess clustering tendency via the spatial histogram approach.

    Contrasts the d-dimensional empirical histogram (EPMF) of the actual
    dataset D with the histogram of uniformly random data sampled from the
    same bounding box. D is considered clusterable if the two distributions
    differ substantially (high KL / JS divergence) -- if D were itself
    uniformly random, its histogram would closely match the random one.

    Method:
        1. Divide each (reduced) dimension into `n_bins` equi-width bins
           and count points per cell -> EPMF of D.
        2. Draw the same number of points uniformly at random from D's
           bounding box, using the *same* bin edges -> EPMF of random data.
        3. Quantify the difference between the two EPMFs with KL divergence.

    Note on dimensionality: a full joint histogram has n_bins**d cells, so
    it is only meaningful in low dimensions. If X has more than `max_dims`
    features, it is PCA-reduced to `max_dims` components first. Set
    max_dims >= X.shape[1] to disable this reduction (only recommended for
    already low-dimensional data).

    Args:
        X: Data points (n_samples, n_features)
        n_bins: Number of equi-width bins per dimension
        max_dims: Max dimensionality for the joint histogram; X is
            PCA-reduced to this many components if it has more features
        random_state: Random seed
        eps: Small epsilon to avoid log(0) / division by 0

    Returns:
        Dictionary with kl_divergence (D || random), kl_divergence_reverse
        (random || D), symmetric_kl, js_divergence, the two EPMFs, bin
        edges, and metadata about any PCA reduction applied.
    """
    X = np.asarray(X)
    X_hist, pca = _reduce_dims_for_histogram(X, max_dims=max_dims, random_state=random_state)

    n_samples, d = X_hist.shape
    mins, maxs = X_hist.min(axis=0), X_hist.max(axis=0)

    # Guard against zero-width dimensions (constant feature after reduction)
    ranges = maxs - mins
    zero_range = ranges == 0
    if np.any(zero_range):
        maxs = maxs.copy()
        maxs[zero_range] = mins[zero_range] + eps

    bins = [n_bins] * d
    bin_range = [(mins[i], maxs[i]) for i in range(d)]

    # EPMF of the actual data
    hist_data, edges = np.histogramdd(X_hist, bins=bins, range=bin_range)
    epmf_data = hist_data / hist_data.sum()

    # EPMF of uniform random data over the same bounding box, same bins
    rng = np.random.RandomState(random_state)
    X_random = rng.uniform(mins, maxs, size=(n_samples, d))
    hist_random, _ = np.histogramdd(X_random, bins=bins, range=bin_range)
    epmf_random = hist_random / hist_random.sum()

    kl = kl_divergence(epmf_data, epmf_random, eps=eps)
    kl_rev = kl_divergence(epmf_random, epmf_data, eps=eps)
    sym_kl = (kl + kl_rev) / 2
    js = js_divergence(epmf_data, epmf_random, eps=eps)

    return {
        'kl_divergence': kl,
        'kl_divergence_reverse': kl_rev,
        'symmetric_kl': sym_kl,
        'js_divergence': js,
        'epmf_data': epmf_data,
        'epmf_random': epmf_random,
        'bin_edges': edges,
        'n_bins': n_bins,
        'n_dims_used': d,
        'n_dims_original': X.shape[1],
        'pca_reduced': pca is not None,
        'explained_variance_ratio': pca.explained_variance_ratio_.tolist() if pca is not None else None,
    }


def interpret_histogram_tendency(js: float) -> str:
    """
    Interpret the JS divergence from the spatial histogram approach.

    Args:
        js: Jensen-Shannon divergence between data and random EPMFs

    Returns:
        Interpretation string
    """
    if np.isnan(js):
        return "Insufficient data for spatial histogram KL computation."
    elif js > 0.3:
        return f"Strong clustering tendency (JS={js:.3f}). Data's spatial distribution differs substantially from random."
    elif js > 0.15:
        return f"Moderate clustering tendency (JS={js:.3f}). Data shows some non-random spatial structure."
    elif js > 0.05:
        return f"Weak clustering tendency (JS={js:.3f}). Data is only slightly different from random."
    else:
        return f"Little to no clustering tendency (JS={js:.3f}). Data closely resembles a uniform/random distribution."


# ============================================================================
# Clustering Tendency Analysis
# ============================================================================

class ClusteringTendency:
    """
    Assess clustering tendency of a dataset.
    
    Combines the Hopkins statistic with the spatial histogram KL/JS
    divergence approach (data EPMF vs. random EPMF) -- both are
    pre-clustering diagnostics that require no labels.
    
    Usage:
        tendency = ClusteringTendency(X)
        H = tendency.compute_hopkins()
        hist_results = tendency.compute_histogram_tendency()
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
        self.histogram_results = None
        
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
    
    def compute_histogram_tendency(
        self,
        n_bins: int = 5,
        max_dims: int = 3,
        verbose: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        Assess clustering tendency using the spatial histogram approach.

        Compares the EPMF of the actual data against the EPMF of uniformly
        random data drawn from the same bounding box (same bin edges).
        This is a pre-clustering, distribution-based companion to the
        Hopkins statistic -- it requires no labels or preliminary
        clustering step.

        Because a full joint histogram has n_bins**max_dims cells, X is
        PCA-reduced to `max_dims` components first if it has more features
        than that (see spatial_histogram_kl for details).

        Args:
            n_bins: Number of equi-width bins per dimension
            max_dims: Max dimensionality for the joint histogram
            verbose: Print progress. Defaults to the instance's verbose
                setting when not given.

        Returns:
            Dictionary with KL/JS divergence results (see spatial_histogram_kl)
        """
        verbose = self.verbose if verbose is None else verbose

        if verbose:
            eff_dims = min(self.X.shape[1], max_dims)
            print(f"Computing spatial histogram KL tendency "
                  f"({self.X.shape[1]}D -> {eff_dims}D, {n_bins} bins/dim)...")

        self.histogram_results = spatial_histogram_kl(
            self.X, n_bins=n_bins, max_dims=max_dims, random_state=self.random_state
        )

        if verbose:
            r = self.histogram_results
            if r['pca_reduced']:
                print(f"PCA-reduced to {r['n_dims_used']}D "
                      f"(explained variance: {sum(r['explained_variance_ratio']):.1%})")
            print(f"KL(data || random): {r['kl_divergence']:.4f}")
            print(f"Symmetric KL: {r['symmetric_kl']:.4f}")
            print(f"JS divergence: {r['js_divergence']:.4f}")
            print(self.interpret_histogram_tendency())

        return self.histogram_results

    def interpret_histogram_tendency(self) -> str:
        """
        Get interpretation of the spatial histogram KL tendency results.

        Returns:
            Interpretation string
        """
        if self.histogram_results is None:
            return "No histogram tendency results available. Run compute_histogram_tendency() first."
        return interpret_histogram_tendency(self.histogram_results['js_divergence'])

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
        figsize: Tuple[int, int] = (15, 5),
        show_random: bool = True,
        show_histogram_kl: bool = True
    ) -> plt.Figure:
        """
        Visualize clustering tendency: Hopkins statistic and, optionally,
        the spatial histogram KL/JS divergence (data vs. random).

        For the fuller histogram-tendency visualization (including EPMF
        heatmaps when applicable), use plot_histogram_tendency() instead.
        
        Args:
            figsize: Figure size
            show_random: Show random data comparison panel
            show_histogram_kl: Show spatial histogram KL/JS summary panel
                (computes it via compute_histogram_tendency() if not
                already computed)
        
        Returns:
            Matplotlib figure
        """
        if self.hopkins_value is None:
            self.compute_hopkins()
        
        # Determine number of subplots
        n_plots = 1  # Hopkins
        if show_random:
            n_plots += 1
        if show_histogram_kl:
            n_plots += 1
        
        # Create figure with subplots
        n_cols = min(3, n_plots)
        n_rows = int(np.ceil(n_plots / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        
        # Flatten axes if needed
        if n_plots == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        plot_idx = 0
        
        # Plot 1: Hopkins statistic value
        ax = axes[plot_idx]
        color = 'green' if self.hopkins_value > 0.7 else 'orange' if self.hopkins_value > 0.5 else 'red'
        
        ax.bar(['Hopkins Statistic'], [self.hopkins_value], color=color, alpha=0.7, edgecolor='black')
        ax.axhline(0.5, color='black', linestyle='--', alpha=0.5, label='Random (0.5)')
        ax.axhline(0.7, color='green', linestyle='--', alpha=0.5, label='Strong (0.7)')
        ax.set_ylim(0, 1)
        ax.set_ylabel('Hopkins Statistic', fontsize=12)
        ax.set_title(f"Clustering Tendency\n{self.interpretation}", fontsize=12)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        plot_idx += 1
        
        # Plot 2: Comparison with random data (optional)
        if show_random and plot_idx < len(axes):
            ax = axes[plot_idx]
            
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
                   color=['steelblue', 'gray'], alpha=0.7, edgecolor='black')
            ax.axhline(0.5, color='black', linestyle='--', alpha=0.5, label='Random (0.5)')
            ax.set_ylim(0, 1)
            ax.set_ylabel('Hopkins Statistic', fontsize=12)
            ax.set_title("Comparison with Random Data", fontsize=12)
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
            plot_idx += 1
        
        # Plot 3: Spatial histogram KL/JS summary (optional)
        if show_histogram_kl and plot_idx < len(axes):
            ax = axes[plot_idx]
            
            if self.histogram_results is None:
                self.compute_histogram_tendency(verbose=False)
            
            r = self.histogram_results
            metrics = ['kl_divergence', 'symmetric_kl', 'js_divergence']
            values = [r.get(m, np.nan) for m in metrics]
            labels_metrics = ['KL(D||R)', 'Sym KL', 'JS Div']
            colors = ['#3498db', '#2ecc71', '#e74c3c']
            
            bars = ax.bar(labels_metrics, values, color=colors, alpha=0.7, edgecolor='black')
            ax.set_ylabel('Divergence Value', fontsize=12)
            ax.set_title(f"Spatial Histogram KL (Data vs. Random)\n{self.interpret_histogram_tendency()}",
                         fontsize=11)
            ax.grid(True, alpha=0.3, axis='y')
            
            for bar, val in zip(bars, values):
                if not np.isnan(val):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{val:.3f}', ha='center', va='bottom', fontsize=10)
            plot_idx += 1
        
        # Hide any unused subplots
        for ax in axes[plot_idx:]:
            ax.axis('off')
        
        plt.tight_layout()
        return fig
    
    def plot_histogram_tendency(self, figsize: Tuple[int, int] = (14, 5)) -> plt.Figure:
        """
        Visualize the spatial histogram clustering tendency results.

        Shows: (1) a bar summary of KL/symmetric-KL/JS divergence between
        the data's EPMF and the random EPMF, and (2) if the histogram was
        computed on exactly 2 dimensions (either because X was 2D or was
        PCA-reduced to 2D), side-by-side heatmaps of the two EPMFs.

        Returns:
            Matplotlib figure
        """
        if self.histogram_results is None:
            self.compute_histogram_tendency()

        r = self.histogram_results
        show_heatmaps = r['n_dims_used'] == 2

        n_cols = 3 if show_heatmaps else 1
        fig, axes = plt.subplots(1, n_cols, figsize=figsize)
        axes = np.atleast_1d(axes)

        # Divergence summary bar chart
        ax = axes[0]
        metrics = ['kl_divergence', 'symmetric_kl', 'js_divergence']
        labels_metrics = ['KL(D||R)', 'Symmetric KL', 'JS Div']
        values = [r[m] for m in metrics]
        colors = ['#3498db', '#2ecc71', '#e74c3c']
        bars = ax.bar(labels_metrics, values, color=colors, alpha=0.8, edgecolor='black')
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f'{val:.3f}', ha='center', va='bottom', fontsize=10)
        ax.set_ylabel('Divergence Value', fontsize=11)
        title_dims = f"{r['n_dims_used']}D" + (" (PCA)" if r['pca_reduced'] else "")
        ax.set_title(f"Spatial Histogram KL\nData vs. Random ({title_dims}, {r['n_bins']} bins/dim)",
                     fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')

        if show_heatmaps:
            vmax = max(r['epmf_data'].max(), r['epmf_random'].max())
            im0 = axes[1].imshow(r['epmf_data'].T, origin='lower', cmap='viridis', vmin=0, vmax=vmax)
            axes[1].set_title("Actual Data EPMF", fontsize=11)
            plt.colorbar(im0, ax=axes[1], fraction=0.046)

            im1 = axes[2].imshow(r['epmf_random'].T, origin='lower', cmap='viridis', vmin=0, vmax=vmax)
            axes[2].set_title("Random Data EPMF", fontsize=11)
            plt.colorbar(im1, ax=axes[2], fraction=0.046)

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
{"="*70}
CLUSTERING TENDENCY REPORT
{"="*70}

Dataset:
  Samples: {self.X.shape[0]:,}
  Features: {self.X.shape[1]}

Hopkins Statistic: {self.hopkins_value:.4f}
Interpretation: {self.interpretation}
Clusterability: {'YES' if self.is_clusterable() else 'NO'}
"""
        
        if self.histogram_results:
            r = self.histogram_results
            report += f"""
{"-"*70}
Spatial Histogram KL Analysis (Data vs. Random, no clustering required):
  Dimensions used: {r['n_dims_used']} of {r['n_dims_original']} ({'PCA-reduced' if r['pca_reduced'] else 'original'})
  Bins per dimension: {r['n_bins']}
  KL(data || random): {r['kl_divergence']:.4f}
  Symmetric KL: {r['symmetric_kl']:.4f}
  Jensen-Shannon Divergence: {r['js_divergence']:.4f}
  Interpretation: {self.interpret_histogram_tendency()}
"""

        report += f"""
{"-"*70}
Recommendation:
  {self._get_recommendation()}
{"="*70}
"""
        return report
    
    def _get_recommendation(self) -> str:
        """Get recommendation based on Hopkins statistic and histogram KL results."""
        has_hist = self.histogram_results is not None
        js = self.histogram_results.get('js_divergence', 0) if has_hist else 0
        
        if self.hopkins_value > 0.7 and (not has_hist or js > 0.15):
            return "Data is strongly clusterable. Proceed with full clustering analysis."
        elif self.hopkins_value > 0.6 and (not has_hist or js > 0.1):
            return "Data has moderate clustering structure. Clustering may be useful."
        elif self.hopkins_value > 0.5 and (not has_hist or js > 0.05):
            return "Data has weak clustering tendency. Consider feature engineering or dimensionality reduction."
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
    from sklearn.cluster import KMeans
    
    # Test with blobs (clusterable)
    print("\n1. Testing with blobs:")
    X_blobs, y_true = make_blobs(n_samples=500, centers=3, random_state=42)
    X_blobs = X_blobs.astype(np.float64)
    
    tendency_blobs = ClusteringTendency(X_blobs, verbose=True)
    H_blobs = tendency_blobs.compute_hopkins()
    
    # Spatial histogram KL tendency (data vs. random)
    hist_results = tendency_blobs.compute_histogram_tendency(n_bins=5, max_dims=3)
    
    print(tendency_blobs.get_report())
    
    # Plot with all visualizations
    fig = tendency_blobs.plot(show_random=True, show_histogram_kl=True)
    plt.show()
    fig2 = tendency_blobs.plot_histogram_tendency()
    plt.show()
    
    print("\nAll tests passed!")