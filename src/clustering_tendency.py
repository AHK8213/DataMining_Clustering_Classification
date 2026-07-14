"""
clustering_tendency.py - Clustering tendency assessment

Provides:
- Hopkins statistic computation
- KL Divergence for measuring distance between distributions
- Comprehensive visualization of clustering tendency
- Interpretation of results
"""

import warnings
from typing import Optional, Tuple, Dict, Any, List, Union

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import NearestNeighbors
from scipy.special import kl_div, rel_entr
from scipy.stats import entropy
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

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


def kl_between_clusters(
    X: np.ndarray,
    labels: np.ndarray,
    method: str = 'gmm',
    n_components: int = 10
) -> Dict[str, Any]:
    """
    Compute KL divergence between cluster distributions.
    
    Measures how different the feature distributions are across clusters.
    Higher KL divergence indicates more distinct clusters.
    
    Args:
        X: Data points (n_samples, n_features)
        labels: Cluster labels
        method: 'gmm' for Gaussian Mixture, 'histogram' for binned histograms
        n_components: Number of components for GMM or bins for histogram
    
    Returns:
        Dictionary containing KL divergence results
    """
    labels = np.asarray(labels)
    unique_labels = [l for l in np.unique(labels) if l != -1]
    n_clusters = len(unique_labels)
    
    if n_clusters < 2:
        return {
            'kl_divergence': np.nan,
            'symmetric_kl': np.nan,
            'js_divergence': np.nan,
            'pairwise_kl': {},
            'method': method,
            'n_clusters': n_clusters,
            'unique_labels': unique_labels
        }
    
    # Standardize data for better GMM fitting
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Fit GMM for each cluster
    cluster_gmms = {}
    for label in unique_labels:
        mask = labels == label
        X_cluster = X_scaled[mask]
        if len(X_cluster) > 2 * n_components:
            gmm = GaussianMixture(
                n_components=min(n_components, max(1, len(X_cluster) // 3)),
                random_state=RANDOM_STATE
            )
            gmm.fit(X_cluster)
            cluster_gmms[label] = gmm
        else:
            # For small clusters, use a simple multivariate normal
            mean = X_cluster.mean(axis=0)
            cov = np.cov(X_cluster.T) + np.eye(X.shape[1]) * 1e-6
            cluster_gmms[label] = {'mean': mean, 'cov': cov, 'type': 'single'}
    
    # Compute KL divergence between clusters
    pairwise_kl = {}
    
    for i, label_i in enumerate(unique_labels):
        for j, label_j in enumerate(unique_labels):
            if i >= j:
                continue
            pair_key = f"{label_i}_{label_j}"
            
            # Compute KL divergence between GMMs
            gmm_i = cluster_gmms[label_i]
            gmm_j = cluster_gmms[label_j]
            
            if isinstance(gmm_i, GaussianMixture) and isinstance(gmm_j, GaussianMixture):
                # Use Monte Carlo approximation for GMM KL
                kl_ij = _kl_gmm_mc(gmm_i, gmm_j, n_samples=1000)
                kl_ji = _kl_gmm_mc(gmm_j, gmm_i, n_samples=1000)
            else:
                # Use analytical formula for single Gaussian
                kl_ij = _kl_gaussian(gmm_i, gmm_j)
                kl_ji = _kl_gaussian(gmm_j, gmm_i)
            
            pairwise_kl[pair_key] = {
                'kl_ij': kl_ij,
                'kl_ji': kl_ji,
                'symmetric_kl': (kl_ij + kl_ji) / 2
            }
    
    # Aggregate results
    all_symmetric = [v['symmetric_kl'] for v in pairwise_kl.values()] if pairwise_kl else [np.nan]
    
    return {
        'kl_divergence': np.mean([v['kl_ij'] for v in pairwise_kl.values()]) if pairwise_kl else np.nan,
        'symmetric_kl': np.mean(all_symmetric),
        'js_divergence': np.mean([_kl_to_js(v['symmetric_kl']) for v in pairwise_kl.values()]) if pairwise_kl else np.nan,
        'pairwise_kl': pairwise_kl,
        'method': method,
        'n_clusters': n_clusters,
        'unique_labels': unique_labels,
        'max_kl': max(all_symmetric) if all_symmetric and not np.isnan(all_symmetric[0]) else np.nan,
        'min_kl': min(all_symmetric) if all_symmetric and not np.isnan(all_symmetric[0]) else np.nan,
    }


def _kl_gmm_mc(gmm_i, gmm_j, n_samples: int = 1000) -> float:
    """
    Approximate KL divergence between two GMMs using Monte Carlo sampling.
    
    KL(GMM_i || GMM_j) ≈ (1/N) * sum(log(p_i(x) / p_j(x)))
    """
    try:
        samples = gmm_i.sample(n_samples)[0]
        log_p_i = gmm_i.score_samples(samples)
        log_p_j = gmm_j.score_samples(samples)
        return np.mean(log_p_i - log_p_j)
    except:
        return np.nan


def _kl_gaussian(gmm_i, gmm_j) -> float:
    """
    Compute KL divergence between two Gaussian distributions.
    
    For multivariate normal distributions:
    KL(N(μ1, Σ1) || N(μ2, Σ2)) = 0.5 * (log(det(Σ2)/det(Σ1)) - d + trace(Σ2^{-1}Σ1) + (μ2-μ1)^T Σ2^{-1} (μ2-μ1))
    """
    try:
        if isinstance(gmm_i, dict) and gmm_i.get('type') == 'single':
            mean_i = gmm_i['mean']
            cov_i = gmm_i['cov']
        else:
            mean_i = gmm_i.means_[0] if gmm_i.means_.shape[0] == 1 else gmm_i.means_[0]
            cov_i = gmm_i.covariances_[0] if gmm_i.covariances_.shape[0] == 1 else gmm_i.covariances_[0]
        
        if isinstance(gmm_j, dict) and gmm_j.get('type') == 'single':
            mean_j = gmm_j['mean']
            cov_j = gmm_j['cov']
        else:
            mean_j = gmm_j.means_[0] if gmm_j.means_.shape[0] == 1 else gmm_j.means_[0]
            cov_j = gmm_j.covariances_[0] if gmm_j.covariances_.shape[0] == 1 else gmm_j.covariances_[0]
        
        d = len(mean_i)
        
        # Add small regularization for numerical stability
        cov_i = cov_i + np.eye(d) * 1e-6
        cov_j = cov_j + np.eye(d) * 1e-6
        
        inv_cov_j = np.linalg.inv(cov_j)
        term1 = np.log(np.linalg.det(cov_j) / np.linalg.det(cov_i))
        term2 = -d
        term3 = np.trace(inv_cov_j @ cov_i)
        diff = mean_j - mean_i
        term4 = diff.T @ inv_cov_j @ diff
        return 0.5 * (term1 + term2 + term3 + term4)
    except:
        return np.nan


def _kl_to_js(kl_value: float) -> float:
    """
    Convert symmetric KL to JS divergence (approximation).
    JS is bounded [0, log(2)] ≈ [0, 0.693].
    """
    return 1 - np.exp(-kl_value if not np.isnan(kl_value) else 0)


# ============================================================================
# Clustering Tendency Analysis (Enhanced with KL)
# ============================================================================

class ClusteringTendency:
    """
    Assess clustering tendency of a dataset.
    
    Enhanced with KL divergence analysis.
    
    Usage:
        tendency = ClusteringTendency(X)
        H = tendency.compute_hopkins()
        kl_results = tendency.compute_kl_divergence(labels)
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
        self.kl_results = None
        self.prelim_labels = None
        
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
    
    def compute_kl_divergence(
        self,
        labels: Optional[np.ndarray] = None,
        k: Optional[int] = None,
        method: str = 'gmm',
        n_components: int = 10,
        auto_cluster: bool = True
    ) -> Dict[str, Any]:
        """
        Compute KL divergence between clusters.
        
        Args:
            labels: Cluster labels (if None, auto-cluster)
            k: Number of clusters for auto-clustering
            method: 'gmm' or 'histogram'
            n_components: Number of components/bins
            auto_cluster: If True and labels is None, run KMeans automatically
        
        Returns:
            Dictionary with KL divergence results
        """
        if labels is None and auto_cluster:
            if self.verbose:
                print("Auto-clustering for KL divergence analysis...")
            
            # Use K=3 as default or specified k
            k = k or 3
            kmeans = KMeans(n_clusters=k, n_init=10, random_state=self.random_state)
            labels = kmeans.fit_predict(self.X)
            self.prelim_labels = labels
            
            if self.verbose:
                print(f"Created {k} preliminary clusters for KL analysis")
        
        if labels is None:
            raise ValueError("Either provide labels or enable auto_cluster=True")
        
        if self.verbose:
            print(f"Computing KL divergence between clusters using {method}...")
        
        self.kl_results = kl_between_clusters(
            self.X, labels, method=method, n_components=n_components
        )
        
        if self.verbose:
            print(f"KL divergence: {self.kl_results['kl_divergence']:.3f}")
            print(f"Symmetric KL: {self.kl_results['symmetric_kl']:.3f}")
            print(f"JS divergence: {self.kl_results['js_divergence']:.3f}")
        
        return self.kl_results
    
    def interpret(self) -> str:
        """
        Get interpretation of the Hopkins statistic.
        
        Returns:
            Interpretation string
        """
        if self.hopkins_value is None:
            self.compute_hopkins()
        return self.interpretation
    
    def interpret_kl(self) -> str:
        """
        Get interpretation of KL divergence results.
        
        Returns:
            Interpretation string
        """
        if self.kl_results is None:
            return "No KL divergence results available. Run compute_kl_divergence() first."
        
        js = self.kl_results['js_divergence']
        
        if np.isnan(js):
            return "Insufficient data for KL divergence computation."
        elif js > 0.3:
            return f"Strong cluster separation (JS={js:.3f}). Clusters are well-distinguished."
        elif js > 0.15:
            return f"Moderate cluster separation (JS={js:.3f}). Clusters show some distinction."
        elif js > 0.05:
            return f"Weak cluster separation (JS={js:.3f}). Clusters overlap significantly."
        else:
            return f"Very weak cluster separation (JS={js:.3f}). Clusters are nearly identical."
    
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
        figsize: Tuple[int, int] = (15, 10),
        show_random: bool = True,
        show_kl: bool = True,
        show_heatmap: bool = True,
        show_distribution: bool = True
    ) -> plt.Figure:
        """
        Visualize clustering tendency with KL divergence.
        
        Args:
            figsize: Figure size
            show_random: Show random data comparison
            show_kl: Show KL divergence visualization
            show_heatmap: Show KL heatmap
            show_distribution: Show KL distribution
        
        Returns:
            Matplotlib figure
        """
        if self.hopkins_value is None:
            self.compute_hopkins()
        
        # Determine number of subplots
        n_plots = 1  # Hopkins
        if show_random:
            n_plots += 1
        if show_kl and self.kl_results:
            n_plots += 1
            if show_heatmap and self.kl_results.get('pairwise_kl'):
                n_plots += 1
            if show_distribution and self.kl_results.get('pairwise_kl'):
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
        
        # Plot 3: KL summary (optional)
        if show_kl and self.kl_results and plot_idx < len(axes):
            ax = axes[plot_idx]
            
            kl_results = self.kl_results
            metrics = ['kl_divergence', 'symmetric_kl', 'js_divergence']
            values = [kl_results.get(m, np.nan) for m in metrics]
            labels_metrics = ['KL Div', 'Sym KL', 'JS Div']
            colors = ['#3498db', '#2ecc71', '#e74c3c']
            
            bars = ax.bar(labels_metrics, values, color=colors, alpha=0.7, edgecolor='black')
            ax.set_ylabel('Divergence Value', fontsize=12)
            ax.set_title(f"KL Divergence Summary\n{self.interpret_kl()}", fontsize=12)
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add value annotations
            for bar, val in zip(bars, values):
                if not np.isnan(val):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{val:.3f}', ha='center', va='bottom', fontsize=10)
            plot_idx += 1
        
        # Plot 4: KL heatmap (optional)
        if show_heatmap and show_kl and self.kl_results and self.kl_results.get('pairwise_kl') and plot_idx < len(axes):
            ax = axes[plot_idx]
            
            pairwise_kl = self.kl_results['pairwise_kl']
            unique_labels = self.kl_results.get('unique_labels', [])
            
            if pairwise_kl and unique_labels:
                n_clusters = len(unique_labels)
                kl_matrix = np.zeros((n_clusters, n_clusters))
                
                for i, label_i in enumerate(unique_labels):
                    for j, label_j in enumerate(unique_labels):
                        if i == j:
                            kl_matrix[i, j] = 0
                        else:
                            pair_key = f"{label_i}_{label_j}"
                            if pair_key in pairwise_kl:
                                kl_matrix[i, j] = pairwise_kl[pair_key]['symmetric_kl']
                            else:
                                pair_key_rev = f"{label_j}_{label_i}"
                                if pair_key_rev in pairwise_kl:
                                    kl_matrix[i, j] = pairwise_kl[pair_key_rev]['symmetric_kl']
                
                # Create heatmap
                im = ax.imshow(kl_matrix, cmap='RdYlBu_r', 
                              vmin=0, vmax=kl_matrix.max() if kl_matrix.max() > 0 else 1)
                
                ax.set_xticks(range(n_clusters))
                ax.set_yticks(range(n_clusters))
                ax.set_xticklabels([f'C{int(l)}' for l in unique_labels])
                ax.set_yticklabels([f'C{int(l)}' for l in unique_labels])
                ax.set_title("Pairwise KL Divergence\n(Between Clusters)", fontsize=12)
                
                # Add text annotations
                for i in range(n_clusters):
                    for j in range(n_clusters):
                        text = ax.text(j, i, f'{kl_matrix[i, j]:.2f}',
                                      ha="center", va="center",
                                      color="white" if kl_matrix[i, j] > kl_matrix.max()/2 else "black",
                                      fontsize=8)
                
                plt.colorbar(im, ax=ax, label='Symmetric KL')
                plot_idx += 1
        
        # Plot 5: KL distribution (optional)
        if show_distribution and show_kl and self.kl_results and self.kl_results.get('pairwise_kl') and plot_idx < len(axes):
            ax = axes[plot_idx]
            
            pairwise_kl = self.kl_results['pairwise_kl']
            
            if pairwise_kl:
                kl_values = [v['symmetric_kl'] for v in pairwise_kl.values() if not np.isnan(v['symmetric_kl'])]
                
                if kl_values:
                    ax.hist(kl_values, bins=max(5, len(kl_values)), color='steelblue', 
                           alpha=0.7, edgecolor='black')
                    ax.axvline(np.mean(kl_values), color='red', linestyle='--', 
                              linewidth=2, label=f"Mean: {np.mean(kl_values):.3f}")
                    ax.axvline(np.median(kl_values), color='green', linestyle='--', 
                              linewidth=2, label=f"Median: {np.median(kl_values):.3f}")
                    ax.set_xlabel("Symmetric KL Divergence", fontsize=12)
                    ax.set_ylabel("Frequency", fontsize=12)
                    ax.set_title("Distribution of Pairwise KL Divergences", fontsize=12)
                    ax.legend(loc='best')
                    ax.grid(True, alpha=0.3)
                    plot_idx += 1
        
        # Hide any unused subplots
        for ax in axes[plot_idx:]:
            ax.axis('off')
        
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
        
        if self.kl_results:
            report += f"""
{"-"*70}
KL Divergence Analysis:
  Method: {self.kl_results['method']}
  Number of clusters: {self.kl_results['n_clusters']}
  Mean KL Divergence: {self.kl_results['kl_divergence']:.4f}
  Mean Symmetric KL: {self.kl_results['symmetric_kl']:.4f}
  Jensen-Shannon Divergence: {self.kl_results['js_divergence']:.4f}
  KL Interpretation: {self.interpret_kl()}
"""
        
        report += f"""
{"-"*70}
Recommendation:
  {self._get_recommendation()}
{"="*70}
"""
        return report
    
    def _get_recommendation(self) -> str:
        """Get recommendation based on Hopkins statistic and KL results."""
        has_kl = self.kl_results is not None
        js = self.kl_results.get('js_divergence', 0) if has_kl else 0
        
        if self.hopkins_value > 0.7 and (not has_kl or js > 0.15):
            return "Data is strongly clusterable. Proceed with full clustering analysis."
        elif self.hopkins_value > 0.6 and (not has_kl or js > 0.1):
            return "Data has moderate clustering structure. Clustering may be useful."
        elif self.hopkins_value > 0.5 and (not has_kl or js > 0.05):
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


def compute_kl_for_clustering(
    X: np.ndarray,
    labels: np.ndarray,
    method: str = 'gmm',
    n_components: int = 10,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Convenience function to compute KL divergence between clusters.
    
    Args:
        X: Data points
        labels: Cluster labels
        method: 'gmm' or 'histogram'
        n_components: Number of components/bins
        verbose: Print progress
    
    Returns:
        Dictionary with KL divergence results
    """
    if verbose:
        print(f"Computing KL divergence between clusters using {method}...")
    
    results = kl_between_clusters(X, labels, method=method, n_components=n_components)
    
    if verbose:
        print(f"Mean KL Divergence: {results['kl_divergence']:.4f}")
        print(f"Symmetric KL: {results['symmetric_kl']:.4f}")
        print(f"JS Divergence: {results['js_divergence']:.4f}")
    
    return results


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
    
    # Auto-compute KL with clustering
    kl_results = tendency_blobs.compute_kl_divergence(k=3, auto_cluster=True)
    
    print(tendency_blobs.get_report())
    
    # Plot with all visualizations
    fig = tendency_blobs.plot(show_kl=True, show_heatmap=True, show_distribution=True)
    plt.show()
    
    print("\nAll tests passed!")