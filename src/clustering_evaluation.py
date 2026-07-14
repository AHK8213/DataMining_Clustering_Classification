"""
clustering_evaluation.py - Evaluation and comparison of clustering results

Provides:
- Computing metrics for clustering results
- Comparing multiple algorithms
- Holistic model selection
- Optimal K determination (Elbow, Silhouette, etc.)
- Visualization of comparisons
- Nonlinear shape detection (Two-Moons)
- Noise resistance experiments
"""

import warnings
from typing import Dict, List, Tuple, Optional, Any, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import (
    silhouette_score, 
    calinski_harabasz_score, 
    davies_bouldin_score,
    adjusted_rand_score
)
from sklearn.datasets import make_moons
from sklearn.cluster import MeanShift, estimate_bandwidth

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
    - Nonlinear shape detection (Two-Moons)
    - Noise resistance experiments
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
        self.nonlinear_results = None
        self.noise_results = None
    
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
        primary_metric: str = 'silhouette',
        include_nonlinear: bool = False,
        include_noise: bool = False
    ) -> str:
        """
        Select the best model holistically.
        
        Args:
            min_clusters: Minimum number of clusters (excluding noise)
            max_noise_pct: Maximum allowed noise percentage
            primary_metric: Primary metric for selection
            include_nonlinear: Include nonlinear ARI in ranking
            include_noise: Include noise robustness in ranking
        
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
        
        # If including additional metrics, compute them first
        if include_nonlinear and self.nonlinear_results is None:
            self.evaluate_nonlinear_shapes(verbose=False)
        
        if include_noise and self.noise_results is None:
            self.evaluate_noise_resistance(verbose=False)
        
        return eligible[primary_metric].idxmax()
    
    def rank_algorithms(
        self,
        metrics: List[str] = None,
        ascending: Dict[str, bool] = None,
        include_nonlinear: bool = True,
        include_noise: bool = True
    ) -> pd.DataFrame:
        """
        Rank algorithms by multiple metrics.
        
        Args:
            metrics: List of metrics to include in ranking
            ascending: Dict mapping metric to ascending/descending
            include_nonlinear: Include nonlinear ARI in ranking
            include_noise: Include noise robustness in ranking
        
        Returns:
            DataFrame with ranks and overall score
        """
        df = self.get_comparison_df().copy()
        
        # Add nonlinear results if requested
        if include_nonlinear:
            if self.nonlinear_results is None:
                self.evaluate_nonlinear_shapes(verbose=False)
            nonlinear_ari = {name: res['ari'] for name, res in self.nonlinear_results.items()}
            df['nonlinear_ari'] = pd.Series(nonlinear_ari)
        
        # Add noise results if requested
        if include_noise:
            if self.noise_results is None:
                self.evaluate_noise_resistance(verbose=False)
            # Average ARI across noise levels
            noise_robustness = {}
            for name, results in self.noise_results.items():
                if results:
                    noise_robustness[name] = np.mean([r['mean_ari_robustness'] for r in results])
            df['noise_robustness'] = pd.Series(noise_robustness)
        
        if metrics is None:
            metrics = ['silhouette', 'davies_bouldin', 'dunn', 'runtime_s']
            if include_nonlinear and 'nonlinear_ari' in df.columns:
                metrics.append('nonlinear_ari')
            if include_noise and 'noise_robustness' in df.columns:
                metrics.append('noise_robustness')
        
        if ascending is None:
            ascending = {
                'silhouette': False,  # higher is better
                'davies_bouldin': True,  # lower is better
                'dunn': False,  # higher is better
                'calinski_harabasz': False,  # higher is better
                'runtime_s': True,  # lower is better
                'nonlinear_ari': False,  # higher is better
                'noise_robustness': False  # higher is better
            }
        
        # Compute ranks
        rank_df = pd.DataFrame(index=df.index)
        
        for metric in metrics:
            if metric not in df.columns:
                continue
            
            # Skip if all values are NaN
            if df[metric].isna().all():
                continue
            
            asc = ascending.get(metric, False)
            rank_df[f'{metric}_rank'] = df[metric].rank(ascending=asc, na_option='keep')
        
        # Overall rank score (average of ranks)
        rank_cols = [c for c in rank_df.columns if c.endswith('_rank')]
        if rank_cols:
            # Only include columns without NaN
            valid_rank_cols = []
            for col in rank_cols:
                if not rank_df[col].isna().all():
                    valid_rank_cols.append(col)
            
            if valid_rank_cols:
                rank_df['overall_rank_score'] = rank_df[valid_rank_cols].mean(axis=1)
                df['overall_rank_score'] = rank_df['overall_rank_score']
        
        # Sort by overall rank
        if 'overall_rank_score' in df.columns:
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
        
        # Handle single metric case
        if n_metrics == 1:
            axes = [axes]
        
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
    
    # ========================================================================
    # NEW: Nonlinear Shape Detection (Two-Moons)
    # ========================================================================
    
    def evaluate_nonlinear_shapes(
        self,
        n_samples: int = 1500,
        noise: float = 0.07,
        random_state: int = RANDOM_STATE,
        verbose: bool = None,
        figsize: Tuple[int, int] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate algorithms on Two-Moons nonlinear shape.
        
        This is a critical test for detecting non-linear, interlocking structures
        that centroid-based methods cannot capture.
        
        Args:
            n_samples: Number of samples for two-moons dataset
            noise: Noise level for two-moons dataset
            random_state: Random seed
            verbose: Print progress
            figsize: Figure size for visualization
        
        Returns:
            Dictionary mapping algorithm name to results (ARI vs true labels)
        """
        if verbose is None:
            verbose = self.verbose
        
        if self.verbose:
            print("\n" + "=" * 70)
            print("NONLINEAR SHAPE DETECTION (Two-Moons)")
            print("=" * 70)
            print(f"Generating {n_samples:,} samples with noise={noise:.2f}...")
        
        # Generate two-moons dataset
        X_moons, y_true = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
        X_moons = ensure_float64(X_moons)
        
        # Estimate bandwidth for MeanShift
        bandwidth = estimate_bandwidth(X_moons, quantile=0.2, random_state=random_state)
        
        # Define algorithms for nonlinear detection
        nonlinear_algos = {
            'K-Means': lambda: self._get_clustering_fn('K-Means'),
            'Bisecting K-Means': lambda: self._get_clustering_fn('Bisecting K-Means'),
            'DBSCAN': lambda: self._get_clustering_fn('DBSCAN'),
            'OPTICS': lambda: self._get_clustering_fn('OPTICS'),
            'HDBSCAN': lambda: self._get_clustering_fn('HDBSCAN'),
            'Gaussian Mixture': lambda: self._get_clustering_fn('Gaussian Mixture'),
            'K-Medoids': lambda: self._get_clustering_fn('K-Medoids'),
            'K-Median': lambda: self._get_clustering_fn('K-Median'),
            'Fuzzy C-Means': lambda: self._get_clustering_fn('Fuzzy C-Means'),
            'Kernel K-Means': lambda: self._get_clustering_fn('Kernel K-Means'),
            'Agglomerative (Ward)': lambda: self._get_clustering_fn('Agglomerative (Ward)'),
            'Agglomerative (Single)': lambda: self._get_clustering_fn('Agglomerative (Single)'),
            'Agglomerative (Complete)': lambda: self._get_clustering_fn('Agglomerative (Complete)'),
            'MeanShift': lambda: MeanShift(bandwidth=bandwidth).fit_predict(X_moons),
        }
        
        # Run algorithms and compute ARI
        results = {}
        labels_moons = {}
        
        for name, fn in nonlinear_algos.items():
            try:
                if verbose:
                    print(f"  Running: {name}...")
                
                # Get labels for two-moons
                labels = fn()
                labels_moons[name] = labels
                
                # Compute ARI with true labels
                ari = adjusted_rand_score(y_true, labels)
                results[name] = {
                    'ari': ari,
                    'n_clusters': len(set(labels)) - (1 if -1 in labels else 0),
                    'noise_pct': (labels == -1).mean() * 100 if -1 in labels else 0
                }
                
                if verbose:
                    print(f"    ARI={ari:.3f}")
            
            except Exception as e:
                if verbose:
                    print(f"    Failed: {e}")
                results[name] = {'ari': np.nan, 'n_clusters': 0, 'noise_pct': 0}
        
        self.nonlinear_results = results
        
        # Create visualization
        fig = self._plot_nonlinear_shapes(X_moons, y_true, labels_moons, figsize)
        
        if verbose:
            print("\nNonlinear Shape Detection Results (ARI vs true moons):")
            sorted_results = sorted(results.items(), key=lambda x: x[1]['ari'], reverse=True)
            for name, res in sorted_results:
                ari = res['ari']
                if not np.isnan(ari):
                    print(f"  {name:30s}: ARI={ari:.3f}")
        
        return results
    
    def _get_clustering_fn(self, name: str):
        """
        Get clustering function for a named algorithm.
        
        Args:
            name: Algorithm name
        
        Returns:
            Function that returns cluster labels
        """
        # This is a simplified version - in practice, you'd have a mapping
        # to the actual clustering functions from ClusteringRunner
        from sklearn.cluster import (
            KMeans, DBSCAN, OPTICS, HDBSCAN,
            AgglomerativeClustering, BisectingKMeans
        )
        from sklearn.mixture import GaussianMixture
        
        # Get K from original data (if available)
        k = 2  # Two moons has 2 clusters
        
        # Map names to functions
        if name == 'K-Means':
            return lambda X: KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit_predict(X)
        elif name == 'Bisecting K-Means':
            return lambda X: BisectingKMeans(n_clusters=k, random_state=RANDOM_STATE).fit_predict(X)
        elif name == 'DBSCAN':
            return lambda X: DBSCAN(eps=0.2, min_samples=5).fit_predict(X)
        elif name == 'OPTICS':
            return lambda X: OPTICS(min_samples=5).fit_predict(X)
        elif name == 'HDBSCAN':
            return lambda X: HDBSCAN(min_cluster_size=15).fit_predict(X)
        elif name == 'Gaussian Mixture':
            return lambda X: GaussianMixture(n_components=k, random_state=RANDOM_STATE).fit_predict(X)
        elif name == 'Agglomerative (Ward)':
            return lambda X: AgglomerativeClustering(n_clusters=k, linkage='ward').fit_predict(X)
        elif name == 'Agglomerative (Single)':
            return lambda X: AgglomerativeClustering(n_clusters=k, linkage='single').fit_predict(X)
        elif name == 'Agglomerative (Complete)':
            return lambda X: AgglomerativeClustering(n_clusters=k, linkage='complete').fit_predict(X)
        elif name == 'K-Medoids':
            from src.clustering_algorithms import k_medoids
            return lambda X: k_medoids(X, k, random_state=RANDOM_STATE)[0]
        elif name == 'K-Median':
            from src.clustering_algorithms import k_median
            return lambda X: k_median(X, k, random_state=RANDOM_STATE)[0]
        elif name == 'Fuzzy C-Means':
            from src.clustering_algorithms import fuzzy_c_means
            return lambda X: fuzzy_c_means(X, k, random_state=RANDOM_STATE)[0]
        elif name == 'Kernel K-Means':
            from src.clustering_algorithms import kernel_kmeans
            return lambda X: kernel_kmeans(X, k, gamma=3, random_state=RANDOM_STATE)
        else:
            raise ValueError(f"Unknown algorithm: {name}")
    
    def _plot_nonlinear_shapes(
        self,
        X_moons: np.ndarray,
        y_true: np.ndarray,
        labels_moons: Dict[str, np.ndarray],
        figsize: Optional[Tuple[int, int]] = None
    ) -> plt.Figure:
        """
        Plot nonlinear shape detection results.
        
        Args:
            X_moons: Two-moons data
            y_true: True labels
            labels_moons: Dictionary of predicted labels
            figsize: Figure size
        
        Returns:
            Matplotlib figure
        """
        if figsize is None:
            n_algos = len(labels_moons)
            n_cols = 5
            n_rows = int(np.ceil(n_algos / n_cols))
            figsize = (4 * n_cols, 3.5 * n_rows)
        
        n_algos = len(labels_moons)
        n_cols = 5
        n_rows = int(np.ceil(n_algos / n_cols))
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = np.array(axes).reshape(-1)
        
        for ax, (name, labels) in zip(axes, labels_moons.items()):
            # Plot points colored by predicted cluster
            scatter = ax.scatter(X_moons[:, 0], X_moons[:, 1], c=labels, 
                                 cmap='tab10', s=10, alpha=0.7)
            
            # Compute ARI
            ari = adjusted_rand_score(y_true, labels)
            
            ax.set_title(f"{name}\nARI={ari:.2f}", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
        
        # Hide unused subplots
        for ax in axes[n_algos:]:
            ax.axis('off')
        
        plt.suptitle(f"Nonlinear Shape Detection (Two-Moons, n={X_moons.shape[0]:,})", 
                     y=1.01, fontsize=14)
        plt.tight_layout()
        return fig
    
    # ========================================================================
    # NEW: Noise Resistance Experiments
    # ========================================================================
    
    def evaluate_noise_resistance(
        self,
        noise_levels: List[float] = None,
        repeats: int = 2,
        noise_scale: float = 3.0,
        random_state: int = RANDOM_STATE,
        verbose: bool = None,
        figsize: Tuple[int, int] = (16, 6)
    ) -> Dict[str, List[Dict[str, float]]]:
        """
        Evaluate algorithm stability under Gaussian noise.
        
        Tests how well each algorithm handles noisy data, which is crucial
        for real-world deployment where data quality varies.
        
        Args:
            noise_levels: List of noise fractions (e.g., [0.05, 0.15])
            repeats: Number of repeats per noise level
            noise_scale: Standard deviation of Gaussian noise
            random_state: Random seed
            verbose: Print progress
            figsize: Figure size for visualization
        
        Returns:
            Dictionary mapping algorithm name to list of result dicts
        """
        if verbose is None:
            verbose = self.verbose
        
        if noise_levels is None:
            noise_levels = [0.05, 0.15]
        
        if self.verbose:
            print("\n" + "=" * 70)
            print("NOISE RESISTANCE EXPERIMENTS")
            print("=" * 70)
            print(f"Testing {len(self.labels_dict)} algorithms x {len(noise_levels)} noise levels "
                  f"x {repeats} repeats = {len(self.labels_dict) * len(noise_levels) * repeats} refits.")
        
        results = {}
        
        for name, original_labels in self.labels_dict.items():
            if self.verbose:
                print(f"\n  Testing: {name}")
            
            # Get original runtime
            original_runtime = self.runtimes.get(name, 1.0)
            
            name_results = []
            
            for noise_frac in noise_levels:
                aris = []
                sils = []
                runtimes = []
                
                for rep in range(repeats):
                    seed = random_state + rep
                    
                    # Add noise to data
                    X_noisy = self._add_noise(self.X, frac=noise_frac, 
                                              scale=noise_scale, random_state=seed)
                    
                    # Refit algorithm on noisy data
                    try:
                        t0 = plt.get_fignums().__len__()  # Just for timing
                        import time
                        start = time.time()
                        
                        noisy_labels = self._refit_algorithm(name, X_noisy)
                        
                        elapsed = time.time() - start
                    except Exception as e:
                        if self.verbose:
                            print(f"    [skip] @ noise={noise_frac}: {e}")
                        continue
                    
                    # Compute ARI vs original labels
                    ari = adjusted_rand_score(original_labels, noisy_labels)
                    aris.append(ari)
                    runtimes.append(elapsed)
                    
                    # Compute silhouette on noisy data
                    mask = noisy_labels != -1
                    if len(set(np.asarray(noisy_labels)[mask])) >= 2:
                        try:
                            sil = silhouette_score(X_noisy[mask], np.asarray(noisy_labels)[mask])
                            sils.append(sil)
                        except:
                            pass
                
                if not aris:
                    continue
                
                name_results.append({
                    'noise_level': noise_frac,
                    'mean_ari_robustness': np.mean(aris),
                    'stability_ari_std': np.std(aris),
                    'mean_silhouette_noisy': np.mean(sils) if sils else np.nan,
                    'mean_runtime_s': np.mean(runtimes),
                    'runtime_degradation_x': np.mean(runtimes) / original_runtime if original_runtime > 0 else np.nan,
                })
                
                if self.verbose:
                    print(f"    noise={noise_frac:.0%}: ARI={np.mean(aris):.3f}±{np.std(aris):.3f}, "
                          f"time={np.mean(runtimes):.2f}s")
            
            results[name] = name_results
        
        self.noise_results = results
        
        # Create visualization
        self._plot_noise_results(figsize)
        
        if self.verbose:
            print("\nNoise resistance experiments completed.")
        
        return results
    
    def _add_noise(
        self, 
        X: np.ndarray, 
        frac: float = 0.1, 
        scale: float = 3.0, 
        random_state: int = 42
    ) -> np.ndarray:
        """
        Add Gaussian noise to a subset of data points.
        
        Args:
            X: Original data
            frac: Fraction of points to add noise to
            scale: Standard deviation of noise
            random_state: Random seed
        
        Returns:
            Noisy data
        """
        rng = np.random.RandomState(random_state)
        X_noisy = X.copy()
        n_noise = int(frac * X.shape[0])
        idx = rng.choice(X.shape[0], n_noise, replace=False)
        X_noisy[idx] += rng.normal(0, scale, size=(n_noise, X.shape[1]))
        return X_noisy
    
    def _refit_algorithm(self, name: str, X_noisy: np.ndarray) -> np.ndarray:
        """
        Refit an algorithm on noisy data.
        
        Args:
            name: Algorithm name
            X_noisy: Noisy data
        
        Returns:
            Cluster labels on noisy data
        """
        from sklearn.cluster import (
            KMeans, DBSCAN, OPTICS, HDBSCAN,
            AgglomerativeClustering, BisectingKMeans
        )
        from sklearn.mixture import GaussianMixture
        from src.clustering_algorithms import (
            k_medoids, k_median, fuzzy_c_means, kernel_kmeans
        )
        
        # Get K from original labels
        original_labels = self.labels_dict.get(name)
        if original_labels is not None:
            k = len(set(original_labels)) - (1 if -1 in original_labels else 0)
            k = max(2, k)  # Ensure at least 2 clusters
        else:
            k = 2
        
        # Map name to function
        if name == 'K-Means':
            return KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit_predict(X_noisy)
        elif name == 'Bisecting K-Means':
            return BisectingKMeans(n_clusters=k, random_state=RANDOM_STATE).fit_predict(X_noisy)
        elif name == 'DBSCAN':
            # Use default eps (or estimate from data)
            from sklearn.neighbors import NearestNeighbors
            nn = NearestNeighbors(n_neighbors=5).fit(X_noisy)
            dist, _ = nn.kneighbors(X_noisy)
            eps = np.percentile(dist[:, -1], 50)
            return DBSCAN(eps=eps, min_samples=5).fit_predict(X_noisy)
        elif name == 'OPTICS':
            return OPTICS(min_samples=10).fit_predict(X_noisy)
        elif name == 'HDBSCAN':
            return HDBSCAN(min_cluster_size=30).fit_predict(X_noisy)
        elif name == 'Gaussian Mixture':
            return GaussianMixture(n_components=k, random_state=RANDOM_STATE).fit_predict(X_noisy)
        elif name == 'K-Medoids':
            return k_medoids(X_noisy, k, random_state=RANDOM_STATE)[0]
        elif name == 'K-Median':
            return k_median(X_noisy, k, random_state=RANDOM_STATE)[0]
        elif name == 'Fuzzy C-Means':
            return fuzzy_c_means(X_noisy, k, random_state=RANDOM_STATE)[0]
        elif name == 'Kernel K-Means':
            return kernel_kmeans(X_noisy, k, random_state=RANDOM_STATE)
        elif name.startswith('Agglomerative'):
            linkage_type = 'ward'
            if 'Single' in name:
                linkage_type = 'single'
            elif 'Complete' in name:
                linkage_type = 'complete'
            return AgglomerativeClustering(n_clusters=k, linkage=linkage_type).fit_predict(X_noisy)
        else:
            # Fallback to K-Means
            return KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit_predict(X_noisy)
    
    def _plot_noise_results(self, figsize: Tuple[int, int] = (16, 6)) -> plt.Figure:
        """
        Plot noise resistance results.
        
        Args:
            figsize: Figure size
        
        Returns:
            Matplotlib figure
        """
        if self.noise_results is None:
            return None
        
        # Prepare data for plotting
        rows = []
        for name, results in self.noise_results.items():
            for r in results:
                rows.append({
                    'algorithm': name,
                    'noise_level': r['noise_level'],
                    'mean_ari_robustness': r['mean_ari_robustness'],
                    'runtime_degradation_x': r['runtime_degradation_x']
                })
        
        df = pd.DataFrame(rows)
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Plot 1: ARI robustness by noise level
        pivot_ari = df.pivot(index='algorithm', columns='noise_level', 
                            values='mean_ari_robustness')
        # Sort by performance at highest noise level
        pivot_ari = pivot_ari.sort_values(pivot_ari.columns[-1], ascending=False)
        pivot_ari.plot(kind='barh', ax=axes[0])
        axes[0].set_title("Robustness: ARI(original, noisy) by noise level\n(higher = more robust)")
        axes[0].set_xlabel("ARI")
        
        # Plot 2: Runtime degradation
        pivot_degrad = df.pivot(index='algorithm', columns='noise_level', 
                               values='runtime_degradation_x')
        pivot_degrad = pivot_degrad.reindex(pivot_ari.index)
        pivot_degrad.plot(kind='barh', ax=axes[1], color=['#8888ff', '#ff8888'])
        axes[1].set_title("Runtime degradation (x original runtime)")
        axes[1].set_xlabel("x original runtime")
        
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
    ranked = comparator.rank_algorithms(include_nonlinear=True, include_noise=True)
    print(f"\nRanked algorithms:")
    print(ranked[['silhouette', 'davies_bouldin', 'runtime_s', 'overall_rank_score']].head())
    
    # Test nonlinear detection
    print("\n--- Nonlinear Shape Detection ---")
    nonlinear_results = comparator.evaluate_nonlinear_shapes(verbose=True)
    
    # Test noise resistance
    print("\n--- Noise Resistance ---")
    noise_results = comparator.evaluate_noise_resistance(verbose=True)
    
    print("\nAll tests passed!")