"""
clustering_algorithms.py - Optimized clustering with subsampling for large datasets
"""

import time
import warnings
from typing import Tuple, Optional, Dict, Any, List, Union

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import (
    KMeans,
    MiniBatchKMeans,
    DBSCAN,
    OPTICS,
    HDBSCAN,
    BisectingKMeans,
    AgglomerativeClustering,
)
from sklearn.mixture import GaussianMixture
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.neighbors import NearestNeighbors

from src.config import (
    RANDOM_STATE,
    K_RANGE,
    DBSCAN_DEFAULT_EPS_PERCENTILE,
    DBSCAN_DEFAULT_MIN_SAMPLES,
    OPTICS_DEFAULT_MIN_SAMPLES,
    HDBSCAN_DEFAULT_MIN_CLUSTER_SIZE,
    USE_FLOAT64,
    VERBOSE,
)
from src.utils import (
    get_rng,
    cleanup,
    timer,
    ensure_float64,
    compute_clustering_metrics,
)

warnings.filterwarnings("ignore")


# ============================================================================
# Subsampling Configuration
# ============================================================================

SUBSAMPLE_LIMITS = {
    'kernel_kmeans': 8000,
    'agglomerative_ward': 15000,
    'agglomerative_single': 15000,
    'agglomerative_complete': 15000,
    'kmedoids': 10000,
    'gmm': None,
    'dbscan': None,
    'optics': None,
    'hdbscan': None,
    'kmeans': None,
    'bisecting_kmeans': None,
    'kmedian': None,
    'fuzzy_cmeans': None,
}

DEFAULT_SUBSAMPLE_LIMIT = 20000


# ============================================================================
# Utility Functions
# ============================================================================

def get_k_distance_eps(
    X: np.ndarray,
    k: int = 5,
    percentile: float = DBSCAN_DEFAULT_EPS_PERCENTILE,
    verbose: bool = VERBOSE
) -> Tuple[float, np.ndarray]:
    """Determine eps parameter for DBSCAN using k-distance plot."""
    nn = NearestNeighbors(n_neighbors=k).fit(X)
    distances, _ = nn.kneighbors(X)
    k_distances = np.sort(distances[:, -1])
    
    eps = np.percentile(k_distances, percentile)
    
    if verbose:
        print(f"Suggested eps (k={k}, {percentile}th percentile): {eps:.3f}")
    
    return eps, k_distances


def plot_k_distance(
    k_distances: np.ndarray,
    figsize: Tuple[int, int] = (8, 5)
):
    """Plot k-distance plot for DBSCAN eps selection."""
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(k_distances)
    ax.set_title("K-distance plot")
    ax.set_xlabel("Points sorted by distance")
    ax.set_ylabel("k-NN distance")
    ax.grid(True, alpha=0.3)
    return fig


def auto_subsample(
    X: np.ndarray,
    algorithm_name: str,
    max_samples: Optional[int] = None,
    random_state: int = RANDOM_STATE
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Subsample data for memory-intensive algorithms.
    
    Args:
        X: Full dataset
        algorithm_name: Name of the algorithm
        max_samples: Override the default max samples
        random_state: Random seed
    
    Returns:
        Tuple of (subsampled_X, indices_used or None)
    """
    n = X.shape[0]
    
    if max_samples is None:
        max_samples = SUBSAMPLE_LIMITS.get(algorithm_name, DEFAULT_SUBSAMPLE_LIMIT)
    
    if max_samples is None or n <= max_samples:
        return X, None
    
    rng = get_rng(random_state)
    indices = rng.choice(n, max_samples, replace=False)
    X_subsampled = X[indices]
    
    if VERBOSE:
        print(f"  ⚡ {algorithm_name}: subsampled from {n:,} to {max_samples:,} points")
    
    return X_subsampled, indices


# ============================================================================
# K-Medoids Implementation (Optimized)
# ============================================================================

def k_medoids(
    X: np.ndarray,
    k: int,
    max_iter: int = 20,
    random_state: int = RANDOM_STATE,
    verbose: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    K-Medoids clustering algorithm with optimized memory usage.
    """
    try:
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        
        n = X.shape[0]
        if k > n:
            raise ValueError(f"k ({k}) cannot be greater than number of samples ({n})")
        
        if n == 0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
        
        rng = get_rng(random_state)
        
        use_precomputed = n <= 20000
        
        if use_precomputed:
            D_full = cdist(X, X, metric='cityblock').astype(np.float32)
            
            medoid_idx = rng.choice(n, k, replace=False)
            labels = np.zeros(n, dtype=np.int64)
            
            for iteration in range(max_iter):
                D = D_full[:, medoid_idx]
                labels = np.argmin(D, axis=1).astype(np.int64)
                
                new_medoid_idx = medoid_idx.copy()
                for j in range(k):
                    members = np.where(labels == j)[0]
                    if len(members) == 0:
                        continue
                    
                    intra = D_full[np.ix_(members, members)].sum(axis=1)
                    new_medoid_idx[j] = members[np.argmin(intra)]
                
                if np.array_equal(new_medoid_idx, medoid_idx):
                    break
                medoid_idx = new_medoid_idx
            
            D = D_full[:, medoid_idx]
            labels = np.argmin(D, axis=1).astype(np.int64)
            
            del D_full
            
        else:
            medoid_idx = rng.choice(n, k, replace=False)
            labels = np.zeros(n, dtype=np.int64)
            
            for iteration in range(max_iter):
                D = cdist(X, X[medoid_idx], metric='cityblock')
                labels = np.argmin(D, axis=1).astype(np.int64)
                
                new_medoid_idx = medoid_idx.copy()
                for j in range(k):
                    members = np.where(labels == j)[0]
                    if len(members) == 0:
                        continue
                    
                    sub = X[members]
                    intra = cdist(sub, sub, metric='cityblock').sum(axis=1)
                    new_medoid_idx[j] = members[np.argmin(intra)]
                
                if np.array_equal(new_medoid_idx, medoid_idx):
                    break
                medoid_idx = new_medoid_idx
            
            D = cdist(X, X[medoid_idx], metric='cityblock')
            labels = np.argmin(D, axis=1).astype(np.int64)
        
        return labels, medoid_idx
        
    except Exception as e:
        print(f"ERROR in k_medoids: {e}")
        n = X.shape[0]
        if n > 0:
            labels = np.zeros(n, dtype=np.int64)
            return labels, np.array([0], dtype=np.int64)
        else:
            return np.array([], dtype=np.int64), np.array([], dtype=np.int64)


# ============================================================================
# K-Median Implementation
# ============================================================================

def k_median(
    X: np.ndarray,
    k: int,
    max_iter: int = 20,
    random_state: int = RANDOM_STATE,
    verbose: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    K-Median clustering algorithm using Manhattan distance.
    
    Args:
        X: Data points (n_samples, n_features)
        k: Number of clusters
        max_iter: Maximum number of iterations
        random_state: Random seed
        verbose: Print progress
    
    Returns:
        Tuple of (cluster labels, cluster centers)
    """
    rng = get_rng(random_state)
    n = X.shape[0]
    
    centers = X[rng.choice(n, k, replace=False)].copy()
    
    for iteration in range(max_iter):
        D = cdist(X, centers, metric='cityblock')
        labels = np.argmin(D, axis=1)
        
        new_centers = centers.copy()
        for j in range(k):
            members = X[labels == j]
            if len(members) > 0:
                new_centers[j] = np.median(members, axis=0)
        
        if np.allclose(new_centers, centers, rtol=1e-4):
            break
        centers = new_centers
    
    D = cdist(X, centers, metric='cityblock')
    labels = np.argmin(D, axis=1)
    
    return labels, centers


# ============================================================================
# Fuzzy C-Means Implementation
# ============================================================================

def fuzzy_c_means(
    X: np.ndarray,
    c: int,
    m: float = 2.0,
    max_iter: int = 150,
    tol: float = 1e-4,
    random_state: int = RANDOM_STATE,
    verbose: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fuzzy C-Means clustering algorithm.
    
    Args:
        X: Data points (n_samples, n_features)
        c: Number of clusters
        m: Fuzziness parameter (m > 1)
        max_iter: Maximum number of iterations
        tol: Convergence tolerance
        random_state: Random seed
        verbose: Print progress
    
    Returns:
        Tuple of (cluster labels, cluster centers, membership matrix)
    """
    rng = get_rng(random_state)
    n = X.shape[0]
    
    U = rng.dirichlet(np.ones(c), size=n)
    
    for iteration in range(max_iter):
        um = U ** m
        centers = (um.T @ X) / um.sum(axis=0)[:, None]
        
        dist = cdist(X, centers) + 1e-10
        inv_dist = dist ** (-2 / (m - 1))
        U_new = inv_dist / inv_dist.sum(axis=1, keepdims=True)
        
        if np.linalg.norm(U_new - U) < tol:
            U = U_new
            break
        U = U_new
    
    labels = np.argmax(U, axis=1)
    
    return labels, centers, U


# ============================================================================
# Kernel K-Means Implementation (Optimized)
# ============================================================================

def kernel_kmeans(
    X: np.ndarray,
    k: int,
    gamma: Optional[float] = None,
    max_iter: int = 50,
    random_state: int = RANDOM_STATE,
    verbose: bool = False,
    batch_size: Optional[int] = None
) -> np.ndarray:
    """
    Kernel K-Means with memory-efficient batch processing.
    
    For large datasets, processes kernel matrix in batches to avoid O(n²) memory.
    """
    if gamma is None:
        gamma = 1.0 / X.shape[1]
    
    rng = get_rng(random_state)
    n = X.shape[0]
    
    if n <= 15000:
        K = rbf_kernel(X, gamma=gamma)
        labels = rng.integers(0, k, size=n)
        
        for iteration in range(max_iter):
            dist = np.zeros((n, k))
            
            for j in range(k):
                members = np.where(labels == j)[0]
                if len(members) == 0:
                    dist[:, j] = np.inf
                    continue
                
                Kjj = K[np.ix_(members, members)].mean()
                Kij = K[:, members].mean(axis=1)
                dist[:, j] = np.diag(K) - 2 * Kij + Kjj
            
            new_labels = np.argmin(dist, axis=1)
            
            if np.array_equal(new_labels, labels):
                break
            labels = new_labels
        
        return labels
    
    if batch_size is None:
        batch_size = min(5000, n // 4)
    
    if verbose:
        print(f"  Kernel K-Means using batch processing (batch_size={batch_size})")
    
    labels = rng.integers(0, k, size=n)
    diag_K = np.ones(n)
    
    for iteration in range(max_iter):
        dist = np.zeros((n, k))
        
        for j in range(k):
            members = np.where(labels == j)[0]
            if len(members) == 0:
                dist[:, j] = np.inf
                continue
            
            if len(members) > 1:
                if len(members) > 1000:
                    sample_members = rng.choice(members, min(1000, len(members)), replace=False)
                    Kjj = rbf_kernel(X[sample_members], gamma=gamma).mean()
                else:
                    Kjj = rbf_kernel(X[members], gamma=gamma).mean()
            else:
                Kjj = 1.0
            
            Kij = np.zeros(n)
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                batch_X = X[start:end]
                if len(members) > 1000 and len(members) > len(members) // 4:
                    sample_members = rng.choice(members, min(1000, len(members)), replace=False)
                    K_batch = rbf_kernel(batch_X, X[sample_members], gamma=gamma).mean(axis=1)
                else:
                    K_batch = rbf_kernel(batch_X, X[members], gamma=gamma).mean(axis=1)
                Kij[start:end] = K_batch
            
            dist[:, j] = diag_K - 2 * Kij + Kjj
        
        new_labels = np.argmin(dist, axis=1)
        
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
    
    return labels


# ============================================================================
# Main Clustering Runner with Subsampling Support
# ============================================================================

class ClusteringRunner:
    """
    Unified interface for running all clustering algorithms with auto-subsampling.
    """
    
    def __init__(
        self,
        X: np.ndarray,
        k: Optional[int] = None,
        random_state: int = RANDOM_STATE,
        verbose: bool = VERBOSE,
        auto_subsample: bool = True
    ):
        self.X_full = ensure_float64(X)
        self.k = k
        self.random_state = random_state
        self.verbose = verbose
        self.auto_subsample = auto_subsample
        self.results = {}
        self.labels = {}
        self.runtimes = {}
        self.subsample_info = {}
        self.n_full = X.shape[0]
    
    def _prepare_data(
        self,
        algorithm_name: str,
        max_samples: Optional[int] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if not self.auto_subsample:
            return self.X_full, None
        
        return auto_subsample(
            self.X_full,
            algorithm_name,
            max_samples=max_samples,
            random_state=self.random_state
        )
    
    def _run_algorithm(
        self,
        name: str,
        func,
        *args,
        use_subsample: bool = True,
        max_samples: Optional[int] = None,
        **kwargs
    ) -> np.ndarray:
        if use_subsample and self.auto_subsample:
            X_use, indices_used = self._prepare_data(name, max_samples)
        else:
            X_use = self.X_full
            indices_used = None
        
        if indices_used is not None:
            self.subsample_info[name] = {
                'n_full': self.n_full,
                'n_subsample': len(X_use),
                'indices': indices_used
            }
        
        t0 = time.time()
        
        if args and isinstance(args[0], np.ndarray) and args[0] is self.X_full:
            args = (X_use,) + args[1:]
        
        result = func(*args, **kwargs)
        
        if isinstance(result, tuple):
            labels = result[0]
        else:
            labels = result
        
        if not isinstance(labels, np.ndarray):
            labels = np.array(labels, dtype=np.int64)
        
        elapsed = time.time() - t0
        
        if indices_used is not None:
            full_labels = -np.ones(self.n_full, dtype=np.int64)
            full_labels[indices_used] = labels
            labels = full_labels
        
        self.runtimes[name] = elapsed
        self.labels[name] = labels
        
        if self.verbose:
            clean_labels = labels[labels != -1] if len(labels) > 0 else labels
            n_clusters = len(set(clean_labels)) if len(clean_labels) > 0 else 0
            
            if len(labels) > 0:
                noise_pct = (labels == -1).mean() * 100
            else:
                noise_pct = 0.0
            
            subsample_msg = ""
            if indices_used is not None:
                subsample_msg = f" [subsampled {len(X_use):,}/{self.n_full:,}]"
            
            print(f"{name:30s} | clusters={n_clusters:3d} | noise={noise_pct:5.1f}% | {elapsed:.2f}s{subsample_msg}")
        
        return labels
    
    def run_kmeans(self, **kwargs) -> np.ndarray:
        if self.k is None:
            raise ValueError("k must be specified for K-Means")
        
        name = 'K-Means'
        model = KMeans(
            n_clusters=self.k,
            n_init=10,
            random_state=self.random_state,
            **kwargs
        )
        labels = self._run_algorithm(
            name, model.fit_predict, self.X_full,
            use_subsample=False
        )
        return labels
    
    def run_bisecting_kmeans(self, **kwargs) -> np.ndarray:
        if self.k is None:
            raise ValueError("k must be specified for Bisecting K-Means")
        
        name = 'Bisecting K-Means'
        model = BisectingKMeans(
            n_clusters=self.k,
            random_state=self.random_state,
            **kwargs
        )
        labels = self._run_algorithm(
            name, model.fit_predict, self.X_full,
            use_subsample=False
        )
        return labels
    
    def run_kmedoids(self, **kwargs) -> np.ndarray:
        if self.k is None:
            raise ValueError("k must be specified for K-Medoids")
        
        name = 'K-Medoids'
        labels = self._run_algorithm(
            name,
            k_medoids,
            self.X_full,
            self.k,
            random_state=self.random_state,
            use_subsample=True,
            **kwargs
        )
        return labels
    
    def run_kmedian(self, **kwargs) -> np.ndarray:
        if self.k is None:
            raise ValueError("k must be specified for K-Median")
        
        name = 'K-Median'
        labels = self._run_algorithm(
            name,
            k_median,
            self.X_full,
            self.k,
            random_state=self.random_state,
            use_subsample=False,
            **kwargs
        )
        return labels
    
    def run_fuzzy_cmeans(self, **kwargs) -> np.ndarray:
        if self.k is None:
            raise ValueError("k must be specified for Fuzzy C-Means")
        
        name = 'Fuzzy C-Means'
        labels = self._run_algorithm(
            name,
            fuzzy_c_means,
            self.X_full,
            self.k,
            random_state=self.random_state,
            use_subsample=False,
            **kwargs
        )
        return labels
    
    def run_kernel_kmeans(self, gamma: Optional[float] = None, **kwargs) -> np.ndarray:
        if self.k is None:
            raise ValueError("k must be specified for Kernel K-Means")
        
        name = 'Kernel K-Means'
        labels = self._run_algorithm(
            name,
            kernel_kmeans,
            self.X_full,
            self.k,
            gamma=gamma,
            random_state=self.random_state,
            use_subsample=True,
            **kwargs
        )
        return labels
    
    def run_dbscan(
        self,
        eps: Optional[float] = None,
        min_samples: int = DBSCAN_DEFAULT_MIN_SAMPLES,
        **kwargs
    ) -> np.ndarray:
        name = 'DBSCAN'
        
        if eps is None:
            eps, _ = get_k_distance_eps(self.X_full, k=min_samples, verbose=False)
        
        model = DBSCAN(eps=eps, min_samples=min_samples, **kwargs)
        labels = self._run_algorithm(
            name, model.fit_predict, self.X_full,
            use_subsample=False
        )
        return labels
    
    def run_optics(
        self,
        min_samples: int = OPTICS_DEFAULT_MIN_SAMPLES,
        **kwargs
    ) -> np.ndarray:
        name = 'OPTICS'
        model = OPTICS(min_samples=min_samples, **kwargs)
        labels = self._run_algorithm(
            name, model.fit_predict, self.X_full,
            use_subsample=False
        )
        return labels
    
    def run_hdbscan(
        self,
        min_cluster_size: int = HDBSCAN_DEFAULT_MIN_CLUSTER_SIZE,
        **kwargs
    ) -> np.ndarray:
        name = 'HDBSCAN'
        model = HDBSCAN(min_cluster_size=min_cluster_size, **kwargs)
        labels = self._run_algorithm(
            name, model.fit_predict, self.X_full,
            use_subsample=False
        )
        return labels
    
    def run_agglomerative(
        self,
        linkage: str = 'ward',
        **kwargs
    ) -> np.ndarray:
        if self.k is None:
            raise ValueError("k must be specified for Agglomerative Clustering")
        
        name = f'Agglomerative ({linkage.capitalize()})'
        model = AgglomerativeClustering(
            n_clusters=self.k,
            linkage=linkage,
            **kwargs
        )
        labels = self._run_algorithm(
            name, model.fit_predict, self.X_full,
            use_subsample=True,
            **kwargs
        )
        return labels
    
    def run_all_linkages(self) -> Dict[str, np.ndarray]:
        results = {}
        for linkage in ['ward', 'single', 'complete']:
            labels = self.run_agglomerative(linkage=linkage)
            results[linkage] = labels
        return results
    
    def run_gmm(self, **kwargs) -> np.ndarray:
        if self.k is None:
            raise ValueError("k must be specified for GMM")
        
        name = 'GMM'
        model = GaussianMixture(
            n_components=self.k,
            random_state=self.random_state,
            **kwargs
        )
        labels = self._run_algorithm(
            name, model.fit_predict, self.X_full,
            use_subsample=False
        )
        return labels
    
    def run_all(
        self,
        include_manual: bool = True,
        **kwargs
    ) -> Dict[str, np.ndarray]:
        if self.k is None:
            raise ValueError("k must be specified to run all algorithms")
        
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"Running all clustering algorithms on {self.n_full:,} points")
            print(f"Auto-subsampling: {'ON' if self.auto_subsample else 'OFF'}")
            print(f"{'='*70}\n")
        
        self.run_kmeans(**kwargs)
        self.run_bisecting_kmeans(**kwargs)
        
        if include_manual:
            self.run_kmedoids(**kwargs)
            self.run_kmedian(**kwargs)
            self.run_fuzzy_cmeans(**kwargs)
            self.run_kernel_kmeans(**kwargs)
        
        self.run_dbscan(**kwargs)
        self.run_optics(**kwargs)
        self.run_hdbscan(**kwargs)
        
        self.run_all_linkages()
        
        self.run_gmm(**kwargs)
        
        cleanup()
        
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"SUBSAMPLING SUMMARY:")
            for name, info in self.subsample_info.items():
                print(f"  {name}: {info['n_subsample']:,}/{info['n_full']:,} "
                      f"({info['n_subsample']/info['n_full']*100:.1f}%)")
            print(f"{'='*70}\n")
        
        return self.labels
    
    def get_results(self) -> Dict[str, Any]:
        return {
            'labels': self.labels,
            'runtimes': self.runtimes,
            'subsample_info': self.subsample_info,
            'n_algorithms': len(self.labels)
        }


# ============================================================================
# Convenience Functions
# ============================================================================

def run_all_clustering(
    X: np.ndarray,
    k: int,
    random_state: int = RANDOM_STATE,
    verbose: bool = VERBOSE,
    auto_subsample: bool = True
) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    runner = ClusteringRunner(
        X, k=k, random_state=random_state,
        verbose=verbose, auto_subsample=auto_subsample
    )
    labels = runner.run_all()
    return labels, runner.runtimes


def run_clustering_algorithm(
    X: np.ndarray,
    algorithm: str,
    k: Optional[int] = None,
    auto_subsample: bool = True,
    **kwargs
) -> Tuple[np.ndarray, float]:
    runner = ClusteringRunner(X, k=k, verbose=False, auto_subsample=auto_subsample)
    
    algorithm_map = {
        'kmeans': runner.run_kmeans,
        'bisecting_kmeans': runner.run_bisecting_kmeans,
        'kmedoids': runner.run_kmedoids,
        'kmedian': runner.run_kmedian,
        'fuzzy_cmeans': runner.run_fuzzy_cmeans,
        'kernel_kmeans': runner.run_kernel_kmeans,
        'dbscan': runner.run_dbscan,
        'optics': runner.run_optics,
        'hdbscan': runner.run_hdbscan,
        'gmm': runner.run_gmm,
        'agglomerative_ward': lambda: runner.run_agglomerative(linkage='ward'),
        'agglomerative_single': lambda: runner.run_agglomerative(linkage='single'),
        'agglomerative_complete': lambda: runner.run_agglomerative(linkage='complete'),
    }
    
    if algorithm not in algorithm_map:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    labels = algorithm_map[algorithm](**kwargs)
    last_name = next(reversed(runner.runtimes))
    return labels, runner.runtimes[last_name]


# ============================================================================
# Testing
# ============================================================================

if __name__ == "__main__":
    print("Testing clustering.py with auto-subsampling...")
    
    from sklearn.datasets import make_blobs
    
    X, y = make_blobs(n_samples=50000, centers=3, random_state=42)
    X = X.astype(np.float64)
    
    print(f"Test data shape: {X.shape} (50,000 samples)")
    
    runner = ClusteringRunner(X, k=3, verbose=True, auto_subsample=True)
    labels = runner.run_all()
    
    print(f"\nRan {len(labels)} algorithms successfully!")
    print(f"Subsampled algorithms: {len(runner.subsample_info)}")
    
    print("\nAll tests passed!")