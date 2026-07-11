"""
clustering.py - Clustering algorithm implementations for Project 3

Provides a unified interface for running all clustering algorithms:
- Centroid-based: K-Means, Bisecting K-Means, K-Medoids, K-Median, Kernel K-Means, Fuzzy C-Means
- Density-based: DBSCAN, OPTICS, HDBSCAN
- Hierarchical: Agglomerative (Ward, Single, Complete)
- Probabilistic: Gaussian Mixture Models

All algorithms return labels and runtime information.
"""

import time
import warnings
from typing import Tuple, Optional, Dict, Any, List, Union

import numpy as np
from scipy.spatial.distance import cdist
from scipy.cluster.hierarchy import dendrogram, linkage
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

# Import configuration and utilities
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
# Utility Functions
# ============================================================================

def get_k_distance_eps(
    X: np.ndarray,
    k: int = 5,
    percentile: float = DBSCAN_DEFAULT_EPS_PERCENTILE,
    verbose: bool = VERBOSE
) -> float:
    """
    Determine eps parameter for DBSCAN using k-distance plot.
    
    Args:
        X: Data points
        k: Number of neighbors
        percentile: Percentile of distances to use as eps
        verbose: Print progress
    
    Returns:
        Suggested eps value
    """
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


# ============================================================================
# K-Medoids Implementation
# ============================================================================

def k_medoids(
    X: np.ndarray,
    k: int,
    max_iter: int = 20,
    random_state: int = RANDOM_STATE,
    verbose: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    K-Medoids clustering algorithm using Manhattan distance.
    
    Args:
        X: Data points (n_samples, n_features)
        k: Number of clusters
        max_iter: Maximum number of iterations
        random_state: Random seed
        verbose: Print progress
    
    Returns:
        Tuple of (cluster labels, medoid indices)
    """
    rng = get_rng(random_state)
    n = X.shape[0]
    
    # Initialize medoids randomly
    medoid_idx = rng.choice(n, k, replace=False)
    
    for iteration in range(max_iter):
        # Assign points to nearest medoid
        D = cdist(X, X[medoid_idx], metric='cityblock')
        labels = np.argmin(D, axis=1)
        
        # Update medoids
        new_medoid_idx = medoid_idx.copy()
        for j in range(k):
            members = np.where(labels == j)[0]
            if len(members) == 0:
                continue
            
            # Find point with minimum total distance to other members
            sub = X[members]
            intra = cdist(sub, sub, metric='cityblock').sum(axis=1)
            new_medoid_idx[j] = members[np.argmin(intra)]
        
        # Check convergence
        if np.array_equal(new_medoid_idx, medoid_idx):
            break
        medoid_idx = new_medoid_idx
    
    # Final assignment
    D = cdist(X, X[medoid_idx], metric='cityblock')
    labels = np.argmin(D, axis=1)
    
    return labels, medoid_idx


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
    
    # Initialize centers randomly
    centers = X[rng.choice(n, k, replace=False)].copy()
    
    for iteration in range(max_iter):
        # Assign points to nearest center
        D = cdist(X, centers, metric='cityblock')
        labels = np.argmin(D, axis=1)
        
        # Update centers (median of each cluster)
        new_centers = centers.copy()
        for j in range(k):
            members = X[labels == j]
            if len(members) > 0:
                new_centers[j] = np.median(members, axis=0)
        
        # Check convergence
        if np.allclose(new_centers, centers, rtol=1e-4):
            break
        centers = new_centers
    
    # Final assignment
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
    
    # Initialize membership matrix randomly
    U = rng.dirichlet(np.ones(c), size=n)
    
    for iteration in range(max_iter):
        # Update centers
        um = U ** m
        centers = (um.T @ X) / um.sum(axis=0)[:, None]
        
        # Update membership matrix
        dist = cdist(X, centers) + 1e-10
        inv_dist = dist ** (-2 / (m - 1))
        U_new = inv_dist / inv_dist.sum(axis=1, keepdims=True)
        
        # Check convergence
        if np.linalg.norm(U_new - U) < tol:
            U = U_new
            break
        U = U_new
    
    # Get hard labels
    labels = np.argmax(U, axis=1)
    
    return labels, centers, U


# ============================================================================
# Kernel K-Means Implementation
# ============================================================================

def kernel_kmeans(
    X: np.ndarray,
    k: int,
    gamma: Optional[float] = None,
    max_iter: int = 50,
    random_state: int = RANDOM_STATE,
    verbose: bool = False
) -> np.ndarray:
    """
    Kernel K-Means clustering using RBF kernel.
    
    Args:
        X: Data points (n_samples, n_features)
        k: Number of clusters
        gamma: RBF kernel parameter (default: 1/n_features)
        max_iter: Maximum number of iterations
        random_state: Random seed
        verbose: Print progress
    
    Returns:
        Cluster labels
    """
    if gamma is None:
        gamma = 1.0 / X.shape[1]
    
    rng = get_rng(random_state)
    n = X.shape[0]
    
    # Compute kernel matrix
    K = rbf_kernel(X, gamma=gamma)
    
    # Initialize labels randomly
    labels = rng.randint(0, k, size=n)
    
    for iteration in range(max_iter):
        # Compute distance to each cluster in kernel space
        dist = np.zeros((n, k))
        
        for j in range(k):
            members = np.where(labels == j)[0]
            if len(members) == 0:
                dist[:, j] = np.inf
                continue
            
            Kjj = K[np.ix_(members, members)].mean()
            Kij = K[:, members].mean(axis=1)
            dist[:, j] = np.diag(K) - 2 * Kij + Kjj
        
        # Update labels
        new_labels = np.argmin(dist, axis=1)
        
        # Check convergence
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
    
    return labels


# ============================================================================
# Main Clustering Runner
# ============================================================================

class ClusteringRunner:
    """
    Unified interface for running all clustering algorithms.
    
    Usage:
        runner = ClusteringRunner(X, verbose=True)
        results = runner.run_all()
    """
    
    def __init__(
        self,
        X: np.ndarray,
        k: Optional[int] = None,
        random_state: int = RANDOM_STATE,
        verbose: bool = VERBOSE
    ):
        """
        Initialize clustering runner.
        
        Args:
            X: Data points
            k: Number of clusters (if fixed K is needed)
            random_state: Random seed
            verbose: Print progress
        """
        self.X = ensure_float64(X)
        self.k = k
        self.random_state = random_state
        self.verbose = verbose
        self.results = {}
        self.labels = {}
        self.runtimes = {}
    
    def _run_algorithm(
        self,
        name: str,
        func,
        *args,
        **kwargs
    ) -> np.ndarray:
        """Run an algorithm and time it."""
        t0 = time.time()
        labels = func(*args, **kwargs)
        elapsed = time.time() - t0
        
        self.runtimes[name] = elapsed
        self.labels[name] = labels
        
        if self.verbose:
            n_clusters = len(set(labels[labels != -1]))
            noise_pct = (labels == -1).mean() * 100 if len(labels) > 0 else 0
            print(f"{name:30s} | clusters={n_clusters:3d} | noise={noise_pct:5.1f}% | {elapsed:.2f}s")
        
        return labels
    
    # ========================================================================
    # Centroid-Based Algorithms
    # ========================================================================
    
    def run_kmeans(self, **kwargs) -> np.ndarray:
        """Run standard K-Means."""
        if self.k is None:
            raise ValueError("k must be specified for K-Means")
        
        name = 'K-Means'
        model = KMeans(
            n_clusters=self.k,
            n_init=10,
            random_state=self.random_state,
            **kwargs
        )
        labels = self._run_algorithm(name, model.fit_predict, self.X)
        return labels
    
    def run_bisecting_kmeans(self, **kwargs) -> np.ndarray:
        """Run Bisecting K-Means."""
        if self.k is None:
            raise ValueError("k must be specified for Bisecting K-Means")
        
        name = 'Bisecting K-Means'
        model = BisectingKMeans(
            n_clusters=self.k,
            random_state=self.random_state,
            **kwargs
        )
        labels = self._run_algorithm(name, model.fit_predict, self.X)
        return labels
    
    def run_kmedoids(self, **kwargs) -> np.ndarray:
        """Run K-Medoids."""
        if self.k is None:
            raise ValueError("k must be specified for K-Medoids")
        
        name = 'K-Medoids'
        labels, _ = self._run_algorithm(
            name,
            k_medoids,
            self.X,
            self.k,
            random_state=self.random_state,
            **kwargs
        )
        return labels
    
    def run_kmedian(self, **kwargs) -> np.ndarray:
        """Run K-Median."""
        if self.k is None:
            raise ValueError("k must be specified for K-Median")
        
        name = 'K-Median'
        labels, _ = self._run_algorithm(
            name,
            k_median,
            self.X,
            self.k,
            random_state=self.random_state,
            **kwargs
        )
        return labels
    
    def run_fuzzy_cmeans(self, **kwargs) -> np.ndarray:
        """Run Fuzzy C-Means."""
        if self.k is None:
            raise ValueError("k must be specified for Fuzzy C-Means")
        
        name = 'Fuzzy C-Means'
        labels, _, _ = self._run_algorithm(
            name,
            fuzzy_c_means,
            self.X,
            self.k,
            random_state=self.random_state,
            **kwargs
        )
        return labels
    
    def run_kernel_kmeans(self, gamma: Optional[float] = None, **kwargs) -> np.ndarray:
        """Run Kernel K-Means."""
        if self.k is None:
            raise ValueError("k must be specified for Kernel K-Means")
        
        name = 'Kernel K-Means'
        labels = self._run_algorithm(
            name,
            kernel_kmeans,
            self.X,
            self.k,
            gamma=gamma,
            random_state=self.random_state,
            **kwargs
        )
        return labels
    
    # ========================================================================
    # Density-Based Algorithms
    # ========================================================================
    
    def run_dbscan(
        self,
        eps: Optional[float] = None,
        min_samples: int = DBSCAN_DEFAULT_MIN_SAMPLES,
        **kwargs
    ) -> np.ndarray:
        """Run DBSCAN."""
        name = 'DBSCAN'
        
        if eps is None:
            eps, _ = get_k_distance_eps(self.X, k=min_samples, verbose=False)
        
        model = DBSCAN(eps=eps, min_samples=min_samples, **kwargs)
        labels = self._run_algorithm(name, model.fit_predict, self.X)
        return labels
    
    def run_optics(
        self,
        min_samples: int = OPTICS_DEFAULT_MIN_SAMPLES,
        **kwargs
    ) -> np.ndarray:
        """Run OPTICS."""
        name = 'OPTICS'
        model = OPTICS(min_samples=min_samples, **kwargs)
        labels = self._run_algorithm(name, model.fit_predict, self.X)
        return labels
    
    def run_hdbscan(
        self,
        min_cluster_size: int = HDBSCAN_DEFAULT_MIN_CLUSTER_SIZE,
        **kwargs
    ) -> np.ndarray:
        """Run HDBSCAN."""
        name = 'HDBSCAN'
        model = HDBSCAN(min_cluster_size=min_cluster_size, **kwargs)
        labels = self._run_algorithm(name, model.fit_predict, self.X)
        return labels
    
    # ========================================================================
    # Hierarchical Algorithms
    # ========================================================================
    
    def run_agglomerative(
        self,
        linkage: str = 'ward',
        **kwargs
    ) -> np.ndarray:
        """Run Agglomerative Clustering."""
        if self.k is None:
            raise ValueError("k must be specified for Agglomerative Clustering")
        
        name = f'Agglomerative ({linkage.capitalize()})'
        model = AgglomerativeClustering(
            n_clusters=self.k,
            linkage=linkage,
            **kwargs
        )
        labels = self._run_algorithm(name, model.fit_predict, self.X)
        return labels
    
    def run_all_linkages(self) -> Dict[str, np.ndarray]:
        """Run Agglomerative with all three linkages."""
        results = {}
        for linkage in ['ward', 'single', 'complete']:
            labels = self.run_agglomerative(linkage=linkage)
            results[linkage] = labels
        return results
    
    # ========================================================================
    # Probabilistic Algorithms
    # ========================================================================
    
    def run_gmm(self, **kwargs) -> np.ndarray:
        """Run Gaussian Mixture Model."""
        if self.k is None:
            raise ValueError("k must be specified for GMM")
        
        name = 'GMM'
        model = GaussianMixture(
            n_components=self.k,
            random_state=self.random_state,
            **kwargs
        )
        labels = self._run_algorithm(name, model.fit_predict, self.X)
        return labels
    
    # ========================================================================
    # Run All Algorithms
    # ========================================================================
    
    def run_all(
        self,
        include_manual: bool = True,
        **kwargs
    ) -> Dict[str, np.ndarray]:
        """
        Run all clustering algorithms.
        
        Args:
            include_manual: Include manual implementations (K-Medoids, etc.)
            **kwargs: Additional arguments passed to algorithms
        
        Returns:
            Dictionary mapping algorithm names to labels
        """
        if self.k is None:
            raise ValueError("k must be specified to run all algorithms")
        
        # Centroid-based
        self.run_kmeans(**kwargs)
        self.run_bisecting_kmeans(**kwargs)
        
        if include_manual:
            self.run_kmedoids(**kwargs)
            self.run_kmedian(**kwargs)
            self.run_fuzzy_cmeans(**kwargs)
            self.run_kernel_kmeans(**kwargs)
        
        # Density-based
        self.run_dbscan(**kwargs)
        self.run_optics(**kwargs)
        self.run_hdbscan(**kwargs)
        
        # Hierarchical
        self.run_all_linkages()
        
        # Probabilistic
        self.run_gmm(**kwargs)
        
        cleanup()
        
        return self.labels
    
    def get_results(self) -> Dict[str, Any]:
        """Get all results."""
        return {
            'labels': self.labels,
            'runtimes': self.runtimes,
            'n_algorithms': len(self.labels)
        }


# ============================================================================
# Convenience Functions
# ============================================================================

def run_all_clustering(
    X: np.ndarray,
    k: int,
    random_state: int = RANDOM_STATE,
    verbose: bool = VERBOSE
) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    """
    Convenience function to run all clustering algorithms.
    
    Args:
        X: Data points
        k: Number of clusters
        random_state: Random seed
        verbose: Print progress
    
    Returns:
        Tuple of (labels_dict, runtimes_dict)
    """
    runner = ClusteringRunner(X, k=k, random_state=random_state, verbose=verbose)
    labels = runner.run_all()
    return labels, runner.runtimes


def run_clustering_algorithm(
    X: np.ndarray,
    algorithm: str,
    k: Optional[int] = None,
    **kwargs
) -> Tuple[np.ndarray, float]:
    """
    Run a single clustering algorithm by name.
    
    Args:
        X: Data points
        algorithm: Algorithm name
        k: Number of clusters (if applicable)
        **kwargs: Algorithm-specific parameters
    
    Returns:
        Tuple of (labels, runtime)
    """
    runner = ClusteringRunner(X, k=k, verbose=False)
    
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
    return labels, runner.runtimes[algorithm]


# ============================================================================
# Testing
# ============================================================================

if __name__ == "__main__":
    print("Testing clustering.py...")
    
    from sklearn.datasets import make_blobs
    
    # Generate test data
    X, y = make_blobs(n_samples=500, centers=3, random_state=42)
    X = X.astype(np.float64)
    
    print(f"Test data shape: {X.shape}")
    
    # Test K-Means
    runner = ClusteringRunner(X, k=3, verbose=True)
    labels = runner.run_kmeans()
    print(f"K-Means labels: {len(set(labels))} clusters")
    
    # Test DBSCAN with auto eps
    labels_db = runner.run_dbscan()
    print(f"DBSCAN labels: {len(set(labels_db))} clusters, "
          f"noise: {(labels_db == -1).mean() * 100:.1f}%")
    
    # Run all algorithms
    print("\nRunning all algorithms...")
    labels_all, runtimes = run_all_clustering(X, k=3)
    print(f"Ran {len(labels_all)} algorithms")
    
    print("\nAll tests passed!")