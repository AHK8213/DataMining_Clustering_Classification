"""
utils.py - Helper functions for Project 3

Provides utility functions for:
- Memory optimization
- Sampling
- Garbage collection
- Clustering evaluation metrics (Dunn index, etc.)
- Hopkins statistic
"""

import gc
import warnings
from typing import Tuple, Optional, Union, List

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score

# Import configuration
from src.config import (
    RANDOM_STATE,
    ENABLE_GC,
    USE_FLOAT64,
    VERBOSE,
    SAMPLE_HOPKINS,
)

# Suppress warnings
warnings.filterwarnings("ignore")


# ============================================================================
# Random State Management
# ============================================================================

def get_rng(seed: Optional[int] = None) -> np.random.RandomState:
    """Get a numpy RandomState instance."""
    return np.random.RandomState(seed if seed is not None else RANDOM_STATE)


# ============================================================================
# Memory Optimization
# ============================================================================

def reduce_memory(df: pd.DataFrame, verbose: bool = VERBOSE) -> pd.DataFrame:
    """
    Reduce memory usage by downcasting numeric columns and converting
    object columns to category dtype.
    
    Args:
        df: Input DataFrame
        verbose: Print memory usage information
    
    Returns:
        Optimized DataFrame
    """
    df_out = df.copy()
    start_mem = df_out.memory_usage(deep=True).sum() / 1024**2
    
    for col in df_out.columns:
        col_dtype = df_out[col].dtype
        
        if col_dtype == 'object':
            df_out[col] = df_out[col].astype('category')
        elif np.issubdtype(col_dtype, np.integer):
            df_out[col] = pd.to_numeric(df_out[col], downcast='integer')
        elif np.issubdtype(col_dtype, np.floating):
            df_out[col] = pd.to_numeric(df_out[col], downcast='float')
    
    end_mem = df_out.memory_usage(deep=True).sum() / 1024**2
    
    if verbose:
        reduction = 100 * (start_mem - end_mem) / start_mem
        print(f"Memory usage: {start_mem:.2f} MB -> {end_mem:.2f} MB "
              f"({reduction:.1f}% reduction)")
    
    return df_out


# ============================================================================
# Sampling
# ============================================================================
# NOTE: The actual sampling logic now lives in src/sampling.py
# (UnifiedSampler). These two functions are kept as thin backward-compatible
# wrappers so existing `from src.utils import get_sample` imports keep
# working unchanged.

def get_sample(
    X: np.ndarray,
    n: int,
    random_state: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get a random sample from the dataset.
    
    Args:
        X: Input array
        n: Number of samples to draw
        random_state: Random seed
    
    Returns:
        Tuple of (sampled data, indices)
    """
    from src.sampling import default_sampler, SamplingStrategy
    return default_sampler.get_sample(X, n, SamplingStrategy.RANDOM, random_state)


def get_sample_df(
    df: pd.DataFrame,
    n: int,
    random_state: Optional[int] = None
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Get a random sample from a DataFrame.
    
    Args:
        df: Input DataFrame
        n: Number of samples to draw
        random_state: Random seed
    
    Returns:
        Tuple of (sampled DataFrame, indices)
    """
    from src.sampling import default_sampler, SamplingStrategy
    return default_sampler.get_sample(df, n, SamplingStrategy.RANDOM, random_state)


# ============================================================================
# Garbage Collection
# ============================================================================

def cleanup():
    """Run garbage collection if enabled."""
    if ENABLE_GC:
        gc.collect()


# ============================================================================
# Clustering Evaluation Metrics
# ============================================================================

def dunn_index(X: np.ndarray, labels: np.ndarray) -> float:
    """
    Calculate the Dunn Index for clustering.
    
    The Dunn Index is the ratio of the minimum inter-cluster distance
    to the maximum intra-cluster distance. Higher values indicate better
    clustering.
    
    Args:
        X: Data points
        labels: Cluster labels (-1 indicates noise for DBSCAN/HDBSCAN)
    
    Returns:
        Dunn Index (NaN if fewer than 2 clusters)
    """
    labels = np.asarray(labels)
    
    # Get unique clusters (exclude noise points with label -1)
    unique_labels = [c for c in set(labels) if c != -1]
    n_clusters = len(unique_labels)
    
    if n_clusters < 2:
        return np.nan
    
    # Get points for each cluster
    clusters = [X[labels == c] for c in unique_labels]
    
    # Calculate inter-cluster distances (minimum distance between clusters)
    inter_cluster_dists = []
    for i in range(n_clusters):
        for j in range(i + 1, n_clusters):
            # Minimum distance between any two points in different clusters
            min_dist = cdist(clusters[i], clusters[j]).min()
            inter_cluster_dists.append(min_dist)
    
    # Calculate intra-cluster distances (maximum distance within each cluster)
    intra_cluster_dists = []
    for c in clusters:
        if len(c) > 1:
            max_dist = cdist(c, c).max()
        else:
            max_dist = 0
        intra_cluster_dists.append(max_dist)
    
    # Dunn Index = min(inter) / max(intra)
    min_inter = np.min(inter_cluster_dists) if inter_cluster_dists else 0
    max_intra = np.max(intra_cluster_dists) if intra_cluster_dists else 1
    
    if max_intra == 0:
        return np.nan
    
    return min_inter / max_intra


def compute_clustering_metrics(
    X: np.ndarray,
    labels: np.ndarray,
    metrics: Optional[List[str]] = None
) -> dict:
    """
    Compute multiple clustering metrics.
    
    Args:
        X: Data points
        labels: Cluster labels (-1 indicates noise)
        metrics: List of metrics to compute ('silhouette', 'davies_bouldin',
                 'calinski_harabasz', 'dunn'). If None, compute all.
    
    Returns:
        Dictionary of metric names to values
    """
    from sklearn.metrics import (
        silhouette_score,
        davies_bouldin_score,
        calinski_harabasz_score
    )
    
    labels = np.asarray(labels)
    mask = labels != -1
    n_clusters = len(set(labels[mask])) if mask.any() else 0
    
    if metrics is None:
        metrics = ['silhouette', 'davies_bouldin', 'calinski_harabasz', 'dunn']
    
    results = {
        'n_clusters': n_clusters,
        'noise_pct': (labels == -1).mean() * 100 if len(labels) > 0 else 0
    }
    
    # Only compute metrics if we have at least 2 clusters
    if n_clusters >= 2 and mask.sum() > 0:
        X_clean = X[mask]
        labels_clean = labels[mask]
        
        if 'silhouette' in metrics:
            try:
                results['silhouette'] = silhouette_score(X_clean, labels_clean)
            except Exception:
                results['silhouette'] = np.nan
        
        if 'davies_bouldin' in metrics:
            try:
                results['davies_bouldin'] = davies_bouldin_score(X_clean, labels_clean)
            except Exception:
                results['davies_bouldin'] = np.nan
        
        if 'calinski_harabasz' in metrics:
            try:
                results['calinski_harabasz'] = calinski_harabasz_score(X_clean, labels_clean)
            except Exception:
                results['calinski_harabasz'] = np.nan
        
        if 'dunn' in metrics:
            results['dunn'] = dunn_index(X_clean, labels_clean)
    else:
        # Set NaN for metrics that couldn't be computed
        for metric in metrics:
            if metric in ['silhouette', 'davies_bouldin', 'calinski_harabasz', 'dunn']:
                results[metric] = np.nan
    
    return results


# ============================================================================
# Hopkins Statistic
# ============================================================================

def hopkins_statistic(
    X: np.ndarray,
    n_samples: int = SAMPLE_HOPKINS,
    random_state: Optional[int] = None
) -> float:
    """
    Calculate the Hopkins statistic to assess clustering tendency.
    
    The Hopkins statistic measures the clustering tendency of a dataset.
    Values near 1 indicate highly clusterable data, values near 0.5
    indicate random data.
    
    Args:
        X: Input data (n_samples, n_features)
        n_samples: Number of points to sample
        random_state: Random seed
    
    Returns:
        Hopkins statistic (0-1)
    """
    rng = get_rng(random_state)
    n, d = X.shape
    n_samples = min(n_samples, n - 1)
    
    # Sample points from the data
    idx = rng.choice(n, n_samples, replace=False)
    X_sample = X[idx]
    
    # Generate random points uniformly in the data space
    mins, maxs = X.min(axis=0), X.max(axis=0)
    X_random = rng.uniform(mins, maxs, size=(n_samples, d))
    
    # Distance from sampled points to nearest neighbor in data
    nbrs_real = NearestNeighbors(n_neighbors=2).fit(X)
    d_data, _ = nbrs_real.kneighbors(X_sample)
    d_data = d_data[:, 1]  # Exclude self-distance
    
    # Distance from random points to nearest neighbor in data
    nbrs_all = NearestNeighbors(n_neighbors=1).fit(X)
    d_rand, _ = nbrs_all.kneighbors(X_random)
    d_rand = d_rand[:, 0]
    
    # Hopkins statistic
    return d_rand.sum() / (d_rand.sum() + d_data.sum())


def interpret_hopkins(H: float) -> str:
    """
    Interpret the Hopkins statistic value.
    
    Args:
        H: Hopkins statistic value
    
    Returns:
        Interpretation string
    """
    if H > 0.7:
        return "High clustering tendency (strongly clusterable)"
    elif H > 0.6:
        return "Moderate clustering tendency (clusterable)"
    elif H > 0.5:
        return "Weak clustering tendency"
    elif H > 0.4:
        return "Random or uniform data (no clustering)"
    else:
        return "Regularly spaced data (anti-clustering)"


# ============================================================================
# Validation Helpers
# ============================================================================

def ensure_float64(X: np.ndarray) -> np.ndarray:
    """Ensure array is float64."""
    if USE_FLOAT64 and X.dtype != np.float64:
        return X.astype(np.float64)
    return X


def is_float64_compatible() -> bool:
    """Return whether float64 is enabled."""
    return USE_FLOAT64


# ============================================================================
# Timing Helpers
# ============================================================================

import time
from contextlib import contextmanager


@contextmanager
def timer(name: str = "Operation", verbose: bool = VERBOSE):
    """Context manager for timing code blocks."""
    t0 = time.time()
    yield
    elapsed = time.time() - t0
    if verbose:
        print(f"{name} completed in {elapsed:.2f}s")


# ============================================================================
# Debug Helpers
# ============================================================================

def print_memory_usage(label: str = ""):
    """Print current memory usage."""
    import psutil
    process = psutil.Process()
    mem_mb = process.memory_info().rss / 1024**2
    print(f"[{label}] Memory usage: {mem_mb:.1f} MB")


# ============================================================================
# Testing
# ============================================================================

if __name__ == "__main__":
    # Quick test of utility functions
    print("Testing utils.py...")
    
    # Test get_rng
    rng = get_rng()
    print(f"RandomState initialized: {rng is not None}")
    
    # Test Hopkins (with random data)
    X_test = np.random.randn(100, 5)
    H = hopkins_statistic(X_test, n_samples=30)
    print(f"Hopkins statistic on random data: {H:.3f} -> {interpret_hopkins(H)}")
    
    # Test metrics on simple clustering
    from sklearn.datasets import make_blobs
    X_blobs, y_blobs = make_blobs(n_samples=200, centers=3, random_state=42)
    metrics = compute_clustering_metrics(X_blobs, y_blobs)
    print(f"Metrics on blobs: {metrics}")
    
    print("All tests passed!")