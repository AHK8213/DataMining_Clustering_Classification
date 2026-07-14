"""
sampling.py - Unified sampling manager for Project 3

Previously, sampling logic was scattered across several places:
    - utils.py: get_sample(), get_sample_df()
    - clustering_algorithms.py: auto_subsample(), SUBSAMPLE_LIMITS
    - notebook: local create_tiered_datasets()
    - clustering_evaluation.py: sample_size parameter in OptimalKDeterminer

This module is now the single source of truth for every sampling
decision in the project. utils.py and clustering_algorithms.py keep
their original function names as thin backward-compatible wrappers
around UnifiedSampler so existing imports keep working unchanged.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd

from src.config import RANDOM_STATE, VERBOSE

ArrayLike = Union[np.ndarray, pd.DataFrame]


# ============================================================================
# Strategy & Configuration
# ============================================================================

class SamplingStrategy(Enum):
    """How a sample should be drawn."""
    FULL = "full"          # no sampling, return everything
    RANDOM = "random"      # simple random sample without replacement
    TIERED = "tiered"      # named fractional subsets (scalability experiments)


# Per-algorithm sample-size ceilings for clustering algorithms.
# None means "no limit, always use the full dataset". This dict was
# previously duplicated as SUBSAMPLE_LIMITS in clustering_algorithms.py.
ALGORITHM_LIMITS: Dict[str, Optional[int]] = {
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

DEFAULT_ALGORITHM_LIMIT = 20000


@dataclass
class SubsamplingConfig:
    """Configuration bundle for a UnifiedSampler instance."""
    limits: Dict[str, Optional[int]] = field(default_factory=lambda: dict(ALGORITHM_LIMITS))
    default_limit: Optional[int] = DEFAULT_ALGORITHM_LIMIT
    random_state: int = RANDOM_STATE


# ============================================================================
# Unified Sampler
# ============================================================================

class UnifiedSampler:
    """
    Single entry point for all sampling decisions: plain random samples,
    algorithm-aware auto-subsampling for expensive clustering algorithms,
    and tiered dataset construction for scalability experiments.
    """

    def __init__(self, config: Optional[SubsamplingConfig] = None):
        self.config = config or SubsamplingConfig()

    # -- plain random sampling (replaces utils.get_sample / get_sample_df) --
    def get_sample(
        self,
        data: ArrayLike,
        n_samples: int,
        strategy: SamplingStrategy = SamplingStrategy.RANDOM,
        random_state: Optional[int] = None,
    ) -> Tuple[ArrayLike, np.ndarray]:
        """
        Draw a random sample of `n_samples` rows from `data`.

        Returns (sampled_data, indices) where indices always covers the
        rows actually returned (np.arange(n) if no sampling was needed).
        """
        n_total = data.shape[0]
        seed = random_state if random_state is not None else self.config.random_state

        if strategy == SamplingStrategy.FULL or n_samples >= n_total:
            idx = np.arange(n_total)
            return data, idx

        rng = np.random.RandomState(seed)
        idx = rng.choice(n_total, n_samples, replace=False)

        if isinstance(data, pd.DataFrame):
            return data.iloc[idx].reset_index(drop=True), idx
        return data[idx], idx

    # -- algorithm-aware auto-subsampling (replaces clustering_algorithms.auto_subsample) --
    def should_subsample(self, algorithm_name: str, n_samples: int) -> bool:
        """Whether `algorithm_name` would need subsampling for `n_samples` rows."""
        limit = self.config.limits.get(algorithm_name, self.config.default_limit)
        return limit is not None and n_samples > limit

    def get_subsample_size(self, algorithm_name: str, n_samples: int) -> int:
        """The sample size that would be used for `algorithm_name`."""
        limit = self.config.limits.get(algorithm_name, self.config.default_limit)
        if limit is None:
            return n_samples
        return min(limit, n_samples)

    def auto_subsample(
        self,
        X: np.ndarray,
        algorithm_name: str,
        max_samples: Optional[int] = None,
        random_state: Optional[int] = None,
        verbose: bool = VERBOSE,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Subsample X for a memory-intensive algorithm if needed.

        Returns (X_used, indices) where indices is None when no
        subsampling occurred (matches the original auto_subsample
        contract relied on by ClusteringRunner for bookkeeping).
        """
        n = X.shape[0]
        limit = max_samples if max_samples is not None else self.config.limits.get(
            algorithm_name, self.config.default_limit
        )

        if limit is None or n <= limit:
            return X, None

        seed = random_state if random_state is not None else self.config.random_state
        rng = np.random.RandomState(seed)
        indices = rng.choice(n, limit, replace=False)

        if verbose:
            print(f"  \u26a1 {algorithm_name}: subsampled from {n:,} to {limit:,} points")

        return X[indices], indices

    # -- tiered datasets for scalability experiments (replaces notebook helper) --
    def create_tiered_datasets(
        self,
        X_full: np.ndarray,
        tiers: Dict[str, float],
        random_state: Optional[int] = None,
        verbose: bool = True,
    ) -> Dict[str, np.ndarray]:
        """
        Build named subsets of X_full at different fractions of the data,
        e.g. {'small': 0.05, 'medium': 0.20, 'large': 0.50, 'full': 1.00}.
        """
        seed = random_state if random_state is not None else self.config.random_state
        n_total = X_full.shape[0]
        rng = np.random.RandomState(seed)

        datasets = {}
        for name, fraction in tiers.items():
            if fraction >= 1.0:
                datasets[name] = X_full
                n_samples = n_total
            else:
                n_samples = int(n_total * fraction)
                indices = rng.choice(n_total, n_samples, replace=False)
                datasets[name] = X_full[indices]

            if verbose:
                print(f"{name}: {n_samples:,} samples ({fraction * 100:.0f}%)")

        return datasets


# ============================================================================
# Module-level default instance & convenience wrappers
# ============================================================================
# Most callers don't need a custom config, so a shared default instance is
# exposed alongside simple function wrappers for drop-in use.

default_sampler = UnifiedSampler()


def get_sample(X: ArrayLike, n: int, random_state: Optional[int] = None):
    """Convenience wrapper around default_sampler.get_sample for arrays/DataFrames."""
    return default_sampler.get_sample(X, n, SamplingStrategy.RANDOM, random_state)


def auto_subsample(
    X: np.ndarray,
    algorithm_name: str,
    max_samples: Optional[int] = None,
    random_state: Optional[int] = None,
    verbose: bool = VERBOSE,
):
    """Convenience wrapper around default_sampler.auto_subsample."""
    return default_sampler.auto_subsample(X, algorithm_name, max_samples, random_state, verbose)


def create_tiered_datasets(
    X_full: np.ndarray,
    tiers: Dict[str, float],
    random_state: Optional[int] = None,
    verbose: bool = True,
):
    """Convenience wrapper around default_sampler.create_tiered_datasets."""
    return default_sampler.create_tiered_datasets(X_full, tiers, random_state, verbose)


if __name__ == "__main__":
    # Quick smoke test
    X_test = np.random.RandomState(0).randn(1000, 4)

    sampler = UnifiedSampler()
    X_sub, idx = sampler.get_sample(X_test, 100)
    print(f"get_sample: {X_sub.shape}, {len(idx)} indices")

    X_sub2, idx2 = sampler.auto_subsample(X_test, "kernel_kmeans", max_samples=50)
    print(f"auto_subsample: {X_sub2.shape}, indices={'None' if idx2 is None else len(idx2)}")

    tiers = sampler.create_tiered_datasets(X_test, {"small": 0.1, "full": 1.0}, verbose=False)
    print({k: v.shape for k, v in tiers.items()})
