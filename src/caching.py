"""
caching.py - Simple disk caching for expensive intermediate results

Without this, every notebook run re-downloads data, re-fits PCA,
re-computes optimal K, and re-runs every clustering algorithm from
scratch. cache_result() pickles the result of a function the first
time it runs and reuses it on subsequent runs, so re-executing the
notebook after the first pass takes seconds instead of minutes.

Usage:
    from src.caching import cache_result

    pca, X_pca_full = cache_result(
        "pca_fit_transform",
        lambda: (fit_pca(X_full), pca.transform(X_full))
    )

Set src.config.DISABLE_CACHE = True (or pass force=True) to bypass
the cache, e.g. after changing feature engineering or the input data.
"""

import pickle
from pathlib import Path
from typing import Any, Callable, Optional

from src.config import CACHE_DIR, DISABLE_CACHE, VERBOSE


def _cache_path(key: str) -> Path:
    safe_key = key.replace("/", "_").replace(" ", "_")
    return CACHE_DIR / f"{safe_key}.pkl"


def cache_result(
    key: str,
    func: Callable[[], Any],
    force: bool = False,
    verbose: bool = VERBOSE,
) -> Any:
    """
    Return the cached result for `key` if present, otherwise call
    `func()`, cache its return value to disk, and return it.

    Args:
        key: Unique cache key (used as the filename under data/cache/)
        func: Zero-argument callable that computes the result
        force: If True, ignore any existing cache and recompute
        verbose: Print cache hit/miss info

    Returns:
        The cached or freshly computed result.
    """
    path = _cache_path(key)

    if not force and not DISABLE_CACHE and path.exists():
        try:
            with open(path, "rb") as f:
                result = pickle.load(f)
            if verbose:
                print(f"  \U0001F4E6 Loaded '{key}' from cache ({path.name})")
            return result
        except Exception as e:
            if verbose:
                print(f"  \u26a0\ufe0f Cache read failed for '{key}' ({e}); recomputing...")

    result = func()

    try:
        with open(path, "wb") as f:
            pickle.dump(result, f)
        if verbose:
            print(f"  \U0001F4BE Cached '{key}' -> {path.name}")
    except Exception as e:
        if verbose:
            print(f"  \u26a0\ufe0f Could not write cache for '{key}' ({e})")

    return result


def clear_cache(key: Optional[str] = None) -> None:
    """
    Delete a specific cache entry, or every cached entry if key is None.
    """
    if key is not None:
        path = _cache_path(key)
        if path.exists():
            path.unlink()
            print(f"Cleared cache entry: {key}")
        return

    n = 0
    for f in CACHE_DIR.glob("*.pkl"):
        f.unlink()
        n += 1
    print(f"Cleared {n} cache entr{'y' if n == 1 else 'ies'} from {CACHE_DIR}")


if __name__ == "__main__":
    calls = {"n": 0}

    def _expensive():
        calls["n"] += 1
        return sum(range(1000))

    r1 = cache_result("smoke_test", _expensive)
    r2 = cache_result("smoke_test", _expensive)
    assert r1 == r2 == sum(range(1000))
    assert calls["n"] == 1, "second call should have hit the cache"
    clear_cache("smoke_test")
    print("caching.py smoke test passed.")
