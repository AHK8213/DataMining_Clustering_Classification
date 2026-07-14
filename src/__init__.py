"""
src - Project 3: Clustering & Classification

Complete implementation for the Bank Marketing dataset analysis,
including clustering experiments, classification models, and
comprehensive evaluation.

Modules:
    - config: Centralized configuration and constants
    - utils: Helper functions for memory, metrics
    - sampling: Unified sampling manager (random sampling, algorithm-aware
      subsampling, tiered datasets) - single source of truth for sampling
    - caching: Disk caching for expensive intermediate results (PCA,
      optimal K, clustering labels)
    - data_preparation: Data loading, cleaning, feature engineering, PCA
    - clustering_algorithms: All clustering algorithm implementations
    - clustering_evaluation: Metrics, comparison, holistic selection
    - clustering_tendency: Hopkins statistic and clustering tendency assessment
    - optimal_k: Optimal K determination
    - cluster_profiling: Cluster characterization and profiling
    - classification_prep: Data preparation for classification
    - base_classifiers: Base classification models with GridSearch
    - neural_network: PyTorch neural network implementation
    - ensemble_methods: Manual bagging and boosting
    - classification_eval: Metrics, cross-validation, ROC/PR curves
    - classification_analysis: Overfitting, learning curves, pre-call analysis
    - visualization: Plotting utilities
    - final_summary: Final tables, extrinsic evaluation, reports

Usage:
    from src import config
    from src.utils import reduce_memory, hopkins_statistic
    from src.clustering_algorithms import ClusteringRunner
    from src.clustering_evaluation import ClusteringComparator
    from src.final_summary import create_complete_summary
"""

# ============================================================================
# Version Information
# ============================================================================

__version__ = "1.0.0"
__author__ = "Data Mining Team"
__date__ = "2026-07-11"


# ============================================================================
# Import Key Classes and Functions for Easy Access
# ============================================================================

# Configuration
from src.config import (
    RANDOM_STATE,
    NUM_COLS,
    CAT_COLS,
    TARGET_COL,
    K_RANGE,
    DATA_URL,
    DEVICE,
    GPU_AVAILABLE,
    XGB_DEVICE,
)

# Utilities
from src.utils import (
    reduce_memory,
    get_sample,
    hopkins_statistic,
    dunn_index,
    compute_clustering_metrics,
    timer,
    cleanup,
)

# Data Preparation
from src.data_preparation import (
    load_data,
    clean_data,
    engineer_features,
    create_preprocessor,
    fit_pca,
    prepare_data_for_clustering,
    prepare_data_for_classification,
)

# Clustering
from src.clustering_algorithms import (
    ClusteringRunner,
    run_all_clustering,
    k_medoids,
    k_median,
    fuzzy_c_means,
    kernel_kmeans,
)

from src.clustering_evaluation import (
    OptimalKDeterminer,
    ClusteringComparator,
    determine_optimal_k,
    compare_clustering_results,
)

from src.clustering_tendency import (
    ClusteringTendency,
    assess_clustering_tendency,
)

from src.optimal_k import (
    OptimalKAnalyzer,
    determine_optimal_k_simple,
    get_optimal_k_report,
)

from src.cluster_profiling import (
    ClusterProfiler,
    profile_clusters,
)

# Classification
from src.classification_prep import (
    prepare_classification_data,
    prepare_pre_call_data,
    get_pre_call_features,
)

from src.base_classifiers import (
    BaseClassifier,
    create_logistic_regression,
    create_decision_tree,
    create_naive_bayes,
    create_random_forest,
    create_xgboost_classifier,
    create_all_base_classifiers,
    train_and_evaluate_classifier,
)

from src.neural_network import (
    PyTorchNeuralNetwork,
    create_neural_network,
)

from src.ensemble_methods import (
    manual_bagging,
    manual_adaboost,
    run_boosting_depth_experiment,
    train_random_forest,
    train_xgboost,
)

from src.classification_eval import (
    compute_classification_metrics,
    cross_validate_classifier,
    cross_validate_multiple_classifiers,
    lift_at_k,
    plot_roc_curve,
    plot_precision_recall_curve,
    plot_confusion_matrix,
    plot_cumulative_gain,
    compare_classifiers,
)

from src.classification_analysis import (
    OverfittingAnalyzer,
    PreCallAnalyzer,
    compare_small_vs_full_dataset,
)

# Visualization
from src.visualization import (
    create_figure,
    save_figure,
    save_all_figures,
    plot_cluster_pca,
    plot_model_comparison,
    plot_training_history,
    plot_feature_importance,
)

# Sampling (unified sampling manager)
from src.sampling import (
    UnifiedSampler,
    SamplingStrategy,
    SubsamplingConfig,
    ALGORITHM_LIMITS,
    create_tiered_datasets,
)

# Caching
from src.caching import cache_result, clear_cache

# Final Summary
from src.final_summary import (
    create_clustering_final_table,
    get_clustering_ranking_table,
    create_classification_final_table,
    get_classification_ranking_table,
    compute_extrinsic_evaluation,
    interpret_extrinsic_evaluation,
    generate_final_discussion,
    ProjectSummary,
    create_complete_summary,
)


# ============================================================================
# Package Metadata
# ============================================================================

__all__ = [
    # Config
    'RANDOM_STATE',
    'NUM_COLS',
    'CAT_COLS',
    'TARGET_COL',
    'K_RANGE',
    'DATA_URL',
    'DEVICE',
    'GPU_AVAILABLE',
    'XGB_DEVICE',
    
    # Utils
    'reduce_memory',
    'get_sample',
    'hopkins_statistic',
    'dunn_index',
    'compute_clustering_metrics',
    'timer',
    'cleanup',
    
    # Data Preparation
    'load_data',
    'clean_data',
    'engineer_features',
    'create_preprocessor',
    'fit_pca',
    'prepare_data_for_clustering',
    'prepare_data_for_classification',
    
    # Clustering
    'ClusteringRunner',
    'run_all_clustering',
    'k_medoids',
    'k_median',
    'fuzzy_c_means',
    'kernel_kmeans',
    'OptimalKDeterminer',
    'ClusteringComparator',
    'determine_optimal_k',
    'compare_clustering_results',
    'ClusteringTendency',
    'assess_clustering_tendency',
    'OptimalKAnalyzer',
    'determine_optimal_k_simple',
    'get_optimal_k_report',
    'ClusterProfiler',
    'profile_clusters',
    
    # Classification
    'prepare_classification_data',
    'prepare_pre_call_data',
    'get_pre_call_features',
    'BaseClassifier',
    'create_logistic_regression',
    'create_decision_tree',
    'create_naive_bayes',
    'create_random_forest',
    'create_xgboost_classifier',
    'create_all_base_classifiers',
    'train_and_evaluate_classifier',
    'PyTorchNeuralNetwork',
    'create_neural_network',
    'manual_bagging',
    'manual_adaboost',
    'run_boosting_depth_experiment',
    'train_random_forest',
    'train_xgboost',
    'compute_classification_metrics',
    'cross_validate_classifier',
    'cross_validate_multiple_classifiers',
    'lift_at_k',
    'plot_roc_curve',
    'plot_precision_recall_curve',
    'plot_confusion_matrix',
    'plot_cumulative_gain',
    'compare_classifiers',
    'OverfittingAnalyzer',
    'PreCallAnalyzer',
    'compare_small_vs_full_dataset',
    
    # Visualization
    'create_figure',
    'save_figure',
    'save_all_figures',
    'plot_cluster_pca',
    'plot_model_comparison',
    'plot_training_history',
    'plot_feature_importance',

    # Sampling
    'UnifiedSampler',
    'SamplingStrategy',
    'SubsamplingConfig',
    'ALGORITHM_LIMITS',
    'create_tiered_datasets',

    # Caching
    'cache_result',
    'clear_cache',
    
    # Final Summary
    'create_clustering_final_table',
    'get_clustering_ranking_table',
    'create_classification_final_table',
    'get_classification_ranking_table',
    'compute_extrinsic_evaluation',
    'interpret_extrinsic_evaluation',
    'generate_final_discussion',
    'ProjectSummary',
    'create_complete_summary',
]


# ============================================================================
# Package Initialization
# ============================================================================

def test_imports():
    """
    Test that all imports work correctly.
    
    Returns:
        Dictionary with import status
    """
    results = {}
    
    # Test critical imports
    try:
        from src import config
        results['config'] = 'OK'
    except ImportError as e:
        results['config'] = f'FAILED: {e}'
    
    try:
        from src import utils
        results['utils'] = 'OK'
    except ImportError as e:
        results['utils'] = f'FAILED: {e}'
    
    try:
        from src import data_preparation
        results['data_preparation'] = 'OK'
    except ImportError as e:
        results['data_preparation'] = f'FAILED: {e}'
    
    try:
        from src import clustering_algorithms
        results['clustering_algorithms'] = 'OK'
    except ImportError as e:
        results['clustering_algorithms'] = f'FAILED: {e}'
    
    try:
        from src import clustering_evaluation
        results['clustering_evaluation'] = 'OK'
    except ImportError as e:
        results['clustering_evaluation'] = f'FAILED: {e}'
    
    try:
        from src import classification_prep
        results['classification_prep'] = 'OK'
    except ImportError as e:
        results['classification_prep'] = f'FAILED: {e}'
    
    try:
        from src import base_classifiers
        results['base_classifiers'] = 'OK'
    except ImportError as e:
        results['base_classifiers'] = f'FAILED: {e}'
    
    try:
        from src import final_summary
        results['final_summary'] = 'OK'
    except ImportError as e:
        results['final_summary'] = f'FAILED: {e}'
    
    return results


def print_import_status():
    """Print import status for all modules."""
    results = test_imports()
    
    print("=" * 50)
    print("SRC PACKAGE IMPORT STATUS")
    print("=" * 50)
    
    for module, status in results.items():
        print(f"  {module:25s} : {status}")
    
    print("=" * 50)


# ============================================================================
# Package Information
# ============================================================================

def info():
    """Print package information."""
    print("=" * 60)
    print("Project 3: Clustering & Classification")
    print("=" * 60)
    print(f"Version: {__version__}")
    print(f"Author: {__author__}")
    print(f"Date: {__date__}")
    print("")
    print("Modules:")
    print("  - config")
    print("  - utils")
    print("  - data_preparation")
    print("  - clustering_algorithms")
    print("  - clustering_evaluation")
    print("  - clustering_tendency")
    print("  - optimal_k")
    print("  - cluster_profiling")
    print("  - classification_prep")
    print("  - base_classifiers")
    print("  - neural_network")
    print("  - ensemble_methods")
    print("  - classification_eval")
    print("  - classification_analysis")
    print("  - visualization")
    print("  - final_summary")
    print("=" * 60)


# ============================================================================
# Self-Test
# ============================================================================

if __name__ == "__main__":
    info()
    print("")
    print_import_status()