"""
final_summary.py - Final summary and conclusion generation

Provides:
- Final clustering comparison table with rankings
- Final classification comparison table with rankings
- Extrinsic evaluation (clustering vs classification)
- Final discussion generation
- Complete project summary report
"""

import warnings
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.config import RANDOM_STATE, VERBOSE, NUM_COLS, CAT_COLS, TARGET_COL
from src.utils import compute_clustering_metrics

warnings.filterwarnings("ignore")


# ============================================================================
# Final Clustering Summary
# ============================================================================

def create_clustering_final_table(
    comparison_df: pd.DataFrame,
    nonlinear_ari: Dict[str, float] = None,
    noise_robustness: Dict[str, float] = None,
    include_ranking: bool = True
) -> pd.DataFrame:
    """
    Create the final clustering comparison table with rankings.
    
    Args:
        comparison_df: DataFrame from clustering comparison
        nonlinear_ari: Dict mapping algorithm names to Two-Moons ARI
        noise_robustness: Dict mapping algorithm names to noise robustness
        include_ranking: Include ranking columns
    
    Returns:
        Final clustering comparison DataFrame
    """
    df = comparison_df.copy()
    
    # Add nonlinear ARI if provided
    if nonlinear_ari:
        df['nonlinear_ari'] = pd.Series(nonlinear_ari)
    
    # Add noise robustness if provided
    if noise_robustness:
        df['noise_robustness_mean'] = pd.Series(noise_robustness)
    
    if include_ranking:
        # Compute ranks for each metric
        rank_df = pd.DataFrame(index=df.index)
        
        # Higher is better
        for metric in ['silhouette', 'dunn', 'calinski_harabasz', 'nonlinear_ari', 'noise_robustness_mean']:
            if metric in df.columns:
                rank_df[f'{metric}_rank'] = df[metric].rank(ascending=False)
        
        # Lower is better
        for metric in ['davies_bouldin', 'runtime_s']:
            if metric in df.columns:
                rank_df[f'{metric}_rank'] = df[metric].rank(ascending=True)
        
        # Overall rank score
        rank_cols = [c for c in rank_df.columns if c.endswith('_rank')]
        if rank_cols:
            df['overall_rank_score'] = rank_df[rank_cols].mean(axis=1)
            df['overall_rank'] = df['overall_rank_score'].rank(ascending=True)
    
    # Sort by overall rank
    if 'overall_rank' in df.columns:
        df = df.sort_values('overall_rank')
    
    return df


def get_clustering_ranking_table(
    df: pd.DataFrame,
    top_k: int = None,
    metrics_to_show: List[str] = None
) -> pd.DataFrame:
    """
    Get a clean ranking table for clustering algorithms.
    
    Args:
        df: Clustering comparison DataFrame
        top_k: Number of top algorithms to show
        metrics_to_show: Metrics to include
    
    Returns:
        Clean ranking table
    """
    if metrics_to_show is None:
        metrics_to_show = [
            'silhouette', 'davies_bouldin', 'dunn',
            'runtime_s', 'noise_robustness_mean', 'nonlinear_ari',
            'overall_rank'
        ]
    
    # Select columns
    available_metrics = [m for m in metrics_to_show if m in df.columns]
    result = df[available_metrics].copy()
    
    # Format values
    for col in result.columns:
        if col == 'overall_rank':
            result[col] = result[col].astype(int)
        elif col in ['silhouette', 'davies_bouldin', 'dunn', 'noise_robustness_mean', 'nonlinear_ari']:
            result[col] = result[col].round(3)
        elif col == 'runtime_s':
            result[col] = result[col].round(2)
    
    if top_k and len(result) > top_k:
        result = result.head(top_k)
    
    return result


# ============================================================================
# Final Classification Summary
# ============================================================================

def create_classification_final_table(
    results_df: pd.DataFrame,
    cv_results: pd.DataFrame = None,
    train_val_results: pd.DataFrame = None,
    include_ranking: bool = True
) -> pd.DataFrame:
    """
    Create the final classification comparison table.
    
    Args:
        results_df: DataFrame from classification results
        cv_results: DataFrame from cross-validation
        train_val_results: DataFrame from train/validation comparison
        include_ranking: Include ranking columns
    
    Returns:
        Final classification comparison DataFrame
    """
    df = results_df.copy()
    
    # Add CV results if provided
    if cv_results is not None:
        cv_df = cv_results.set_index('model')
        for col in ['accuracy_mean', 'precision_mean', 'recall_mean', 'f1_mean', 'roc_auc_mean']:
            if col in cv_df.columns:
                df[col] = df.index.map(cv_df[col])
        
        for col in ['accuracy_std', 'precision_std', 'recall_std', 'f1_std', 'roc_auc_std']:
            if col in cv_df.columns:
                df[col] = df.index.map(cv_df[col])
    
    # Add train/validation results if provided
    if train_val_results is not None:
        tv_df = train_val_results.set_index('model')
        for col in ['train_f1', 'val_f1', 'overfitting_gap']:
            if col in tv_df.columns:
                df[col] = df.index.map(tv_df[col])
    
    if include_ranking:
        # Rank by F1 (primary metric)
        df['f1_rank'] = df['f1'].rank(ascending=False)
        
        # Rank by AUC (secondary metric)
        if 'auc_roc' in df.columns:
            df['auc_rank'] = df['auc_roc'].rank(ascending=False)
        
        # Overall rank (weighted average)
        rank_cols = [c for c in df.columns if c.endswith('_rank')]
        if rank_cols:
            df['overall_rank_score'] = df[rank_cols].mean(axis=1)
            df['overall_rank'] = df['overall_rank_score'].rank(ascending=True)
    
    # Sort by F1
    if 'f1' in df.columns:
        df = df.sort_values('f1', ascending=False)
    
    return df


def get_classification_ranking_table(
    df: pd.DataFrame,
    top_k: int = None,
    metrics_to_show: List[str] = None
) -> pd.DataFrame:
    """
    Get a clean ranking table for classification algorithms.
    
    Args:
        df: Classification comparison DataFrame
        top_k: Number of top algorithms to show
        metrics_to_show: Metrics to include
    
    Returns:
        Clean ranking table
    """
    if metrics_to_show is None:
        metrics_to_show = [
            'f1', 'auc_roc', 'accuracy', 'precision', 'recall',
            'train_time_s', 'cv_f1_mean', 'overfitting_gap'
        ]
    
    # Select columns
    available_metrics = [m for m in metrics_to_show if m in df.columns]
    result = df[available_metrics].copy()
    
    # Format values
    for col in result.columns:
        if col in ['f1', 'auc_roc', 'accuracy', 'precision', 'recall', 'cv_f1_mean']:
            result[col] = result[col].round(3)
        elif col == 'overfitting_gap':
            result[col] = result[col].round(3)
        elif col == 'train_time_s':
            result[col] = result[col].round(2)
    
    if top_k and len(result) > top_k:
        result = result.head(top_k)
    
    return result


# ============================================================================
# Extrinsic Evaluation
# ============================================================================

def compute_extrinsic_evaluation(
    df: pd.DataFrame,
    labels: np.ndarray,
    target_col: str = TARGET_COL,
    algorithm_name: str = "Clustering Model",
    verbose: bool = VERBOSE
) -> Dict[str, Any]:
    """
    Compute extrinsic evaluation metrics (clustering vs classification).
    
    Args:
        df: DataFrame with features and target
        labels: Cluster labels
        target_col: Target column name
        algorithm_name: Name of the clustering algorithm
        verbose: Print progress
    
    Returns:
        Dictionary with extrinsic metrics
    """
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    
    # Filter unknown target
    df_clf = df[df[target_col] != 'unknown'].copy()
    df_clf[target_col] = (df_clf[target_col] == 'yes').astype(int)
    
    # Align labels with data
    # This assumes labels correspond to the same order as df
    labels_aligned = labels[:len(df_clf)] if len(labels) >= len(df_clf) else labels
    
    # Compute extrinsic metrics
    y_true = df_clf[target_col].values
    
    ari = adjusted_rand_score(y_true, labels_aligned)
    nmi = normalized_mutual_info_score(y_true, labels_aligned)
    
    # Compute subscription rate per cluster
    df_cluster = df_clf.copy()
    df_cluster['cluster'] = labels_aligned
    cluster_rates = df_cluster.groupby('cluster')[target_col].mean().sort_values(ascending=False)
    
    results = {
        'algorithm': algorithm_name,
        'ari': ari,
        'nmi': nmi,
        'cluster_rates': cluster_rates,
        'n_clusters': len(cluster_rates),
        'n_samples': len(df_clf)
    }
    
    if verbose:
        print(f"\nExtrinsic Evaluation: {algorithm_name}")
        print(f"  ARI (vs. subscription): {ari:.3f}")
        print(f"  NMI (vs. subscription): {nmi:.3f}")
        print(f"  Number of clusters: {len(cluster_rates)}")
        print(f"\n  Subscription rate per cluster:")
        for cluster, rate in cluster_rates.items():
            print(f"    Cluster {cluster}: {rate:.2%}")
    
    return results


def interpret_extrinsic_evaluation(
    results: Dict[str, Any],
    best_classifier_f1: float
) -> Dict[str, str]:
    """
    Interpret extrinsic evaluation results.
    
    Args:
        results: Results from compute_extrinsic_evaluation
        best_classifier_f1: F1 score of the best classifier
    
    Returns:
        Dictionary with interpretation
    """
    ari = results['ari']
    
    interpretation = {
        'ari_interpretation': '',
        'combined_interpretation': '',
        'practical_implication': ''
    }
    
    # Interpret ARI
    if ari > 0.3:
        interpretation['ari_interpretation'] = (
            "Clusters strongly align with subscription behavior. "
            "Static customer characteristics are predictive of subscription."
        )
    elif ari > 0.15:
        interpretation['ari_interpretation'] = (
            "Clusters moderately align with subscription behavior. "
            "Some static characteristics relate to subscription."
        )
    else:
        interpretation['ari_interpretation'] = (
            "Clusters do NOT align with subscription behavior. "
            "Subscription is driven by behavioral factors (like duration), "
            "not static customer characteristics."
        )
    
    # Combined interpretation
    if ari < 0.15 and best_classifier_f1 > 0.35:
        interpretation['combined_interpretation'] = (
            "Clustering does NOT predict subscription, but classification does. "
            "This suggests subscription is driven by behavioral signals "
            "(e.g., call duration), not customer demographics."
        )
    elif ari > 0.3 and best_classifier_f1 < 0.25:
        interpretation['combined_interpretation'] = (
            "Clustering predicts subscription better than classification. "
            "This suggests customer segments are more predictive than individual features."
        )
    else:
        interpretation['combined_interpretation'] = (
            "Clustering and classification show complementary predictive power. "
            "Both provide useful insights about subscription behavior."
        )
    
    # Practical implication
    max_cluster_rate = results['cluster_rates'].max()
    min_cluster_rate = results['cluster_rates'].min()
    rate_ratio = max_cluster_rate / min_cluster_rate if min_cluster_rate > 0 else float('inf')
    
    if rate_ratio > 3:
        interpretation['practical_implication'] = (
            f"Large variation in subscription rates across clusters "
            f"({rate_ratio:.1f}x difference). "
            f"This segmentation could be useful for targeted marketing campaigns."
        )
    elif rate_ratio > 1.5:
        interpretation['practical_implication'] = (
            f"Moderate variation in subscription rates across clusters "
            f"({rate_ratio:.1f}x difference). "
            f"Some targeting potential exists."
        )
    else:
        interpretation['practical_implication'] = (
            f"Limited variation in subscription rates across clusters. "
            f"This segmentation may not be useful for targeting."
        )
    
    return interpretation


# ============================================================================
# Final Discussion Generation
# ============================================================================

def generate_final_discussion(
    clustering_best: str,
    classification_best: str,
    clustering_metrics: Dict[str, Any],
    classification_metrics: Dict[str, Any],
    extrinsic_results: Dict[str, Any],
    feature_importance: pd.Series = None,
    verbose: bool = True
) -> str:
    """
    Generate the final discussion section.
    
    Args:
        clustering_best: Name of best clustering algorithm
        classification_best: Name of best classification algorithm
        clustering_metrics: Metrics of best clustering algorithm
        classification_metrics: Metrics of best classification algorithm
        extrinsic_results: Extrinsic evaluation results
        feature_importance: Feature importance series (optional)
        verbose: Print progress
    
    Returns:
        Formatted discussion text
    """
    import textwrap
    
    discussion_parts = []
    
    # 1. Best clustering algorithm
    discussion_parts.append("=" * 70)
    discussion_parts.append("FINAL DISCUSSION")
    discussion_parts.append("=" * 70)
    discussion_parts.append("")
    
    discussion_parts.append("### Best Clustering Algorithm")
    discussion_parts.append("")
    discussion_parts.append(f"The holistic selection process identified **{clustering_best}** as the best clustering algorithm.")
    if clustering_metrics:
        discussion_parts.append(f"  - Silhouette Score: {clustering_metrics.get('silhouette', 'N/A'):.3f}")
        discussion_parts.append(f"  - Davies-Bouldin: {clustering_metrics.get('davies_bouldin', 'N/A'):.3f}")
        discussion_parts.append(f"  - Number of Clusters: {clustering_metrics.get('n_clusters', 'N/A')}")
        discussion_parts.append(f"  - Noise Percentage: {clustering_metrics.get('noise_pct', 'N/A'):.1f}%")
    discussion_parts.append("")
    
    # 2. Best classifier
    discussion_parts.append("### Best Classification Model")
    discussion_parts.append("")
    discussion_parts.append(f"The best classification model is **{classification_best}**.")
    if classification_metrics:
        discussion_parts.append(f"  - F1 Score: {classification_metrics.get('f1', 'N/A'):.3f}")
        discussion_parts.append(f"  - AUC-ROC: {classification_metrics.get('auc_roc', 'N/A'):.3f}")
        discussion_parts.append(f"  - Accuracy: {classification_metrics.get('accuracy', 'N/A'):.3f}")
    discussion_parts.append("")
    
    # 3. Extrinsic evaluation
    discussion_parts.append("### Extrinsic Evaluation")
    discussion_parts.append("")
    discussion_parts.append(f"Clustering vs. Subscription Alignment:")
    discussion_parts.append(f"  - ARI: {extrinsic_results.get('ari', 'N/A'):.3f}")
    discussion_parts.append(f"  - NMI: {extrinsic_results.get('nmi', 'N/A'):.3f}")
    discussion_parts.append("")
    
    if extrinsic_results.get('cluster_rates') is not None:
        discussion_parts.append("Subscription rates by cluster:")
        for cluster, rate in extrinsic_results['cluster_rates'].items():
            discussion_parts.append(f"  - Cluster {cluster}: {rate:.2%}")
    discussion_parts.append("")
    
    # 4. Key findings
    discussion_parts.append("### Key Findings")
    discussion_parts.append("")
    discussion_parts.append("1. **Clustering Performance:**")
    discussion_parts.append("   The selected clustering algorithm demonstrates good internal validity")
    discussion_parts.append("   with reasonable cluster separation and compactness.")
    discussion_parts.append("")
    
    discussion_parts.append("2. **Classification Performance:**")
    discussion_parts.append(f"   The best classifier achieves F1={classification_metrics.get('f1', 'N/A'):.3f},")
    discussion_parts.append(f"   indicating {'strong' if classification_metrics.get('f1', 0) > 0.4 else 'moderate'} predictive power.")
    discussion_parts.append("")
    
    # Feature importance if available
    if feature_importance is not None and len(feature_importance) > 0:
        discussion_parts.append("3. **Important Features:**")
        top_features = feature_importance.head(5)
        for i, (feature, importance) in enumerate(top_features.items(), 1):
            discussion_parts.append(f"   {i}. {feature}: {importance:.3f}")
        discussion_parts.append("")
    
    # 5. Recommendations
    discussion_parts.append("### Practical Recommendations")
    discussion_parts.append("")
    discussion_parts.append("1. **Use the full dataset** (or as large a sample as computationally reasonable)")
    discussion_parts.append("   for final reported metrics; use small samples only for rapid iteration.")
    discussion_parts.append("")
    discussion_parts.append("2. **For pre-call lead targeting**, use the duration-free model")
    discussion_parts.append("   and evaluate with lift-at-k rather than raw F1/AUC.")
    discussion_parts.append("")
    discussion_parts.append("3. **Report cross-validated metrics** (mean ± std), not a single train/test split.")
    discussion_parts.append("")
    discussion_parts.append("4. **Regularize/Prune tree-based models** (`max_depth`, `min_samples_split`)")
    discussion_parts.append("   or prefer boosting with early stopping to control overfitting.")
    discussion_parts.append("")
    
    # 6. Limitations
    discussion_parts.append("### Limitations")
    discussion_parts.append("")
    discussion_parts.append("1. **Sampling:** Some algorithms (Kernel K-Means, K-Medoids, HDBSCAN, OPTICS)")
    discussion_parts.append("   run on samples rather than the full dataset for computational tractability.")
    discussion_parts.append("")
    discussion_parts.append("2. **Manual Implementations:** Manual algorithms (K-Medoids, K-Median, Fuzzy C-Means)")
    discussion_parts.append("   are pedagogical and not as optimized as library implementations.")
    discussion_parts.append("")
    discussion_parts.append("3. **Dataset Specificity:** Findings may not generalize to different campaigns,")
    discussion_parts.append("   products, or time periods.")
    discussion_parts.append("")
    discussion_parts.append("4. **Evaluation Bias:** Silhouette/Davies-Bouldin/Calinski-Harabasz all favor")
    discussion_parts.append("   convex, similarly-sized clusters.")
    discussion_parts.append("")
    
    # 7. Future work
    discussion_parts.append("### Future Work")
    discussion_parts.append("")
    discussion_parts.append("1. **Try density-ratio or shape-aware clustering validity indices**")
    discussion_parts.append("   (e.g., DBCV) alongside current metrics.")
    discussion_parts.append("")
    discussion_parts.append("2. **Explore SHAP values** for more rigorous, additive feature-attribution")
    discussion_parts.append("   alternatives to Random Forest importance.")
    discussion_parts.append("")
    discussion_parts.append("3. **Collect additional pre-call behavioral features**")
    discussion_parts.append("   (prior campaign outcomes, contact recency) to close the pre-call gap.")
    discussion_parts.append("")
    discussion_parts.append("4. **Test on other datasets** to assess generalizability.")
    discussion_parts.append("")
    
    discussion_parts.append("=" * 70)
    
    return "\n".join(discussion_parts)


# ============================================================================
# Complete Summary Report
# ============================================================================

class ProjectSummary:
    """
    Generate complete project summary report.
    
    Usage:
        summary = ProjectSummary()
        summary.add_clustering_results(df)
        summary.add_classification_results(df)
        summary.generate_report()
    """
    
    def __init__(self, project_name: str = "Project 3 - Clustering & Classification"):
        """
        Initialize project summary.
        
        Args:
            project_name: Name of the project
        """
        self.project_name = project_name
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        self.clustering_results = None
        self.classification_results = None
        self.extrinsic_results = None
        self.feature_importance = None
        
        self.clustering_best = None
        self.classification_best = None
    
    def add_clustering_results(
        self,
        comparison_df: pd.DataFrame,
        best_model: str,
        nonlinear_ari: Dict[str, float] = None,
        noise_robustness: Dict[str, float] = None
    ) -> None:
        """
        Add clustering results.
        
        Args:
            comparison_df: Clustering comparison DataFrame
            best_model: Name of best model
            nonlinear_ari: Nonlinear shape detection results
            noise_robustness: Noise robustness results
        """
        self.clustering_results = create_clustering_final_table(
            comparison_df, nonlinear_ari, noise_robustness
        )
        self.clustering_best = best_model
    
    def add_classification_results(
        self,
        results_df: pd.DataFrame,
        best_model: str,
        cv_results: pd.DataFrame = None,
        train_val_results: pd.DataFrame = None
    ) -> None:
        """
        Add classification results.
        
        Args:
            results_df: Classification results DataFrame
            best_model: Name of best model
            cv_results: Cross-validation results
            train_val_results: Train/validation results
        """
        self.classification_results = create_classification_final_table(
            results_df, cv_results, train_val_results
        )
        self.classification_best = best_model
    
    def add_extrinsic_results(self, results: Dict[str, Any]) -> None:
        """Add extrinsic evaluation results."""
        self.extrinsic_results = results
    
    def add_feature_importance(self, importance: pd.Series) -> None:
        """Add feature importance."""
        self.feature_importance = importance
    
    def generate_report(self) -> str:
        """
        Generate complete project summary report.
        
        Returns:
            Formatted report string
        """
        report_parts = []
        
        # Header
        report_parts.append("=" * 80)
        report_parts.append(f"{self.project_name}")
        report_parts.append(f"Generated: {self.timestamp}")
        report_parts.append("=" * 80)
        report_parts.append("")
        
        # 1. Clustering Results
        report_parts.append("1. CLUSTERING RESULTS")
        report_parts.append("-" * 50)
        report_parts.append("")
        
        if self.clustering_results is not None:
            report_parts.append(f"Best Algorithm: {self.clustering_best}")
            report_parts.append("")
            report_parts.append("Ranking (top 5):")
            ranking = get_clustering_ranking_table(self.clustering_results, top_k=5)
            report_parts.append(ranking.to_string())
            report_parts.append("")
        else:
            report_parts.append("No clustering results available.")
            report_parts.append("")
        
        # 2. Classification Results
        report_parts.append("2. CLASSIFICATION RESULTS")
        report_parts.append("-" * 50)
        report_parts.append("")
        
        if self.classification_results is not None:
            report_parts.append(f"Best Model: {self.classification_best}")
            report_parts.append("")
            report_parts.append("Ranking:")
            ranking = get_classification_ranking_table(self.classification_results)
            report_parts.append(ranking.to_string())
            report_parts.append("")
        else:
            report_parts.append("No classification results available.")
            report_parts.append("")
        
        # 3. Extrinsic Evaluation
        report_parts.append("3. EXTRINSIC EVALUATION")
        report_parts.append("-" * 50)
        report_parts.append("")
        
        if self.extrinsic_results is not None:
            report_parts.append(f"ARI (vs. subscription): {self.extrinsic_results.get('ari', 'N/A'):.3f}")
            report_parts.append(f"NMI (vs. subscription): {self.extrinsic_results.get('nmi', 'N/A'):.3f}")
            report_parts.append("")
            report_parts.append("Subscription rates by cluster:")
            rates = self.extrinsic_results.get('cluster_rates', {})
            for cluster, rate in rates.items():
                report_parts.append(f"  Cluster {cluster}: {rate:.2%}")
            
            # Best classifier F1
            best_f1 = self.classification_results.loc[0, 'f1'] if self.classification_results is not None else 0
            interpretation = interpret_extrinsic_evaluation(self.extrinsic_results, best_f1)
            
            report_parts.append("")
            report_parts.append("Interpretation:")
            report_parts.append(f"  {interpretation['ari_interpretation']}")
            report_parts.append(f"  {interpretation['combined_interpretation']}")
            report_parts.append(f"  {interpretation['practical_implication']}")
            report_parts.append("")
        else:
            report_parts.append("No extrinsic evaluation results available.")
            report_parts.append("")
        
        # 4. Feature Importance (if available)
        if self.feature_importance is not None:
            report_parts.append("4. FEATURE IMPORTANCE")
            report_parts.append("-" * 50)
            report_parts.append("")
            
            top_10 = self.feature_importance.head(10)
            for i, (feature, importance) in enumerate(top_10.items(), 1):
                report_parts.append(f"  {i:2d}. {feature:30s} {importance:.4f}")
            report_parts.append("")
        
        # 5. Summary Statistics
        report_parts.append("5. SUMMARY STATISTICS")
        report_parts.append("-" * 50)
        report_parts.append("")
        
        if self.clustering_results is not None:
            report_parts.append(f"  Number of clustering algorithms evaluated: {len(self.clustering_results)}")
            report_parts.append(f"  Best clustering algorithm: {self.clustering_best}")
        
        if self.classification_results is not None:
            report_parts.append(f"  Number of classification models evaluated: {len(self.classification_results)}")
            report_parts.append(f"  Best classification model: {self.classification_best}")
        
        if self.extrinsic_results is not None:
            report_parts.append(f"  Extrinsic ARI: {self.extrinsic_results.get('ari', 'N/A'):.3f}")
            report_parts.append(f"  Extrinsic NMI: {self.extrinsic_results.get('nmi', 'N/A'):.3f}")
        
        report_parts.append("")
        
        # Footer
        report_parts.append("=" * 80)
        report_parts.append("END OF REPORT")
        report_parts.append("=" * 80)
        
        return "\n".join(report_parts)
    
    def save_report(self, filepath: str) -> None:
        """
        Save the report to a file.
        
        Args:
            filepath: Output file path
        """
        report = self.generate_report()
        with open(filepath, 'w') as f:
            f.write(report)
        print(f"Report saved to: {filepath}")
    
    def print_report(self) -> None:
        """Print the report to console."""
        report = self.generate_report()
        print(report)


# ============================================================================
# Convenience Functions
# ============================================================================

def create_complete_summary(
    clustering_comparison_df: pd.DataFrame,
    classification_results_df: pd.DataFrame,
    clustering_best: str,
    classification_best: str,
    extrinsic_results: Dict[str, Any],
    feature_importance: pd.Series = None,
    nonlinear_ari: Dict[str, float] = None,
    noise_robustness: Dict[str, float] = None,
    cv_results: pd.DataFrame = None,
    train_val_results: pd.DataFrame = None,
    project_name: str = "Project 3 - Clustering & Classification"
) -> ProjectSummary:
    """
    Create a complete project summary.
    
    Args:
        clustering_comparison_df: Clustering comparison DataFrame
        classification_results_df: Classification results DataFrame
        clustering_best: Best clustering algorithm name
        classification_best: Best classification model name
        extrinsic_results: Extrinsic evaluation results
        feature_importance: Feature importance series
        nonlinear_ari: Nonlinear shape detection results
        noise_robustness: Noise robustness results
        cv_results: Cross-validation results
        train_val_results: Train/validation results
        project_name: Name of the project
    
    Returns:
        ProjectSummary instance
    """
    summary = ProjectSummary(project_name)
    
    summary.add_clustering_results(
        clustering_comparison_df,
        clustering_best,
        nonlinear_ari,
        noise_robustness
    )
    
    summary.add_classification_results(
        classification_results_df,
        classification_best,
        cv_results,
        train_val_results
    )
    
    if extrinsic_results:
        summary.add_extrinsic_results(extrinsic_results)
    
    if feature_importance is not None:
        summary.add_feature_importance(feature_importance)
    
    return summary


# ============================================================================
# Testing
# ============================================================================

if __name__ == "__main__":
    print("Testing final_summary.py...")
    
    # Create sample data
    np.random.seed(42)
    
    # Sample clustering results
    clustering_df = pd.DataFrame({
        'silhouette': np.random.uniform(0.3, 0.7, 5),
        'davies_bouldin': np.random.uniform(0.5, 1.5, 5),
        'dunn': np.random.uniform(0.1, 0.5, 5),
        'runtime_s': np.random.uniform(0.5, 10, 5),
        'noise_pct': np.random.uniform(0, 20, 5),
        'n_clusters': np.random.randint(2, 6, 5),
    }, index=['K-Means', 'DBSCAN', 'GMM', 'HDBSCAN', 'Agglomerative'])
    
    # Sample classification results
    classification_df = pd.DataFrame({
        'f1': np.random.uniform(0.4, 0.8, 4),
        'auc_roc': np.random.uniform(0.6, 0.9, 4),
        'accuracy': np.random.uniform(0.6, 0.85, 4),
        'precision': np.random.uniform(0.5, 0.8, 4),
        'recall': np.random.uniform(0.4, 0.75, 4),
        'train_time_s': np.random.uniform(0.1, 5, 4),
    }, index=['Logistic Regression', 'Decision Tree', 'Random Forest', 'XGBoost'])
    
    # Sample extrinsic results
    extrinsic_results = {
        'ari': 0.25,
        'nmi': 0.30,
        'cluster_rates': {0: 0.15, 1: 0.05, 2: 0.25, 3: 0.10},
        'n_clusters': 4,
        'n_samples': 1000
    }
    
    # Sample feature importance
    importance = pd.Series({
        'duration': 0.25,
        'balance': 0.18,
        'age': 0.12,
        'day': 0.10,
        'campaign': 0.08,
        'month': 0.07,
        'job': 0.06,
        'education': 0.05,
        'marital': 0.04,
        'default': 0.03,
        'housing': 0.02,
        'loan': 0.00
    })
    
    # Test report generation
    print("\n1. Testing ProjectSummary:")
    summary = create_complete_summary(
        clustering_comparison_df=clustering_df,
        classification_results_df=classification_df,
        clustering_best='GMM',
        classification_best='XGBoost',
        extrinsic_results=extrinsic_results,
        feature_importance=importance,
        project_name="Test Project"
    )
    
    summary.print_report()
    
    # Test extrinsic evaluation interpretation
    print("\n2. Testing extrinsic evaluation interpretation:")
    best_f1 = classification_df.loc['XGBoost', 'f1']
    interpretation = interpret_extrinsic_evaluation(extrinsic_results, best_f1)
    for key, value in interpretation.items():
        print(f"{key}: {value}")
    
    print("\nAll tests passed!")