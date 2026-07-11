"""
visualization.py - Visualization utilities for Project 3

Provides:
- Plotting utilities for clustering and classification
- Consistent styling
- Figure management
- Report-quality visualizations
"""

import warnings
from typing import Optional, List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import FIGURE_DPI, PLOT_STYLE, COLORS

warnings.filterwarnings("ignore")

# Set plotting style
sns.set_style(PLOT_STYLE)
plt.rcParams['figure.dpi'] = FIGURE_DPI
plt.rcParams['savefig.dpi'] = FIGURE_DPI


# ============================================================================
# Figure Management
# ============================================================================

def create_figure(
    figsize: Tuple[int, int] = (10, 6),
    title: Optional[str] = None,
    dpi: int = FIGURE_DPI
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Create a figure with consistent styling.
    
    Args:
        figsize: Figure size
        title: Figure title
        dpi: Figure DPI
    
    Returns:
        Tuple of (figure, axes)
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    
    ax.grid(True, alpha=0.3)
    
    return fig, ax


def save_figure(
    fig: plt.Figure,
    filename: str,
    dpi: int = FIGURE_DPI,
    bbox_inches: str = 'tight'
) -> None:
    """
    Save figure to file.
    
    Args:
        fig: Matplotlib figure
        filename: Output filename
        dpi: DPI for saving
        bbox_inches: Bounding box mode
    """
    fig.savefig(filename, dpi=dpi, bbox_inches=bbox_inches)
    plt.close(fig)


# ============================================================================
# Clustering Visualizations
# ============================================================================

def plot_cluster_pca(
    X_pca: np.ndarray,
    labels: np.ndarray,
    title: str = "PCA Projection",
    figsize: Tuple[int, int] = (8, 6),
    alpha: float = 0.7,
    s: float = 12,
    cmap: str = 'tab10',
    show_noise: bool = True
) -> plt.Figure:
    """
    Plot PCA projection colored by clusters.
    
    Args:
        X_pca: PCA-transformed data (2D)
        labels: Cluster labels
        title: Plot title
        figsize: Figure size
        alpha: Point transparency
        s: Point size
        cmap: Colormap
        show_noise: Show noise points (label -1)
    
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Filter noise if needed
    if show_noise:
        mask = labels != -1
        scatter = ax.scatter(
            X_pca[mask, 0],
            X_pca[mask, 1],
            c=labels[mask],
            cmap=cmap,
            alpha=alpha,
            s=s
        )
        
        # Show noise in gray
        if (labels == -1).any():
            ax.scatter(
                X_pca[labels == -1, 0],
                X_pca[labels == -1, 1],
                c='gray',
                alpha=0.3,
                s=s,
                label='Noise'
            )
            ax.legend()
    else:
        scatter = ax.scatter(
            X_pca[:, 0],
            X_pca[:, 1],
            c=labels,
            cmap=cmap,
            alpha=alpha,
            s=s
        )
    
    ax.set_xlabel("PC1", fontsize=12)
    ax.set_ylabel("PC2", fontsize=12)
    ax.set_title(title, fontsize=14)
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Cluster')
    
    return fig


def plot_dendrogram(
    linkage_matrix: np.ndarray,
    title: str = "Dendrogram",
    figsize: Tuple[int, int] = (14, 6),
    truncate_mode: str = 'lastp',
    p: int = 30,
    color_threshold: float = None
) -> plt.Figure:
    """
    Plot dendrogram.
    
    Args:
        linkage_matrix: Linkage matrix from scipy.cluster.hierarchy.linkage
        title: Plot title
        figsize: Figure size
        truncate_mode: Truncation mode
        p: Number of clusters to show
        color_threshold: Color threshold
    
    Returns:
        Matplotlib figure
    """
    from scipy.cluster.hierarchy import dendrogram
    
    fig, ax = plt.subplots(figsize=figsize)
    
    dendrogram(
        linkage_matrix,
        ax=ax,
        truncate_mode=truncate_mode,
        p=p,
        color_threshold=color_threshold,
        above_threshold_color='gray'
    )
    
    ax.set_xlabel("Sample Index / (Cluster Size)", fontsize=12)
    ax.set_ylabel("Distance", fontsize=12)
    ax.set_title(title, fontsize=14)
    
    return fig


def plot_elbow(
    k_values: List[int],
    sse: List[float],
    best_k: Optional[int] = None,
    title: str = "Elbow Method",
    figsize: Tuple[int, int] = (8, 5)
) -> plt.Figure:
    """
    Plot elbow curve.
    
    Args:
        k_values: K values
        sse: SSE values
        best_k: Optimal K (optional)
        title: Plot title
        figsize: Figure size
    
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(k_values, sse, marker='o', linewidth=2, color=COLORS['primary'])
    
    if best_k:
        ax.axvline(best_k, color=COLORS['danger'], linestyle='--', alpha=0.7,
                   label=f'Optimal K = {best_k}')
        ax.legend()
    
    ax.set_xlabel("Number of Clusters (K)", fontsize=12)
    ax.set_ylabel("SSE (Within-Cluster Sum of Squares)", fontsize=12)
    ax.set_title(title, fontsize=14)
    
    return fig


def plot_silhouette_scores(
    k_values: List[int],
    scores: List[float],
    best_k: Optional[int] = None,
    title: str = "Silhouette Scores",
    figsize: Tuple[int, int] = (8, 5)
) -> plt.Figure:
    """
    Plot silhouette scores.
    
    Args:
        k_values: K values
        scores: Silhouette scores
        best_k: Optimal K (optional)
        title: Plot title
        figsize: Figure size
    
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(k_values, scores, marker='o', linewidth=2, color=COLORS['secondary'])
    
    if best_k:
        ax.axvline(best_k, color=COLORS['danger'], linestyle='--', alpha=0.7,
                   label=f'Optimal K = {best_k}')
        ax.legend()
    
    ax.set_xlabel("Number of Clusters (K)", fontsize=12)
    ax.set_ylabel("Silhouette Score", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_ylim(0, 1)
    
    return fig


# ============================================================================
# Classification Visualizations
# ============================================================================

def plot_roc_curves(
    y_true: np.ndarray,
    y_probas: Dict[str, np.ndarray],
    title: str = "ROC Curves",
    figsize: Tuple[int, int] = (8, 6)
) -> plt.Figure:
    """
    Plot multiple ROC curves.
    
    Args:
        y_true: True labels
        y_probas: Dictionary mapping model names to probabilities
        title: Plot title
        figsize: Figure size
    
    Returns:
        Matplotlib figure
    """
    from sklearn.metrics import roc_curve, roc_auc_score
    
    fig, ax = plt.subplots(figsize=figsize)
    
    for name, y_proba in y_probas.items():
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc = roc_auc_score(y_true, y_proba)
        ax.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC={auc:.3f})")
    
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.4, label='Random (AUC=0.5)')
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='lower right', fontsize=10)
    
    return fig


def plot_precision_recall_curves(
    y_true: np.ndarray,
    y_probas: Dict[str, np.ndarray],
    title: str = "Precision-Recall Curves",
    figsize: Tuple[int, int] = (8, 6)
) -> plt.Figure:
    """
    Plot multiple Precision-Recall curves.
    
    Args:
        y_true: True labels
        y_probas: Dictionary mapping model names to probabilities
        title: Plot title
        figsize: Figure size
    
    Returns:
        Matplotlib figure
    """
    from sklearn.metrics import precision_recall_curve, average_precision_score
    
    fig, ax = plt.subplots(figsize=figsize)
    
    for name, y_proba in y_probas.items():
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        ap = average_precision_score(y_true, y_proba)
        ax.plot(recall, precision, linewidth=2, label=f"{name} (AP={ap:.3f})")
    
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='lower left', fontsize=10)
    
    return fig


def plot_confusion_matrices(
    y_true: np.ndarray,
    y_preds: Dict[str, np.ndarray],
    labels: List[str] = ['no', 'yes'],
    title: str = "Confusion Matrices",
    figsize: Tuple[int, int] = None
) -> plt.Figure:
    """
    Plot multiple confusion matrices.
    
    Args:
        y_true: True labels
        y_preds: Dictionary mapping model names to predictions
        labels: Label names
        title: Plot title
        figsize: Figure size
    
    Returns:
        Matplotlib figure
    """
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    
    n_models = len(y_preds)
    if figsize is None:
        figsize = (4 * n_models, 4)
    
    fig, axes = plt.subplots(1, n_models, figsize=figsize)
    if n_models == 1:
        axes = [axes]
    
    for ax, (name, y_pred) in zip(axes, y_preds.items()):
        ConfusionMatrixDisplay(
            confusion_matrix(y_true, y_pred),
            display_labels=labels
        ).plot(ax=ax, colorbar=False)
        ax.set_title(name, fontsize=10)
    
    plt.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    
    return fig


def plot_feature_importance(
    importance: pd.Series,
    top_k: int = 10,
    title: str = "Feature Importance",
    figsize: Tuple[int, int] = (8, 5),
    color: str = None
) -> plt.Figure:
    """
    Plot feature importance.
    
    Args:
        importance: Series of feature importances
        top_k: Number of top features to show
        title: Plot title
        figsize: Figure size
        color: Bar color
    
    Returns:
        Matplotlib figure
    """
    if color is None:
        color = COLORS['primary']
    
    fig, ax = plt.subplots(figsize=figsize)
    
    top_features = importance.head(top_k)
    top_features.sort_values().plot(kind='barh', ax=ax, color=color)
    
    ax.set_xlabel("Importance", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.invert_yaxis()
    
    return fig


# ============================================================================
# Model Comparison Visualizations
# ============================================================================

def plot_model_comparison(
    results_df: pd.DataFrame,
    metrics: List[str] = ['f1', 'auc_roc', 'accuracy', 'precision', 'recall'],
    title: str = "Model Comparison",
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Plot comparison of multiple models.
    
    Args:
        results_df: DataFrame with model results
        metrics: Metrics to plot
        title: Plot title
        figsize: Figure size
    
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Filter available metrics
    available_metrics = [m for m in metrics if m in results_df.columns]
    
    if not available_metrics:
        raise ValueError("No available metrics to plot.")
    
    # Plot
    results_df[available_metrics].plot(
        kind='bar',
        ax=ax,
        width=0.8,
        rot=20
    )
    
    ax.set_title(title, fontsize=14)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_ylim(0, 1)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig


def plot_training_history(
    train_losses: List[float],
    val_losses: List[float],
    best_epoch: Optional[int] = None,
    title: str = "Training History",
    figsize: Tuple[int, int] = (10, 5)
) -> plt.Figure:
    """
    Plot training and validation loss history.
    
    Args:
        train_losses: Training losses
        val_losses: Validation losses
        best_epoch: Best epoch (optional)
        title: Plot title
        figsize: Figure size
    
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    epochs = range(1, len(train_losses) + 1)
    
    ax.plot(epochs, train_losses, label='Training Loss', linewidth=2, color=COLORS['primary'])
    ax.plot(epochs, val_losses, label='Validation Loss', linewidth=2, color=COLORS['secondary'])
    
    if best_epoch:
        ax.axvline(best_epoch + 1, color=COLORS['danger'], linestyle='--', alpha=0.5,
                   label=f'Best epoch ({best_epoch + 1})')
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='best')
    
    return fig


# ============================================================================
# Distribution Visualizations
# ============================================================================

def plot_distributions(
    df: pd.DataFrame,
    columns: List[str],
    kind: str = 'hist',
    figsize: Tuple[int, int] = (15, 8),
    n_cols: int = 3
) -> plt.Figure:
    """
    Plot distributions of multiple columns.
    
    Args:
        df: DataFrame
        columns: Columns to plot
        kind: 'hist' or 'box'
        figsize: Figure size
        n_cols: Number of columns in subplot grid
    
    Returns:
        Matplotlib figure
    """
    n_rows = int(np.ceil(len(columns) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = np.array(axes).flatten()
    
    for ax, col in zip(axes, columns):
        if kind == 'hist':
            df[col].hist(ax=ax, bins=30, alpha=0.7, color=COLORS['primary'])
            ax.axvline(df[col].mean(), color=COLORS['danger'], linestyle='--', 
                       alpha=0.7, label='Mean')
            ax.axvline(df[col].median(), color=COLORS['success'], linestyle='--', 
                       alpha=0.7, label='Median')
            ax.legend()
        else:
            df.boxplot(column=col, ax=ax)
        
        ax.set_title(col, fontsize=10)
        ax.set_xlabel('')
    
    # Hide unused subplots
    for ax in axes[len(columns):]:
        ax.axis('off')
    
    plt.tight_layout()
    return fig


def plot_correlation_heatmap(
    df: pd.DataFrame,
    columns: List[str],
    title: str = "Correlation Matrix",
    figsize: Tuple[int, int] = (8, 6),
    annot: bool = True,
    cmap: str = 'coolwarm'
) -> plt.Figure:
    """
    Plot correlation heatmap.
    
    Args:
        df: DataFrame
        columns: Columns to include
        title: Plot title
        figsize: Figure size
        annot: Show correlation values
        cmap: Colormap
    
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    corr = df[columns].corr()
    
    sns.heatmap(
        corr,
        ax=ax,
        annot=annot,
        fmt='.2f',
        cmap=cmap,
        vmin=-1,
        vmax=1,
        square=True,
        cbar_kws={'label': 'Correlation'}
    )
    
    ax.set_title(title, fontsize=14)
    
    plt.tight_layout()
    return fig


# ============================================================================
# Testing
# ============================================================================

if __name__ == "__main__":
    print("Testing visualization.py...")
    
    from sklearn.datasets import make_blobs, make_classification
    from sklearn.cluster import KMeans
    from sklearn.linear_model import LogisticRegression
    from sklearn.decomposition import PCA
    
    # Test clustering visualization
    print("\n1. Testing clustering visualizations:")
    X, y = make_blobs(n_samples=300, centers=3, random_state=42)
    X = X.astype(np.float64)
    
    km = KMeans(n_clusters=3, random_state=42)
    labels = km.fit_predict(X)
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    fig = plot_cluster_pca(X_pca, labels, title="Test Clusters")
    plt.show()
    
    # Test classification visualization
    print("\n2. Testing classification visualizations:")
    X, y = make_classification(n_samples=200, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    fig = plot_roc_curves(y_test, {'Logistic Regression': y_proba})
    plt.show()
    
    print("\nAll tests passed!")