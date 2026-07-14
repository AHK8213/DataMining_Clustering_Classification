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


def save_all_figures(
    figures: Dict[str, plt.Figure],
    prefix: str = "",
    subdir: str = "",
    dpi: int = FIGURE_DPI,
) -> List[str]:
    """
    Save several figures at once into FIGURES_DIR, organized by section.

    Args:
        figures: Mapping of {name: matplotlib Figure}
        prefix: Optional filename prefix, e.g. "clustering"
        subdir: Optional subfolder under FIGURES_DIR, e.g. "optimal_k"
        dpi: DPI for saving

    Returns:
        List of the file paths written, e.g.:
        save_all_figures({'hopkins_plot': fig}, subdir='clustering_tendency')
        -> figures/clustering_tendency/hopkins_plot.png
    """
    from src.config import FIGURES_DIR

    out_dir = FIGURES_DIR / subdir if subdir else FIGURES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    written = []
    for name, fig in figures.items():
        fname = f"{prefix}_{name}.png" if prefix else f"{name}.png"
        path = out_dir / fname
        fig.savefig(path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        written.append(str(path))

    return written


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


# ============================================================================
# Classification Visualizations
# ============================================================================

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
    
    importances = pd.Series(
        np.abs(model.coef_[0]), index=['f0', 'f1', 'f2', 'f3']
    ).sort_values(ascending=False)
    fig = plot_feature_importance(importances, title="Test Importances")
    plt.show()

    # Test batch figure saving
    print("\n3. Testing save_all_figures:")
    saved = save_all_figures({'test_pca': fig}, subdir='_smoke_test')
    print(f"Saved: {saved}")

    print("\nAll tests passed!")