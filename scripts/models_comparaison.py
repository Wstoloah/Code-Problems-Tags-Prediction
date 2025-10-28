"""
Model Comparison Visualization Script

This script generates comprehensive plots to compare ML models from experiment logs.
Reads global_metrics.csv and per_tag_metrics.csv to identify the best performing model.

Usage:
    python models_comparaison.py --data-dir <path> --output-dir <path> [--top-n 10]
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import click
from typing import Tuple, List


def load_data(data_dir: Path = Path(".")) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load global and per-tag metrics from CSV files.
    
    Args:
        data_dir: Directory containing the CSV files
        
    Returns:
        Tuple of (global_df, per_tag_df)
    """
    global_path = data_dir / "global_metrics.csv"
    per_tag_path = data_dir / "per_tag_metrics.csv"
    
    if not global_path.exists():
        raise FileNotFoundError(f"Global metrics file not found: {global_path}")
    if not per_tag_path.exists():
        raise FileNotFoundError(f"Per-tag metrics file not found: {per_tag_path}")
    
    global_df = pd.read_csv(global_path)
    per_tag_df = pd.read_csv(per_tag_path)
    
    # Convert date column
    if "Date" in global_df.columns:
        global_df["Date"] = pd.to_datetime(global_df["Date"])
    
    return global_df, per_tag_df


def plot_global_metrics_comparison(global_df: pd.DataFrame, output_dir: Path, top_n: int = 10):
    """Create bar plot comparing global metrics across models.
    
    Shows precision, recall, F1, and hamming loss for top N models by F1 score.
    """
    # Get top N models by F1 score
    top_models = global_df.nlargest(top_n, "F1")
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Prepare data
    x = np.arange(len(top_models))
    width = 0.2
    
    metrics = ["Precision", "Recall", "F1"]
    colors = ["#3498db", "#e74c3c", "#2ecc71"]
    
    # Plot bars for each metric
    for i, (metric, color) in enumerate(zip(metrics, colors)):
        offset = width * (i - 1)
        ax.bar(x + offset, top_models[metric], width, label=metric, color=color, alpha=0.8)
    
    # Customize plot
    ax.set_xlabel("Model ID", fontsize=12, fontweight='bold')
    ax.set_ylabel("Score", fontsize=12, fontweight='bold')
    ax.set_title(f"Top {top_n} Models: Global Metrics Comparison", fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(top_models["ID"], rotation=45, ha='right')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim([0, 1.0])
    
    plt.tight_layout()
    plt.savefig(output_dir / "01_global_metrics_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    click.echo(f"===> Created: 01_global_metrics_comparison.png")

def plot_f1_vs_hamming(global_df: pd.DataFrame, output_dir: Path):
    """Create scatter plot of F1 score vs Hamming loss.
    
    Helps identify models with high F1 and low error rate.
    Includes a zoomed inset of the region with best performing models.
    """
    # Filter out models with Hamming score > 0.5
    global_df = global_df[global_df["Hamming"] <= 0.5].copy()
    
    if len(global_df) == 0:
        click.echo(click.style("/!\ Skipped: F1 vs Hamming plot (no models with Hamming <= 0.5)", fg='yellow'))
        return
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Get top 5 models
    top_5_models = global_df.nlargest(5, "F1")
    top_5_ids = set(top_5_models["ID"].tolist())
    
    # Calculate zoom region based on top models (safe padding)
    if len(top_5_models) > 0:
        x_min = float(top_5_models["Hamming"].min())
        x_max = float(top_5_models["Hamming"].max())
        y_min = float(top_5_models["F1"].min())
        y_max = float(top_5_models["F1"].max())

        # Compute a padding that's 10% of the range, fallback to small absolute pad
        x_range = x_max - x_min
        y_range = y_max - y_min
        pad_x = (x_range * 0.1) if x_range > 0 else 0.01
        pad_y = (y_range * 0.1) if y_range > 0 else 0.01

        zoom_x_min = x_min - pad_x
        zoom_x_max = x_max + pad_x
        zoom_y_min = y_min - pad_y
        zoom_y_max = y_max + pad_y
    else:
        # No top models found - skip inset by setting zoom to None
        zoom_x_min = zoom_x_max = zoom_y_min = zoom_y_max = None
    
    # Find baseline model
    baseline_models = global_df[global_df["ID"] == "BaselineTagPredictor-001"]
    baseline_id = baseline_models["ID"].iloc[0] if len(baseline_models) > 0 else None
    
    # Add baseline to special models if not already in top 5
    if baseline_id and baseline_id not in top_5_ids:
        baseline_row = global_df[global_df["ID"] == baseline_id].iloc[0]
        special_models = pd.concat([top_5_models, pd.DataFrame([baseline_row])])
    else:
        special_models = top_5_models
    
    special_ids = set(special_models["ID"].tolist())
    
    # Colors for special models (top 5 + baseline)
    special_colors = ['#FFD700', '#FF6347', '#32CD32', '#1E90FF', '#FF1493', '#FFA500']
    
    # Create scatter plot
    models = global_df["Model"].unique()
    print("====> Models: {models}")
    base_colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
    
    for model, color in zip(models, base_colors):
        model_data = global_df[global_df["Model"] == model]
        
        # Separate special models from rest
        special_mask = model_data["ID"].isin(special_ids)
        other_data = model_data[~special_mask]
        
        # Plot non-special models with original colors
        if len(other_data) > 0:
            ax.scatter(other_data["Hamming"], other_data["F1"], 
                      label=model, alpha=0.5, s=80, color=color, 
                      edgecolors='black', linewidth=0.5)
    
    # Plot top 5 models with special colors and larger markers
    for i, (_, model) in enumerate(top_5_models.iterrows()):
        ax.scatter(model["Hamming"], model["F1"], 
                  s=50, color=special_colors[i], alpha=0.9,
                  edgecolors='black', linewidth=0.5, marker='*',
                  label=f'#{i+1}: {model["ID"]} (F1={model["F1"]:.4f})',
                  zorder=5)
    
    # Plot baseline model if it exists and not in top 5
    if baseline_id and baseline_id not in top_5_ids:
        baseline_row = global_df[global_df["ID"] == baseline_id].iloc[0]
        ax.scatter(baseline_row["Hamming"], baseline_row["F1"], 
                  s=50, color=special_colors[5], alpha=0.9,
                  edgecolors='red', linewidth=0.5, marker='D',
                  label=f'Baseline: {baseline_id} (F1={baseline_row["F1"]:.4f})',
                  zorder=5)
    
    ax.set_xlabel("Hamming Loss (lower is better)", fontsize=12, fontweight='bold')
    ax.set_ylabel("F1 Score (higher is better)", fontsize=12, fontweight='bold')
    ax.set_title("F1 Score vs Hamming Loss: Top 5 Models & Baseline", fontsize=14, fontweight='bold')
    
    # Create legend
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', 
             fontsize=9, title='Models', title_fontsize=10)
    
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Create zoomed inset only if we computed valid zoom bounds
    if zoom_x_min is not None:
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
        from matplotlib.patches import Rectangle

        # Draw a rectangle on the main axes showing the zoom region
        rect = Rectangle((zoom_x_min, zoom_y_min), zoom_x_max - zoom_x_min,
                         zoom_y_max - zoom_y_min, linewidth=1, edgecolor='gray',
                         facecolor='none', linestyle='--', zorder=2)
        ax.add_patch(rect)

        # Create inset axes positioned inside the figure (lower-right by default)
        axins = inset_axes(ax, width="38%", height="38%",
                          bbox_to_anchor=(0.60, 0.06, 0.35, 0.35),
                          bbox_transform=ax.transAxes)

        # Plot only the points that fall inside the zoom region to keep inset clean
        for model, color in zip(models, base_colors):
            model_data = global_df[global_df["Model"] == model]
            special_mask = model_data["ID"].isin(special_ids)
            other_data = model_data[~special_mask]

            in_bounds = other_data[
                (other_data["Hamming"] >= zoom_x_min) & (other_data["Hamming"] <= zoom_x_max) &
                (other_data["F1"] >= zoom_y_min) & (other_data["F1"] <= zoom_y_max)
            ]

            if len(in_bounds) > 0:
                axins.scatter(in_bounds["Hamming"], in_bounds["F1"],
                              alpha=0.6, s=80, color=color,
                              edgecolors='black', linewidth=0.5)

        # Plot special models in inset (top models always highlighted)
        for i, (_, row) in enumerate(top_5_models.iterrows()):
            axins.scatter(row["Hamming"], row["F1"],
                         s=120, color=special_colors[i], alpha=0.95,
                         edgecolors='black', linewidth=0.6, marker='*', zorder=5)

        if baseline_id and baseline_id not in top_5_ids:
            baseline_row = global_df[global_df["ID"] == baseline_id].iloc[0]
            axins.scatter(baseline_row["Hamming"], baseline_row["F1"],
                         s=100, color=special_colors[5], alpha=0.95,
                         edgecolors='red', linewidth=0.6, marker='D', zorder=5)

        # Set the limits for the inset
        axins.set_xlim(zoom_x_min, zoom_x_max)
        axins.set_ylim(zoom_y_min, zoom_y_max)

        # Add connecting lines between the axes and inset
        try:
            mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
        except Exception:
            # Fall back silently if mark_inset is not compatible with current axes
            pass

        # Add grid to inset and reduce tick label size
        axins.grid(True, alpha=0.3, linestyle='--')
        axins.tick_params(axis='both', which='major', labelsize=8)
    
    plt.tight_layout()
    plt.savefig(output_dir / "02_f1_vs_hamming.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    click.echo(f"===> Created: 02_f1_vs_hamming.png")

def plot_precision_recall_tradeoff(global_df: pd.DataFrame, output_dir: Path):
    """Create precision-recall scatter plot.
    
    Shows the tradeoff between precision and recall for each model.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Scatter plot
    scatter = ax.scatter(global_df["Recall"], global_df["Precision"], 
                        c=global_df["F1"], cmap='viridis', 
                        s=150, alpha=0.7, edgecolors='black', linewidth=0.5)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('F1 Score', fontsize=11, fontweight='bold')
    
    # Annotate top 3 models
    top_5 = global_df.nlargest(3, "F1")
    for _, row in top_5.iterrows():
        ax.annotate(row["ID"], 
                   xy=(row["Recall"], row["Precision"]),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=8, alpha=0.8)
    
    # Add diagonal line (perfect balance)
    ax.plot([0, 1], [0, 1], 'r--', alpha=0.3, linewidth=1, label='Perfect Balance')
    
    ax.set_xlabel("Recall", fontsize=12, fontweight='bold')
    ax.set_ylabel("Precision", fontsize=12, fontweight='bold')
    ax.set_title("Precision-Recall Tradeoff", fontsize=14, fontweight='bold')
    ax.set_xlim([0, 1.0])
    ax.set_ylim([0, 1.0])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_dir / "03_precision_recall_tradeoff.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    click.echo(f"===> Created: 03_precision_recall_tradeoff.png")

def plot_per_tag_heatmap(per_tag_df: pd.DataFrame, global_df: pd.DataFrame, 
                         output_dir: Path, top_n: int = 10):
    """Create heatmap of F1 scores per tag for top models.
    
    Shows which models perform best on which tags.
    """
    # Get top N model IDs from global metrics
    top_model_ids = global_df.nlargest(top_n, "F1")["ID"].tolist()
    
    # Filter per-tag data
    filtered = per_tag_df[per_tag_df["ID"].isin(top_model_ids)]
    
    # Pivot to create matrix: rows = models, columns = tags
    pivot = filtered.pivot_table(values="F1", index="ID", columns="Tag", aggfunc='mean')
    
    # Reorder rows by mean F1 across tags
    pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]
    
    fig, ax = plt.subplots(figsize=(14, max(8, len(pivot) * 0.4)))
    
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn', 
                center=0.5, vmin=0, vmax=1, 
                cbar_kws={'label': 'F1 Score'},
                linewidths=0.5, linecolor='gray', ax=ax)
    
    ax.set_title(f"Per-Tag F1 Scores: Top {top_n} Models", fontsize=14, fontweight='bold')
    ax.set_xlabel("Tag", fontsize=12, fontweight='bold')
    ax.set_ylabel("Model ID", fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / "04_per_tag_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    click.echo(f"===> Created: 04_per_tag_heatmap.png")

def plot_tag_difficulty(per_tag_df: pd.DataFrame, output_dir: Path):
    """Analyze which tags are hardest to predict.
    
    Box plot showing F1 score distribution for each tag across all models.
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Group by tag and calculate statistics
    tag_stats = per_tag_df.groupby("Tag", observed=True)["F1"].agg(['mean', 'std', 'count']).reset_index()
    tag_stats = tag_stats.sort_values("mean", ascending=False)
    
    # Create box plot
    tags_order = tag_stats["Tag"].tolist()
    per_tag_df['Tag'] = pd.Categorical(per_tag_df['Tag'], categories=tags_order, ordered=True)
    
    sns.boxplot(data=per_tag_df, x="Tag", y="F1", ax=ax, hue="Tag", palette="Set2", legend=False)
    
    # Add mean line
    means = per_tag_df.groupby("Tag", observed=True)["F1"].mean().reindex(tags_order)
    ax.plot(range(len(means)), means, color='red', marker='D', 
            linestyle='--', linewidth=2, markersize=8, label='Mean F1', alpha=0.7)
    
    ax.set_xlabel("Tag", fontsize=12, fontweight='bold')
    ax.set_ylabel("F1 Score", fontsize=12, fontweight='bold')
    ax.set_title("Tag Difficulty Analysis: F1 Score Distribution Across All Models", 
                fontsize=14, fontweight='bold')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_dir / "05_tag_difficulty.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    click.echo(f"===> Created: 05_tag_difficulty.png")

def plot_model_consistency(per_tag_df: pd.DataFrame, global_df: pd.DataFrame, 
                           output_dir: Path, top_n: int = 10):
    """Compare model consistency across tags.
    
    Shows standard deviation of F1 scores across tags for each model.
    Lower std = more consistent performance.
    """
    # Get top N model IDs
    top_model_ids = global_df.nlargest(top_n, "F1")["ID"].tolist()
    
    # Calculate per-model statistics
    filtered = per_tag_df[per_tag_df["ID"].isin(top_model_ids)]
    model_stats = filtered.groupby("ID", observed=True)["F1"].agg(['mean', 'std']).reset_index()
    model_stats = model_stats.sort_values("mean", ascending=False)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create scatter plot: mean F1 vs std
    scatter = ax.scatter(model_stats["std"], model_stats["mean"], 
                        s=200, alpha=0.6, edgecolors='black', linewidth=1.5,
                        c=range(len(model_stats)), cmap='coolwarm')
    
    # Annotate each point
    for _, row in model_stats.iterrows():
        ax.annotate(row["ID"], 
                   xy=(row["std"], row["mean"]),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=9, fontweight='bold')
    
    # Add quadrant lines
    mean_std = model_stats["std"].mean()
    mean_f1 = model_stats["mean"].mean()
    ax.axvline(mean_std, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.axhline(mean_f1, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    
    # Add ideal region annotation
    ax.text(0.02, 0.98, 'Ideal:\nHigh Mean, Low Std', 
           transform=ax.transAxes, fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    ax.set_xlabel("Standard Deviation (consistency)", fontsize=12, fontweight='bold')
    ax.set_ylabel("Mean F1 Score (performance)", fontsize=12, fontweight='bold')
    ax.set_title("Model Consistency: Performance vs Stability Across Tags", 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_dir / "06_model_consistency.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    click.echo(f"===> Created: 06_model_consistency.png")

def plot_feature_impact(global_df: pd.DataFrame, output_dir: Path):
    """Analyze impact of different features (code, stats, embeddings).
    
    Shows how different feature combinations affect performance.
    """
    if not all(col in global_df.columns for col in ["Use code", "Use stats features"]):
        click.echo(click.style("/!\ Skipped: Feature impact plot (missing feature columns)", fg='yellow'))
        return
    
    # Create feature combination categories
    global_df["Features"] = global_df.apply(
        lambda row: f"Code={row['Use code']}, Stats={row['Use stats features']}", axis=1
    )
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: F1 by feature combination
    feature_stats = global_df.groupby("Features", observed=True)["F1"].agg(['mean', 'std', 'count']).reset_index()
    axes[0, 0].bar(range(len(feature_stats)), feature_stats["mean"], 
                   yerr=feature_stats["std"], capsize=5, alpha=0.7, color='skyblue')
    axes[0, 0].set_xticks(range(len(feature_stats)))
    axes[0, 0].set_xticklabels(feature_stats["Features"], rotation=45, ha='right', fontsize=9)
    axes[0, 0].set_ylabel("Mean F1 Score", fontweight='bold')
    axes[0, 0].set_title("F1 Score by Feature Combination", fontweight='bold')
    axes[0, 0].grid(axis='y', alpha=0.3, linestyle='--')
    
    # Plot 2: F1 by embedding type
    if "Embedding" in global_df.columns:
        embedding_stats = global_df.groupby("Embedding", observed=True)["F1"].agg(['mean', 'std']).reset_index()
        axes[0, 1].bar(range(len(embedding_stats)), embedding_stats["mean"],
                      yerr=embedding_stats["std"], capsize=5, alpha=0.7, color='lightcoral')
        axes[0, 1].set_xticks(range(len(embedding_stats)))
        axes[0, 1].set_xticklabels(embedding_stats["Embedding"], rotation=45, ha='right')
        axes[0, 1].set_ylabel("Mean F1 Score", fontweight='bold')
        axes[0, 1].set_title("F1 Score by Embedding Type", fontweight='bold')
        axes[0, 1].grid(axis='y', alpha=0.3, linestyle='--')
    
    # Plot 3: F1 by classifier
    if "Classifier" in global_df.columns:
        classifier_stats = global_df.groupby("Classifier", observed=True)["F1"].agg(['mean', 'std']).reset_index()
        axes[1, 0].bar(range(len(classifier_stats)), classifier_stats["mean"],
                      yerr=classifier_stats["std"], capsize=5, alpha=0.7, color='lightgreen')
        axes[1, 0].set_xticks(range(len(classifier_stats)))
        axes[1, 0].set_xticklabels(classifier_stats["Classifier"], rotation=45, ha='right')
        axes[1, 0].set_ylabel("Mean F1 Score", fontweight='bold')
        axes[1, 0].set_title("F1 Score by Classifier", fontweight='bold')
        axes[1, 0].grid(axis='y', alpha=0.3, linestyle='--')
    
    # Plot 4: Count of experiments by configuration
    config_counts = global_df.groupby("Features", observed=True).size().reset_index(name='count')
    axes[1, 1].bar(range(len(config_counts)), config_counts["count"], 
                   alpha=0.7, color='plum')
    axes[1, 1].set_xticks(range(len(config_counts)))
    axes[1, 1].set_xticklabels(config_counts["Features"], rotation=45, ha='right', fontsize=9)
    axes[1, 1].set_ylabel("Number of Experiments", fontweight='bold')
    axes[1, 1].set_title("Experiment Count by Configuration", fontweight='bold')
    axes[1, 1].grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.suptitle("Feature Impact Analysis", fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(output_dir / "07_feature_impact.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    click.echo(f"===> Created: 07_feature_impact.png")

def plot_support_vs_performance(per_tag_df: pd.DataFrame, output_dir: Path):
    """Analyze if tag support (sample count) correlates with performance.
    
    Scatter plot of support vs F1 score for each tag.
    """
    # Average metrics per tag across all models
    tag_avg = per_tag_df.groupby("Tag", observed=True).agg({
        "F1": "mean",
        "Support": "mean",
        "Precision": "mean",
        "Recall": "mean"
    }).reset_index()
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Scatter plot colored by precision-recall balance
    scatter = ax.scatter(tag_avg["Support"], tag_avg["F1"], 
                        s=200, alpha=0.6, edgecolors='black', linewidth=1,
                        c=tag_avg["Precision"] - tag_avg["Recall"], 
                        cmap='RdYlBu', vmin=-0.3, vmax=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Precision - Recall Balance', fontsize=10, fontweight='bold')
    
    # Annotate tags
    for _, row in tag_avg.iterrows():
        ax.annotate(row["Tag"], 
                   xy=(row["Support"], row["F1"]),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=9, alpha=0.7)
    
    # Add trend line
    z = np.polyfit(tag_avg["Support"], tag_avg["F1"], 1)
    p = np.poly1d(z)
    ax.plot(tag_avg["Support"], p(tag_avg["Support"]), 
           "r--", alpha=0.5, linewidth=2, label=f'Trend line')
    
    ax.set_xlabel("Average Support (samples per tag)", fontsize=12, fontweight='bold')
    ax.set_ylabel("Average F1 Score", fontsize=12, fontweight='bold')
    ax.set_title("Tag Support vs Performance: Does More Data Help?", 
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_dir / "08_support_vs_performance.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    click.echo(f"===> Created: 08_support_vs_performance.png")

@click.command()
@click.option(
    '--data-dir',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    required=True,
    help='Directory containing CSV files (global_metrics.csv and per_tag_metrics.csv)'
)
@click.option(
    '--output-dir',
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    required=True,
    help='Output directory where plots will be saved'
)
@click.option(
    '--top-n',
    type=int,
    default=10,
    show_default=True,
    help='Number of top models to display in plots'
)
def main(data_dir: Path, output_dir: Path, top_n: int):
    """Generate comprehensive model comparison plots and analysis.
    
    This tool reads experiment results from CSV files and generates
    visualizations to help identify the best performing models.
    """
    # Setup output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    click.echo("\n" + "="*60)
    click.echo("MODEL COMPARISON VISUALIZATION")
    click.echo("="*60 + "\n")
    
    # Load data
    click.echo("===> Loading data...")
    try:
        global_df, per_tag_df = load_data(data_dir)
        click.echo(f"   - Global metrics: {len(global_df)} experiments")
        click.echo(f"   - Per-tag metrics: {len(per_tag_df)} records")
        click.echo(f"   - Unique models: {global_df['Model'].nunique()}")
        click.echo(f"   - Unique tags: {per_tag_df['Tag'].nunique()}\n")
    except FileNotFoundError as e:
        click.echo(click.style(f"XXX Error: {e}", fg='red'), err=True)
        raise click.Abort()
    except Exception as e:
        click.echo(click.style(f"XXX Unexpected error loading data: {e}", fg='red'), err=True)
        raise click.Abort()
    
    # Generate plots
    click.echo("===> Generating plots...\n")
    
    try:
        plot_global_metrics_comparison(global_df, output_dir, top_n)
    except Exception as e:
        click.echo(click.style(f"XXX Error creating global metrics comparison: {e}", fg='red'), err=True)
    
    try:
        plot_f1_vs_hamming(global_df, output_dir)
    except Exception as e:
        click.echo(click.style(f"XXX Error creating F1 vs Hamming plot: {e}", fg='red'), err=True)
    
    try:
        plot_precision_recall_tradeoff(global_df, output_dir)
    except Exception as e:
        click.echo(click.style(f"XXX Error creating precision-recall tradeoff: {e}", fg='red'), err=True)
    
    try:
        plot_per_tag_heatmap(per_tag_df, global_df, output_dir, top_n)
    except Exception as e:
        click.echo(click.style(f"XXX Error creating per-tag heatmap: {e}", fg='red'), err=True)
    
    try:
        plot_tag_difficulty(per_tag_df, output_dir)
    except Exception as e:
        click.echo(click.style(f"XXX Error creating tag difficulty plot: {e}", fg='red'), err=True)
    
    try:
        plot_model_consistency(per_tag_df, global_df, output_dir, top_n)
    except Exception as e:
        click.echo(click.style(f"XXX Error creating model consistency plot: {e}", fg='red'), err=True)
    
    try:
        plot_feature_impact(global_df, output_dir)
    except Exception as e:
        click.echo(click.style(f"XXX Error creating feature impact plot: {e}", fg='red'), err=True)
    
    try:
        plot_support_vs_performance(per_tag_df, output_dir)
    except Exception as e:
        click.echo(click.style(f"XXX Error creating support vs performance plot: {e}", fg='red'), err=True)

    
    best_model = global_df.loc[global_df["F1"].idxmax()]
    click.echo(f"\n=== Best Model: {best_model['ID']} ===")
    click.echo(f"   F1 Score: {best_model['F1']:.4f}")
    click.echo(f"   Precision: {best_model['Precision']:.4f}")
    click.echo(f"   Recall: {best_model['Recall']:.4f}")
    click.echo(f"   Hamming Loss: {best_model['Hamming']:.4f}")
    
    click.echo(f"\n===> All outputs saved to: {click.style(str(output_dir.absolute()), fg='green', bold=True)}")
    click.echo("\n=== Done! ===\n")


if __name__ == "__main__":
    # Set style for better-looking plots
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    
    main()