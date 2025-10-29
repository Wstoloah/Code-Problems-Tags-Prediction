"""
Model Comparison Analysis Script

Generates detailed visualizations comparing any two machine learning models.
Analyzes differences in performance metrics, per-tag behavior, and prediction patterns.

Can compare:
- Baseline vs Tuned models (hyperparameter impact)
- Baseline vs Advanced models (e.g., XGBoost vs ModernBERT)
- Tuned vs Fine-tuned models (training approach comparison)
- Any two models from the same evaluation dataset

Usage:
    python scripts/compare_two_models.py --data-dir <path> --output-dir <path> --model1-id <model1-id> --model2-id <model2-id>

    Optional: Use --model1-label and --model2-label to customize legend names
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import click
from typing import Tuple


def load_data(data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load global and per-tag metrics from CSV files."""
    global_path = data_dir / "global_metrics.csv"
    per_tag_path = data_dir / "per_tag_metrics.csv"
    
    if not global_path.exists():
        raise FileNotFoundError(f"Global metrics file not found: {global_path}")
    if not per_tag_path.exists():
        raise FileNotFoundError(f"Per-tag metrics file not found: {per_tag_path}")
    
    global_df = pd.read_csv(global_path)
    per_tag_df = pd.read_csv(per_tag_path)
    
    return global_df, per_tag_df


def plot_global_comparison(model1_row: pd.Series, model2_row: pd.Series, output_dir: Path, labels: dict = None):
    """Compare global metrics between two models."""
    if labels is None:
        labels = {'model1': 'Model 1', 'model2': 'Model 2'}
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Bar chart of main metrics
    metrics = ['Precision', 'Recall', 'F1']
    model1_values = [model1_row['Precision'], model1_row['Recall'], model1_row['F1']]
    model2_values = [model2_row['Precision'], model2_row['Recall'], model2_row['F1']]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = axes[0].bar(x - width/2, model1_values, width, label=f"{labels['model1']}: {model1_row['ID']}", 
                        color='#3498db', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = axes[0].bar(x + width/2, model2_values, width, label=f"{labels['model2']}: {model2_row['ID']}", 
                        color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add percentage change labels
    for i, (model1_val, model2_val) in enumerate(zip(model1_values, model2_values)):
        change = ((model2_val - model1_val) / model1_val) * 100
        y_pos = max(model1_val, model2_val) + 0.02
        color = 'green' if change > 0 else 'red'
        axes[0].text(x[i], y_pos, f'{change:+.1f}%', ha='center', fontsize=10, 
                    fontweight='bold', color=color)
    
    axes[0].set_ylabel('Score', fontsize=12, fontweight='bold')
    axes[0].set_title(f'Global Metrics: {labels["model1"]} vs {labels["model2"]}', fontsize=14, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(metrics, fontsize=11)
    axes[0].legend(fontsize=12, loc='upper left')
    axes[0].set_ylim([0, 1.0])
    axes[0].grid(axis='y', alpha=0.3, linestyle='--')
    
    # Plot 2: Hamming Loss comparison
    hamming_data = [model1_row['Hamming'], model2_row['Hamming']]
    hamming_labels_list = [labels['model1'], labels['model2']]
    colors_hamming = ['#3498db', '#e74c3c']
    
    bars = axes[1].bar(hamming_labels_list, hamming_data, color=colors_hamming, 
                       alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add percentage change label
    change = ((model2_row['Hamming'] - model1_row['Hamming']) / model1_row['Hamming']) * 100
    y_pos = max(hamming_data) + 0.005
    color = 'green' if change < 0 else 'red'
    axes[1].text(0.5, y_pos, f'{change:+.1f}%', ha='center', fontsize=11, 
                fontweight='bold', color=color)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{height:.4f}', ha='center', va='bottom', fontsize=10)
    
    axes[1].set_ylabel('Hamming Loss (lower is better)', fontsize=12, fontweight='bold')
    axes[1].set_title(f'Error Rate: {labels["model1"]} vs {labels["model2"]}', fontsize=14, fontweight='bold')
    axes[1].set_ylim([0, max(hamming_data) * 1.2])
    axes[1].grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_dir / "01_global_metrics_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    click.echo(f"===> Created: 01_global_metrics_comparison.png")


def plot_per_tag_comparison(model1_tags: pd.DataFrame, model2_tags: pd.DataFrame, 
                           output_dir: Path, labels: dict = None):
    """Compare per-tag F1 scores between two models."""
    if labels is None:
        labels = {'model1': 'Model 1', 'model2': 'Model 2'}
    # Merge model data
    comparison = model1_tags.merge(model2_tags, on='Tag', suffixes=('_model1', '_model2'))
    comparison['F1_change'] = comparison['F1_model2'] - comparison['F1_model1']
    comparison['F1_change_pct'] = (comparison['F1_change'] / comparison['F1_model1']) * 100
    comparison = comparison.sort_values('F1_change', ascending=True)
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 12))
    
    # Plot 1: F1 Score comparison by tag
    tags = comparison['Tag'].tolist()
    x = np.arange(len(tags))
    width = 0.35
    
    bars1 = axes[0].barh(x - width/2, comparison['F1_model1'], width, 
                         label=f"{labels['model1']}: {model1_tags['ID'].iloc[0]}", color='#3498db', alpha=0.8, 
                         edgecolor='black', linewidth=1)
    bars2 = axes[0].barh(x + width/2, comparison['F1_model2'], width, 
                         label=f"{labels['model2']}: {model2_tags['ID'].iloc[0]}", color='#e74c3c', alpha=0.8, 
                         edgecolor='black', linewidth=1)
    
    axes[0].set_yticks(x)
    axes[0].set_yticklabels(tags, fontsize=11)
    axes[0].set_xlabel('F1 Score', fontsize=12, fontweight='bold')
    axes[0].set_title(f'Per-Tag F1 Scores: {labels["model1"]} vs {labels["model2"]}', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=12, loc='lower right')
    axes[0].set_xlim([0, 1.0])
    axes[0].grid(axis='x', alpha=0.3, linestyle='--')
    
    # Add change annotations
    for i, row in enumerate(comparison.itertuples()):
        if abs(row.F1_change_pct) > 5:  # Only show significant changes
            x_pos = max(row.F1_model1, row.F1_model2) + 0.03
            color = 'green' if row.F1_change > 0 else 'red'
            axes[0].text(x_pos, i, f'{row.F1_change_pct:+.1f}%', 
                        va='center', fontsize=9, color=color, fontweight='bold')
    
    # Plot 2: F1 Change waterfall
    comparison_sorted = comparison.sort_values('F1_change_pct', ascending=False)
    tags_sorted = comparison_sorted['Tag'].tolist()
    changes = comparison_sorted['F1_change_pct'].tolist()
    colors_change = ['green' if c > 0 else 'red' for c in changes]
    
    bars = axes[1].barh(tags_sorted, changes, color=colors_change, alpha=0.7, 
                        edgecolor='black', linewidth=1)
    
    axes[1].set_xlabel('F1 Change (%)', fontsize=12, fontweight='bold')
    axes[1].set_title('F1 Score Difference by Tag', fontsize=14, fontweight='bold')
    axes[1].axvline(0, color='black', linewidth=1.5, linestyle='-')
    axes[1].grid(axis='x', alpha=0.3, linestyle='--')
    
    # Add value labels
    for i, (tag, change) in enumerate(zip(tags_sorted, changes)):
        x_pos = change + (1 if change > 0 else -1)
        ha = 'left' if change > 0 else 'right'
        axes[1].text(x_pos, i, f'{change:+.1f}%', va='center', ha=ha, 
                    fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / "02_per_tag_f1_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    click.echo(f"===> Created: 02_per_tag_f1_comparison.png")


def plot_precision_recall_comparison(model1_tags: pd.DataFrame, model2_tags: pd.DataFrame,
                                     output_dir: Path, labels: dict = None):
    """Compare precision and recall changes per tag between two models."""
    if labels is None:
        labels = {'model1': 'Model 1', 'model2': 'Model 2'}
    comparison = model1_tags.merge(model2_tags, on='Tag', suffixes=('_model1', '_model2'))
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Plot 1: Precision-Recall scatter - Model 1
    axes[0].scatter(comparison['Recall_model1'], comparison['Precision_model1'],
                   s=200, alpha=0.6, color='#3498db', edgecolors='black', linewidth=1.5,
                   label=f"{labels['model1']}: {model1_tags['ID'].iloc[0]}", marker='o')
    
    # Add labels
    for _, row in comparison.iterrows():
        axes[0].annotate(row['Tag'], 
                        xy=(row['Recall_model1'], row['Precision_model1']),
                        xytext=(5, 5), textcoords='offset points', fontsize=9, alpha=0.7)
    
    axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1, label='Balance line')
    axes[0].set_xlabel('Recall', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Precision', fontsize=12, fontweight='bold')
    axes[0].set_title(f'{labels["model1"]}: Precision vs Recall', fontsize=13, fontweight='bold')
    axes[0].set_xlim([0, 1.0])
    axes[0].set_ylim([0, 1.0])
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3, linestyle='--')
    
    # Plot 2: Precision-Recall scatter - Model 2 (with arrows showing movement)
    axes[1].scatter(comparison['Recall_model2'], comparison['Precision_model2'],
                   s=200, alpha=0.6, color='#e74c3c', edgecolors='black', linewidth=1.5,
                   label=f"{labels['model2']}: {model2_tags['ID'].iloc[0]}", marker='s', zorder=5)
    
    # Add arrows from model1 to model2
    for _, row in comparison.iterrows():
        axes[1].annotate('',
                        xy=(row['Recall_model2'], row['Precision_model2']),
                        xytext=(row['Recall_model1'], row['Precision_model1']),
                        arrowprops=dict(arrowstyle='->', lw=1.5, color='gray', alpha=0.5))
        
        # Add label at model2 position
        axes[1].annotate(row['Tag'],
                        xy=(row['Recall_model2'], row['Precision_model2']),
                        xytext=(5, 5), textcoords='offset points', fontsize=9, alpha=0.7)
    
    # Add model1 points for reference
    axes[1].scatter(comparison['Recall_model1'], comparison['Precision_model1'],
                   s=80, alpha=0.3, color='#3498db', edgecolors='black', linewidth=0.5,
                   label=f"{labels['model1']}: {model1_tags['ID'].iloc[0]}", marker='o', zorder=3)
    
    axes[1].plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1, label='Balance line')
    axes[1].set_xlabel('Recall', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Precision', fontsize=12, fontweight='bold')
    axes[1].set_title('Model Comparison: Precision-Recall Movement', fontsize=13, fontweight='bold')
    axes[1].set_xlim([0, 1.0])
    axes[1].set_ylim([0, 1.0])
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_dir / "03_precision_recall_movement.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    click.echo(f"===> Created: 03_precision_recall_movement.png")


def plot_prediction_behavior(model1_tags: pd.DataFrame, model2_tags: pd.DataFrame,
                             output_dir: Path, labels: dict = None):
    """Analyze how prediction counts differ between two models."""
    if labels is None:
        labels = {'model1': 'Model 1', 'model2': 'Model 2'}
    comparison = model1_tags.merge(model2_tags, on='Tag', suffixes=('_model1', '_model2'))
    comparison['Predicted_change'] = comparison['Predicted_model2'] - comparison['Predicted_model1']
    comparison['Predicted_change_pct'] = (comparison['Predicted_change'] / comparison['Predicted_model1']) * 100
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Number of predictions - Model 1 vs Model 2
    tags = comparison['Tag'].tolist()
    x = np.arange(len(tags))
    width = 0.35
    
    bars1 = axes[0, 0].bar(x - width/2, comparison['Predicted_model1'], width,
                          label=f"{labels['model1']}: {model1_tags['ID'].iloc[0]}", color='#3498db', alpha=0.8,
                          edgecolor='black', linewidth=1)
    bars2 = axes[0, 0].bar(x + width/2, comparison['Predicted_model2'], width,
                          label=f"{labels['model2']}: {model2_tags['ID'].iloc[0]}", color='#e74c3c', alpha=0.8,
                          edgecolor='black', linewidth=1)
    
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(tags, rotation=45, ha='right', fontsize=10)
    axes[0, 0].set_ylabel('Number of Predictions', fontsize=11, fontweight='bold')
    axes[0, 0].set_title(f'Prediction Volume: {labels["model1"]} vs {labels["model2"]}', fontsize=13, fontweight='bold')
    axes[0, 0].legend(fontsize=11)
    axes[0, 0].grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add support reference line
    axes[0, 0].plot(x, comparison['Support_model1'], 'go--', linewidth=2, 
                   markersize=6, label='Actual Support', alpha=0.7)
    axes[0, 0].legend(fontsize=11)
    
    # Plot 2: Prediction change percentage
    comparison_sorted = comparison.sort_values('Predicted_change_pct', ascending=False)
    tags_sorted = comparison_sorted['Tag'].tolist()
    changes = comparison_sorted['Predicted_change_pct'].tolist()
    colors = ['green' if c > 0 else 'red' for c in changes]
    
    bars = axes[0, 1].barh(tags_sorted, changes, color=colors, alpha=0.7,
                          edgecolor='black', linewidth=1)
    
    axes[0, 1].set_xlabel('Change in Predictions (%)', fontsize=11, fontweight='bold')
    axes[0, 1].set_title('Prediction Volume Change', fontsize=13, fontweight='bold')
    axes[0, 1].axvline(0, color='black', linewidth=1.5)
    axes[0, 1].grid(axis='x', alpha=0.3, linestyle='--')
    
    # Add value labels
    for i, change in enumerate(changes):
        x_pos = change + (2 if change > 0 else -2)
        ha = 'left' if change > 0 else 'right'
        axes[0, 1].text(x_pos, i, f'{change:+.1f}%', va='center', ha=ha,
                       fontsize=9, fontweight='bold')
    
    # Plot 3: Precision change vs Prediction change
    axes[1, 0].scatter(comparison['Predicted_change_pct'], 
                      (comparison['Precision_model2'] - comparison['Precision_model1']) * 100,
                      s=200, alpha=0.6, color='purple', edgecolors='black', linewidth=1.5)
    
    for _, row in comparison.iterrows():
        axes[1, 0].annotate(row['Tag'],
                           xy=(row['Predicted_change_pct'], 
                               (row['Precision_model2'] - row['Precision_model1']) * 100),
                           xytext=(5, 5), textcoords='offset points', fontsize=9, alpha=0.7)
    
    axes[1, 0].axhline(0, color='black', linewidth=1, linestyle='--', alpha=0.5)
    axes[1, 0].axvline(0, color='black', linewidth=1, linestyle='--', alpha=0.5)
    axes[1, 0].set_xlabel('Prediction Count Change (%)', fontsize=11, fontweight='bold')
    axes[1, 0].set_ylabel('Precision Change (%)', fontsize=11, fontweight='bold')
    axes[1, 0].set_title('Selectivity vs Precision', fontsize=13, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3, linestyle='--')
    
    # Plot 4: Recall change vs Prediction change
    axes[1, 1].scatter(comparison['Predicted_change_pct'],
                      (comparison['Recall_model2'] - comparison['Recall_model1']) * 100,
                      s=200, alpha=0.6, color='orange', edgecolors='black', linewidth=1.5)
    
    for _, row in comparison.iterrows():
        axes[1, 1].annotate(row['Tag'],
                           xy=(row['Predicted_change_pct'],
                               (row['Recall_model2'] - row['Recall_model1']) * 100),
                           xytext=(5, 5), textcoords='offset points', fontsize=9, alpha=0.7)
    
    axes[1, 1].axhline(0, color='black', linewidth=1, linestyle='--', alpha=0.5)
    axes[1, 1].axvline(0, color='black', linewidth=1, linestyle='--', alpha=0.5)
    axes[1, 1].set_xlabel('Prediction Count Change (%)', fontsize=11, fontweight='bold')
    axes[1, 1].set_ylabel('Recall Change (%)', fontsize=11, fontweight='bold')
    axes[1, 1].set_title('Selectivity vs Recall', fontsize=13, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_dir / "04_prediction_behavior.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    click.echo(f"===> Created: 04_prediction_behavior.png")


def generate_summary_table(model1_row: pd.Series, model2_row: pd.Series,
                           model1_tags: pd.DataFrame, model2_tags: pd.DataFrame,
                           output_dir: Path, labels: dict = None):
    """Generate a summary table image comparing two models."""
    if labels is None:
        labels = {'model1': 'Model 1', 'model2': 'Model 2'}
    comparison = model1_tags.merge(model2_tags, on='Tag', suffixes=('_model1', '_model2'))
    
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare data
    table_data = []
    
    # Global metrics section
    table_data.append(['GLOBAL METRICS', '', '', '', ''])
    table_data.append(['Metric', labels['model1'], labels['model2'], 'Change', 'Change %'])
    
    for metric in ['Precision', 'Recall', 'F1', 'Hamming']:
        model1_val = model1_row[metric]
        model2_val = model2_row[metric]
        change = model2_val - model1_val
        change_pct = (change / model1_val) * 100 if model1_val != 0 else 0
        table_data.append([
            metric,
            f'{model1_val:.4f}',
            f'{model2_val:.4f}',
            f'{change:+.4f}',
            f'{change_pct:+.1f}%'
        ])
    
    table_data.append(['', '', '', '', ''])
    table_data.append(['PER-TAG SUMMARY', '', '', '', ''])
    table_data.append(['Tag', f'{labels["model1"]} F1', f'{labels["model2"]} F1', 'Change', 'Status'])
    
    for _, row in comparison.sort_values('F1_model2', ascending=False).iterrows():
        change = row['F1_model2'] - row['F1_model1']
        change_pct = (change / row['F1_model1']) * 100 if row['F1_model1'] != 0 else 0
        status = '++ Better' if change > 0.01 else (' -- Worse' if change < -0.01 else '= Similar')
        table_data.append([
            row['Tag'],
            f'{row["F1_model1"]:.4f}',
            f'{row["F1_model2"]:.4f}',
            f'{change_pct:+.1f}%',
            status
        ])
    
    # Create table
    table = ax.table(cellText=table_data, cellLoc='left', loc='center',
                    colWidths=[0.25, 0.15, 0.15, 0.15, 0.2])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header rows
    for i in [0, 1, 6, 7]:
        for j in range(5):
            cell = table[(i, j)]
            cell.set_facecolor('#34495e')
            cell.set_text_props(weight='bold', color='white')
    
    # Color code the status column
    for i in range(8, len(table_data)):
        cell = table[(i, 4)]
        if '++' in table_data[i][4]:
            cell.set_facecolor('#d4edda')
        elif '--' in table_data[i][4]:
            cell.set_facecolor('#f8d7da')
    
    plt.title(f'Model Comparison Summary: {labels["model1"]} vs {labels["model2"]}', 
             fontsize=16, fontweight='bold', pad=20)
    plt.savefig(output_dir / "05_summary_table.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    click.echo(f"===> Created: 05_summary_table.png")


@click.command()
@click.option(
    '--data-dir',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    required=True,
    help='Directory containing CSV files'
)
@click.option(
    '--output-dir',
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    required=True,
    help='Output directory for plots'
)
@click.option(
    '--model1-id',
    type=str,
    default='BaselineTagPredictor-013',
    show_default=True,
    help='First model ID to compare'
)
@click.option(
    '--model2-id',
    type=str,
    default='TunedBaselineTagPredictor-013',
    show_default=True,
    help='Second model ID to compare'
)
@click.option(
    '--model1-label',
    type=str,
    default='model1',
    help='Custom label for first model (default: auto-detect from ID)'
)
@click.option(
    '--model2-label',
    type=str,
    default='model2',
    help='Custom label for second model (default: auto-detect from ID)'
)
def main(data_dir: Path, output_dir: Path, model1_id: str, model2_id: str,
         model1_label: str = None, model2_label: str = None):
    """Generate detailed visualizations comparing two models.
    
    Can compare:
    - Baseline vs Tuned models
    - Tuned vs ModernBERT models
    - Any two models in the dataset
    """
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    click.echo("\n" + "="*60)
    click.echo("MODEL COMPARISON ANALYSIS")
    click.echo("="*60 + "\n")
    
    # Load data
    click.echo("===> Loading data...")
    try:
        global_df, per_tag_df = load_data(data_dir)
        click.echo(f"   - Loaded {len(global_df)} experiments")
        click.echo(f"   - Loaded {len(per_tag_df)} per-tag records\n")
    except FileNotFoundError as e:
        click.echo(click.style(f"XXX Error: {e}", fg='red'), err=True)
        raise click.Abort()
    
    # Get model data
    model1_row = global_df[global_df['ID'] == model1_id]
    model2_row = global_df[global_df['ID'] == model2_id]
    
    if len(model1_row) == 0:
        click.echo(click.style(f"XXX Error: First model '{model1_id}' not found", fg='red'), err=True)
        raise click.Abort()
    if len(model2_row) == 0:
        click.echo(click.style(f"XXX Error: Second model '{model2_id}' not found", fg='red'), err=True)
        raise click.Abort()
    
    model1_row = model1_row.iloc[0]
    model2_row = model2_row.iloc[0]
    
    model1_tags = per_tag_df[per_tag_df['ID'] == model1_id].copy()
    model2_tags = per_tag_df[per_tag_df['ID'] == model2_id].copy()
    
    # Auto-detect labels if not provided
    if model1_label is None:
        if 'ModernBert' in model1_id or 'modernbert' in model1_id.lower():
            model1_label = 'ModernBERT'
        elif 'Tuned' in model1_id:
            model1_label = 'Tuned'
        elif 'Baseline' in model1_id:
            model1_label = 'Baseline'
        else:
            model1_label = 'Model 1'
    
    if model2_label is None:
        if 'ModernBert' in model2_id or 'modernbert' in model2_id.lower():
            model2_label = 'ModernBERT'
        elif 'Tuned' in model2_id:
            model2_label = 'Tuned'
        elif 'Baseline' in model2_id:
            model2_label = 'Baseline'
        else:
            model2_label = 'Model 2'
    
    # Store labels for use in plotting functions
    labels = {'model1': model1_label, 'model2': model2_label}
    
    click.echo(f"===> Comparing:")
    click.echo(f"   - {model1_label}: {model1_id} (F1={model1_row['F1']:.4f})")
    click.echo(f"   - {model2_label}: {model2_id} (F1={model2_row['F1']:.4f})")
    click.echo(f"   - Change: {((model2_row['F1'] - model1_row['F1']) / model1_row['F1'] * 100):+.2f}%\n")
    
    # Generate plots
    click.echo("===> Generating visualizations...\n")
    
    try:
        plot_global_comparison(model1_row, model2_row, output_dir, labels)
    except Exception as e:
        click.echo(click.style(f"XXX Error creating global comparison: {e}", fg='red'), err=True)
    
    try:
        plot_per_tag_comparison(model1_tags, model2_tags, output_dir, labels)
    except Exception as e:
        click.echo(click.style(f"XXX Error creating per-tag comparison: {e}", fg='red'), err=True)
    
    try:
        plot_precision_recall_comparison(model1_tags, model2_tags, output_dir, labels)
    except Exception as e:
        click.echo(click.style(f"XXX Error creating precision-recall comparison: {e}", fg='red'), err=True)
    
    try:
        plot_prediction_behavior(model1_tags, model2_tags, output_dir, labels)
    except Exception as e:
        click.echo(click.style(f"XXX Error creating prediction behavior plot: {e}", fg='red'), err=True)
    
    try:
        generate_summary_table(model1_row, model2_row, model1_tags, model2_tags, output_dir, labels)
    except Exception as e:
        click.echo(click.style(f"XXX Error creating summary table: {e}", fg='red'), err=True)
    
    click.echo(f"\n===> All outputs saved to: {click.style(str(output_dir.absolute()), fg='green', bold=True)}")
    click.echo("\n=== Done! ===\n")


if __name__ == "__main__":
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    
    main()
