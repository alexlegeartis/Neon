import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path
from typing import Dict, List, Union


def plot_norm_comparison(
    log_files: Dict[str, str],
    metrics: List[str],
    x_axis: str = "step",
    output_dir: str = None,
    figsize: tuple = (10, 6),
    dpi: int = 300
):
    """
    Plot and compare different norm logs.
    
    Args:
        log_files: Dictionary mapping title/name to CSV file path
                   e.g., {"Muon": "grad_norms_Muon_run0.csv", "NormalizedMuon": "grad_norms_NormalizedMuon_run0.csv"}
        metrics: List of metrics to plot. Can be:
                 - Exact column names (e.g., "total_frobenius", "layers.1.conv1.weight_nuclear")
                 - Partial matches (e.g., "nuclear norm for conv1" will match columns containing "conv1" and "nuclear")
                 - Pattern matches (e.g., "total_*" for all total metrics)
        x_axis: Column to use for x-axis (default: "step", can also use "epoch")
        output_dir: Directory to save PDF files (default: same directory as first log file)
        figsize: Figure size tuple (width, height)
        dpi: Resolution for saved figures
    """
    # Load all dataframes
    dataframes = {}
    for title, filepath in log_files.items():
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Log file not found: {filepath}")
        df = pd.read_csv(filepath)
        # Remove empty rows
        df = df.dropna(how='all')
        dataframes[title] = df
    
    # Determine output directory
    if output_dir is None:
        first_file = list(log_files.values())[0]
        output_dir = os.path.dirname(os.path.abspath(first_file))
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all available columns (excluding x_axis and loss)
    # Note: train_acc and val_acc are included and will be plotted separately
    all_columns = set()
    for df in dataframes.values():
        all_columns.update(df.columns)
    all_columns.discard(x_axis)
    all_columns.discard('loss')
    # Keep epoch, train_acc, and val_acc as they are valid metrics to plot
    
    # Resolve metrics to actual column names
    resolved_metrics = []
    for metric in metrics:
        matched_columns = _resolve_metric(metric, all_columns)
        if not matched_columns:
            print(f"Warning: No columns found matching '{metric}'. Skipping.")
            continue
        resolved_metrics.extend(matched_columns)
    
    if not resolved_metrics:
        raise ValueError("No valid metrics found to plot!")
    
    # Plot each metric separately
    for metric_col in resolved_metrics:
        fig, ax = plt.subplots(figsize=figsize)
        
        # Special handling for accuracy metrics
        is_accuracy = metric_col in ['train_acc', 'val_acc']
        
        for title, df in dataframes.items():
            if metric_col in df.columns:
                # Filter out NaN and inf values for plotting
                mask = pd.notna(df[metric_col]) & pd.notna(df[x_axis])
                mask = mask & (df[metric_col] != float('inf')) & (df[metric_col] != float('-inf'))
                
                if mask.any():
                    y_values = df.loc[mask, metric_col]
                    # Convert accuracy to percentage for display
                    if is_accuracy:
                        y_values = y_values * 100
                    
                    ax.plot(
                        df.loc[mask, x_axis],
                        y_values,
                        label=title,
                        marker='o',
                        markersize=3,
                        linewidth=1.5
                    )
        
        ax.set_xlabel(x_axis.replace('_', ' ').title(), fontsize=12)
        
        # Format y-axis label for accuracy metrics
        if is_accuracy:
            ax.set_ylabel(f"{_format_metric_name(metric_col)} (%)", fontsize=12)
        else:
            ax.set_ylabel(_format_metric_name(metric_col), fontsize=12)
        
        # ax.set_title(_format_metric_name(metric_col), fontsize=14, fontweight='bold') - we need no title
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Save figure
        safe_filename = _sanitize_filename(metric_col)
        output_path = os.path.join(output_dir, f"{safe_filename}.pdf")
        plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=dpi)
        plt.close()
        print(f"Saved: {output_path}")


def _resolve_metric(metric: str, available_columns: set) -> List[str]:
    """
    Resolve a metric specification to actual column names.
    
    Supports:
    - Exact column name match
    - Partial keyword matching (e.g., "nuclear norm for conv1")
    - Wildcard patterns (e.g., "total_*")
    """
    metric_lower = metric.lower()
    matched = []
    
    # Check for exact match
    if metric in available_columns:
        return [metric]
    
    # Check for wildcard pattern
    if '*' in metric:
        pattern = metric.replace('*', '.*')
        import re
        regex = re.compile(pattern, re.IGNORECASE)
        for col in available_columns:
            if regex.match(col):
                matched.append(col)
        return matched
    
    # Keyword-based matching
    # Extract keywords from metric (e.g., "nuclear norm for conv1" -> ["nuclear", "conv1"])
    keywords = metric_lower.split()
    # Remove common words
    keywords = [k for k in keywords if k not in ['norm', 'for', 'of', 'the', 'and', 'or']]
    
    for col in available_columns:
        col_lower = col.lower()
        # Check if all keywords are present in column name
        if all(keyword in col_lower for keyword in keywords):
            matched.append(col)
    
    return matched


def _format_metric_name(metric: str) -> str:
    """Format metric column name for display."""
    # Special formatting for common metrics
    if metric == 'train_acc':
        return 'Train Accuracy'
    elif metric == 'val_acc':
        return 'Validation Accuracy'
    
    # Replace underscores with spaces and capitalize
    formatted = metric.replace('_', ' ').replace('.', ' ')
    # Capitalize first letter of each word
    words = formatted.split()
    formatted = ' '.join(word.capitalize() for word in words)
    return formatted


def _sanitize_filename(filename: str) -> str:
    """Sanitize filename by replacing invalid characters."""
    # Replace invalid filename characters
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    # Replace dots and spaces with underscores
    filename = filename.replace('.', '_').replace(' ', '_')
    return filename

import re

import os
import re
from datetime import datetime


def collect_grad_norm_files(folder_path, after_dt=datetime(2025, 11, 26, 10, 50, 0)):
    """
    Collect grad_norm CSV logs whose timestamp is strictly after `after_dt`.

    Parameters
    ----------
    folder_path : str
        Directory containing the CSV files.
    after_dt : datetime
        Only files with date/time > after_dt will be included.

    Returns
    -------
    dict : {key: filename}
    """
    pattern = re.compile(
        r"grad_norms_(.*?)_run\d+_(\d{8})_(\d{6})\.csv$"
    )
    
    result = {}

    for filename in os.listdir(folder_path):
        if not filename.endswith(".csv"):
            continue

        m = pattern.match(filename)
        if not m:
            continue

        key = m.group(1)
        date_str = m.group(2)   # YYYYMMDD
        time_str = m.group(3)   # HHMMSS

        # Parse into datetime
        file_dt = datetime.strptime(date_str + time_str, "%Y%m%d%H%M%S")

        if file_dt > after_dt:
            result[key] = filename

    return result


if __name__ == "__main__":
    # Example usage
    log_files = collect_grad_norm_files(".")
    
    metrics = [
        "total_frobenius",
        "total_spectral",
        "total_nuclear",
        "nuclear norm for conv1",  # Will match layers.*.conv1.weight_nuclear
        "nuclear norm for conv2",
        "Frobenius norm for conv1",
        "Frobenius norm for conv2",

        # "Frobenius norm for norm2",
        "spectral norm for conv1",
        "spectral norm for conv2",
        "train_acc",  # Train accuracy (plotted separately)
        "val_acc"     # Validation accuracy (plotted separately)
    ]
    
    plot_norm_comparison(log_files, metrics)

