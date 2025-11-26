import math
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def plot_from_descriptions(
    panel_descriptions: List[Dict[str, Any]],
    ncols: int = 2,
    title_suffix: str = "",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 10),
) -> None:
    """
    Generic plotting utility.

    panel_descriptions: list of panels; each panel is a dict with keys:
      - title: str
      - x_label: str
      - y_label: str
      - series: list of dict, each with keys:
          - name: str
          - x: list/array-like
          - y: list/array-like
          - color: Optional[str]
          - linestyle: Optional[str]
          - linewidth: Optional[float]
      - yscale: Optional[str] (e.g., 'log')
    """

    num_panels = len(panel_descriptions)
    nrows = math.ceil(num_panels / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes_list = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
    deg = -3
    for idx, panel in enumerate(panel_descriptions):
        ax = axes_list[idx]
        yscale = panel.get("yscale")
        for s in panel.get("series", []):
            y_values = s["y"]
            if yscale == "log":
                y_array = np.asarray(y_values, dtype=float)
                y_values = np.maximum(y_array, 10**deg)
            ax.plot(
                s["x"],
                y_values,
                label=s.get("name", "series"),
                color=s.get("color", None),
                linestyle=s.get("linestyle", "-"),
                linewidth=s.get("linewidth", 2),
            )
        ax.set_xlabel(panel.get("x_label", ""))
        ax.set_ylabel(panel.get("y_label", ""))
        base_title = panel.get("title", "")
        # ax.set_title(base_title + (f" ({title_suffix})" if title_suffix else ""))
        yscale = panel.get("yscale")
        if yscale:
            ax.set_yscale(yscale)
        ax.grid(True, alpha=0.3)
        ax.legend(ncol=2, frameon=True, fancybox=True, loc='upper right', bbox_to_anchor=(1, 0.85), framealpha=0.95)

    # Hide any unused subplots
    for j in range(num_panels, len(axes_list)):
        fig.delaxes(axes_list[j])

    fig.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight", format='pdf')
    plt.show()


def plot_and_save_individual_panels(
    panel_descriptions: List[Dict[str, Any]],
    base_save_dir: str = "simple_lls",
    title_suffix: str = "",
    figsize: Tuple[int, int] = (10, 8),
) -> None:
    """
    Plot and save each panel individually as a separate PDF file in the specified directory.
    
    Args:
        panel_descriptions: List of panel descriptions (same format as plot_from_descriptions)
        base_save_dir: Directory to save the PDF files (default: "simple_lls")
        title_suffix: Optional suffix to add to plot titles
        figsize: Figure size for individual plots
    """
    
    # Create the save directory if it doesn't exist
    os.makedirs(base_save_dir, exist_ok=True)
    
    for idx, panel in enumerate(panel_descriptions):
        # Create a new figure for each panel
        fig, ax = plt.subplots(figsize=figsize)
        
        yscale = panel.get("yscale")
        deg = -3
        
        # Plot all series for this panel
        for s in panel.get("series", []):
            y_values = s["y"]
            if yscale == "log":
                y_array = np.asarray(y_values, dtype=float)
                y_values = np.maximum(y_array, 10**deg)
            
            ax.plot(
                s["x"],
                y_values,
                label=s.get("name", "series"),
                color=s.get("color", None),
                linestyle=s.get("linestyle", "-"),
                linewidth=s.get("linewidth", 2),
            )
        
        # Set labels and title
        ax.set_xlabel(panel.get("x_label", ""))
        ax.set_ylabel(panel.get("y_label", ""))
        base_title = panel.get("title", "")
        full_title = base_title + (f" ({title_suffix})" if title_suffix else "")
        # ax.set_title(full_title)
        
        # Set yscale if specified
        if yscale:
            ax.set_yscale(yscale)
        
        # Add grid and legend
        ax.grid(True, alpha=0.3)
        ax.legend(ncol=2, frameon=True, fancybox=True, loc='upper right', bbox_to_anchor=(1, 0.85), framealpha=0.95)
        
        # Create filename from title (sanitized for filesystem)
        filename = base_title.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("||", "").replace("∇", "grad").replace("f", "f").replace("X", "X").replace(":", "").replace("__", "_")
        if title_suffix:
            filename += f"_{title_suffix.lower().replace(' ', '_')}"
        filename += ".pdf"
        
        # Save the individual plot
        save_path = os.path.join(base_save_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches="tight", format='pdf')
        print(f"Saved plot to: {save_path}")
        
        # Close the figure to free memory
        plt.close(fig)


def build_default_panels(experiments: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Build default panels from results dicts keyed by experiment name (str):
      - Loss vs Iteration
      - Loss vs Time
      - Gradient Spectral Norm (||∇f(X)||_2) vs Iteration
      - Gradient Spectral Norm (||∇f(X)||_2) vs Time
      - Gradient Frobenius Norm (||∇f(X)||_F) vs Iteration
      - Gradient Frobenius Norm (||∇f(X)||_F) vs Time
      - Gradient Nuclear Norm (||∇f(X)||_*) vs Iteration
      - Gradient Nuclear Norm (||∇f(X)||_*) vs Time
    Each value in experiments must contain keys: iterations, cumulative_time, losses,
    grad_spectral_norms, grad_frobenius_norms, grad_nuclear_norms.
    """

    def collect_series(x_key: str, y_key: str) -> List[Dict[str, Any]]:
        series: List[Dict[str, Any]] = []
        # Use preserved order if available, otherwise use dict iteration order
        experiment_order = experiments.get("_experiment_order", list(experiments.keys()))
        # Filter out the special _experiment_order key
        experiment_order = [name for name in experiment_order if name != "_experiment_order" and name in experiments]
        
        for name in experiment_order:
            res = experiments[name]
            # Pass through optional style fields if present in the experiment results
            series_item: Dict[str, Any] = {
                "name": name,
                "x": res[x_key],
                "y": res[y_key],
            }
            if "color" in res:
                series_item["color"] = res["color"]
            if "linestyle" in res:
                series_item["linestyle"] = res["linestyle"]
            series.append(series_item)
        return series

    panels: List[Dict[str, Any]] = [
        {
            "title": "Loss vs Iteration",
            "x_label": "Iteration",
            "y_label": "Loss",
            "series": collect_series("iterations", "losses"),
            "yscale": "log",
        },
        {
            "title": "Loss vs Time",
            "x_label": "Time (s)",
            "y_label": "Loss",
            "series": collect_series("cumulative_time", "losses"),
            "yscale": "log",
        },
        {
            "title": "Gradient Spectral Norm vs Iteration",
            "x_label": "Iteration",
            "y_label": r"$||∇f(X)||_{op}$",
            "series": collect_series("iterations", "grad_spectral_norms"),
            "yscale": "log",
        },
        {
            "title": "Gradient Spectral Norm vs Time",
            "x_label": "Time (s)",
            "y_label": r"$||∇f(X)||_{op}$",
            "series": collect_series("cumulative_time", "grad_spectral_norms"),
            "yscale": "log",
        },
        {
            "title": "Gradient Frobenius Norm vs Iteration",
            "x_label": "Iteration",
            "y_label": "$||∇f(X)||_F$",
            "series": collect_series("iterations", "grad_frobenius_norms"),
            "yscale": "log",
        },
        {
            "title": "Gradient Frobenius Norm vs Time",
            "x_label": "Time (s)",
            "y_label": "$||∇f(X)||_F$",
            "series": collect_series("cumulative_time", "grad_frobenius_norms"),
            "yscale": "log",
        },
        {
            "title": "Gradient Nuclear Norm vs Iteration",
            "x_label": "Iteration",
            "y_label": r"$||∇f(X)||_{nuc}$",
            "series": collect_series("iterations", "grad_nuclear_norms"),
            "yscale": "log",
        },
        {
            "title": "Gradient Nuclear Norm vs Time",
            "x_label": "Time (s)",
            "y_label": r"$||∇f(X)||_{nuc}$",
            "series": collect_series("cumulative_time", "grad_nuclear_norms"),
            "yscale": "log",
        },
    ]

    return panels


def plot_and_save_default_panels(
    experiments: Dict[str, Dict[str, Any]],
    base_save_dir: str = "L_smooth_tests/simple_lls", # because it is run as a module
    title_suffix: str = "",
) -> None:
    """
    Convenience function to build default panels and save them individually.
    
    Args:
        experiments: Dictionary of experiment results
        base_save_dir: Directory to save the PDF files (default: "simple_lls")
        title_suffix: Optional suffix to add to plot titles
    """
    panels = build_default_panels(experiments)
    plot_and_save_individual_panels(panels, base_save_dir, title_suffix)



def experiments_to_dataframe(experiments: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """
    Convert experiments dict into a long-form DataFrame with columns:
      - experiment: str
      - x_key: str in {"iterations", "cumulative_time"}
      - x: float
      - y_key: str in {"losses", "grad_spectral_norms", "grad_frobenius_norms", "grad_nuclear_norms"}
      - y: float
      - color: Optional[str]
      - linestyle: Optional[str]
    """
    rows: List[Dict[str, Any]] = []
    y_keys = [
        "losses",
        "grad_spectral_norms",
        "grad_frobenius_norms",
        "grad_nuclear_norms",
    ]
    x_keys = ["iterations", "cumulative_time"]

    for exp_name, res in experiments.items():
        color = res.get("color")
        linestyle = res.get("linestyle")
        for y_key in y_keys:
            y_values = res.get(y_key)
            if y_values is None:
                continue
            for x_key in x_keys:
                x_values = res.get(x_key)
                if x_values is None:
                    continue
                # Pair by index (assume equal length)
                length = min(len(x_values), len(y_values))
                for i in range(length):
                    rows.append(
                        {
                            "experiment": exp_name,
                            "x_key": x_key,
                            "x": float(x_values[i]),
                            "y_key": y_key,
                            "y": float(y_values[i]),
                            "color": color,
                            "linestyle": linestyle,
                        }
                    )

    return pd.DataFrame(rows)


def save_experiments_to_csv(experiments: Dict[str, Dict[str, Any]], csv_path: str) -> None:
    """Save experiments dict into a single CSV file at csv_path."""
    df = experiments_to_dataframe(experiments)
    # Create parent dirs if needed
    import os
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    df.to_csv(csv_path, index=False)


def load_experiments_from_csv(csv_path: str) -> Dict[str, Dict[str, Any]]:
    """
    Load experiments from CSV produced by save_experiments_to_csv and
    reconstruct the original experiments dict structure expected by plotting helpers.
    Preserves the order of experiments as they first appear in the CSV.
    """
    df = pd.read_csv(csv_path)
    # Ensure types
    if "x" in df.columns:
        df["x"] = df["x"].astype(float)
    if "y" in df.columns:
        df["y"] = df["y"].astype(float)

    # Track the order of experiments as they first appear in the CSV
    experiment_order = []
    seen_experiments = set()
    for exp_name in df["experiment"]:
        if exp_name not in seen_experiments:
            experiment_order.append(exp_name)
            seen_experiments.add(exp_name)
    
    experiments: Dict[str, Dict[str, Any]] = {}
    for exp_name, g in df.groupby("experiment"):
        exp: Dict[str, Any] = {}
        # Preserve style if present
        styles = g[[c for c in ["color", "linestyle"] if c in g.columns]].dropna(how="all").head(1)
        if not styles.empty:
            srow = styles.iloc[0]
            if "color" in srow and pd.notna(srow["color"]):
                color_val = srow["color"]
                # Convert numpy array string representation to tuple if needed
                if isinstance(color_val, str) and color_val.startswith('['):
                    try:
                        # Handle space-separated values in brackets like '[0.87 0.62 0.84 1.0]'
                        color_str = color_val.strip('[]')
                        color_vals = [float(x) for x in color_str.split()]
                        if len(color_vals) == 4:
                            # RGBA tuple
                            exp["color"] = tuple(color_vals)
                        elif len(color_vals) == 3:
                            # RGB tuple
                            exp["color"] = tuple(color_vals)
                        else:
                            exp["color"] = color_val
                    except:
                        exp["color"] = color_val
                else:
                    exp["color"] = color_val
            if "linestyle" in srow and pd.notna(srow["linestyle"]):
                linestyle_val = srow["linestyle"]
                # Convert tuple string representation if needed
                if isinstance(linestyle_val, str) and linestyle_val.startswith('('):
                    import ast
                    try:
                        linestyle_val = ast.literal_eval(linestyle_val)
                    except:
                        pass
                exp["linestyle"] = linestyle_val

        # Rebuild each series - pair x_key and y_key together
        for (x_key, y_key), gxy in g.groupby(["x_key", "y_key"]):
            gxy_sorted = gxy.sort_values("x")
            # Store x values for this x_key (only once per x_key)
            if x_key not in exp:
                exp[x_key] = gxy_sorted["x"].tolist()
            # Store y values for this y_key, paired with the same x_key
            # Use the length of the corresponding x_key to ensure matching lengths
            x_values = exp.get(x_key, [])
            y_values = gxy_sorted["y"].tolist()
            # Truncate to match x length
            min_len = min(len(x_values), len(y_values))
            exp[y_key] = y_values[:min_len]
            # Also truncate x to match if needed
            if len(x_values) > min_len:
                exp[x_key] = x_values[:min_len]

        experiments[exp_name] = exp

    # Store the order for use in plotting
    experiments["_experiment_order"] = experiment_order
    
    return experiments


def plot_from_csv(
    csv_path: str,
    ncols: int = 2,
    title_suffix: str = "",
    combined_save_path: Optional[str] = None,
    base_save_dir: str = "L_smooth_tests/simple_lls",
) -> None:
    """
    Load experiments from a CSV file and generate the default panels.
    - Saves individual PDFs to base_save_dir
    - If combined_save_path is provided, also saves the combined panel figure there
    """
    experiments = load_experiments_from_csv(csv_path)
    panels = build_default_panels(experiments)
    plot_and_save_individual_panels(panels, base_save_dir=base_save_dir, title_suffix=title_suffix)
    if combined_save_path is not None:
        plot_from_descriptions(panels, ncols=ncols, title_suffix=title_suffix, save_path=combined_save_path)
