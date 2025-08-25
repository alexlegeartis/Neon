import math
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


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
        ax.legend()

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
        ax.legend()
        
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
    Build four default panels from results dicts keyed by experiment name (str):
      - Loss vs Iteration
      - Loss vs Time
      - ||∇f(X)||_F vs Iteration
      - ||∇f(X)||_F vs Time
    Each value in experiments must contain keys: iterations, cumulative_time, losses, grad_frobenius_norms.
    """

    def collect_series(x_key: str, y_key: str) -> List[Dict[str, Any]]:
        series: List[Dict[str, Any]] = []
        for name, res in experiments.items():
            series.append({
                "name": name,
                "x": res[x_key],
                "y": res[y_key],
            })
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
            "y_label": "||∇f(X)||_2",
            "series": collect_series("iterations", "grad_frobenius_norms"),
            "yscale": "log",
        },
        {
            "title": "Gradient Spectral Norm vs Time",
            "x_label": "Time (s)",
            "y_label": "||∇f(X)||_2",
            "series": collect_series("cumulative_time", "grad_frobenius_norms"),
            "yscale": "log",
        },
    ]

    return panels


def plot_and_save_default_panels(
    experiments: Dict[str, Dict[str, Any]],
    base_save_dir: str = "simple_lls",
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


