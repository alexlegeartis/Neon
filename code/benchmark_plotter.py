import math
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
        ax.set_title(base_title + (f" ({title_suffix})" if title_suffix else ""))
        yscale = panel.get("yscale")
        if yscale:
            ax.set_yscale(yscale)
            #if yscale == "log":
            #    ax.set_ylim(bottom=10**deg)
        ax.grid(True, alpha=0.3)
        ax.legend()

    # Hide any unused subplots
    for j in range(num_panels, len(axes_list)):
        fig.delaxes(axes_list[j])

    fig.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


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


