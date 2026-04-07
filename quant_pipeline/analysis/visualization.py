"""
Visualization for sensitivity analysis and robustness results.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _clean_layer_name(name):
    """Shorten layer names for readability."""
    name = name.replace("distilbert.transformer.", "")
    name = name.replace("layer.", "L")
    name = name.replace(".attention", " Attn")
    name = name.replace(".ffn", " FFN")
    return name


def plot_sensitivity(results, baseline_acc, save_path="outputs/sensitivity.png"):
    """
    Plot layer sensitivity — clean, presentation-ready.
    Only shows components with non-zero delta.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Only keep meaningful deltas
    meaningful = {k: v for k, v in results.items() if abs(v["delta"]) > 0.0001}

    # Sort by delta (most negative first)
    sorted_items = sorted(meaningful.items(), key=lambda x: x[1]["delta"])

    layer_names = [_clean_layer_name(name) for name, _ in sorted_items]
    deltas = [item["delta"] * 100 for _, item in sorted_items]

    # Color by severity
    def get_color(d):
        if d <= -2.0:
            return "#501313"
        elif d <= -0.8:
            return "#A32D2D"
        elif d <= -0.3:
            return "#D85A30"
        elif d < -0.05:
            return "#EF9F27"
        elif d > 0.3:
            return "#0F6E56"
        elif d > 0.05:
            return "#5DCAA5"
        else:
            return "#B4B2A9"

    colors = [get_color(d) for d in deltas]

    # Figure
    fig, ax = plt.subplots(figsize=(9, max(4, len(layer_names) * 0.45 + 1.5)))

    # Bars
    bars = ax.barh(range(len(layer_names)), deltas, color=colors,
                   height=0.65, edgecolor="none", zorder=2)

    # Value labels — all on the outside
    for i, (bar, delta) in enumerate(zip(bars, deltas)):
        label = f"{delta:+.2f}%"
        if delta < 0:
            ax.text(delta - 0.12, i, label, ha="right", va="center",
                    fontsize=9, fontweight="500", color=colors[i])
        else:
            ax.text(delta + 0.12, i, label, ha="left", va="center",
                    fontsize=9, fontweight="500", color=colors[i])

    # Zero line
    ax.axvline(x=0, color="#374151", linewidth=0.8, zorder=3)

    # Y-axis
    ax.set_yticks(range(len(layer_names)))
    ax.set_yticklabels(layer_names, fontsize=10.5, fontweight="500")
    ax.invert_yaxis()

    # X-axis
    ax.set_xlabel("Accuracy delta vs FP32 baseline (%)", fontsize=11, labelpad=10)

    # Title
    ax.set_title(
        f"Layer Sensitivity to INT8 Quantization (baseline = {baseline_acc:.1%})",
        fontsize=13, fontweight="bold", pad=16
    )

    # Clean axes
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(left=False)
    ax.grid(axis="x", alpha=0.12, linestyle="-", zorder=0)

    # X padding
    x_min = min(deltas) - 1.0
    x_max = max(deltas) + 1.0
    ax.set_xlim(x_min, x_max)

    # Legend — upper right
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#501313", label="Critical (>2%)"),
        Patch(facecolor="#A32D2D", label="High (0.8–2%)"),
        Patch(facecolor="#D85A30", label="Moderate (0.3–0.8%)"),
        Patch(facecolor="#EF9F27", label="Mild (<0.3%)"),
        Patch(facecolor="#0F6E56", label="Improved"),
    ]
    ax.legend(
        handles=legend_elements, loc="upper right",
        fontsize=7.5, framealpha=0.95, edgecolor="#e2e8f0",
        title="Sensitivity level", title_fontsize=8,
        borderpad=0.6, labelspacing=0.4
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.05)
    plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Sensitivity plot saved to {save_path}")


def plot_robustness_comparison(robustness_results, save_path="outputs/robustness.png"):
    """
    Plot robustness comparison across modes and perturbations.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    modes = list(robustness_results.keys())
    perturbations = list(robustness_results[modes[0]].keys())

    x = np.arange(len(perturbations))
    width = 0.8 / len(modes)

    mode_colors = {
        "fp32": "#378ADD", "int8_ptq": "#D85A30",
        "int8_qat": "#7F77DD", "fp16": "#1D9E75"
    }

    for i, mode in enumerate(modes):
        values = [robustness_results[mode][p] for p in perturbations]
        offset = (i - len(modes) / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=mode.upper(),
                       color=mode_colors.get(mode, "#888780"), edgecolor="none")

        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.0%}", ha="center", va="bottom", fontsize=8, fontweight="500")

    ax.set_xticks(x)
    ax.set_xticklabels(perturbations, rotation=25, ha="right", fontsize=10)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_title("Robustness: FP32 vs quantized under distribution shift",
                 fontsize=13, fontweight="bold", pad=15)
    ax.legend(fontsize=9, framealpha=0.9, edgecolor="#e2e8f0")
    ax.set_ylim(0, 1.12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.15)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Robustness plot saved to {save_path}")
