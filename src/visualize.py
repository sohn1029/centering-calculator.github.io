"""Stage 4 - visual report of the centering measurement.

Produces a single figure: the rectified card with the detected outer (card edge)
and inner (artwork) borders overlaid and each side's width labelled, plus
left/right and top/bottom ratio gauges and a summary panel.
"""
from __future__ import annotations

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from .pipeline import Result

_GREEN = "#00c853"
_RED = "#ff1744"
_INK = "#222222"


def _gauge(ax, a_pct: float, left_label: str, right_label: str, title: str):
    """Horizontal 0..100 gauge with the split marked and the ideal 50 line."""
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title(title, fontsize=11, loc="left", color=_INK, pad=6)
    # two-tone bar
    ax.add_patch(Rectangle((0, 0.35), a_pct, 0.3, color="#42a5f5"))
    ax.add_patch(Rectangle((a_pct, 0.35), 100 - a_pct, 0.3, color="#ffa726"))
    ax.axvline(50, 0.2, 0.8, color=_INK, ls="--", lw=1)          # ideal centre
    ax.plot([a_pct], [0.5], marker="v", color=_INK, ms=9)
    off = abs(a_pct - 50)
    ax.text(1, 0.85, f"{left_label} {a_pct:.1f}%", fontsize=10, color="#1565c0", va="bottom")
    ax.text(99, 0.85, f"{100 - a_pct:.1f}% {right_label}", fontsize=10,
            color="#e65100", va="bottom", ha="right")
    ax.text(50, 0.05, f"offset {off:.1f}%p from 50/50", fontsize=8,
            color="#666", ha="center", va="bottom")


def render(result: Result, out_path: str):
    warped = cv2.cvtColor(result.rectification.warped, cv2.COLOR_BGR2RGB)
    b = result.borders
    c = result.centering
    W, H = b.size

    fig = plt.figure(figsize=(12, 8), dpi=130)
    gs = fig.add_gridspec(3, 2, width_ratios=[1.05, 1], height_ratios=[1, 1, 1.1],
                          wspace=0.18, hspace=0.35)

    # --- card with overlays (spans the left column) ---
    axc = fig.add_subplot(gs[:, 0])
    axc.imshow(warped)
    axc.set_xlim(0, W)
    axc.set_ylim(H, 0)
    axc.axis("off")
    axc.set_title(f"{result.name}  -  rectified card", fontsize=12, color=_INK)

    ol, oi = b.outer, b.inner
    axc.add_patch(Rectangle((ol["left"], ol["top"]), ol["right"] - ol["left"],
                            ol["bottom"] - ol["top"], fill=False, ec=_GREEN, lw=2))
    axc.add_patch(Rectangle((oi["left"], oi["top"]), oi["right"] - oi["left"],
                            oi["bottom"] - oi["top"], fill=False, ec=_RED, lw=2))
    midx = 0.5 * (oi["left"] + oi["right"])
    midy = 0.5 * (oi["top"] + oi["bottom"])
    bbox = dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8)
    axc.text(midx, 0.5 * (ol["top"] + oi["top"]), f"{b.widths['top']:.0f}px",
             color=_RED, ha="center", va="center", fontsize=10, bbox=bbox)
    axc.text(midx, 0.5 * (ol["bottom"] + oi["bottom"]), f"{b.widths['bottom']:.0f}px",
             color=_RED, ha="center", va="center", fontsize=10, bbox=bbox)
    axc.text(0.5 * (ol["left"] + oi["left"]), midy, f"{b.widths['left']:.0f}px",
             color=_RED, ha="center", va="center", fontsize=10, rotation=90, bbox=bbox)
    axc.text(0.5 * (ol["right"] + oi["right"]), midy, f"{b.widths['right']:.0f}px",
             color=_RED, ha="center", va="center", fontsize=10, rotation=90, bbox=bbox)

    # --- gauges ---
    _gauge(fig.add_subplot(gs[0, 1]), c.lr, "L", "R", "Left / Right centering")
    _gauge(fig.add_subplot(gs[1, 1]), c.tb, "T", "B", "Top / Bottom centering")

    # --- summary panel ---
    axs = fig.add_subplot(gs[2, 1])
    axs.axis("off")
    conf_color = {"high": "#2e7d32", "medium": "#ef6c00", "low": "#c62828"}[c.confidence]
    lines = [
        (f"Left / Right :  {c.lr_text}", _INK),
        (f"Top / Bottom :  {c.tb_text}", _INK),
        (f"Worst axis   :  {c.worst_pct:.1f} / {100 - c.worst_pct:.1f}", _INK),
        (f"Centering grade :  {c.grade:.0f}  ({c.grade_label})", "#1565c0"),
        (f"Confidence :  {c.confidence.upper()}", conf_color),
    ]
    y = 0.92
    for text, color in lines:
        axs.text(0.02, y, text, fontsize=13, color=color, va="top",
                 family="monospace")
        y -= 0.17
    axs.text(0.02, y - 0.02,
             "green = card edge (outer)   red = artwork edge (inner)",
             fontsize=8, color="#666", va="top")

    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path
