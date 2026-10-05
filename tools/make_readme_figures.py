#!/usr/bin/env python3
"""
Render the figures used in README.md.

    python tools/make_readme_figures.py            # writes into ./assets

Only matplotlib is required. Edit the constants below if numbers in the
paper change, then re-run the script.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")

# ---------------------------------------------------------------------------
# Numbers from the paper (DiA, NeurIPS 2026).
# ---------------------------------------------------------------------------
# Table 5: Diving48, A100-40GB, FP16, batch of 8 clips of shape [3, 32, 224, 224].
EFFICIENCY = [
    # (metric, AIM, DiA, unit, lower_is_better, value format)
    ("GFLOPs per clip", 812.06, 558.37, "GFLOP", True, "{:.0f}"),
    ("Throughput", 28.15, 61.20, "clips/s", False, "{:.2f}"),
    ("p50 batch latency", 284.14, 130.73, "ms", True, "{:.0f}"),
]

# Tables 1-4: Top-1 (%) of AIM vs. DiA on the seven benchmarks.
BENCHMARKS = [
    # (name, AIM, DiA)
    ("Diving48", 88.9, 92.8),
    ("FineGym99", 94.9, 96.2),
    ("FineGym288", 89.1, 91.7),
    ("HAA500", 80.2, 82.4),
    ("Toyota Smarthome", 84.3, 86.8),
    ("MPII Cooking 2", 76.1, 78.7),
    ("HMDB51", 82.4, 85.6),
]

# Table 1: Diving48, tunable params (M) vs. Top-1 (%).
# EVL (32.9M, 51.4%) is left out to keep the y-axis readable.
DIVING48 = [
    # (method, params, top1, clip_based)
    ("ORViT", 160.0, 88.0, False),
    ("TimeSformer", 121.4, 78.0, False),
    ("BEVT", 88.1, 86.7, False),
    ("VideoSwin-B", 88.1, 81.9, False),
    ("SIFAR-B-14", 87.0, 87.3, False),
    ("SlowFast R101", 62.8, 77.6, False),
    ("GC-TDN", 27.4, 87.6, False),
    ("CTM", 24.4, 86.5, False),
    ("ART", 365.0, 87.9, False),
    ("BDC-CLIP", 125.4, 76.6, True),
    ("PeVL (video)", 28.0, 84.5, True),
    ("DiST", 19.0, 76.85, True),
    ("AIM", 11.0, 88.9, True),
    ("DualPath", 10.0, 88.7, True),
    ("ST-Adapter", 7.2, 84.2, True),
]
DIA_DIVING48 = (3.6, 92.8)

# Palette matched to the DiA infographic.
BG = "#06090d"
CARD = "#0d1218"
EDGE = "#1f2a33"
CYAN = "#38c6f4"
LIME = "#9be15d"
YELLOW = "#e8e84a"
GREY = "#5b6670"
TEXT = "#e9eef2"
MUTED = "#9aa6b0"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "text.color": TEXT,
    "axes.labelcolor": TEXT,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
})


def _card(ax, x, y, w, h, edge=EDGE, face=CARD, lw=1.2, r=0.02):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                linewidth=lw, edgecolor=edge, facecolor=face,
                                transform=ax.transAxes, zorder=0))


# ---------------------------------------------------------------------------
# Figure 1: efficiency (Table 5)
# ---------------------------------------------------------------------------
def efficiency_figure(path):
    fig = plt.figure(figsize=(12, 3.9), dpi=200, facecolor=BG)
    fig.text(0.03, 0.9, "EFFICIENCY", color=CYAN, fontsize=13, fontweight="bold")
    fig.text(0.155, 0.9, "DiA vs. AIM on Diving48  ·  A100, FP16, batch of 8 × 32-frame clips",
             color=MUTED, fontsize=11.5)

    n = len(EFFICIENCY)
    left, gap, top, height = 0.03, 0.025, 0.80, 0.70
    width = (1 - 2 * left - (n - 1) * gap) / n
    for i, (name, base, dia, unit, lower_better, fmt) in enumerate(EFFICIENCY):
        x0 = left + i * (width + gap)
        ax = fig.add_axes([x0, top - height, width, height])
        ax.set_facecolor(CARD)
        for s_ in ax.spines.values():
            s_.set_visible(False)
        _card(ax, 0, 0, 1, 1, edge=EDGE)

        if lower_better:
            delta, arrow = f"{(1 - dia / base) * 100:.0f}% lower", "↓"
        else:
            delta, arrow = f"{dia / base:.2f}× higher", "↑"
        ax.text(0.06, 0.86, name.upper(), transform=ax.transAxes, fontsize=11,
                fontweight="bold", color=MUTED)
        ax.text(0.06, 0.62, f"{delta} {arrow}", transform=ax.transAxes, fontsize=19,
                fontweight="bold", color=LIME)

        vmax = max(base, dia) * 1.05
        rows = [("AIM", base, GREY), ("DiA", dia, CYAN)]
        for j, (lab, val, col) in enumerate(rows):
            yb = 0.40 - j * 0.19
            bw = 0.50 * val / vmax
            ax.add_patch(FancyBboxPatch((0.20, yb - 0.055), bw, 0.11,
                                        boxstyle="round,pad=0,rounding_size=0.02",
                                        transform=ax.transAxes, facecolor=col,
                                        edgecolor="none"))
            ax.text(0.17, yb, lab, transform=ax.transAxes, ha="right", va="center",
                    fontsize=10.5, color=TEXT if lab == "DiA" else MUTED,
                    fontweight="bold" if lab == "DiA" else "normal")
            ax.text(0.20 + bw + 0.02, yb, f"{fmt.format(val)} {unit}", transform=ax.transAxes,
                    va="center", fontsize=10.5, color=TEXT)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    fig.savefig(path, facecolor=BG)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3: DiA vs. AIM across the seven benchmarks (Tables 1-4)
# ---------------------------------------------------------------------------
def benchmarks_figure(path):
    fig = plt.figure(figsize=(12, 5.2), dpi=200, facecolor=BG)
    fig.text(0.03, 0.92, "ACCURACY", color=CYAN, fontsize=13, fontweight="bold")
    fig.text(0.135, 0.92, "Top-1 (%) on seven benchmarks  ·  CLIP ViT-B/16, RGB only",
             color=MUTED, fontsize=11.5)
    ax = fig.add_axes([0.17, 0.12, 0.70, 0.74])
    ax.set_facecolor(BG)
    for s_ in ax.spines.values():
        s_.set_visible(False)

    names = [b[0] for b in BENCHMARKS][::-1]
    aim = [b[1] for b in BENCHMARKS][::-1]
    dia = [b[2] for b in BENCHMARKS][::-1]
    ys = range(len(names))
    for y, a, d in zip(ys, aim, dia):
        ax.plot([a, d], [y, y], color=EDGE, lw=5, solid_capstyle="round", zorder=1)
        ax.scatter([a], [y], s=110, color=GREY, zorder=3, edgecolors=BG, linewidths=1.5)
        ax.scatter([d], [y], s=150, color=CYAN, zorder=4, edgecolors=BG, linewidths=1.5)
        ax.text(a - 0.45, y, f"{a:.1f}", ha="right", va="center", color=MUTED, fontsize=10)
        ax.text(d + 0.45, y, f"{d:.1f}", ha="left", va="center", color=TEXT, fontsize=10.5,
                fontweight="bold")
        fig.text(0.895, 0.12 + 0.74 * (y + 0.5) / len(names), f"+{d - a:.1f}",
                 color=LIME, fontsize=11.5, fontweight="bold", va="center")
    ax.set_yticks(list(ys))
    ax.set_yticklabels(names, color=TEXT, fontsize=11)
    ax.set_ylim(-0.5, len(names) - 0.5)
    ax.set_xlim(72, 100)
    ax.tick_params(axis="y", length=0, pad=10)
    ax.tick_params(axis="x", colors=MUTED, labelsize=9.5)
    ax.grid(axis="x", color=EDGE, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel("Top-1 accuracy (%)", color=MUTED, fontsize=10)

    fig.text(0.885, 0.88, "gain", color=LIME, fontsize=10.5)
    fig.text(0.655, 0.92, "●", color=GREY, fontsize=13)
    fig.text(0.672, 0.92, "AIM (11.0M)", color=MUTED, fontsize=10.5)
    fig.text(0.775, 0.92, "●", color=CYAN, fontsize=13)
    fig.text(0.792, 0.92, "DiA (2.3–3.7M)", color=TEXT, fontsize=10.5)

    fig.savefig(path, facecolor=BG)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4: tunable parameters vs. accuracy on Diving48 (Table 1)
# ---------------------------------------------------------------------------
def params_figure(path):
    import numpy as np
    fig = plt.figure(figsize=(12, 5.6), dpi=200, facecolor=BG)
    fig.text(0.03, 0.93, "DIVING48", color=CYAN, fontsize=13, fontweight="bold")
    fig.text(0.13, 0.93, "Top-1 accuracy vs. tunable parameters (log scale)",
             color=MUTED, fontsize=11.5)
    ax = fig.add_axes([0.07, 0.12, 0.90, 0.75])
    ax.set_facecolor(BG)
    for s_ in ax.spines.values():
        s_.set_color(EDGE)

    offsets = {
        "ORViT": (6, 6), "TimeSformer": (6, -12), "BEVT": (-34, 7), "VideoSwin-B": (-74, -4),
        "SIFAR-B-14": (-38, 8), "SlowFast R101": (-90, -3), "GC-TDN": (6, 5), "CTM": (-30, -12),
        "ART": (-26, 7), "BDC-CLIP": (6, -4), "PeVL (video)": (7, -3), "DiST": (7, -4),
        "AIM": (5, 6), "DualPath": (-58, -12), "ST-Adapter": (-24, -15),
    }
    for name, p, acc, clip_based in DIVING48:
        col = "#8a6bd1" if clip_based else GREY
        ax.scatter([p], [acc], s=70, color=col, edgecolors=BG, linewidths=1, zorder=3)
        dx, dy = offsets.get(name, (6, 4))
        ax.annotate(name, (p, acc), xytext=(dx, dy), textcoords="offset points",
                    color=MUTED, fontsize=9)
    p, acc = DIA_DIVING48
    ax.scatter([p], [acc], s=420, marker="*", color=YELLOW, edgecolors=BG, linewidths=1, zorder=5)
    ax.annotate(f"DiA (ours)  {acc}%  ·  {p}M", (p, acc), xytext=(12, -4),
                textcoords="offset points", color=YELLOW, fontsize=11.5, fontweight="bold")

    ax.set_xscale("log")
    ax.set_xlim(2.5, 500)
    ax.set_ylim(74, 95)
    ticks = [3, 5, 10, 20, 50, 100, 200, 400]
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks])
    ax.minorticks_off()
    ax.tick_params(colors=MUTED, labelsize=9.5)
    ax.grid(color=EDGE, lw=0.7)
    ax.set_axisbelow(True)
    ax.set_xlabel("Tunable parameters (M)", color=MUTED, fontsize=10)
    ax.set_ylabel("Top-1 (%)", color=MUTED, fontsize=10)

    for lab, col, x in [("non-CLIP", GREY, 0.62), ("CLIP-based", "#8a6bd1", 0.73)]:
        fig.text(x, 0.93, "●", color=col, fontsize=12)
        fig.text(x + 0.016, 0.932, lab, color=MUTED, fontsize=10.5)
    fig.text(0.85, 0.932, "★ DiA", color=YELLOW, fontsize=10.5, fontweight="bold")

    fig.savefig(path, facecolor=BG)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: causal / anti-causal receptive fields
# ---------------------------------------------------------------------------
def directional_figure(path, k=5, n_frames=13):
    fig = plt.figure(figsize=(12, 5.0), dpi=200, facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor(BG)
    ax.set_xlim(0, 12); ax.set_ylim(0, 5)
    ax.axis("off")

    ax.text(0.35, 4.6, "HOW DiA READS TIME", color=CYAN, fontsize=13, fontweight="bold")
    ax.text(3.35, 4.6, "two depth-wise temporal filters per token, one per direction",
            color=MUTED, fontsize=11.5)

    xs = [2.35 + i * 0.55 for i in range(n_frames)]
    t = n_frames // 2

    def row(y, label, sub, lo, hi, color, arrow_dir):
        ax.text(0.35, y + 0.08, label, color=color, fontsize=12.5, fontweight="bold",
                va="center")
        ax.text(0.35, y - 0.25, sub, color=MUTED, fontsize=9.5, va="center")
        # window
        x_lo, x_hi = xs[lo] - 0.22, xs[hi] + 0.22
        ax.add_patch(FancyBboxPatch((x_lo, y - 0.27), x_hi - x_lo, 0.54,
                                    boxstyle="round,pad=0,rounding_size=0.12",
                                    facecolor=color + "22", edgecolor=color, lw=1.4))
        ax.plot([xs[0] - 0.3, xs[-1] + 0.3], [y, y], color=EDGE, lw=1.5, zorder=1)
        for i, x in enumerate(xs):
            inside = lo <= i <= hi
            c = color if inside else "#2a3640"
            s = 150 if i == t else 70
            ax.scatter([x], [y], s=s, color=YELLOW if i == t else c, zorder=3,
                       edgecolors=BG, linewidths=1.2)
        # direction arrow
        if arrow_dir:
            a0, a1 = (xs[lo], xs[hi]) if arrow_dir > 0 else (xs[hi], xs[lo])
            ax.add_patch(FancyArrowPatch((a0, y + 0.45), (a1, y + 0.45),
                                         arrowstyle="-|>", mutation_scale=14,
                                         color=color, lw=1.6))

    row(3.55, "Causal", f"past → present  (k={k})", t - (k - 1), t, LIME, +1)
    row(2.35, "Anti-causal", f"future → present  (k={k})", t, t + (k - 1), CYAN, -1)
    row(1.15, "Fused (DiA)", "softmax-weighted sum", t - (k - 1), t + (k - 1), YELLOW, 0)

    # frame axis label
    ax.text(xs[t], 0.45, "frame t", color=YELLOW, ha="center", fontsize=10.5,
            fontweight="bold")
    ax.text(xs[0], 0.45, "← past", color=MUTED, ha="left", fontsize=10)
    ax.text(xs[-1], 0.45, "future →", color=MUTED, ha="right", fontsize=10)

    # right-hand explainer card
    cx = 9.95
    ax.add_patch(FancyBboxPatch((cx - 0.2, 0.55), 2.05, 3.75,
                                boxstyle="round,pad=0,rounding_size=0.12",
                                facecolor=CARD, edgecolor=EDGE, lw=1.2))
    lines = [
        ("c = DWConv₁(pad_L(y))", LIME),
        ("a = DWConv₂(pad_R(y))", CYAN),
        ("w = softmax(α)", MUTED),
        ("out = w₀·c + w₁·a", YELLOW),
    ]
    for i, (s, c) in enumerate(lines):
        ax.text(cx, 3.85 - i * 0.55, s, color=c, fontsize=9.4, family="DejaVu Sans Mono")
    ax.text(cx, 1.42, "Separate weights for", color=TEXT, fontsize=9.6)
    ax.text(cx, 1.12, "past and future make", color=TEXT, fontsize=9.6)
    ax.text(cx, 0.82, "temporal order explicit.", color=TEXT, fontsize=9.6)

    fig.savefig(path, facecolor=BG)
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    efficiency_figure(os.path.join(OUT_DIR, "dia_efficiency.png"))
    directional_figure(os.path.join(OUT_DIR, "dia_directional.png"))
    benchmarks_figure(os.path.join(OUT_DIR, "dia_benchmarks.png"))
    params_figure(os.path.join(OUT_DIR, "dia_params_vs_acc.png"))
    print("Figures written to", OUT_DIR)
