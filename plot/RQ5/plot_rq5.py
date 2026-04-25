"""RQ5 figure: router confusion + per-query-type strategy comparison.

Left: 3x3 confusion matrix (oracle row, router prediction column).
Right: grouped bar chart, four query types x five strategies.

Reads:  plot/RQ5/rq5_data.json
Writes: plot/RQ5/fig_rq5.{pdf, png}
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
DATA = HERE / "rq5_data.json"
OUT_PDF = HERE / "fig_rq5.pdf"
OUT_PNG = HERE / "fig_rq5.png"

STRAT_COLOR = {
    "sub-only":   "#4e79a7",
    "com-only":   "#f28e2b",
    "disc-only":  "#76b7b2",
    "oracle":     "#59a14f",
    "KC router":  "#c0392b",
}


def main():
    d = json.loads(DATA.read_text())
    angles = d["angles"]
    qtypes = d["query_types"]
    strategies = d["strategies"]
    conf = d["confusion_recall"]
    conf_n = d["confusion_count"]
    means = d["means"]
    stds = d["stds"]

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    })

    fig = plt.figure(figsize=(13.0, 4.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[0.85, 1.6], wspace=0.28,
                          left=0.06, right=0.985, top=0.86, bottom=0.18)

    # ── Confusion matrix ────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    M = np.array([[conf[a][b] for b in angles] for a in angles])
    im = ax.imshow(M, cmap="Blues", vmin=0, vmax=1, aspect="equal")
    ax.set_xticks(range(len(angles))); ax.set_xticklabels(angles)
    ax.set_yticks(range(len(angles))); ax.set_yticklabels(angles)
    ax.set_xlabel("Router prediction")
    ax.set_ylabel("Oracle angle")
    ax.set_title("(a) KC router confusion (row-normalised)")
    for i in range(len(angles)):
        for j in range(len(angles)):
            v = M[i, j]
            color = "white" if v > 0.55 else "#222"
            ax.text(j, i, f"{v*100:.1f}%\n(n={conf_n[angles[i]][angles[j]]})",
                    ha="center", va="center", fontsize=9, color=color)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("recall")

    # diagonal accuracy annotation
    diag_acc = sum(M[i, i] for i in range(len(angles))) / len(angles)
    ax.text(0.5, -0.18, f"macro recall = {diag_acc*100:.1f}%",
            transform=ax.transAxes, ha="center",
            fontsize=10, color="#1f4e79", weight="bold")

    # ── Grouped bar: per-query-type per-strategy ────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    width = 0.16
    xs = np.arange(len(qtypes))
    for i, s in enumerate(strategies):
        ys = [means[s][q] for q in qtypes]
        es = [stds[s][q] for q in qtypes]
        ax2.bar(xs + (i - 2) * width, ys, width, label=s,
                color=STRAT_COLOR[s], edgecolor="white", linewidth=0.5,
                yerr=es, capsize=2,
                error_kw=dict(elinewidth=0.6, ecolor="#444"))
    ax2.set_xticks(xs)
    ax2.set_xticklabels(qtypes)
    ax2.set_ylabel("application score")
    ax2.set_title("(b) Per-query-type performance across five routing strategies")
    ax2.set_ylim(0, 100)
    ax2.grid(True, axis="y", alpha=0.25, linewidth=0.6)
    ax2.legend(loc="lower right", ncol=5, frameon=True, framealpha=0.92,
               fontsize=8.5, columnspacing=0.8, handlelength=1.6,
               bbox_to_anchor=(1.0, -0.32))

    fig.suptitle("RQ5  ·  Multi-angle routing approaches the oracle while single-angle baselines collapse on off-target queries",
                 y=0.985, fontsize=11.5, weight="bold")

    fig.savefig(OUT_PDF, bbox_inches="tight")
    fig.savefig(OUT_PNG, bbox_inches="tight", dpi=180)
    print(f"[RQ5] wrote {OUT_PDF.name} and {OUT_PNG.name}")


if __name__ == "__main__":
    main()
