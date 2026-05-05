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

FS = 18  # one font size for the whole figure

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
        "font.family":      "Georgia",
        "font.size":        FS,
        "axes.titlesize":   FS,
        "axes.labelsize":   FS,
        "xtick.labelsize":  FS,
        "ytick.labelsize":  FS,
        "legend.fontsize":  FS,
    })

    fig = plt.figure(figsize=(13.0, 3.2))

    # Manual axes positioning. Goal: visible-content tops AND bottoms of both
    # panels coincide at OUTER_TOP / OUTER_BOT — counting the macro-recall
    # annotation and the xlabel below, not just the data axes.  The constants
    # below were calibrated by measuring rendered pixel offsets for FS=18 at
    # this figure size.
    OUTER_TOP = 0.965
    OUTER_BOT = 0.218
    LEG_BOT   = 0.03
    LEG_H     = 0.16

    YTICK_OVERHANG = 0.005    # ytick "100" cap top above data axis (small)
    BAR_XTICK_DROP = 0.103    # bar xtick visible bottom below data axis
    HM_TOTAL_DROP  = 0.180    # heatmap xlabel visible bottom below data axis

    BAR_DATA_TOP = OUTER_TOP - YTICK_OVERHANG
    BAR_DATA_BOT = OUTER_BOT + BAR_XTICK_DROP

    # Heatmap occupies the FULL right-side stack: its xlabel visible bottom
    # is aligned with the legend frame's visible bottom so the left side has
    # no whitespace under the xlabel.  LEG_FRAME_BOT_OFFSET accounts for the
    # legend's internal borderpad inside ax_leg.
    LEG_FRAME_BOT_OFFSET = 0.04
    HM_DATA_BOT  = LEG_BOT + LEG_FRAME_BOT_OFFSET + HM_TOTAL_DROP
    HM_DATA_TOP  = 0.875                       # leaves room above for macro recall

    HM_L     = 0.06
    HM_W     = 0.16
    CB_GAP   = 0.012
    CB_W     = 0.013
    BAR_L    = HM_L + HM_W + CB_GAP + CB_W + 0.13
    BAR_R    = 0.99
    BAR_W    = BAR_R - BAR_L

    # ── Confusion matrix ─────────────────────────────────────────────────
    ax  = fig.add_axes([HM_L, HM_DATA_BOT, HM_W, HM_DATA_TOP - HM_DATA_BOT])
    cax = fig.add_axes([HM_L + HM_W + CB_GAP, HM_DATA_BOT, CB_W, HM_DATA_TOP - HM_DATA_BOT])
    M = np.array([[conf[a][b] for b in angles] for a in angles])
    im = ax.imshow(M, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(angles))); ax.set_xticklabels(angles)
    ax.set_yticks(range(len(angles))); ax.set_yticklabels(angles)
    ax.set_xlabel("Router prediction")
    ax.set_ylabel("Oracle angle")
    ax.tick_params(axis="both", which="both", length=0)
    for i in range(len(angles)):
        for j in range(len(angles)):
            v = M[i, j]
            color = "white" if v > 0.55 else "#222"
            ax.text(j, i, f"{v*100:.0f}%",
                    ha="center", va="center", fontsize=FS, color=color)
    cb = fig.colorbar(im, cax=cax, ticks=[0.0, 0.5, 1.0])
    cb.ax.tick_params(labelsize=FS)

    # macro recall annotation — anchored at figure level so its top
    # edge sits exactly at OUTER_TOP, matching the bar chart's "100" tick top.
    diag_acc = sum(M[i, i] for i in range(len(angles))) / len(angles)
    fig.text(HM_L + HM_W / 2, OUTER_TOP,
             f"macro recall = {diag_acc*100:.1f}%",
             ha="center", va="top",
             fontsize=FS, color="#1f4e79", weight="bold")

    # ── Grouped bar: per-query-type per-strategy ────────────────────────
    ax2     = fig.add_axes([BAR_L, BAR_DATA_BOT, BAR_W, BAR_DATA_TOP - BAR_DATA_BOT])
    ax_leg  = fig.add_axes([BAR_L, LEG_BOT, BAR_W, LEG_H])
    ax_leg.axis("off")
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
    ax2.set_ylim(0, 100)
    ax2.set_yticks([0, 50, 100])
    ax2.grid(True, axis="y", alpha=0.25, linewidth=0.6)

    # Legend hosted in dedicated axis directly under right panel
    handles, labels = ax2.get_legend_handles_labels()
    ax_leg.legend(handles, labels, loc="center",
                  ncol=5, frameon=True, framealpha=0.92,
                  fontsize=FS, columnspacing=1.1, handlelength=1.4,
                  handletextpad=0.45, borderpad=0.35)

    fig.savefig(OUT_PDF, bbox_inches="tight")
    fig.savefig(OUT_PNG, bbox_inches="tight", dpi=180)
    print(f"[RQ5] wrote {OUT_PDF.name} and {OUT_PNG.name}")


if __name__ == "__main__":
    main()
