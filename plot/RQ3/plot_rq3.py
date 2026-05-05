"""RQ3 figure: Where does omnimodality help?

Two-panel figure:
  (a) Persona x Task heatmap of KC - max(baseline), diverging colormap.
  (b) Modality x Task heatmap of full_KC - ablated_KC (drop when this
      modality is removed), sequential colormap.

Reads:  plot/RQ3/rq3_data.json
Writes: plot/RQ3/fig_rq3.pdf, plot/RQ3/fig_rq3.png
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

HERE = Path(__file__).resolve().parent
DATA = HERE / "rq3_data.json"
OUT_PDF = HERE / "fig_rq3.pdf"
OUT_PNG = HERE / "fig_rq3.png"

FS = 18  # one font size for the whole figure

P_SHORT = {
    "Alice_Healthcare": "Healthcare",
    "Bob_Finance":      "Finance",
    "Carol_Education":  "Education",
    "Dave_Law":         "Law",
    "Eve_Psychology":   "Psychology",
    "Frank_Music":      "Music",
    "Grace_Game":       "Game",
    "Heidi_Building":   "Building",
    "Ivan_Software":    "Software",
    "Judy_Robotics":    "Robotics",
}


def main():
    d = json.loads(DATA.read_text())
    tasks = d["tasks"]
    personas = d["personas"]
    mods = d["modalities"]

    # (a) Δ matrix: rows = personas, cols = tasks
    A = np.array([[d["delta"][p][t] for t in tasks] for p in personas], dtype=float)

    # (b) drop matrix: rows = modalities, cols = tasks
    B = np.array([[d["modality_drop"][m][t] for t in tasks] for m in mods], dtype=float)

    plt.rcParams.update({
        "font.family":      "Georgia",
        "font.size":        FS,
        "axes.titlesize":   FS,
        "axes.labelsize":   FS,
        "xtick.labelsize":  FS,
        "ytick.labelsize":  FS,
        "legend.fontsize":  FS,
    })

    fig = plt.figure(figsize=(13.0, 2.9))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.65, 1.0], wspace=0.30,
                          left=0.07, right=0.985, top=0.97, bottom=0.10)

    # ── Panel (a) ────────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    vmax = max(abs(A.min()), abs(A.max()))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    im1 = ax1.imshow(A, cmap="RdBu_r", norm=norm, aspect="auto")
    ax1.set_xticks(range(len(tasks))); ax1.set_xticklabels(tasks)
    ax1.set_yticks(range(len(personas)))
    ax1.set_yticklabels([P_SHORT[p] for p in personas])
    ax1.tick_params(axis="both", which="both", length=0)

    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            v = A[i, j]
            color = "white" if abs(v) > 0.55 * vmax else "#222222"
            ax1.text(j, i, f"{v:+.1f}", ha="center", va="center",
                     fontsize=FS, color=color)

    cbar1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.02)
    cbar1.ax.tick_params(labelsize=FS)

    # ── Panel (b) ────────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(B, cmap="YlOrRd", aspect="auto", vmin=0, vmax=max(B.max(), 1.0))
    ax2.set_xticks(range(len(tasks))); ax2.set_xticklabels(tasks)
    ax2.set_yticks(range(len(mods))); ax2.set_yticklabels([m.upper() for m in mods])
    ax2.tick_params(axis="both", which="both", length=0)
    for i in range(B.shape[0]):
        for j in range(B.shape[1]):
            v = B[i, j]
            color = "white" if v > 0.6 * B.max() else "#222222"
            ax2.text(j, i, f"{v:.1f}", ha="center", va="center",
                     fontsize=FS, color=color)
    cbar2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.02)
    cbar2.ax.tick_params(labelsize=FS)

    fig.savefig(OUT_PDF, bbox_inches="tight")
    fig.savefig(OUT_PNG, bbox_inches="tight", dpi=180)
    print(f"[RQ3] wrote {OUT_PDF.name} and {OUT_PNG.name}")


if __name__ == "__main__":
    main()
