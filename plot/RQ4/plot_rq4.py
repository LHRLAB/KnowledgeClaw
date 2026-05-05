"""RQ4 figure: lifecycle drift over a 200-event CRUD stream.

Left  (large): T5(t) line plot — six methods, 5-seed mean ± 1 std shaded band.
Right (small): per-method, per-event-type average score impact (grouped bars).

Reads:  plot/RQ4/rq4_data.json
Writes: plot/RQ4/fig_rq4.{pdf, png}
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
DATA = HERE / "rq4_data.json"
OUT_PDF = HERE / "fig_rq4.pdf"
OUT_PNG = HERE / "fig_rq4.png"

FS = 18  # one font size for the whole figure


def main():
    d = json.loads(DATA.read_text())
    methods = d["methods"]
    meta = d["method_meta"]
    mean = d["mean"]
    std = d["std"]
    events_x = np.array(d["events"])
    bar = d["by_event_type"]
    evtypes = d["event_types"]

    plt.rcParams.update({
        "font.family":      "Georgia",
        "font.size":        FS,
        "axes.titlesize":   FS,
        "axes.labelsize":   FS,
        "xtick.labelsize":  FS,
        "ytick.labelsize":  FS,
        "legend.fontsize":  FS,
    })

    fig = plt.figure(figsize=(13.0, 3.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.0, 1.25], wspace=0.20,
                          left=0.065, right=0.99, top=0.94, bottom=0.36)

    # ── Panel: T5(t) trajectory ─────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    for m in methods:
        c = meta[m]["color"]; lw = meta[m]["linewidth"]
        mu = np.array(mean[m]); sd = np.array(std[m])
        ax.plot(events_x, mu, label=m, color=c, linewidth=lw,
                zorder=3 if m == "KC" else 2)
        ax.fill_between(events_x, mu - sd, mu + sd, color=c, alpha=0.13, linewidth=0)

    ax.set_xlabel("Number of CRUD events")
    ax.set_ylabel("T5 consistency score")
    ax.set_xlim(0, max(events_x))
    ax.set_ylim(0, 60)
    ax.grid(True, alpha=0.25, linewidth=0.6)

    # In-plot annotation: highlight final gap between KC and best memory baseline
    final_kc = mean["KC"][-1]
    final_amem = mean["A-MEM"][-1]
    ax.annotate(f"KC vs. A-MEM: +{final_kc - final_amem:.1f}",
                xy=(events_x[-1] - 4, final_kc + 1.2),
                ha="right", va="bottom",
                fontsize=FS, weight="bold", color="#c0392b")

    # ── Panel: per-event-type average impact ────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    width = 0.14
    xs = np.arange(len(evtypes))
    for i, m in enumerate(methods):
        c = meta[m]["color"]
        ys = [bar[m][ev] for ev in evtypes]
        ax2.bar(xs + (i - 2.5) * width, ys, width, label=m, color=c,
                edgecolor="white", linewidth=0.4)
    ax2.set_xticks(xs)
    ax2.set_xticklabels([e.capitalize() for e in evtypes], rotation=0)
    ax2.set_ylabel("avg |ΔT5| per event")
    ax2.grid(True, axis="y", alpha=0.25, linewidth=0.6)

    # Figure-level legend (one row at bottom, shared by both panels)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center",
               bbox_to_anchor=(0.5, 0.02),
               ncol=6, frameon=True, framealpha=0.92,
               fontsize=FS, columnspacing=1.4, handlelength=1.8)

    fig.savefig(OUT_PDF, bbox_inches="tight")
    fig.savefig(OUT_PNG, bbox_inches="tight", dpi=180)
    print(f"[RQ4] wrote {OUT_PDF.name} and {OUT_PNG.name}")


if __name__ == "__main__":
    main()
