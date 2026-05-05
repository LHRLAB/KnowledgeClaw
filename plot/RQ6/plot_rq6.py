"""RQ6 figure: cost-OKE Pareto + construction-time bars at full-coverage tier.

Left  : scatter, x = USD per persona (log scale), y = OKE.  KC drawn as a
        starred marker on the Pareto frontier.  Because KC is purpose-built
        for KE (C1 structured extraction skips redundant LLM calls, C2
        incremental updates avoid full rebuilds, C3 routing avoids over-
        invocation at query time), KC sits to the LEFT of the agentic-KE
        cluster while still ranking highest in OKE.
Right : horizontal bars of construction time for the full-coverage tier
        (methods that score on every T1-T6).  Sorted by time; KC is the
        fastest in the tier and is annotated with cost / OKE.

Reads:  plot/RQ6/rq6_data.json
Writes: plot/RQ6/fig_rq6.{pdf, png}
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter
from adjustText import adjust_text

HERE = Path(__file__).resolve().parent
DATA = HERE / "rq6_data.json"
OUT_PDF = HERE / "fig_rq6.pdf"
OUT_PNG = HERE / "fig_rq6.png"

FS = 18

KC_RED = "#c0392b"
PEER_GRAY = "#8a96a3"

GROUP_COLOR = {
    "naive_rag": "#7f8c8d", "raptor": "#7f8c8d",
    "graphrag": "#2980b9", "lightrag": "#2980b9",
    "hypergraphrag": "#16a085", "ontokg": "#16a085", "itext2kg": "#16a085",
    "hipporag2": "#9b59b6",
    "kag": "#f39c12", "graphiti": "#f39c12",
    "memgpt": "#e67e22", "mem0": "#e67e22", "amem": "#e67e22",
    "omnirag": "#34495e",
    "openclaw": "#1f4e79", "cognee": "#1f4e79", "llm_wiki": "#1f4e79", "graphify": "#1f4e79",
    "kc_no_c1": "#bdc3c7",
    "knowledgeclaw": KC_RED,
}

# Methods to label on the scatter even if they fall off the Pareto frontier
# (they form the agentic-KE comparison cluster KC dominates).
ALWAYS_LABEL = {"openclaw", "cognee", "llm_wiki", "graphify"}

# Methods hidden from Panel A entirely (e.g., ablation rows that distract
# from the cross-method comparison story).
HIDE_FROM_SCATTER = {"kc_no_c1"}


def main():
    d = json.loads(DATA.read_text())
    rows   = d["rows"]
    by_key = {r["key"]: r for r in rows}
    pareto = set(d["pareto"])
    full_cov = d["full_coverage_methods"]

    plt.rcParams.update({
        "font.family":      "Georgia",
        "font.size":        FS,
        "axes.titlesize":   FS,
        "axes.labelsize":   FS,
        "xtick.labelsize":  FS,
        "ytick.labelsize":  FS,
        "legend.fontsize":  FS,
    })

    fig = plt.figure(figsize=(13.0, 3.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.55, 1.0], wspace=0.28,
                          left=0.06, right=0.99, top=0.96, bottom=0.18)

    # ── Panel A: cost-OKE Pareto scatter ─────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    texts = []
    for r in rows:
        x, y, k, name = r["cost_total"], r["OKE"], r["key"], r["display"]
        if k in HIDE_FROM_SCATTER:
            continue
        if k == "knowledgeclaw":
            ax.scatter([x], [y], s=320, marker="*", color=KC_RED,
                       edgecolor="white", linewidth=1.6, zorder=5,
                       label="KnowledgeClaw")
        else:
            ax.scatter([x], [y], s=70, color=GROUP_COLOR.get(k, "#888"),
                       edgecolor="white", linewidth=0.8, zorder=3, alpha=0.92)
        if k in pareto or k in ALWAYS_LABEL:
            weight = "bold" if k == "knowledgeclaw" else "normal"
            texts.append(ax.text(x, y, name, fontsize=FS, weight=weight, color="#222"))

    pf = sorted([r for r in rows
                 if r["key"] in pareto and r["key"] not in HIDE_FROM_SCATTER],
                key=lambda r: r["cost_total"])
    ax.plot([r["cost_total"] for r in pf], [r["OKE"] for r in pf],
            "--", color="#aaaaaa", linewidth=1.2, zorder=2,
            label="Pareto frontier")

    # KC ceiling line (dotted) emphasises that no method reaches KC's OKE
    kc_row = by_key["knowledgeclaw"]
    ax.axhline(kc_row["OKE"], color=KC_RED, linestyle=":", linewidth=0.9,
               alpha=0.45, zorder=1)

    ax.set_xscale("log")
    ax.set_xlabel("Cost per persona (USD, summed over T1–T6; log scale)")
    ax.set_ylabel("OKE Score")
    ax.set_xlim(0.18, 4.0)
    ax.set_ylim(0, 80)
    ax.grid(True, which="both", alpha=0.22, linewidth=0.6)

    # More log-x ticks so the cost axis is readable
    ax.xaxis.set_major_locator(FixedLocator([0.2, 0.3, 0.5, 1.0, 2.0, 3.0]))
    ax.xaxis.set_major_formatter(FixedFormatter(["0.2", "0.3", "0.5", "1", "2", "3"]))
    ax.xaxis.set_minor_locator(FixedLocator([]))

    adjust_text(texts, ax=ax,
                arrowprops=dict(arrowstyle="-", color="#888", lw=0.7, alpha=0.7),
                expand=(1.6, 1.8), force_text=(0.6, 0.9),
                only_move={"text": "xy"})

    # In-axes legend, top-left — replaces the previous figure-level legend
    ax.legend(loc="upper left", frameon=True, framealpha=0.92,
              fontsize=FS - 3, handlelength=1.4, handletextpad=0.5,
              borderpad=0.4)

    # ── Panel B: construction time at full-coverage tier ─────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    fc_rows = sorted([by_key[k] for k in full_cov],
                     key=lambda r: r["time_construction_min"])
    labels = [r["display"] for r in fc_rows]
    times  = np.array([r["time_construction_min"] for r in fc_rows])

    y_pos  = np.arange(len(fc_rows))
    colors = [KC_RED if r["key"] == "knowledgeclaw" else PEER_GRAY for r in fc_rows]
    ax2.barh(y_pos, times, color=colors, edgecolor="white", linewidth=0.5,
             alpha=0.95)

    # Cost / OKE annotation outside the right edge of each bar
    for i, r in enumerate(fc_rows):
        kc = (r["key"] == "knowledgeclaw")
        col    = KC_RED if kc else "#444"
        weight = "bold" if kc else "normal"
        ax2.text(r["time_construction_min"] + 1.0, i,
                 f"  ${r['cost_total']:.2f}  /  OKE {r['OKE']:.1f}",
                 va="center", fontsize=FS - 4, color=col, weight=weight)

    # KC's lead vs next-fastest peer (vertical dotted line + delta)
    kc_idx = next(i for i, r in enumerate(fc_rows) if r["key"] == "knowledgeclaw")
    next_t = min(t for i, t in enumerate(times) if i != kc_idx)
    ax2.axvline(next_t, color="#777", linestyle=":", linewidth=1.0, zorder=1)
    ax2.annotate(f"−{next_t - times[kc_idx]:.0f} min",
                 xy=(times[kc_idx] - 1.0, kc_idx),
                 ha="right", va="center", color="white",
                 fontsize=FS - 3, weight="bold")

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(labels)
    ax2.invert_yaxis()                        # fastest (KC) at the top
    ax2.set_xlabel("Time per persona (min, T1–T6)")
    ax2.set_xlim(0, times.max() * 1.55)        # leave room for annotation
    ax2.grid(True, axis="x", alpha=0.22, linewidth=0.6)

    fig.savefig(OUT_PDF, bbox_inches="tight")
    fig.savefig(OUT_PNG, bbox_inches="tight", dpi=180)
    print(f"[RQ6] wrote {OUT_PDF.name} and {OUT_PNG.name}")


if __name__ == "__main__":
    main()
