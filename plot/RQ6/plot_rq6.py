"""RQ6 figure: cost-performance Pareto + cost-component breakdown.

Left  : scatter, x = USD per persona (log scale), y = OKE.
        KC drawn as a starred marker on the Pareto frontier.
Right : stacked horizontal bars of per-method cost components for ~6
        representative methods.

Reads:  plot/RQ6/rq6_data.json
Writes: plot/RQ6/fig_rq6.{pdf, png}
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
DATA = HERE / "rq6_data.json"
OUT_PDF = HERE / "fig_rq6.pdf"
OUT_PNG = HERE / "fig_rq6.png"

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
    "knowledgeclaw": "#c0392b",
}

COMP_COLOR = {
    "llm":   "#2c3e50",
    "embed": "#3498db",
    "store": "#1abc9c",
    "misc":  "#e67e22",
}


def main():
    d = json.loads(DATA.read_text())
    rows = d["rows"]
    pareto = set(d["pareto"])
    reps = d["representatives_for_breakdown"]

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    })

    fig = plt.figure(figsize=(13.0, 4.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.55, 1.0], wspace=0.28,
                          left=0.06, right=0.985, top=0.88, bottom=0.16)

    # ── Pareto scatter ──────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    for r in rows:
        x = r["cost_total"]; y = r["OKE"]; k = r["key"]; name = r["display"]
        if k == "knowledgeclaw":
            ax.scatter([x], [y], s=320, marker="*", color=GROUP_COLOR[k],
                       edgecolor="white", linewidth=1.6, zorder=5,
                       label="KnowledgeClaw")
        else:
            ax.scatter([x], [y], s=70, color=GROUP_COLOR.get(k, "#888"),
                       edgecolor="white", linewidth=0.8, zorder=3,
                       alpha=0.92)
        # label every point with the display name (small offset so they don't
        # collide; KC and pareto-frontier methods get bold)
        weight = "bold" if k in pareto or k == "knowledgeclaw" else "normal"
        dy = 1.2 if k == "knowledgeclaw" else 0.8
        ax.annotate(name, xy=(x, y), xytext=(5, dy),
                    textcoords="offset points",
                    fontsize=8.2, weight=weight,
                    color="#222")

    # Pareto frontier line: walk by cost ascending, plot points whose key is in pareto
    pf = sorted([r for r in rows if r["key"] in pareto], key=lambda r: r["cost_total"])
    ax.plot([r["cost_total"] for r in pf], [r["OKE"] for r in pf],
            "--", color="#aaaaaa", linewidth=1.2, zorder=2,
            label="Pareto frontier")

    ax.set_xscale("log")
    ax.set_xlabel("Cost per persona (USD, log scale)")
    ax.set_ylabel("OKE Score")
    ax.set_title("(a) OKE vs end-to-end cost per persona")
    ax.set_xlim(0.18, 4.5)
    ax.set_ylim(0, 70)
    ax.grid(True, which="both", alpha=0.22, linewidth=0.6)
    ax.legend(loc="lower right", frameon=True, framealpha=0.92, fontsize=9)

    # ── Cost-component stacked bar ──────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    rep_rows = [next(r for r in rows if r["key"] == k) for k in reps]
    labels = [r["display"] for r in rep_rows]
    comps = ["llm", "embed", "store", "misc"]
    arrays = {c: np.array([r["cost_breakdown"][c] for r in rep_rows]) for c in comps}

    y_pos = np.arange(len(rep_rows))
    left = np.zeros(len(rep_rows))
    for c in comps:
        ax2.barh(y_pos, arrays[c], left=left, color=COMP_COLOR[c],
                 edgecolor="white", linewidth=0.5, label=c)
        left += arrays[c]

    # Total cost annotation
    for i, r in enumerate(rep_rows):
        ax2.text(left[i] + 0.04, i, f"${r['cost_total']:.2f}",
                 ha="left", va="center", fontsize=9, color="#222")

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(labels)
    ax2.invert_yaxis()
    ax2.set_xlabel("Cost (USD per persona)")
    ax2.set_title("(b) Cost breakdown across representative methods")
    ax2.set_xlim(0, max(left) * 1.18)
    ax2.legend(loc="lower right", ncol=4, frameon=True, framealpha=0.92,
               fontsize=8.5, columnspacing=1.0, handlelength=1.4)
    ax2.grid(True, axis="x", alpha=0.22, linewidth=0.6)

    fig.suptitle("RQ6  ·  KnowledgeClaw is the rightmost point on the Pareto frontier — top OKE, modest cost markup",
                 y=0.985, fontsize=11.5, weight="bold")

    fig.savefig(OUT_PDF, bbox_inches="tight")
    fig.savefig(OUT_PNG, bbox_inches="tight", dpi=180)
    print(f"[RQ6] wrote {OUT_PDF.name} and {OUT_PNG.name}")


if __name__ == "__main__":
    main()
