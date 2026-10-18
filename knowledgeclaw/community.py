"""C3, the community tier: K projected onto a weighted entity graph, w(u, v) = number of hyperedges u and v
share; Louvain (fixed seed) partitions it; the model writes one report per community from its propositions
(the texts of its members' record hyperedges) and modalities; reports are embedded into I_C. Reports are derived data: every edit invalidates them and they
are rebuilt on the next query that needs them.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from itertools import combinations

from knowledgeclaw.hypergraph import Hypergraph

MAX_COMMUNITIES = 64
MIN_SIZE = 2
MAX_PROPS = 12
MAX_MEMBERS = 40            # a hyperedge with more members than this adds no pairwise weight (hubs)

REPORT_PROMPT = """You are summarising one community of a personal knowledge hypergraph.
Entities: {entities}
Propositions:
{props}
Modalities: {mods}
Write a report of at most 200 words, plain prose: (1) THEME, one sentence; (2) KEY ENTITIES, the 3-5 most \
central and their role; (3) SUMMARY of the facts and relationships; (4) CROSS-MODAL NOTES if several \
modalities occur. No generic boilerplate."""


def build(kg: Hypergraph) -> list[dict]:
    import networkx as nx
    from networkx.algorithms.community import louvain_communities
    g = nx.Graph()
    for x in kg.edges.values():
        if x.kind != "record" or len(x.members) > MAX_MEMBERS:
            continue
        for a, b in combinations(sorted(set(x.members)), 2):
            w = g.get_edge_data(a, b, {"weight": 0})["weight"]
            g.add_edge(a, b, weight=w + 1)
    if g.number_of_edges() == 0:
        return []
    comms = sorted(louvain_communities(g, weight="weight", seed=0), key=len, reverse=True)
    comms = [sorted(c) for c in comms if len(c) >= MIN_SIZE][:MAX_COMMUNITIES]

    def report(i_c):
        i, c = i_c
        props, mods, seen = [], set(), set()
        for e in c:
            for h in kg.ent_edges.get(e, ()):
                x = kg.edges.get(h)
                if x is None or h in seen or x.kind != "record":
                    continue
                seen.add(h)
                props.append(f"- {x.text}")
                mods.add(x.modality)
                if len(props) >= MAX_PROPS:
                    break
            if len(props) >= MAX_PROPS:
                break
        names = ", ".join(kg.entities[e].name for e in c[:15] if e in kg.entities)
        text = kg.llm.chat([{"role": "user", "content": REPORT_PROMPT.format(
            entities=names, props="\n".join(props), mods=", ".join(sorted(mods)))}],
            json_mode=False, max_tokens=400) or f"Community of {names}."
        files = sorted({f for e in c for f in kg.entity_files(e)})
        return {"id": f"c{i:03d}", "members": c, "files": files, "modalities": sorted(mods), "report": text.strip()}

    with ThreadPoolExecutor(16) as ex:
        reps = list(ex.map(report, enumerate(comms)))
    kg.I_C.drop(list(kg.I_C.row))
    kg.I_C.add([r["id"] for r in reps], [r["report"] for r in reps])
    return reps
