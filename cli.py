#!/usr/bin/env python
"""KnowledgeClaw CLI — command-line interface for the knowledge engineering system."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def _make_kc(args):
    from knowledgeclaw import KnowledgeClaw
    storage = getattr(args, "storage", None)
    return KnowledgeClaw(kb_dir=args.kb, storage_dir=storage)


def _dump(obj):
    print(json.dumps(obj, indent=2, ensure_ascii=False))


# ---------------------------------------------------------------------------
# Construction / Update / Delete (C2 lifecycle)
# ---------------------------------------------------------------------------

def cmd_ingest(args):
    kc = _make_kc(args)
    _dump(kc.lifecycle.construct(args.folder))


def cmd_update(args):
    kc = _make_kc(args)
    _dump(kc.lifecycle.update(args.file, force=args.force))


def cmd_delete(args):
    kc = _make_kc(args)
    _dump(kc.lifecycle.delete(args.source))


def cmd_consistency(args):
    kc = _make_kc(args)
    _dump(kc.consistency_check())


def cmd_communities(args):
    kc = _make_kc(args)
    if args.list:
        for r in kc.list_communities():
            print(f"[{r['community_id']}] size={r['size']} modalities={r.get('modalities', [])}")
            print(f"  {r['report'][:200]}")
            print()
    else:
        _dump(kc.lifecycle.rebuild_communities())


# ---------------------------------------------------------------------------
# Querying / Reasoning (C3)
# ---------------------------------------------------------------------------

def cmd_query(args):
    kc = _make_kc(args)
    result = kc.lifecycle.query(args.question, top_k=args.top_k)["result"]
    print(result["answer"])
    print(f"\n[intent: {result.get('intent', 'unknown')}, "
          f"chunks: {result.get('chunks_used', 0)}, "
          f"communities: {result.get('communities_used', 0)}]")
    if args.sources and result.get("sources"):
        print("\nSources:")
        for s in result["sources"]:
            print(f"  - {s}")


def cmd_infer(args):
    kc = _make_kc(args)
    result = kc.lifecycle.reason(args.topic, max_rounds=args.max_rounds)["result"]
    print(result["insight"])
    if args.verbose:
        print("\nQueries used:")
        for q in result.get("queries_used", []):
            print(f"  - {q}")
        print("\nSources:")
        for s in result.get("sources", []):
            print(f"  - {s}")


def cmd_research(args):
    kc = _make_kc(args)
    result = kc.research(args.topic)
    print(f"Coverage: {result.get('coverage', 'unknown')}\n")
    print(result.get("current_knowledge", ""))
    if result.get("gaps"):
        print("\nKnowledge Gaps:")
        for g in result["gaps"]:
            print(f"  - {g}")
    if result.get("suggestions"):
        print("\nSuggestions:")
        for s in result["suggestions"]:
            print(f"  - {s}")


def cmd_eval(args):
    kc = _make_kc(args)
    _dump(kc.evaluate())


def cmd_stats(args):
    kc = _make_kc(args)
    stats = kc.stats()
    max_key_len = max((len(k) for k in stats), default=10)
    for k, v in stats.items():
        print(f"  {k:<{max_key_len}} : {v}")


def cmd_history(args):
    kc = _make_kc(args)
    for ev in kc.lifecycle.history(limit=args.limit):
        print(f"[{ev['ts']}] {ev['stage']:13s} ok={ev['ok']} "
              f"elapsed={ev['elapsed_s']}s params={ev['params']}")


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------

def cmd_bench(args):
    from benchmarks.run_table1 import main as bench_main
    bench_main(out_dir=args.out, methods=args.methods, sample_size=args.sample_size)


def cmd_bench_generate(args):
    from benchmarks.generate import generate_all
    generate_all(out_dir=args.out, n_per_split=args.size)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="KnowledgeClaw — closed-loop omnimodal knowledge hypergraph CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli.py ingest ./my_docs
  python cli.py update ./new_paper.pdf
  python cli.py delete old_paper.pdf
  python cli.py consistency
  python cli.py communities          # rebuild
  python cli.py communities --list   # show summaries
  python cli.py query "What did I read about contrastive learning?"
  python cli.py infer "Future of multimodal RAG"
  python cli.py history
  python cli.py bench --out ./results --methods all
  python cli.py bench-generate --out benchmarks/data --size 50
""",
    )
    parser.add_argument("--kb", default="./kb", metavar="DIR",
                        help="Knowledge base root directory (default: ./kb)")
    parser.add_argument("--storage", default=None, metavar="DIR",
                        help="Storage directory for index files (default: kb/.knowledgeclaw)")

    sub = parser.add_subparsers(dest="command", metavar="COMMAND")
    sub.required = True

    # ingest / update / delete / consistency
    p = sub.add_parser("ingest", help="Ingest a folder or file (CONSTRUCTION stage)")
    p.add_argument("folder", help="Path to folder or file to ingest")
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("update", help="Incrementally add a single file (UPDATE stage)")
    p.add_argument("file", help="Path to file")
    p.add_argument("--force", action="store_true", help="Re-ingest even if already processed")
    p.set_defaults(func=cmd_update)

    p = sub.add_parser("delete", help="Delete a source from the hypergraph (UPDATE stage)")
    p.add_argument("source", help="Source filename to delete (e.g. paper.pdf)")
    p.set_defaults(func=cmd_delete)

    p = sub.add_parser("consistency", help="Run consistency check / orphan removal")
    p.set_defaults(func=cmd_consistency)

    p = sub.add_parser("communities", help="Rebuild or list community summaries (C3 coarse tier)")
    p.add_argument("--list", action="store_true", help="List existing community summaries instead of rebuilding")
    p.set_defaults(func=cmd_communities)

    # query / infer / research
    p = sub.add_parser("query", help="Hierarchical Q&A (C3) over the hypergraph")
    p.add_argument("question", help="Your question")
    p.add_argument("--top-k", type=int, default=10, metavar="N",
                   help="Number of chunks to retrieve (default: 10)")
    p.add_argument("--sources", action="store_true", help="Show source files")
    p.set_defaults(func=cmd_query)

    p = sub.add_parser("infer", help="ReACT-style inference / 推演")
    p.add_argument("topic", help="Topic to infer about")
    p.add_argument("--max-rounds", type=int, default=5)
    p.add_argument("--verbose", "-v", action="store_true", help="Show queries and sources")
    p.set_defaults(func=cmd_infer)

    p = sub.add_parser("research", help="Analyse KB coverage and identify knowledge gaps")
    p.add_argument("topic", help="Topic to analyse")
    p.set_defaults(func=cmd_research)

    p = sub.add_parser("eval", help="Evaluate KB health")
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("stats", help="Show KB statistics")
    p.set_defaults(func=cmd_stats)

    p = sub.add_parser("history", help="Show lifecycle audit log")
    p.add_argument("--limit", type=int, default=20)
    p.set_defaults(func=cmd_history)

    # benchmarks
    p = sub.add_parser("bench", help="Run the PAPER.md Table 1 benchmark harness")
    p.add_argument("--out", default="./results", metavar="DIR")
    p.add_argument("--methods", default="all",
                   help="Comma-separated list (default: all). "
                        "Options: NaiveRAG, GraphRAG-Edge, GraphRAG-Community, "
                        "HyperGraphRAG, MemGPT, Mem0, KarpathyKB, "
                        "KCwoC1, KCwoC2, KCwoC3, KnowledgeClaw, all")
    p.add_argument("--sample-size", type=int, default=0,
                   help="Subsample N items per benchmark split (0=full)")
    p.set_defaults(func=cmd_bench)

    p = sub.add_parser("bench-generate", help="Generate the synthetic benchmark splits")
    p.add_argument("--out", default="./benchmarks/data", metavar="DIR")
    p.add_argument("--size", type=int, default=50, help="Items per split")
    p.set_defaults(func=cmd_bench_generate)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
