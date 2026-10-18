"""Run KnowledgeClaw (or an ablation) on OmniKE-Bench and score it.

    python -m evaluation.run                                        # KnowledgeClaw, all personas
    python -m evaluation.run --method knowledgeclaw_wo_c1 --persona healthcare
    python -m evaluation.run --method knowledgeclaw,knowledgeclaw_wo_c1,knowledgeclaw_wo_c2,knowledgeclaw_wo_c3
    python -m evaluation.run --score-only                           # re-score saved predictions

Per persona: build over the base corpus; every non-T5 item is answered; T5 answers every phase-A question,
commits every increment, then answers every phase-B question, so no phase-A answer can see an increment.
Predictions go to results/<method>/<persona>/<task>.jsonl, the artefact to results/<method>/<persona>/artefact/.
Task scores pool every persona's items; a persona's score averages the tasks it instantiates; OKE averages
the ten persona scores.
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor

from evaluation import data as D
from evaluation import scoring as S
from evaluation.method import METHODS

OUT = D.ROOT / "results"


def pmap(fn, xs, workers):
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(fn, xs))


def run_persona(key: str, persona: str, workers: int) -> None:
    todo = {t: D.items(t, persona) for t in D.TASKS}
    todo = {t: v for t, v in todo.items() if v}
    if not todo:
        return
    out = OUT / key / persona
    out.mkdir(parents=True, exist_ok=True)
    m = METHODS[key](out / "artefact")
    t0 = time.time()
    m.build(persona, list(D.corpus(persona)))
    t_build = time.time() - t0
    print(f"[{key}/{persona}] built in {t_build:.0f}s", flush=True)
    for task, pairs in todo.items():
        if task != "T5":
            preds = pmap(lambda qg: m.answer(qg[0]), pairs, workers)
        else:
            a = pmap(lambda qg: m.answer(qg[0], "A"), pairs, workers)
            seen, delta = set(), []
            for q, _ in pairs:
                for d in D.delta_docs(q):
                    if d.file not in seen:
                        seen.add(d.file)
                        delta.append(d)
            m.commit(delta)
            b = pmap(lambda qg: m.answer(qg[0], "B"), pairs, workers)
            preds = [{"A": x, "B": y} for x, y in zip(a, b)]
        rows = [{"qid": g["qid"], "pred": p, "score": S.score(g, p)} for (q, g), p in zip(pairs, preds)]
        (out / f"{task}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        print(f"[{key}/{persona}] {task}: n={len(rows)} score={100 * sum(r['score'] for r in rows) / len(rows):.1f}",
              flush=True)
    (out / "timing.json").write_text(json.dumps({"build_s": round(t_build, 1), "total_s": round(time.time() - t0, 1)}))


def scores(key: str) -> tuple[dict, dict]:
    task_rows = {t: [] for t in D.TASKS}
    persona = {}
    for p in D.PERSONAS:
        per = []
        for t in D.TASKS:
            f = OUT / key / p / f"{t}.jsonl"
            if not f.exists():
                continue
            gold = {g["qid"]: g for _, g in D.items(t, p)}
            rows = [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
            s = [S.score(gold[r["qid"]], r["pred"]) for r in rows if r["qid"] in gold]
            if s:
                task_rows[t] += s
                per.append(100 * sum(s) / len(s))
        if per:
            persona[p] = sum(per) / len(per)
    task = {t: 100 * sum(v) / len(v) for t, v in task_rows.items() if v}
    return task, persona


def table(keys: list[str]) -> None:
    print("\n| Method | " + " | ".join(D.TASKS) + " | " + " | ".join(D.SHORT[p] for p in D.PERSONAS) + " | OKE |")
    print("|---|" + "---:|" * (len(D.TASKS) + len(D.PERSONAS) + 1))
    for key in keys:
        task, persona = scores(key)
        cells = [f"{task[t]:.1f}" if t in task else "-" for t in D.TASKS]
        cells += [f"{persona[p]:.1f}" if p in persona else "-" for p in D.PERSONAS]
        cells.append(f"{sum(persona.values()) / len(persona):.1f}" if len(persona) == len(D.PERSONAS) else "-")
        print(f"| {key} | " + " | ".join(cells) + " |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="knowledgeclaw", help="comma-separated: " + ", ".join(METHODS))
    ap.add_argument("--persona", default="all")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--score-only", action="store_true")
    a = ap.parse_args()
    keys = a.method.split(",")
    personas = D.PERSONAS if a.persona == "all" else a.persona.split(",")
    if not a.score_only:
        for key in keys:
            for p in personas:
                run_persona(key, p, a.workers)
    table(keys)


if __name__ == "__main__":
    main()
