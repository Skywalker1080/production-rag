"""Eval on the user's golden set (data/Eval Set, 2 x 10 probes).

RAGAS (GLM 5 judge) + deterministic checks:
  - number-match: expected numeric tokens present in answer (sign-insensitive)
  - page-hit: gold pdf_page in retrieved pages (exact + off-by-one)
Usage: uv run python evals/run_user_eval.py
"""
import datetime
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.logging_setup import setup_logging

setup_logging()

EVAL_DIR = r"C:\Projects\RAG\data\Eval Set"
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def load_user_golden():
    probes = []
    for part, fname in (("p1", "golden_dataset_jiofin_part_1.json"),
                        ("p2", "golden_dataset_jiofin_part2.json")):
        d = json.load(open(os.path.join(EVAL_DIR, fname), encoding="utf-8"))
        for q in d["questions"]:
            probes.append({
                "id": f"{part}-{q['id']}",
                "type": q.get("type", "?"),
                "difficulty": q.get("difficulty", "?"),
                "question": q["question"],
                "ground_truth": q["expected_answer"],
                "gold_pages": [q["citation"]["pdf_page"]],
                "section": q["citation"].get("section", ""),
            })
    return probes


def num_tokens(s: str) -> set:
    """Digit sequences, commas stripped, abs values (parens = negatives in
    statements but formatting varies, so compare magnitude only)."""
    out = set()
    for t in re.findall(r"[\d,]+\.?\d*", s):
        t = t.replace(",", "").strip().rstrip(".")
        if not t or not any(c.isdigit() for c in t):
            continue
        try:
            out.add(str(float(t)))
        except ValueError:
            continue
    return out


def run_pipeline(probes, snapshot, cached):
    from app import rag

    rows = []
    for g in probes:
        if g["id"] in cached:
            rows.append(cached[g["id"]])
            print(f"  cached {g['id']}", flush=True)
            continue
        answer, sources = rag.query(g["question"])
        row = {
            **{k: g[k] for k in ("id", "type", "difficulty", "question",
                                 "ground_truth", "gold_pages", "section")},
            "answer": answer,
            "contexts": [s["content"] for s in sources],
            "got_pages": sorted({s["metadata"].get("page") for s in sources}),
        }
        rows.append(row)
        cached[g["id"]] = row
        save_cache(snapshot, cached)  # crash-safe: every probe persisted
        print(f"  answered {g['id']} (pages={row['got_pages']})", flush=True)
    return rows


def score(rows):
    from ragas import EvaluationDataset, SingleTurnSample, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (answer_relevancy, context_precision,
                               context_recall, faithfulness)

    from app import rag

    samples = [SingleTurnSample(
        user_input=r["question"], retrieved_contexts=r["contexts"],
        response=r["answer"], reference=r["ground_truth"]) for r in rows]
    return evaluate(
        EvaluationDataset(samples=samples),
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=LangchainLLMWrapper(rag.get_llm()),
        embeddings=LangchainEmbeddingsWrapper(rag.get_embeddings()),
    )


def config_snapshot():
    from app import config

    return {"embed_provider": config.EMBED_PROVIDER,
            "ollama_embed_model": config.OLLAMA_EMBED_MODEL,
            "hybrid": config.HYBRID_SEARCH, "rerank": config.RERANK,
            "top_k": config.TOP_K,
            "chunk": [config.CHUNK_SIZE, config.CHUNK_OVERLAP],
            "llm": config.BEDROCK_MODEL_ID}


CACHE_PATH = os.path.join(os.path.dirname(__file__), "results",
                           ".cache_user_golden.json")


def load_cache(snapshot):
    """Reuse previously answered probes only if config is identical."""
    try:
        c = json.load(open(CACHE_PATH, encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    if c.get("config") != snapshot:
        print("cache config changed — starting fresh", flush=True)
        return {}
    print(f"cache hit: {len(c.get('answers', {}))} probes reused", flush=True)
    return c.get("answers", {})


def save_cache(snapshot, answers):
    tmp = CACHE_PATH + ".tmp"
    json.dump({"config": snapshot, "answers": answers},
              open(tmp, "w", encoding="utf-8"), ensure_ascii=False)
    os.replace(tmp, CACHE_PATH)


def deterministic(rows):
    checks = {}
    for r in rows:
        exp, ans = num_tokens(r["ground_truth"]), num_tokens(r["answer"])
        num_ok = bool(exp) and exp <= ans
        pages = [p for p in r["got_pages"] if isinstance(p, int)]
        golds = [gp for gp in r["gold_pages"] if isinstance(gp, int)]
        page_exact = bool(golds) and any(p in pages for p in golds)
        page_near = bool(golds) and any(abs(p - gp) <= 1 for p in pages for gp in golds)
        checks[r["id"]] = {"number_match": num_ok, "page_exact": page_exact,
                           "page_pm1": page_exact or page_near}
    return checks


def main():
    from app import config

    probes = load_user_golden()
    only = None
    if "--ids" in sys.argv:
        only = sys.argv[sys.argv.index("--ids") + 1].split(",")
        probes = [p for p in probes if p["id"] in only]
    print(f"user probes: {len(probes)}", flush=True)
    snapshot = config_snapshot()
    cached = load_cache(snapshot)
    rows = run_pipeline(probes, snapshot, cached)
    result = score(rows)
    checks = deterministic(rows)

    df = result.to_pandas()
    cols = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    scores = {c: round(float(df[c].mean()), 4) for c in cols if c in df.columns}
    agg = {
        "number_match": round(sum(v["number_match"] for v in checks.values()) / len(checks), 4),
        "page_exact": round(sum(v["page_exact"] for v in checks.values()) / len(checks), 4),
        "page_pm1": round(sum(v["page_pm1"] for v in checks.values()) / len(checks), 4),
    }
    payload = {
        "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "suite": "user-golden-jiofin-20",
        "config": snapshot,
        "ragas_means": scores, "deterministic_rates": agg,
        "per_probe": [{**{k: r[k] for k in ("id", "type", "difficulty", "gold_pages",
                                            "got_pages", "section")},
                       **checks[r["id"]],
                       "answer": r["answer"][:400]} for r in rows],
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"user-golden-{len(probes)}-{payload['ts'][:10]}.json".replace(":", ""))
    json.dump(payload, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    print("\n=== RAGAS means (user set) ===")
    for k, v in scores.items():
        print(f"  {k}: {v}")
    print("=== deterministic rates ===")
    for k, v in agg.items():
        print(f"  {k}: {v}")
    print("=== per-probe (id, nums, page, pages) ===")
    for r in rows:
        c = checks[r["id"]]
        print(f"  {r['id']}: num={c['number_match']} page={c['page_exact']}/{c['page_pm1']} "
              f"gold={r['gold_pages']} got={r['got_pages']}")
    print(f"saved {path}")


if __name__ == "__main__":
    main()
