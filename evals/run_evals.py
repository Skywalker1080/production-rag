"""RAGAS evals: score the pipeline on the golden set, save + diff vs last run.

Usage:  uv run python evals/run_evals.py [--probes rev-consol,pat-standalone]

Judge: Bedrock GLM 5 (same model, LLM-as-judge). Judge embeddings: local bge-m3.
Metrics (answer probes): faithfulness, answer_relevancy, context_precision,
context_recall. Refusal probes: rule-checked.
Results: evals/results/<utc-timestamp>.json + stdout table with delta.
"""
import datetime
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.logging_setup import setup_logging

setup_logging()

from evals.golden import GOLDEN  # noqa: E402

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def run_pipeline():
    from app import rag

    rows = []
    for g in GOLDEN:
        answer, sources = rag.query(g["question"])
        rows.append({
            "id": g["id"], "type": g["type"], "question": g["question"],
            "ground_truth": g["ground_truth"], "gold_pages": g["gold_pages"],
            "answer": answer,
            "contexts": [s["content"] for s in sources],
            "got_pages": sorted({s["metadata"].get("page") for s in sources}),
        })
        print(f"  answered {g['id']} (pages={rows[-1]['got_pages']})", flush=True)
    return rows


def score_answer_probes(rows):
    from ragas import EvaluationDataset, SingleTurnSample, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    from app import rag

    samples = [
        SingleTurnSample(
            user_input=r["question"],
            retrieved_contexts=r["contexts"],
            response=r["answer"],
            reference=r["ground_truth"],
        )
        for r in rows if r["type"] == "answer"
    ]
    wllm = LangchainLLMWrapper(rag.get_llm())
    wemb = LangchainEmbeddingsWrapper(rag.get_embeddings())
    result = evaluate(
        EvaluationDataset(samples=samples),
        metrics=[faithfulness, answer_relevancy, context_precision,
                 context_recall],
        llm=wllm,
        embeddings=wemb,
    )
    return result


def check_refusals(rows):
    """Refusal probe passes if the answer declines instead of inventing."""
    markers = ["do not contain", "only answer", "unable to", "cannot answer",
               "cannot write", "not able to", "specialize only",
               "outside that scope", "decline", "specialist in answering"]
    out = {}
    for r in rows:
        if r["type"] == "refuse":
            a = r["answer"].lower()
            out[r["id"]] = any(m in a for m in markers)
    return out


def load_last():
    files = sorted(glob.glob(os.path.join(RESULTS_DIR, "*.json")))
    if not files:
        return None
    return json.load(open(files[-1], encoding="utf-8"))


def main():
    only = None
    if "--probes" in sys.argv:
        only = sys.argv[sys.argv.index("--probes") + 1].split(",")

    global GOLDEN
    if only:
        GOLDEN = [g for g in GOLDEN if g["id"] in only]
    print(f"probes: {[g['id'] for g in GOLDEN]}", flush=True)

    from app import config, rag  # noqa: E402

    rows = run_pipeline()
    ragas_result = score_answer_probes(rows)
    refusals = check_refusals(rows)

    METRIC_COLS = ("faithfulness", "answer_relevancy", "context_precision",
                   "context_recall")
    try:
        df = ragas_result.to_pandas()
        scores = {c: round(float(df[c].mean()), 4) for c in METRIC_COLS
                  if c in df.columns}
    except Exception:
        scores = {}
    try:
        per_probe = ragas_result.to_pandas().to_dict(orient="records")
    except Exception:
        per_probe = []

    payload = {
        "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "config": {
            "embed_provider": config.EMBED_PROVIDER,
            "ollama_embed_model": config.OLLAMA_EMBED_MODEL,
            "hybrid": config.HYBRID_SEARCH,
            "top_k": config.TOP_K,
            "chunk": [config.CHUNK_SIZE, config.CHUNK_OVERLAP],
            "llm": config.BEDROCK_MODEL_ID,
        },
        "scores_mean": scores,
        "refusals": refusals,
        "probes": [
            {"id": r["id"], "type": r["type"], "got_pages": r["got_pages"],
             "gold_pages": r["gold_pages"],
             "answer": r["answer"][:500]}
            for r in rows
        ],
        "per_probe_scores": per_probe,
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    prev = load_last()
    stamp = payload["ts"].replace(":", "").replace("+", "UTC")
    path = os.path.join(RESULTS_DIR, f"{stamp}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)

    print("\n=== RAGAS means ===")
    for k, v in scores.items():
        print(f"  {k}: {v}")
    print("=== refusals ===")
    for k, v in refusals.items():
        print(f"  {k}: {'PASS' if v else 'FAIL'}")

    if prev:
        print("=== delta vs previous run ===")
        for k, v in scores.items():
            old = prev.get("scores_mean", {}).get(k)
            if old is not None:
                print(f"  {k}: {old} -> {v} ({v - old:+.4f})")
    print(f"saved {path}")


if __name__ == "__main__":
    main()
