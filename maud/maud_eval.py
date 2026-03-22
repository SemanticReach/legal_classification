"""
maud_eval.py — Evaluate MAUD Query System Against Ground Truth
==============================================================

Evaluates the system using two honest approaches, both on the held-out
TEST split only — fixing the in-sample criticism from the CUAD evaluation.

EVAL MODE 1: Classification (--mode classify)
    Given raw clause text with no label, predict the question type.
    No ground truth leakage — system sees text only, not the label.
    Evaluated on test split (6,651 records, never seen at ingest time
    for classification purposes since label is withheld).

EVAL MODE 2: Answer Label Prediction (--mode answer)
    Given clause text AND question type, predict the answer label.
    Tests whether the system correctly characterizes deal terms.
    Compared against a majority-class baseline (always predict the
    most common answer for each question type).

Usage:
    # Classification eval on held-out test split
    python maud_eval.py --mode classify

    # Answer label prediction eval
    python maud_eval.py --mode answer

    # Quick test on 100 records
    python maud_eval.py --mode classify --limit 100
    python maud_eval.py --mode answer --limit 100
"""

from __future__ import annotations

import argparse
import os
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import requests
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────

SERVER_URL = os.environ.get("HB_SERVER_URL", os.environ.get("SERVER_URL", ""))
API_KEY    = os.environ.get("HB_API_KEY",    os.environ.get("API_KEY",    ""))
DB_NAME    = os.environ.get("HB_DB_NAME",    "fraud_db")
NAMESPACE  = "maud_clauses"

QUESTION_TYPES = [
    "Absence of Litigation Closing Condition",
    "Accuracy of Target R&W Closing Condition",
    "Agreement provides for matching rights in connection with COR",
    "Agreement provides for matching rights in connection with FTR",
    "Breach of Meeting Covenant",
    "Breach of No Shop",
    "Compliance with Covenant Closing Condition",
    "FTR Triggers",
    "Fiduciary exception to COR covenant",
    "Fiduciary exception:  Board determination (no-shop)",
    "General Antitrust Efforts Standard",
    "Intervening Event Definition",
    "Knowledge Definition",
    "Limitations on FTR Exercise",
    "MAE Definition",
    "Negative interim operating covenant",
    "No-Shop",
    "Ordinary course covenant",
    "Specific Performance",
    "Superior Offer Definition",
    "Tail Period & Acquisition Proposal Details",
    "Type of Consideration",
]


# ── Search ────────────────────────────────────────────────────────────────────

def search_slots(slot_queries: dict, top_k: int = 5, retries: int = 3) -> list:
    import time as _time
    for attempt in range(retries):
        try:
            resp = requests.post(
                f"{SERVER_URL}/compose/search_slots/{DB_NAME}/{NAMESPACE}",
                headers={"X-API-Key": API_KEY},
                json={"slot_queries": slot_queries, "top_k": top_k},
                timeout=30,
            )
            if resp.status_code == 429:
                _time.sleep(2 ** attempt)
                continue
            resp.raise_for_status()
            return resp.json().get("results", [])
        except Exception as e:
            if attempt < retries - 1:
                _time.sleep(2 ** attempt)
            else:
                print(f"  ✗ Query failed: {e}")
    return []


# ── Load MAUD test split ──────────────────────────────────────────────────────

def load_test_records(limit: int = None) -> list[dict]:
    """
    Load held-out test split from HuggingFace MAUD.
    Shuffles and stratifies by question type so --limit gives
    representative coverage across all 22 question types.
    """
    from datasets import load_dataset
    import re as re_mod
    import random
    from collections import defaultdict

    print("  Loading MAUD test split from HuggingFace...")
    ds = load_dataset("theatticusproject/maud")
    test_data = ds["test"]

    records = []
    for record in test_data:
        text      = record.get("text", "") or ""
        answer    = record.get("answer", "") or ""
        text_type = record.get("text_type", "") or ""

        if not text.strip() or not answer.strip() or not text_type.strip():
            continue
        if answer is None or str(answer).strip().lower() in ("none", "null", ""):
            continue

        clean = text.strip()
        # Strip trailing page references
        import re
        clean = re.sub(r"\s*\(Pages?\s+[\d\-]+\)\s*$", "", clean).strip()
        if not clean:
            continue

        records.append({
            "text":          clean[:2000],
            "answer":        str(answer).strip(),
            "text_type":     text_type,
            "contract_name": record.get("contract_name", ""),
            "category":      record.get("category", ""),
        })

    # Shuffle so ordering does not bias results
    random.seed(42)
    random.shuffle(records)

    if limit:
        # Stratified sample: take proportional records from each question type
        by_type = defaultdict(list)
        for r in records:
            by_type[r["text_type"]].append(r)

        per_type   = max(1, limit // len(by_type))
        stratified = []
        for qt_records in by_type.values():
            stratified.extend(qt_records[:per_type])

        # Top up to limit if needed
        used   = set(id(r) for r in stratified)
        extras = [r for r in records if id(r) not in used]
        stratified.extend(extras[:max(0, limit - len(stratified))])
        records = stratified[:limit]

    n_types = len(set(r["text_type"] for r in records))
    print(f"  Loaded {len(records):,} test records ({n_types} question types covered)")
    return records


def load_train_distributions() -> dict[str, Counter]:
    """Load answer label distributions from train split for baseline."""
    from datasets import load_dataset

    print("  Loading train split for baseline distributions...")
    ds = load_dataset("theatticusproject/maud")

    dist = {}
    for qt in QUESTION_TYPES:
        answers = [
            r["answer"] for r in ds["train"]
            if r.get("text_type") == qt and r.get("answer")
        ]
        dist[qt] = Counter(answers)

    print(f"  ✓ Loaded distributions for {len(dist)} question types")
    return dist


# ── Eval mode 1: Classification ───────────────────────────────────────────────

def eval_classify_one(clause_text: str, top_k: int = 3) -> dict:
    """
    Predict question type from raw text using exact symbolic filtering.
    Queries all question types in parallel, picks highest scoring.
    No leakage: system sees text only, not the question_type label.
    """
    def query_one(qt):
        hits  = search_slots({
            "question_type": {"query": qt,              "weight": 0.01, "encoding": "exact"},
            "object":        {"query": clause_text[:500], "weight": 1.0,  "encoding": "semantic"},
        }, top_k=top_k)
        score = hits[0].get("_score", 0.0) if hits else 0.0
        return qt, score

    all_scores = {}
    with ThreadPoolExecutor(max_workers=3) as executor:
        for qt, score in executor.map(query_one, QUESTION_TYPES):
            all_scores[qt] = score

    ranked       = sorted(all_scores.items(), key=lambda x: -x[1])
    ranked_types = [qt for qt, _ in ranked]
    best_type    = ranked_types[0]
    best_score   = all_scores[best_type]

    return {
        "predicted_type": best_type,
        "top1_score":     round(best_score, 4),
        "ranked_types":   ranked_types,
    }


def run_classify_eval(
    records:      list[dict],
    out_path:     str,
    summary_path: str,
) -> None:
    import pandas as pd

    print(f"\n{'='*70}")
    print("  MODE: Classification — predict question type from raw text")
    print("  Evaluated on held-out TEST split — no in-sample contamination")
    print(f"{'='*70}\n")

    results = []
    t0      = time.time()

    for i, rec in enumerate(records):
        pred = eval_classify_one(rec["text"])

        top1_correct = rec["text_type"] == pred["ranked_types"][0]
        top3_correct = rec["text_type"] in pred["ranked_types"][:3]
        top5_correct = rec["text_type"] in pred["ranked_types"][:5]

        results.append({
            "contract_name":   rec["contract_name"],
            "text_type":       rec["text_type"],
            "predicted_type":  pred["predicted_type"],
            "top1_correct":    top1_correct,
            "top3_correct":    top3_correct,
            "top5_correct":    top5_correct,
            "top1_score":      pred["top1_score"],
            "category":        rec["category"],
            "text_preview":    rec["text"][:80],
        })

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate    = (i + 1) / elapsed
            print(f"  [{i+1:4d}/{len(records)}]  {rate:.1f}/s  "
                  f"running P@1: {sum(r['top1_correct'] for r in results)/len(results):.3f}")

    elapsed = time.time() - t0
    print(f"\n  Done — {len(results):,} records in {elapsed:.1f}s")

    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"  Per-record results -> {out_path}")

    # ── Per-question-type metrics ─────────────────────────────────────────────
    metrics = []
    for qt in QUESTION_TYPES:
        rows = [r for r in results if r["text_type"] == qt]
        if not rows:
            continue
        top1 = sum(r["top1_correct"] for r in rows) / len(rows)
        top3 = sum(r["top3_correct"] for r in rows) / len(rows)
        top5 = sum(r["top5_correct"] for r in rows) / len(rows)
        avg  = sum(r["top1_score"]   for r in rows) / len(rows)
        metrics.append({
            "question_type": qt,
            "count": len(rows),
            "top1":  round(top1, 4),
            "top3":  round(top3, 4),
            "top5":  round(top5, 4),
            "avg_score": round(avg, 4),
        })

    pd.DataFrame(metrics).to_csv(summary_path, index=False)
    print(f"  Summary -> {summary_path}")

    # ── Print table ───────────────────────────────────────────────────────────
    print(f"\n{'='*75}")
    print(f"  {'Question Type':<50} {'N':>4}  {'P@1':>5}  {'P@3':>5}  {'P@5':>5}")
    print(f"  {'─'*50} {'─'*4}  {'─'*5}  {'─'*5}  {'─'*5}")

    total_top1 = total_top3 = total_top5 = 0.0
    n_types = 0

    for m in sorted(metrics, key=lambda x: -x["top1"]):
        print(
            f"  {m['question_type']:<50} {m['count']:>4}"
            f"  {m['top1']:>5.3f}  {m['top3']:>5.3f}  {m['top5']:>5.3f}"
        )
        total_top1 += m["top1"]
        total_top3 += m["top3"]
        total_top5 += m["top5"]
        n_types    += 1

    print(f"  {'─'*50} {'─'*4}  {'─'*5}  {'─'*5}  {'─'*5}")
    print(
        f"  {'MACRO AVERAGE':<50} {'':>4}"
        f"  {total_top1/n_types:>5.3f}"
        f"  {total_top3/n_types:>5.3f}"
        f"  {total_top5/n_types:>5.3f}"
    )
    print(f"{'='*75}\n")

    # ── Misclassifications ────────────────────────────────────────────────────
    wrong = [r for r in results if not r["top1_correct"]]
    if wrong:
        print("  Most common misclassifications:")
        for (true, pred), count in Counter(
            (r["text_type"], r["predicted_type"]) for r in wrong
        ).most_common(10):
            print(f"    {true[:45]:<45} -> {pred[:40]:<40} ({count}x)")
    print()


# ── Eval mode 2: Answer label prediction ─────────────────────────────────────

def eval_answer_one(
    clause_text:   str,
    question_type: str,
    top_k:         int = 5,
) -> dict:
    """
    Given clause text AND question type, predict the answer label by
    finding the most similar clause in the index for that question type
    and returning its answer label.
    """
    hits = search_slots({
        "question_type": {"query": question_type, "weight": 0.01, "encoding": "exact"},
        "object":        {"query": clause_text[:500], "weight": 1.0,  "encoding": "semantic"},
    }, top_k=top_k)

    if not hits:
        return {"predicted_answer": "Unknown", "top1_score": 0.0}

    predicted_answer = hits[0].get("data", {}).get("answer", "Unknown")
    top1_score       = hits[0].get("_score", 0.0)

    return {
        "predicted_answer": predicted_answer,
        "top1_score":       round(top1_score, 4),
    }


def run_answer_eval(
    records:      list[dict],
    train_dist:   dict[str, Counter],
    out_path:     str,
    summary_path: str,
) -> None:
    import pandas as pd

    print(f"\n{'='*70}")
    print("  MODE: Answer label prediction — predict deal term characterization")
    print("  Evaluated on held-out TEST split")
    print("  Baseline: always predict most common answer for each question type")
    print(f"{'='*70}\n")

    results = []
    t0      = time.time()

    for i, rec in enumerate(records):
        pred = eval_answer_one(rec["text"], rec["text_type"])

        # Majority-class baseline prediction
        dist     = train_dist.get(rec["text_type"], Counter())
        baseline = dist.most_common(1)[0][0] if dist else "Unknown"

        sys_correct      = rec["answer"] == pred["predicted_answer"]
        baseline_correct = rec["answer"] == baseline

        results.append({
            "contract_name":      rec["contract_name"],
            "text_type":          rec["text_type"],
            "true_answer":        rec["answer"],
            "predicted_answer":   pred["predicted_answer"],
            "baseline_answer":    baseline,
            "sys_correct":        sys_correct,
            "baseline_correct":   baseline_correct,
            "top1_score":         pred["top1_score"],
            "category":           rec["category"],
        })

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate    = (i + 1) / elapsed
            sys_acc = sum(r["sys_correct"] for r in results) / len(results)
            bas_acc = sum(r["baseline_correct"] for r in results) / len(results)
            print(f"  [{i+1:4d}/{len(records)}]  {rate:.1f}/s  "
                  f"sys={sys_acc:.3f}  baseline={bas_acc:.3f}")

    elapsed = time.time() - t0
    print(f"\n  Done — {len(results):,} records in {elapsed:.1f}s")

    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"  Per-record results -> {out_path}")

    # ── Per-question-type metrics ─────────────────────────────────────────────
    metrics = []
    for qt in QUESTION_TYPES:
        rows = [r for r in results if r["text_type"] == qt]
        if not rows:
            continue
        sys_acc = sum(r["sys_correct"]      for r in rows) / len(rows)
        bas_acc = sum(r["baseline_correct"] for r in rows) / len(rows)
        metrics.append({
            "question_type":    qt,
            "count":            len(rows),
            "sys_accuracy":     round(sys_acc, 4),
            "baseline_accuracy":round(bas_acc, 4),
            "vs_baseline":      round(sys_acc - bas_acc, 4),
        })

    pd.DataFrame(metrics).to_csv(summary_path, index=False)
    print(f"  Summary -> {summary_path}")

    # ── Print table ───────────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"  {'Question Type':<50} {'N':>4}  {'Sys':>5}  {'Base':>5}  {'vs Base':>7}")
    print(f"  {'─'*50} {'─'*4}  {'─'*5}  {'─'*5}  {'─'*7}")

    total_sys = total_bas = total_vs = 0.0
    n_types = 0

    for m in sorted(metrics, key=lambda x: -x["vs_baseline"]):
        vs     = m["vs_baseline"]
        vs_str = f"+{vs:.3f}" if vs >= 0 else f"{vs:.3f}"
        print(
            f"  {m['question_type']:<50} {m['count']:>4}"
            f"  {m['sys_accuracy']:>5.3f}"
            f"  {m['baseline_accuracy']:>5.3f}"
            f"  {vs_str:>7}"
        )
        total_sys += m["sys_accuracy"]
        total_bas += m["baseline_accuracy"]
        total_vs  += vs
        n_types   += 1

    n = n_types
    vs_avg     = total_vs / n
    vs_avg_str = f"+{vs_avg:.3f}" if vs_avg >= 0 else f"{vs_avg:.3f}"
    print(f"  {'─'*50} {'─'*4}  {'─'*5}  {'─'*5}  {'─'*7}")
    print(
        f"  {'MACRO AVERAGE':<50} {'':>4}"
        f"  {total_sys/n:>5.3f}"
        f"  {total_bas/n:>5.3f}"
        f"  {vs_avg_str:>7}"
    )
    print(f"{'='*80}\n")

    beats = [m for m in metrics if m["vs_baseline"] > 0.01]
    hurts = [m for m in metrics if m["vs_baseline"] < -0.01]

    if beats:
        print("  Beats majority-class baseline:")
        for m in beats:
            print(f"    +  {m['question_type']:<50}  +{m['vs_baseline']:.3f}")
    if hurts:
        print("\n  WORSE than majority-class baseline:")
        for m in hurts:
            print(f"    x  {m['question_type']:<50}  {m['vs_baseline']:.3f}")
    print()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate MAUD system on held-out test split"
    )
    parser.add_argument("--mode",    choices=["classify", "answer"],
                        default="classify",
                        help="classify: predict question type from text | "
                             "answer: predict answer label given question type + text")
    parser.add_argument("--limit",   type=int, default=None,
                        help="Max test records to evaluate (default: all ~6,500)")
    parser.add_argument("--out",     default="maud_eval_results.csv")
    parser.add_argument("--summary", default="maud_eval_summary.csv")
    args = parser.parse_args()

    records = load_test_records(limit=args.limit)

    if args.mode == "classify":
        run_classify_eval(records, args.out, args.summary)
    else:
        train_dist = load_train_distributions()
        run_answer_eval(records, train_dist, args.out, args.summary)


if __name__ == "__main__":
    main()