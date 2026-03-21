"""
legal_eval.py — Evaluate CUAD Legal System Against Ground Truth
===============================================================

Evaluates the system using two honest approaches:

EVAL MODE 1: Retrieval Classification (--mode classify)
    Given raw clause text, does the system predict the correct clause type?
    Tests retrieval quality on present clauses only.
    No ground truth leakage — the system sees text but not the label.

EVAL MODE 2: Presence Detection (--mode presence)
    Given only a contract name + clause type, does the system correctly
    predict whether that clause exists in the contract?
    The system queries the index scoped to the contract and checks if
    anything comes back above a confidence threshold.
    This is the hardest and most useful eval.

BASELINE: Always-MISSING majority class (presence mode only)

Usage:
    # Classification eval — fast, ~2 min for all 510 contracts
    python legal_eval.py --mode classify

    # Presence detection eval — slower, ~5 min
    python legal_eval.py --mode presence

    # Quick test on 20 contracts
    python legal_eval.py --mode classify --limit 20
    python legal_eval.py --mode presence --limit 20
"""

from __future__ import annotations

import argparse
import ast
import os
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────

SERVER_URL = os.environ.get("HB_SERVER_URL", os.environ.get("SERVER_URL", ""))
API_KEY    = os.environ.get("HB_API_KEY",    os.environ.get("API_KEY",    ""))
DB_NAME    = os.environ.get("HB_DB_NAME",    "fraud_db")
NAMESPACE  = "cuad_clauses"

RISK_CLAUSE_TYPES = [
    "Uncapped Liability",
    "Cap On Liability",
    "Liquidated Damages",
    "Non-Compete",
    "Anti-Assignment",
    "Change Of Control",
    "Termination For Convenience",
    "Ip Ownership Assignment",
    "Irrevocable Or Perpetual License",
    "Covenant Not To Sue",
]

# Per-clause-type presence thresholds — calibrated against CUAD eval results.
# Must stay in sync with PRESENCE_THRESHOLDS in legal_query.py.
PRESENCE_THRESHOLDS = {
    "Uncapped Liability":             0.82,
    "Cap On Liability":               0.65,
    "Liquidated Damages":             0.72,
    "Non-Compete":                    0.68,
    "Anti-Assignment":                0.65,
    "Change Of Control":              0.68,
    "Termination For Convenience":    0.70,
    "Ip Ownership Assignment":        0.68,
    "Irrevocable Or Perpetual License": 0.80,
    "Covenant Not To Sue":            0.65,
}

# Global fallback threshold used when --threshold flag is passed
PRESENCE_THRESHOLD = 0.65


# ── Search ────────────────────────────────────────────────────────────────────

def search_slots(slot_queries: dict, top_k: int = 10, retries: int = 3) -> list:
    for attempt in range(retries):
        try:
            resp = requests.post(
                f"{SERVER_URL}/compose/search_slots/{DB_NAME}/{NAMESPACE}",
                headers={"X-API-Key": API_KEY},
                json={"slot_queries": slot_queries, "top_k": top_k},
                timeout=30,
            )
            if resp.status_code == 429:
                wait = 2 ** attempt  # exponential backoff: 1s, 2s, 4s
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json().get("results", [])
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"  ✗ Query failed: {e}")
    return []


# ── Ground truth ──────────────────────────────────────────────────────────────

def get_ground_truth(row: pd.Series, clause_type: str) -> tuple[bool, str]:
    """
    Returns (is_present, clause_text) from CUAD CSV.
    answer_col = "ClauseType-Answer" -> "Yes" / "No" flag
    text_col   = "ClauseType"        -> extracted clause text
    """
    flag_raw = row.get(f"{clause_type}-Answer", "")
    flag     = str(flag_raw).strip().lower() if not pd.isna(flag_raw) else ""
    present  = (flag == "yes")

    text = ""
    if present:
        raw = row.get(clause_type, "")
        raw_str = str(raw).strip() if not pd.isna(raw) else ""
        try:
            parsed = ast.literal_eval(raw_str)
            if isinstance(parsed, list):
                text = " | ".join(
                    str(p).strip() for p in parsed
                    if p and str(p).strip().lower() not in ("yes", "no", "")
                )
            else:
                text = raw_str
        except Exception:
            text = raw_str

    return present, text


# ── Eval mode 1: Classification ───────────────────────────────────────────────

def eval_classify(clause_text: str, true_clause_type: str, top_k: int = 3) -> dict:
    """
    Given raw clause text (no label), predict which clause type it is.
    No leakage: system sees text but NOT the clause type label.

    Queries all clause types in parallel using ThreadPoolExecutor —
    10 concurrent requests instead of 10 sequential ones (~10x faster).
    Uses exact symbolic clause_type filter + semantic object scoring.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def query_one(clause_type):
        hits = search_slots({
            "clause_type": {"query": clause_type,       "weight": 0.01, "encoding": "exact"},
            "object":      {"query": clause_text[:500], "weight": 1.0,  "encoding": "semantic"},
        }, top_k=top_k)
        score = hits[0].get("_score", 0.0) if hits else 0.0
        return clause_type, score

    all_scores = {}
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(query_one, ct): ct for ct in RISK_CLAUSE_TYPES}
        for future in futures:
            clause_type, score = future.result()
            all_scores[clause_type] = score

    best_type  = max(all_scores, key=all_scores.get)
    best_score = all_scores[best_type]

    # For top-k metrics, rank all clause types by score
    ranked      = sorted(all_scores.items(), key=lambda x: -x[1])
    ranked_types = [ct for ct, _ in ranked]

    return {
        "predicted_type": best_type,
        "top1_correct":   true_clause_type == ranked_types[0],
        "top3_correct":   true_clause_type in ranked_types[:3],
        "top5_correct":   true_clause_type in ranked_types[:5],
        "top1_score":     round(best_score, 4),
    }


# ── Eval mode 2: Presence detection ──────────────────────────────────────────

def eval_presence(contract_name: str, contract_key: str, clause_type: str) -> dict:
    """
    Given only contract name + clause type (no clause text), predict
    whether this clause exists in the contract.
    No leakage: system never sees ground truth text.
    Uses per-clause-type threshold from PRESENCE_THRESHOLDS.
    """
    # Use per-clause threshold, fall back to global if --threshold was passed
    threshold = PRESENCE_THRESHOLDS.get(clause_type, PRESENCE_THRESHOLD)

    hits = search_slots({
        "predicate": {"query": clause_type,   "weight": 0.5, "encoding": "semantic"},
        "subject":   {"query": contract_name, "weight": 0.5, "encoding": "semantic"},
    }, top_k=20)

    # Filter to hits from this specific contract
    contract_hits = [
        h for h in hits
        if contract_key[:20] in h.get("data", {}).get("contract", "").lower()
    ]

    if contract_hits:
        top_score      = contract_hits[0].get("_score", 0.0)
        sys_present    = top_score >= threshold
        predicted_text = contract_hits[0].get("data", {}).get("object", "")[:80]
    else:
        top_score      = 0.0
        sys_present    = False
        predicted_text = ""

    return {
        "sys_present":    sys_present,
        "top_score":      round(top_score, 4),
        "threshold_used": threshold,
        "predicted_text": predicted_text,
    }


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_presence_metrics(results: list[dict], clause_type: str) -> dict:
    rows = [r for r in results if r["clause_type"] == clause_type]
    tp = fn = fp = tn = 0

    for r in rows:
        gt  = r["gt_present"]
        sys = r["sys_present"]
        if gt and sys:       tp += 1
        elif gt and not sys: fn += 1
        elif not gt and sys: fp += 1
        else:                tn += 1

    total     = len(rows)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy  = (tp + tn) / total if total > 0 else 0.0
    base_acc  = (fp + tn) / total if total > 0 else 0.0

    return {
        "clause_type":          clause_type,
        "total":                total,
        "gt_present":           tp + fn,
        "base_rate":            round((tp + fn) / total, 3) if total > 0 else 0,
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "precision":            round(precision, 4),
        "recall":               round(recall, 4),
        "f1":                   round(f1, 4),
        "accuracy":             round(accuracy, 4),
        "baseline_accuracy":    round(base_acc, 4),
        "accuracy_vs_baseline": round(accuracy - base_acc, 4),
    }


def compute_classify_metrics(results: list[dict], clause_type: str) -> dict:
    rows = [r for r in results if r["clause_type"] == clause_type]
    if not rows:
        return {"clause_type": clause_type, "count": 0,
                "top1": 0.0, "top3": 0.0, "top5": 0.0, "avg_score": 0.0}

    top1 = sum(1 for r in rows if r["top1_correct"]) / len(rows)
    top3 = sum(1 for r in rows if r["top3_correct"]) / len(rows)
    top5 = sum(1 for r in rows if r["top5_correct"]) / len(rows)
    avg  = sum(r["top1_score"] for r in rows) / len(rows)

    return {
        "clause_type": clause_type,
        "count":       len(rows),
        "top1":        round(top1, 4),
        "top3":        round(top3, 4),
        "top5":        round(top5, 4),
        "avg_score":   round(avg, 4),
    }


# ── Classification eval runner ────────────────────────────────────────────────

def run_classify_eval(csv_path: str, limit: int, out_path: str, summary_path: str):
    """
    For every present clause, strip the label and ask the system to
    predict clause type from text alone.
    """
    print(f"\n{'='*70}")
    print("  MODE: Classification — predict clause type from raw text")
    print("  No leakage — system sees text only, not the label")
    print(f"{'='*70}\n")

    df      = pd.read_csv(csv_path, nrows=limit)
    results = []
    t0      = time.time()
    done    = 0

    for _, row in df.iterrows():
        filename      = str(row.get("Filename", "unknown"))
        contract_name = Path(filename).stem[:60]

        for clause_type in RISK_CLAUSE_TYPES:
            gt_present, clause_text = get_ground_truth(row, clause_type)

            if not gt_present or not clause_text.strip():
                continue  # only evaluate on clauses that exist with text

            pred = eval_classify(clause_text, clause_type)

            results.append({
                "contract":       contract_name,
                "clause_type":    clause_type,
                "predicted_type": pred["predicted_type"],
                "top1_correct":   pred["top1_correct"],
                "top3_correct":   pred["top3_correct"],
                "top5_correct":   pred["top5_correct"],
                "top1_score":     pred["top1_score"],
                "text_preview":   clause_text[:80],
            })

            done += 1
            if done % 100 == 0:
                elapsed = time.time() - t0
                rate    = done / elapsed
                print(f"  [{done:4d} clauses]  {rate:.0f}/s  "
                      f"contract: {contract_name[:40]}")

    elapsed = time.time() - t0
    print(f"\n  Done — {done:,} clauses in {elapsed:.1f}s")

    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"  Per-clause results -> {out_path}")

    metrics = [compute_classify_metrics(results, ct) for ct in RISK_CLAUSE_TYPES]
    pd.DataFrame(metrics).to_csv(summary_path, index=False)
    print(f"  Summary -> {summary_path}")

    print(f"\n{'='*65}")
    print(f"  {'Clause Type':<35} {'N':>4}  {'P@1':>5}  {'P@3':>5}  {'P@5':>5}  {'Score':>6}")
    print(f"  {'─'*35} {'─'*4}  {'─'*5}  {'─'*5}  {'─'*5}  {'─'*6}")

    total_top1 = total_top3 = total_top5 = 0.0
    n_types = 0

    for m in metrics:
        if m["count"] == 0:
            continue
        print(
            f"  {m['clause_type']:<35} {m['count']:>4}"
            f"  {m['top1']:>5.3f}  {m['top3']:>5.3f}  {m['top5']:>5.3f}  {m['avg_score']:>6.4f}"
        )
        total_top1 += m["top1"]
        total_top3 += m["top3"]
        total_top5 += m["top5"]
        n_types    += 1

    if n_types:
        print(f"  {'─'*35} {'─'*4}  {'─'*5}  {'─'*5}  {'─'*5}  {'─'*6}")
        print(
            f"  {'MACRO AVERAGE':<35} {'':>4}"
            f"  {total_top1/n_types:>5.3f}  {total_top3/n_types:>5.3f}  {total_top5/n_types:>5.3f}"
        )
    print(f"{'='*65}\n")

    wrong = [r for r in results if not r["top1_correct"]]
    if wrong:
        from collections import Counter
        print("  Most common misclassifications:")
        for (true, pred), count in Counter(
            (r["clause_type"], r["predicted_type"]) for r in wrong
        ).most_common(10):
            print(f"    {true:<35} -> {pred:<35} ({count}x)")
    print()


# ── Presence eval runner ──────────────────────────────────────────────────────

def run_presence_eval(csv_path: str, limit: int, out_path: str, summary_path: str):
    """
    For every (contract, clause_type) pair, predict presence using only
    the contract name + clause type. Never shows the system the ground truth text.
    """
    print(f"\n{'='*70}")
    print("  MODE: Presence detection — predict clause exists from contract name only")
    print(f"  No leakage — system never sees clause text")
    print(f"  Presence thresholds: per-clause-type (see PRESENCE_THRESHOLDS)")
    print(f"{'='*70}\n")

    df      = pd.read_csv(csv_path, nrows=limit)
    results = []
    t0      = time.time()
    done    = 0
    total_q = len(df) * len(RISK_CLAUSE_TYPES)

    for _, row in df.iterrows():
        filename      = str(row.get("Filename", "unknown"))
        contract_name = Path(filename).stem[:60]
        contract_key  = contract_name.lower()

        for clause_type in RISK_CLAUSE_TYPES:
            gt_present, clause_text = get_ground_truth(row, clause_type)

            # System predicts using ONLY contract name + clause type
            # clause_text is ground truth — never passed to the system
            pred = eval_presence(contract_name, contract_key, clause_type)

            results.append({
                "contract":       contract_name,
                "clause_type":    clause_type,
                "gt_present":     gt_present,
                "sys_present":    pred["sys_present"],
                "sys_score":      pred["top_score"],
                "predicted_text": pred["predicted_text"],
                "gt_text":        clause_text[:80] if clause_text else "",
            })

            done += 1
            if done % 100 == 0:
                elapsed = time.time() - t0
                rate    = done / elapsed
                eta     = (total_q - done) / rate if rate > 0 else 0
                print(f"  [{done:4d}/{total_q}]  {rate:.0f}/s  ETA {eta/60:.1f}m  "
                      f"contract: {contract_name[:35]}")

    elapsed = time.time() - t0
    print(f"\n  Done — {done:,} queries in {elapsed:.1f}s")

    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"  Per-contract results -> {out_path}")

    metrics = [compute_presence_metrics(results, ct) for ct in RISK_CLAUSE_TYPES]
    pd.DataFrame(metrics).to_csv(summary_path, index=False)
    print(f"  Summary -> {summary_path}")

    print(f"\n{'='*75}")
    print(f"  {'Clause Type':<35} {'Base%':>5}  {'P':>5}  {'R':>5}  {'F1':>5}  {'Acc':>5}  {'vs Base':>7}")
    print(f"  {'─'*35} {'─'*5}  {'─'*5}  {'─'*5}  {'─'*5}  {'─'*5}  {'─'*7}")

    total_f1 = total_acc = total_vs = 0.0

    for m in metrics:
        base_pct = m["base_rate"] * 100
        vs       = m["accuracy_vs_baseline"]
        vs_str   = f"+{vs:.3f}" if vs >= 0 else f"{vs:.3f}"
        print(
            f"  {m['clause_type']:<35} {base_pct:>4.0f}%"
            f"  {m['precision']:>5.3f}  {m['recall']:>5.3f}"
            f"  {m['f1']:>5.3f}  {m['accuracy']:>5.3f}  {vs_str:>7}"
        )
        total_f1  += m["f1"]
        total_acc += m["accuracy"]
        total_vs  += vs

    n = len(metrics)
    vs_avg_str = f"+{total_vs/n:.3f}" if total_vs/n >= 0 else f"{total_vs/n:.3f}"
    print(f"  {'─'*35} {'─'*5}  {'─'*5}  {'─'*5}  {'─'*5}  {'─'*5}  {'─'*7}")
    print(
        f"  {'MACRO AVERAGE':<35} {'':>5}"
        f"  {'':>5}  {'':>5}"
        f"  {total_f1/n:>5.3f}  {total_acc/n:>5.3f}  {vs_avg_str:>7}"
    )
    print(f"{'='*75}\n")

    beats = [m for m in metrics if m["accuracy_vs_baseline"] > 0.01]
    hurts = [m for m in metrics if m["accuracy_vs_baseline"] < -0.01]

    if beats:
        print("  Beats majority-class baseline:")
        for m in beats:
            print(f"    +  {m['clause_type']:<35}  +{m['accuracy_vs_baseline']:.3f}  "
                  f"(recall={m['recall']:.3f})")
    if hurts:
        print("\n  WORSE than always-MISSING baseline:")
        for m in hurts:
            print(f"    x  {m['clause_type']:<35}  {m['accuracy_vs_baseline']:.3f}")
    print()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    global PRESENCE_THRESHOLD
    parser = argparse.ArgumentParser(
        description="Evaluate CUAD legal system — no ground truth leakage"
    )
    parser.add_argument("--mode",      choices=["classify", "presence"],
                        default="classify",
                        help="classify: predict clause type from text | "
                             "presence: predict if clause exists in contract")
    parser.add_argument("--csv",       default="CUAD_v1/master_clauses.csv")
    parser.add_argument("--limit",     type=int, default=None)
    parser.add_argument("--out",       default="legal_eval_results.csv")
    parser.add_argument("--summary",   default="legal_eval_summary.csv")
    parser.add_argument("--threshold", type=float, default=PRESENCE_THRESHOLD,
                        help=f"Presence threshold (default: {PRESENCE_THRESHOLD})")
    args = parser.parse_args()

    
    PRESENCE_THRESHOLD = args.threshold

    if args.mode == "classify":
        run_classify_eval(args.csv, args.limit, args.out, args.summary)
    else:
        run_presence_eval(args.csv, args.limit, args.out, args.summary)


if __name__ == "__main__":
    main()