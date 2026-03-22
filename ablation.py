"""
ablation.py — Vanilla k-NN vs Multi-Slot Ablation
===================================================

Tests whether HyperBinder's multi-slot symbolic filter adds value
over plain nearest-neighbor retrieval with Legal-BERT embeddings.

Three conditions:
    1. MULTI-SLOT  — one query per clause type with exact symbolic filter
                     (the full HyperBinder architecture)
    2. VANILLA-KNN — single query over full index, pick label of top-1 hit
    3. KNN-VOTE    — single query over full index, majority vote over top-k hits

Usage:
    python ablation.py
    python ablation.py --limit 50     # quick test on first 50 contracts
    python ablation.py --top-k 5      # change k for voting baseline
"""

from __future__ import annotations

import argparse
import ast
import os
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

SERVER_URL = os.environ.get("HB_SERVER_URL", os.environ.get("SERVER_URL", ""))
API_KEY    = os.environ.get("HB_API_KEY",    os.environ.get("API_KEY", ""))
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


# ── Search ────────────────────────────────────────────────────────────────────

def search_slots(slot_queries: dict, top_k: int = 10, retries: int = 3) -> list:
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
                print(f"  Query failed: {e}")
    return []


# ── Ground truth ──────────────────────────────────────────────────────────────

def get_clause_text(row: pd.Series, clause_type: str) -> str | None:
    flag_raw = row.get(f"{clause_type}-Answer", "")
    flag     = str(flag_raw).strip().lower() if not pd.isna(flag_raw) else ""
    if flag != "yes":
        return None

    raw = row.get(clause_type, "")
    raw_str = str(raw).strip() if not pd.isna(raw) else ""
    if not raw_str or raw_str in ("[]", ""):
        return None

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

    return text.strip() if text.strip() else None


# ── Condition 1: Multi-slot (full HyperBinder architecture) ──────────────────

def classify_multislot(clause_text: str, top_k: int = 3) -> str:
    """
    One query per clause type with exact symbolic filter.
    Picks the clause type whose top hit scores highest.
    This is the full HyperBinder architecture.
    """
    all_scores = {}

    def query_one(clause_type):
        hits = search_slots({
            "clause_type": {"query": clause_type,       "weight": 0.01, "encoding": "exact"},
            "object":      {"query": clause_text[:500], "weight": 1.0,  "encoding": "semantic"},
        }, top_k=top_k)
        score = hits[0].get("_score", 0.0) if hits else 0.0
        return clause_type, score

    with ThreadPoolExecutor(max_workers=3) as executor:
        for ct, score in executor.map(query_one, RISK_CLAUSE_TYPES):
            all_scores[ct] = score

    return max(all_scores, key=all_scores.get)


# ── Condition 2: Vanilla k-NN (single query, full index, top-1 label) ────────

def classify_vanilla_knn(clause_text: str) -> str:
    """
    Single semantic query over the full index with no clause_type filter.
    Returns the clause_type label of the top-1 hit.
    This is what FAISS or any standard vector DB would do.
    """
    hits = search_slots({
        "object": {"query": clause_text[:500], "weight": 1.0, "encoding": "semantic"},
    }, top_k=1)

    if not hits:
        return RISK_CLAUSE_TYPES[0]

    return hits[0].get("data", {}).get("clause_type", RISK_CLAUSE_TYPES[0])


# ── Condition 3: k-NN with majority vote ─────────────────────────────────────

def classify_knn_vote(clause_text: str, top_k: int = 5) -> str:
    """
    Single semantic query over the full index with no clause_type filter.
    Majority vote over top-k hits.
    A slightly stronger baseline than top-1.
    """
    hits = search_slots({
        "object": {"query": clause_text[:500], "weight": 1.0, "encoding": "semantic"},
    }, top_k=top_k)

    if not hits:
        return RISK_CLAUSE_TYPES[0]

    labels = [h.get("data", {}).get("clause_type", "") for h in hits]
    labels = [l for l in labels if l]

    if not labels:
        return RISK_CLAUSE_TYPES[0]

    return Counter(labels).most_common(1)[0][0]


# ── Runner ────────────────────────────────────────────────────────────────────

def run_ablation(
    csv_path: str = "CUAD_v1/master_clauses.csv",
    limit: int | None = None,
    top_k: int = 5,
):
    print(f"\n{'='*70}")
    print("  ablation.py — Vanilla k-NN vs Multi-Slot")
    print(f"  CSV    : {csv_path}")
    print(f"  Limit  : {limit or 'all 510 contracts'}")
    print(f"  Top-k  : {top_k} (for k-NN vote baseline)")
    print(f"{'='*70}\n")

    df = pd.read_csv(csv_path, nrows=limit)

    results = []
    done    = 0
    t0      = time.time()

    for _, row in df.iterrows():
        filename      = str(row.get("Filename", "unknown"))
        contract_name = Path(filename).stem[:60]

        for clause_type in RISK_CLAUSE_TYPES:
            text = get_clause_text(row, clause_type)
            if not text:
                continue

            pred_multislot = classify_multislot(text, top_k=3)
            pred_knn       = classify_vanilla_knn(text)
            pred_knn_vote  = classify_knn_vote(text, top_k=top_k)

            results.append({
                "contract":           contract_name,
                "clause_type":        clause_type,
                "multislot_correct":  pred_multislot == clause_type,
                "knn_correct":        pred_knn       == clause_type,
                "knn_vote_correct":   pred_knn_vote  == clause_type,
                "pred_multislot":     pred_multislot,
                "pred_knn":           pred_knn,
                "pred_knn_vote":      pred_knn_vote,
            })

            done += 1
            if done % 50 == 0:
                elapsed = time.time() - t0
                rate    = done / elapsed
                print(f"  [{done:4d} clauses]  {rate:.1f}/s  contract: {contract_name[:40]}")

    # ── Summary ───────────────────────────────────────────────────────────────

    total         = len(results)
    ms_correct    = sum(1 for r in results if r["multislot_correct"])
    knn_correct   = sum(1 for r in results if r["knn_correct"])
    vote_correct  = sum(1 for r in results if r["knn_vote_correct"])

    print(f"\n{'='*60}")
    print(f"  {'Method':<35} {'P@1':>6}  {'Correct':>7} / {total}")
    print(f"  {'─'*35} {'─'*6}  {'─'*7}")
    print(f"  {'HyperBinder multi-slot (ours)':<35} {ms_correct/total:>6.3f}  {ms_correct:>7}")
    print(f"  {'Vanilla k-NN (top-1 label)':<35} {knn_correct/total:>6.3f}  {knn_correct:>7}")
    print(f"  {'k-NN majority vote (top-{top_k})':<35} {vote_correct/total:>6.3f}  {vote_correct:>7}")
    print(f"{'='*60}\n")

    # ── Per-clause-type breakdown ─────────────────────────────────────────────

    print(f"  {'Clause Type':<35} {'Multi-slot':>10}  {'k-NN top-1':>10}  {'k-NN vote':>10}  {'Delta':>6}")
    print(f"  {'─'*35} {'─'*10}  {'─'*10}  {'─'*10}  {'─'*6}")

    for ct in RISK_CLAUSE_TYPES:
        rows = [r for r in results if r["clause_type"] == ct]
        if not rows:
            continue
        n    = len(rows)
        ms   = sum(1 for r in rows if r["multislot_correct"]) / n
        knn  = sum(1 for r in rows if r["knn_correct"]) / n
        vote = sum(1 for r in rows if r["knn_vote_correct"]) / n
        delta = ms - max(knn, vote)
        print(f"  {ct:<35} {ms:>10.3f}  {knn:>10.3f}  {vote:>10.3f}  {delta:>+6.3f}")

    print()

    # ── Save results ──────────────────────────────────────────────────────────

    pd.DataFrame(results).to_csv("ablation_results.csv", index=False)
    print(f"  Per-clause results saved to ablation_results.csv\n")

    # ── Cases where vanilla k-NN fails but multi-slot succeeds ───────────────

    rescued = [r for r in results if r["multislot_correct"] and not r["knn_correct"]]
    broken  = [r for r in results if not r["multislot_correct"] and r["knn_correct"]]

    print(f"  Cases rescued by symbolic filter (kNN wrong, multi-slot right): {len(rescued)}")
    print(f"  Cases hurt by symbolic filter (kNN right, multi-slot wrong)   : {len(broken)}")

    if rescued:
        print(f"\n  Top rescued clause types:")
        for ct, count in Counter(r["clause_type"] for r in rescued).most_common(5):
            print(f"    {ct:<40} {count}")

    if broken:
        print(f"\n  Top hurt clause types:")
        for ct, count in Counter(r["clause_type"] for r in broken).most_common(5):
            print(f"    {ct:<40} {count}")

    print()


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ablation: vanilla k-NN vs HyperBinder multi-slot"
    )
    parser.add_argument("--csv",    default="CUAD_v1/master_clauses.csv")
    parser.add_argument("--limit",  type=int, default=None)
    parser.add_argument("--top-k",  type=int, default=5)
    args = parser.parse_args()

    run_ablation(csv_path=args.csv, limit=args.limit, top_k=args.top_k)