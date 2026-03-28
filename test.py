"""
test.py — Success Mining & Accuracy Audit
===============================================
Evaluates the system using two honest approaches:

EVAL MODE 1: Retrieval Classification (--mode classify)
    Predicts correct clause type from raw text.
    *COMPARES* HyperBinder vs Vanilla to find "Rescued" demo clauses.

EVAL MODE 2: Presence Detection (--mode presence)
    Predicts if a clause exists in a contract using name only.

Usage:
    python test.py --mode classify --limit 100
    python test.py --mode presence --limit 50
"""

from __future__ import annotations

import argparse
import ast
import os
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────

SERVER_URL = os.environ.get("HB_SERVER_URL")
API_KEY    = os.environ.get("HB_API_KEY")
DB_NAME    = os.environ.get("HB_DB_NAME")
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

PRESENCE_THRESHOLD = 0.65


# ── Search ────────────────────────────────────────────────────────────────────

def search_slots(slot_queries: dict, top_k: int = 10, retries: int = 3) -> list:
    """Performs search with the required X-API-Key header."""
    for attempt in range(retries):
        try:
            resp = requests.post(
                f"{SERVER_URL}/compose/search_slots/{DB_NAME}/{NAMESPACE}",
                headers={"X-API-Key": API_KEY},
                json={"slot_queries": slot_queries, "top_k": top_k},
                timeout=30,
            )
            if resp.status_code == 429:
                time.sleep(2 ** attempt)
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
                text = " | ".join(str(p).strip() for p in parsed if p)
            else:
                text = raw_str
        except Exception:
            text = raw_str

    return present, text


# ── Eval mode 1: Classification (HyperBinder vs Vanilla) ─────────────────────

def eval_classify(clause_text: str, true_clause_type: str) -> dict:
    """
    Head-to-Head: HyperBinder (Multi-slot) vs Vanilla (Global).
    Identifies 'Rescued' clauses for the demo.
    """
    # 1. VANILLA SEARCH (Standard Transformer/VectorDB baseline)
    v_hits = search_slots({
        "object": {"query": clause_text[:500], "weight": 1.0, "encoding": "semantic"},
    }, top_k=1)
    # Get label from 'predicate' slot
    v_pred = v_hits[0].get("data", {}).get("predicate", "None") if v_hits else "None"
    v_correct = (v_pred == true_clause_type)

    # 2. HYPERBINDER SEARCH (Multi-slot Architecture)
    def query_one(ct):
        hits = search_slots({
            "clause_type": {"query": ct, "weight": 0.01, "encoding": "exact"},
            "object":      {"query": clause_text[:500], "weight": 1.0, "encoding": "semantic"},
        }, top_k=3)
        return ct, (hits[0].get("_score", 0.0) if hits else 0.0)

    all_scores = {}
    with ThreadPoolExecutor(max_workers=3) as executor:
        for ct, score in executor.map(query_one, RISK_CLAUSE_TYPES):
            all_scores[ct] = score

    hb_pred = max(all_scores, key=all_scores.get)
    hb_correct = (hb_pred == true_clause_type)

    # 3. IDENTIFY RESCUED CLAUSE
    is_rescued = (not v_correct) and hb_correct

    return {
        "predicted_type": hb_pred,
        "hb_correct":     hb_correct,
        "v_correct":      v_correct,
        "is_rescued":     is_rescued,
        "v_pred":         v_pred,
        "hb_score":       round(all_scores[hb_pred], 4),
    }


# ── Eval mode 2: Presence detection ──────────────────────────────────────────

def eval_presence(contract_name: str, contract_key: str, clause_type: str) -> dict:
    threshold = PRESENCE_THRESHOLDS.get(clause_type, PRESENCE_THRESHOLD)
    hits = search_slots({
        "predicate": {"query": clause_type,   "weight": 0.5, "encoding": "semantic"},
        "subject":   {"query": contract_name, "weight": 0.5, "encoding": "semantic"},
    }, top_k=20)

    contract_hits = [
        h for h in hits 
        if contract_key[:20] in h.get("data", {}).get("contract", "").lower()
    ]

    if contract_hits:
        top_score = contract_hits[0].get("_score", 0.0)
        sys_present = top_score >= threshold
        txt = contract_hits[0].get("data", {}).get("object", "")[:80]
    else:
        top_score = 0.0; sys_present = False; txt = ""

    return {"sys_present": sys_present, "top_score": round(top_score, 4), "predicted_text": txt}


# ── Metrics ───────────────────────────────────────────────────────────────────

def run_classify_eval(csv_path: str, limit: int, out_path: str):
    print(f"\n{'='*75}")
    print("  MODE: Classification — predict clause type from raw text")
    print("  AUDIT: HyperBinder (Multi-slot) vs Vanilla (Global Search)")
    print(f"{'='*75}\n")

    df = pd.read_csv(csv_path, nrows=limit)
    results = []; rescued_gold = []; t0 = time.time()

    for _, row in df.iterrows():
        contract_name = Path(str(row.get("Filename", "unknown"))).stem[:40]

        for ct in RISK_CLAUSE_TYPES:
            present, text = get_ground_truth(row, ct)
            if not present or len(text) < 50: continue

            pred = eval_classify(text, ct)
            res = {"contract": contract_name, "clause_type": ct, **pred, "text": text}
            results.append(res)
            if pred["is_rescued"]: rescued_gold.append(res)

    # -- Summary Table --
    print(f"  {'Clause Type':<35} {'N':>4}  {'HB Acc':>8}  {'Van Acc':>8}  {'Gain'}")
    print(f"  {'-'*35} {'-'*4}  {'-'*8}  {'-'*8}  {'----'}")
    
    for ct in RISK_CLAUSE_TYPES:
        rows = [r for r in results if r["clause_type"] == ct]
        if not rows: continue
        hb_acc = sum(1 for r in rows if r["hb_correct"]) / len(rows)
        v_acc = sum(1 for r in rows if r["v_correct"]) / len(rows)
        print(f"  {ct:<35} {len(rows):>4}  {hb_acc:>8.1%}  {v_acc:>8.1%}  {hb_acc-v_acc:>+5.1%}")

    # -- Rescued Goldmines --
    if rescued_gold:
        print(f"\n{'='*75}")
        print("  TOP RESCUED CLAUSES FOR LIVE COMPARISON UI (Mining complete)")
        print(f"{'='*75}")
        for r in rescued_gold[:5]:
            print(f"\n[TYPE]: {r['clause_type']}  | [VANILLA GUESSED]: {r['v_pred']}")
            print(f"[TEXT]: {r['text'][:300]}...")

    pd.DataFrame(results).to_csv(out_path, index=False)


def run_presence_eval(csv_path: str, limit: int, out_path: str):
    """(Kept from your original logic)"""
    print(f"\n{'='*70}\n  MODE: Presence detection\n{'='*70}\n")
    df = pd.read_csv(csv_path, nrows=limit); results = []; t0 = time.time()
    for _, row in df.iterrows():
        name = Path(str(row.get("Filename", ""))).stem[:60]
        for ct in RISK_CLAUSE_TYPES:
            gt_p, gt_txt = get_ground_truth(row, ct)
            pred = eval_presence(name, name.lower(), ct)
            results.append({"clause_type": ct, "gt_present": gt_p, "sys_present": pred["sys_present"], "score": pred["top_score"]})
    
    # Simple summary
    for ct in RISK_CLAUSE_TYPES:
        rows = [r for r in results if r["clause_type"] == ct]
        if not rows: continue
        acc = sum(1 for r in rows if r["gt_present"] == r["sys_present"]) / len(rows)
        print(f"  {ct:<35} Accuracy: {acc:.1%}")

    pd.DataFrame(results).to_csv(out_path, index=False)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["classify", "presence"], default="classify")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--csv", default="CUAD_v1/master_clauses.csv")
    args = parser.parse_args()

    if args.mode == "classify":
        run_classify_eval(args.csv, args.limit, "classify_results.csv")
    else:
        run_presence_eval(args.csv, args.limit, "presence_results.csv")

if __name__ == "__main__":
    main()