"""
legal_ingest.py — CUAD Contract Clause HyperBinder Ingest
==========================================================

Ingests 510 contracts × 41 clause types = ~20k clause triples.

Slot design:
    subject   — semantic: contract filename
    predicate — semantic: clause type
    object    — semantic: actual clause text
    answer    — exact: "Yes" / "No"
    clause_type — exact: normalized clause type name
    contract  — exact: contract filename (for post-filtering)

Workflow:
    # Step 1 — wipe and re-ingest (server generates embeddings)
    python legal_ingest.py --wipe --new-db

    # Test run (first 50 contracts)
    python legal_ingest.py --wipe --limit 50 --new-db
"""

from __future__ import annotations

import argparse
import ast
import io
import json
import os
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from dotenv import load_dotenv

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────

HB_SERVER_URL = os.environ.get("HB_SERVER_URL", os.environ.get("SERVER_URL", ""))
HB_API_KEY = os.environ.get("HB_API_KEY", os.environ.get("API_KEY", ""))
HB_DB_NAME = os.environ.get("HB_DB_NAME", "fraud_db_v2")  # ← New database name
NAMESPACE = "cuad_clauses"

BATCH_SIZE = 500
EMBED_DIM = 1024  # ← Higher dimension to support phase_dim=512
TIMEOUT = 300
MAX_RETRIES = 3

# All 41 CUAD clause types
CLAUSE_TYPES = [
    "Parties",
    "Agreement Date",
    "Effective Date",
    "Expiration Date",
    "Renewal Term",
    "Notice Period To Terminate Renewal",
    "Governing Law",
    "Most Favored Nation",
    "Competitive Restriction Exception",
    "Non-Compete",
    "Exclusivity",
    "No-Solicit Of Customers",
    "No-Solicit Of Employees",
    "Non-Disparagement",
    "Termination For Convenience",
    "Rofr/Rofo/Rofn",
    "Change Of Control",
    "Anti-Assignment",
    "Revenue/Profit Sharing",
    "Price Restrictions",
    "Minimum Commitment",
    "Volume Restriction",
    "Ip Ownership Assignment",
    "Joint Ip Ownership",
    "License Grant",
    "Non-Transferable License",
    "Affiliate License-Licensor",
    "Affiliate License-Licensee",
    "Unlimited/All-You-Can-Eat-License",
    "Irrevocable Or Perpetual License",
    "Source Code Escrow",
    "Post-Termination Services",
    "Audit Rights",
    "Uncapped Liability",
    "Cap On Liability",
    "Liquidated Damages",
    "Warranty Duration",
    "Insurance",
    "Covenant Not To Sue",
    "Third Party Beneficiary",
]

# Schema with semantic fields — server will generate embeddings
TEMPLATE_SCHEMA = json.dumps({
    "molecule": "Row",
    "primary_key": {"name": "clause_id", "encoding": "exact"},
    "fields": {
        "clause_id": {"name": "clause_id", "encoding": "exact"},
        "subject": {"name": "subject", "encoding": "semantic"},
        "predicate": {"name": "predicate", "encoding": "semantic"},
        "object": {"name": "object", "encoding": "semantic"},
        "answer": {"name": "answer", "encoding": "exact"},
        "clause_type": {"name": "clause_type", "encoding": "exact"},
        "contract": {"name": "contract", "encoding": "exact"},
    },
    "field_order": [
        "clause_id", "subject", "predicate", "object",
        "answer", "clause_type", "contract"
    ]
})


# ── Feature Engineering ───────────────────────────────────────────────────────

def extract_clause_text(raw) -> str:
    """
    Extract readable text from CUAD answer field.
    Fields are stored as Python list strings like "['text here']"
    or plain strings or NaN.
    """
    if pd.isna(raw) or raw == "" or raw == "[]":
        return ""
    try:
        parsed = ast.literal_eval(str(raw))
        if isinstance(parsed, list) and parsed:
            return " | ".join(str(p).strip() for p in parsed if p)
        return str(parsed).strip()
    except Exception:
        return str(raw).strip()


def build_legal_triples(
    csv_path: str,
    limit: Optional[int] = None,
) -> list[dict]:
    """
    Build one triple per (contract, clause_type) pair where clause text exists.
    """
    print(f"\n  Loading {csv_path}...")
    df = pd.read_csv(csv_path, nrows=limit)
    print(f"  Loaded {len(df):,} contracts")

    triples = []
    skipped = 0

    for _, row in df.iterrows():
        contract = str(row.get("Filename", "unknown"))
        contract_short = Path(contract).stem[:60]

        for clause_type in CLAUSE_TYPES:
            answer_col = f"{clause_type}-Answer"
            text_col = clause_type

            flag_raw = row.get(answer_col, "")
            flag = str(flag_raw).strip().lower() if not pd.isna(flag_raw) else ""
            answer = "Yes" if flag == "yes" else "No"

            if answer != "Yes":
                skipped += 1
                continue

            clause_text = extract_clause_text(row.get(text_col, ""))

            if not clause_text:
                skipped += 1
                continue

            clause_id = f"{contract_short}__{clause_type.replace(' ', '_')}"

            triples.append({
                "clause_id": clause_id,
                "subject": contract_short,
                "predicate": clause_type,
                "object": clause_text[:2000],
                "answer": answer,
                "clause_type": clause_type,
                "contract": contract_short,
            })

    print(f"  Built {len(triples):,} clause triples "
          f"({skipped:,} empty clauses skipped)")
    return triples


# ── Database management ──────────────────────────────────────────────────────

def delete_database(db_name: str) -> None:
    """Delete an entire database."""
    print(f"  Deleting database '{db_name}'...")
    resp = requests.delete(
        f"{HB_SERVER_URL}/db/{db_name}",
        headers={"X-API-Key": HB_API_KEY},
        timeout=30,
    )
    if resp.status_code in (200, 404):
        print(f"  ✓ Deleted (status {resp.status_code})")
    else:
        print(f"  ⚠️  {resp.status_code}: {resp.text[:100]}")


def wipe_namespace() -> None:
    print(f"  Wiping namespace '{NAMESPACE}'...")
    resp = requests.delete(
        f"{HB_SERVER_URL}/db/{HB_DB_NAME}/namespace/{NAMESPACE}",
        headers={"X-API-Key": HB_API_KEY},
        timeout=30,
    )
    if resp.status_code in (200, 404):
        print(f"  ✓ Wiped (status {resp.status_code})")
    else:
        print(f"  ⚠️  {resp.status_code}: {resp.text[:100]}")


def get_namespace_count() -> int:
    try:
        resp = requests.get(
            f"{HB_SERVER_URL}/namespace/{HB_DB_NAME}/{NAMESPACE}/count",
            headers={"X-API-Key": HB_API_KEY},
            timeout=10,
        )
        if resp.status_code == 200:
            return resp.json().get("count", 0)
    except Exception:
        pass
    return -1


# ── Batched ingest ────────────────────────────────────────────────────────────

def ingest_batch(
    triples: list[dict],
    batch_num: int,
    is_first: bool,
) -> tuple[int, int]:
    rows = []

    for t in triples:
        rows.append({
            "clause_id": t["clause_id"],
            "subject": t["subject"],
            "predicate": t["predicate"],
            "object": t["object"],
            "answer": t["answer"],
            "clause_type": t["clause_type"],
            "contract": t["contract"],
        })

    if not rows:
        return 0, 0

    df = pd.DataFrame(rows)
    buf = io.BytesIO()
    df.to_csv(buf, index=False)

    for attempt in range(1, MAX_RETRIES + 1):
        buf.seek(0)
        try:
            resp = requests.post(
                f"{HB_SERVER_URL}/build_ingest_data/",
                headers={"X-API-Key": HB_API_KEY},
                files={"file": (f"batch_{batch_num:04d}.csv", buf, "text/csv")},
                data={
                    "dim": EMBED_DIM,
                    "seed": 42,
                    "depth": 3,
                    "db_name": HB_DB_NAME,
                    "namespace": NAMESPACE,
                    "template_schema": TEMPLATE_SCHEMA,
                    "on_conflict": "error",
                    "skip_semantic_encoding": "false",
                },
                timeout=TIMEOUT,
            )

            if resp.status_code == 200:
                result = resp.json()
                rows_added = result.get("rows_added", len(rows))
                vec_source = result.get("vector_source", "unknown")
                if batch_num == 1:
                    print(f"  ✓ vector_source = {vec_source}")
                return rows_added, 0

            else:
                print(f"  ⚠️  Batch {batch_num} attempt {attempt} "
                      f"— {resp.status_code}: {resp.text[:80]}")

        except requests.exceptions.ReadTimeout:
            print(f"  ⏱️  Batch {batch_num} attempt {attempt} timed out")
        except requests.exceptions.ConnectionError as e:
            print(f"  ✗  Connection error: {e}")

        if attempt < MAX_RETRIES:
            wait = 10 * attempt
            print(f"     Retrying in {wait}s...")
            time.sleep(wait)

    return 0, 0


# ── Main ingest ───────────────────────────────────────────────────────────────

def run(
    csv_path: str = "CUAD_v1/master_clauses.csv",
    limit: Optional[int] = None,
    batch_size: int = BATCH_SIZE,
    wipe: bool = False,
    new_db: bool = False,
) -> None:
    print("\n" + "=" * 65)
    print("  legal_ingest.py — CUAD Contract Clause Ingest")
    print(f"  Server    : {HB_SERVER_URL}")
    print(f"  DB        : {HB_DB_NAME} / {NAMESPACE}")
    print(f"  CSV       : {csv_path}")
    print(f"  Limit     : {limit or 'all 510 contracts'}")
    print(f"  Clauses   : {len(CLAUSE_TYPES)} types per contract")
    print(f"  Wipe      : {wipe}")
    print(f"  New DB    : {new_db}")
    print(f"  Dim       : {EMBED_DIM}")
    print("  Vectors   : Server-generated from semantic fields")
    print("=" * 65)

    if new_db:
        delete_database(HB_DB_NAME)

    triples = build_legal_triples(csv_path, limit=limit)
    if not triples:
        print("  ✗ No triples built")
        return

    print(f"\n  ✓ {len(triples):,} clauses to ingest")
    print("  ✓ Server will generate embeddings for: subject, predicate, object")

    if wipe and not new_db:
        wipe_namespace()

    total_added = 0
    total_batches = (len(triples) + batch_size - 1) // batch_size
    t0 = time.time()

    print(f"\n  Ingesting {len(triples):,} clauses in {total_batches} batch(es)...\n")

    for i in range(0, len(triples), batch_size):
        batch = triples[i:i + batch_size]
        batch_num = i // batch_size + 1
        is_first = (i == 0) and (wipe or new_db)

        added, _ = ingest_batch(batch, batch_num, is_first)
        total_added += added

        pct = batch_num / total_batches * 100
        elapsed = time.time() - t0
        rate = total_added / elapsed if elapsed > 0 else 0
        remaining = (len(triples) - total_added) / rate if rate > 0 else 0

        print(
            f"  Batch {batch_num:3d}/{total_batches}  "
            f"[{pct:5.1f}%]  "
            f"+{added:4d} clauses  "
            f"{rate:5.0f}/s  "
            f"ETA {remaining / 60:.1f}m"
        )

    elapsed = time.time() - t0
    final_count = get_namespace_count()

    print(f"\n{'=' * 65}")
    print(f"  ✓ INGEST COMPLETE")
    print(f"  Clauses ingested : {total_added:,}")
    print(f"  Total time       : {elapsed:.1f}s")
    if final_count >= 0:
        print(f"  Namespace count  : {final_count:,}")
    print(f"{'=' * 65}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ingest CUAD contract clauses into HyperBinder"
    )
    parser.add_argument("--csv", default="CUAD_v1/master_clauses.csv")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max contracts to process (default: all 510)")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--wipe", action="store_true",
                        help="Wipe the namespace before ingesting")
    parser.add_argument("--new-db", action="store_true",
                        help="Delete and recreate the database")
    parser.add_argument("--namespace", default=NAMESPACE)
    parser.add_argument("--db", default=HB_DB_NAME)
    args = parser.parse_args()

    NAMESPACE = args.namespace
    HB_DB_NAME = args.db

    run(
        csv_path=args.csv,
        limit=args.limit,
        batch_size=args.batch_size,
        wipe=args.wipe,
        new_db=args.new_db,
    )