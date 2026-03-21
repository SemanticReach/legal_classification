"""
legal_ingest.py — CUAD Contract Clause HyperBinder Ingest
==========================================================

Ingests 510 contracts × 41 clause types = ~20k clause triples.

Slot design:
    subject   — semantic: contract filename (embedded so query-time subject
                matching works; use filter_by_contract() in legal_query.py
                for exact contract scoping via post-filter)
    predicate — semantic: clause type ("Termination For Convenience",
                "Governing Law", "Non-Compete", etc.)
    object    — semantic: actual clause text extracted from contract
    answer    — exact: "Yes" / "No" (clause present or not)
    clause_type — exact: normalized clause type name

NOTE on subject encoding:
    subject is now "semantic" (was "exact") so that contract name embeddings
    are precomputed and stored. This allows partial/fuzzy contract name
    matching at query time. For strict contract scoping, use the
    post-filter approach in legal_query.py (filter_by_contract) which
    does a substring match against the stored 'contract' exact field.

Embedding model: nlpaueb/legal-bert-base-uncased (768-dim)
    To switch back to MiniLM, set EMBED_MODEL and EMBED_DIM in config.

Workflow:
    # Step 1 — delete old cache (different dim than MiniLM)
    del legal_embeddings_cache.pkl

    # Step 2 — precompute embeddings with Legal-BERT (once, ~8-10 min)
    python legal_ingest.py --precompute

    # Step 3 — wipe and re-ingest
    python legal_ingest.py --wipe

    # Test run (first 50 contracts, ~1 min)
    python legal_ingest.py --precompute --limit 50
    python legal_ingest.py --wipe --limit 50
"""

from __future__ import annotations

import argparse
import ast
import io
import json
import os
import pickle
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from dotenv import load_dotenv

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["CUDA_VISIBLE_DEVICES"]  = ""

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────

HB_SERVER_URL = os.environ.get("HB_SERVER_URL", os.environ.get("SERVER_URL", ""))
HB_API_KEY    = os.environ.get("HB_API_KEY",    os.environ.get("API_KEY",    ""))
HB_DB_NAME    = os.environ.get("HB_DB_NAME",    "fraud_db")
NAMESPACE     = "cuad_clauses"

CACHE_PATH    = "legal_embeddings_cache_legalbert.pkl"  # separate from MiniLM cache
BATCH_SIZE    = 500
VECTOR_COL    = "precomputed_vectors"
EMBED_DIM     = 768   # legal-bert-base-uncased is 768-dim (was 384 for MiniLM)
TIMEOUT       = 300
MAX_RETRIES   = 3

# Embedding model — swap here to change models
# Options:
#   "nlpaueb/legal-bert-base-uncased"  — legal domain, 768-dim, best quality (default)
#   "all-MiniLM-L6-v2"                 — general purpose, 384-dim, faster
EMBED_MODEL   = "nlpaueb/legal-bert-base-uncased"

# All 41 CUAD clause types
CLAUSE_TYPES = [
    "Parties", "Agreement Date", "Effective Date", "Expiration Date",
    "Renewal Term", "Notice Period To Terminate Renewal", "Governing Law",
    "Most Favored Nation", "Competitive Restriction Exception", "Non-Compete",
    "Exclusivity", "No-Solicit Of Customers", "No-Solicit Of Employees",
    "Non-Disparagement", "Termination For Convenience", "Rofr/Rofo/Rofn",
    "Change Of Control", "Anti-Assignment", "Revenue/Profit Sharing",
    "Price Restrictions", "Minimum Commitment", "Volume Restriction",
    "Ip Ownership Assignment", "Joint Ip Ownership", "License Grant",
    "Non-Transferable License", "Affiliate License-Licensor",
    "Affiliate License-Licensee", "Unlimited/All-You-Can-Eat-License",
    "Irrevocable Or Perpetual License", "Source Code Escrow",
    "Post-Termination Services", "Audit Rights", "Uncapped Liability",
    "Cap On Liability", "Liquidated Damages", "Warranty Duration",
    "Insurance", "Covenant Not To Sue", "Third Party Beneficiary",
]

# subject is now "semantic" so contract name embeddings are precomputed
# and stored — enabling query-time matching against the subject slot.
TEMPLATE_SCHEMA = json.dumps({
    "molecule": "Row",
    "semantic_fields": ["subject", "predicate", "object"],
    "primary_key": "clause_id",
    "fields": {
        "clause_id":   {"encoding": "exact"},
        "subject":     {"encoding": "semantic"},   # was "exact" — now embedded
        "predicate":   {"encoding": "semantic"},
        "object":      {"encoding": "semantic"},
        "answer":      {"encoding": "exact"},
        "clause_type": {"encoding": "exact"},
        "contract":    {"encoding": "exact"},      # kept exact for post-filter
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
            # Join multiple excerpts with separator
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
    Skips empty clauses (answer = No with no text).
    """
    print(f"\n  Loading {csv_path}...")
    df = pd.read_csv(csv_path, nrows=limit)
    print(f"  Loaded {len(df):,} contracts")

    triples = []
    skipped = 0

    for _, row in df.iterrows():
        contract = str(row.get("Filename", "unknown"))
        # Clean up filename for display
        contract_short = Path(contract).stem[:60]

        for clause_type in CLAUSE_TYPES:
            # CUAD column convention:
            #   "ClauseType-Answer" = "Yes" / "No" presence flag
            #   "ClauseType"        = actual extracted clause text (list-string)
            answer_col = f"{clause_type}-Answer"
            text_col   = clause_type

            # Check presence flag — skip if clause not in this contract
            flag_raw = row.get(answer_col, "")
            flag     = str(flag_raw).strip().lower() if not pd.isna(flag_raw) else ""
            answer   = "Yes" if flag == "yes" else "No"

            if answer != "Yes":
                skipped += 1
                continue

            # Get actual clause text from the bare clause_type column
            clause_text = extract_clause_text(row.get(text_col, ""))

            # Skip if no actual text — not useful for semantic retrieval
            if not clause_text:
                skipped += 1
                continue

            clause_id = f"{contract_short}__{clause_type.replace(' ', '_')}"

            triples.append({
                "clause_id":   clause_id,
                "subject":     contract_short,
                "predicate":   clause_type,
                "object":      clause_text[:2000],  # cap at 2000 chars
                "answer":      answer,
                "clause_type": clause_type,
                "contract":    contract_short,
            })

    print(f"  Built {len(triples):,} clause triples "
          f"({skipped:,} empty clauses skipped)")
    return triples


# ── Precompute embeddings ─────────────────────────────────────────────────────

def precompute_embeddings(
    triples:    list[dict],
    cache_path: str = CACHE_PATH,
) -> dict[str, list[float]]:
    """
    Batch-encode all unique subject, predicate, and object strings.
    All three semantic slots are embedded so query-time matching works
    correctly for contract name, clause type, and clause content.
    """
    from sentence_transformers import SentenceTransformer

    cache: dict[str, list[float]] = {}
    if Path(cache_path).exists():
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
        print(f"  ✓ Loaded cache: {len(cache):,} embeddings")

    # Encode subject, predicate, AND object (subject was missing before)
    all_texts = list(
        {t["subject"]   for t in triples} |
        {t["predicate"] for t in triples} |
        {t["object"]    for t in triples}
    )
    new_texts = [t for t in all_texts if t not in cache]
    print(f"  Texts to encode: {len(new_texts):,} new / {len(all_texts):,} total")

    if not new_texts:
        print("  ✓ Cache complete")
        return cache

    print(f"  Loading sentence-transformer ({EMBED_MODEL})...")
    print(f"  Embedding dim: {EMBED_DIM}")
    model      = SentenceTransformer(EMBED_MODEL)
    batch_size = 64   # smaller batches for Legal-BERT (larger model)
    total      = len(new_texts)
    t0         = time.time()

    for i in range(0, total, batch_size):
        batch   = new_texts[i : i + batch_size]
        vectors = model.encode(batch, show_progress_bar=False)
        for text, vec in zip(batch, vectors):
            cache[text] = vec.tolist()

        pct  = min((i + batch_size) / total * 100, 100)
        rate = (i + batch_size) / (time.time() - t0 + 0.001)
        eta  = (total - i - batch_size) / rate if rate > 0 else 0
        print(f"  [{pct:5.1f}%] {min(i+batch_size,total):,}/{total:,}  "
              f"{rate:.0f}/s  ETA {eta:.0f}s")

    with open(cache_path, "wb") as f:
        pickle.dump(cache, f)
    print(f"  ✓ Cache saved: {len(cache):,} embeddings → {cache_path}")
    return cache


# ── Namespace management ──────────────────────────────────────────────────────

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
    triples:   list[dict],
    cache:     dict[str, list[float]],
    batch_num: int,
    is_first:  bool,
) -> tuple[int, int]:
    rows    = []
    missing = 0

    for t in triples:
        # Use object (clause text) embedding as the primary precomputed vector.
        # subject and predicate embeddings are also in cache and will be
        # used by HyperBinder for their respective semantic slots.
        vec = cache.get(t["object"])
        if vec is None:
            missing += 1
            vec = [0.0] * EMBED_DIM

        rows.append({
            "clause_id":   t["clause_id"],
            "subject":     t["subject"],
            "predicate":   t["predicate"],
            "object":      t["object"],
            "answer":      t["answer"],
            "clause_type": t["clause_type"],
            "contract":    t["contract"],
            VECTOR_COL:    json.dumps(vec),
        })

    if not rows:
        return 0, missing

    df  = pd.DataFrame(rows)
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    mode = "Create" if is_first else "Append"

    for attempt in range(1, MAX_RETRIES + 1):
        buf.seek(0)
        try:
            resp = requests.post(
                f"{HB_SERVER_URL}/build_ingest_data/",
                headers={"X-API-Key": HB_API_KEY},
                files={"file": (f"batch_{batch_num:04d}.csv", buf, "text/csv")},
                data={
                    "dim":             EMBED_DIM,
                    "seed":            42,
                    "depth":           3,
                    "db_name":         HB_DB_NAME,
                    "namespace":       NAMESPACE,
                    "template_schema": TEMPLATE_SCHEMA,
                    "vector_col":      VECTOR_COL,
                    "mode":            mode,
                },
                timeout=TIMEOUT,
            )

            if resp.status_code == 200:
                result     = resp.json()
                rows_added = result.get("rows_added", len(rows))
                vec_source = result.get("vector_source", "unknown")
                if batch_num == 1:
                    print(f"  ✓ vector_source = {vec_source}")
                    if vec_source != "precomputed":
                        print("  ⚠️  WARNING: not using precomputed vectors!")
                return rows_added, missing

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

    return 0, missing


# ── Main ingest ───────────────────────────────────────────────────────────────

def run(
    csv_path:   str           = "CUAD_v1/master_clauses.csv",
    limit:      Optional[int] = None,
    batch_size: int           = BATCH_SIZE,
    wipe:       bool          = False,
    cache_path: str           = CACHE_PATH,
) -> None:
    print("\n" + "=" * 65)
    print("  legal_ingest.py — CUAD Contract Clause Ingest")
    print(f"  Server    : {HB_SERVER_URL}")
    print(f"  DB        : {HB_DB_NAME} / {NAMESPACE}")
    print(f"  CSV       : {csv_path}")
    print(f"  Limit     : {limit or 'all 510 contracts'}")
    print(f"  Clauses   : {len(CLAUSE_TYPES)} types per contract")
    print(f"  Wipe      : {wipe}")
    print("=" * 65)

    triples = build_legal_triples(csv_path, limit=limit)
    if not triples:
        print("  ✗ No triples built")
        return

    # Load or build cache
    if Path(cache_path).exists():
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
        print(f"\n  ✓ Loaded cache: {len(cache):,} embeddings")
    else:
        print("\n  No cache — building now (run --precompute first next time)")
        cache = precompute_embeddings(triples, cache_path)

    missing = sum(1 for t in triples if t["object"] not in cache)
    coverage = (1 - missing / max(len(triples), 1)) * 100
    print(f"  Cache coverage: {coverage:.1f}%")

    if wipe:
        wipe_namespace()

    total_added   = 0
    total_missing = 0
    total_batches = (len(triples) + batch_size - 1) // batch_size
    t0            = time.time()

    print(f"\n  Ingesting {len(triples):,} clauses in {total_batches} batch(es)...\n")

    for i in range(0, len(triples), batch_size):
        batch     = triples[i : i + batch_size]
        batch_num = i // batch_size + 1
        is_first  = (i == 0) and wipe

        added, miss   = ingest_batch(batch, cache, batch_num, is_first)
        total_added   += added
        total_missing += miss

        pct       = batch_num / total_batches * 100
        elapsed   = time.time() - t0
        rate      = total_added / elapsed if elapsed > 0 else 0
        remaining = (len(triples) - total_added) / rate if rate > 0 else 0

        print(
            f"  Batch {batch_num:3d}/{total_batches}  "
            f"[{pct:5.1f}%]  "
            f"+{added:4d} clauses  "
            f"{rate:5.0f}/s  "
            f"ETA {remaining/60:.1f}m"
        )

    elapsed     = time.time() - t0
    final_count = get_namespace_count()

    print(f"\n{'='*65}")
    print(f"  ✓ INGEST COMPLETE")
    print(f"  Clauses ingested : {total_added:,}")
    print(f"  Cache misses     : {total_missing:,}")
    print(f"  Total time       : {elapsed:.1f}s")
    if final_count >= 0:
        print(f"  Namespace count  : {final_count:,}")
    print(f"{'='*65}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ingest CUAD contract clauses into HyperBinder"
    )
    parser.add_argument("--csv",        default="CUAD_v1/master_clauses.csv")
    parser.add_argument("--limit",      type=int, default=None,
                        help="Max contracts to process (default: all 510)")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--cache",      default=CACHE_PATH)
    parser.add_argument("--wipe",       action="store_true")
    parser.add_argument("--precompute", action="store_true",
                        help="Precompute embeddings only, do not ingest")
    parser.add_argument("--namespace",  default=NAMESPACE)
    parser.add_argument("--db",         default=HB_DB_NAME)
    args = parser.parse_args()

    NAMESPACE  = args.namespace
    HB_DB_NAME = args.db

    if args.precompute:
        print("\n" + "=" * 65)
        print("  legal_ingest.py — precompute embeddings only")
        print("=" * 65)
        triples = build_legal_triples(args.csv, limit=args.limit)
        precompute_embeddings(triples, args.cache)
    else:
        run(
            csv_path   = args.csv,
            limit      = args.limit,
            batch_size = args.batch_size,
            wipe       = args.wipe,
            cache_path = args.cache,
        )