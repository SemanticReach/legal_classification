"""
maud_ingest.py — MAUD Merger Agreement HyperBinder Ingest
==========================================================

Ingests 39,231 annotated clause triples from the MAUD dataset
(152 M&A merger agreements × 22 legal question types).

Slot design:
    subject     — semantic: contract name (merger agreement identity)
    predicate   — semantic: text_type (legal question category)
    object      — semantic: clause text extracted from agreement
    answer      — exact: human-annotated answer label (229 unique values)
    question_type — exact: normalized text_type for symbolic filtering
    category    — exact: high-level grouping (Deal Protection, MAE, etc.)
    split       — exact: "train" / "validation" / "test" (for held-out eval)

NOTE on splits:
    All three splits (train/val/test) are ingested into the index so the
    full 152-contract knowledge base is available for retrieval and risk
    detection. The split field is stored so held-out evaluation can filter
    to test-only records — fixing the in-sample criticism from CUAD.

Embedding model: nlpaueb/legal-bert-base-uncased (768-dim)
    Same model as the CUAD system for consistency.

Workflow:
    # Step 1 — precompute embeddings (once, ~15-20 min)
    python maud_ingest.py --precompute

    # Step 2 — wipe and ingest
    python maud_ingest.py --wipe

    # Test run (train split only, ~5 min)
    python maud_ingest.py --precompute --split train
    python maud_ingest.py --wipe --split train
"""


from __future__ import annotations

import argparse
import io
import json
import os
import pickle
import re
import time
from pathlib import Path
from typing import Optional

import requests
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────

HB_SERVER_URL = os.environ.get("HB_SERVER_URL", os.environ.get("SERVER_URL", ""))
HB_API_KEY    = os.environ.get("HB_API_KEY",    os.environ.get("API_KEY",    ""))
HB_DB_NAME    = os.environ.get("HB_DB_NAME",    "fraud_db")
NAMESPACE     = "maud_clauses"

CACHE_PATH    = "maud_embeddings_cache.pkl"
BATCH_SIZE    = 500
VECTOR_COL    = "precomputed_vectors"
EMBED_DIM     = 768
TIMEOUT       = 300
MAX_RETRIES   = 3

EMBED_MODEL   = "nlpaueb/legal-bert-base-uncased"

# All 22 MAUD text types (legal question categories)
TEXT_TYPES = [
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

TEMPLATE_SCHEMA = json.dumps({
    "molecule": "Row",
    "semantic_fields": ["subject", "predicate", "object"],
    "primary_key": "record_id",
    "fields": {
        "record_id":      {"encoding": "exact"},
        "subject":        {"encoding": "semantic"},
        "predicate":      {"encoding": "semantic"},
        "object":         {"encoding": "semantic"},
        "answer":         {"encoding": "exact"},
        "question_type":  {"encoding": "exact"},   # symbolic anchor for filtering
        "category":       {"encoding": "exact"},
        "split":          {"encoding": "exact"},
        "contract_name":  {"encoding": "exact"},
    },
    "field_order": [
        "record_id", "subject", "predicate", "object",
        "answer", "question_type", "category", "split", "contract_name"
    ]
})


# ── Feature Engineering ───────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Strip page references from clause text.
    MAUD stores text as "...clause language...  (Page N)"
    """
    if not text:
        return ""
    # Remove trailing (Page N) or (Pages N-M)
    text = re.sub(r'\s*\(Pages?\s+[\d\-–]+\)\s*$', '', text.strip())
    return text.strip()


def build_maud_triples(
    splits: list[str] = None,
) -> list[dict]:
    """
    Load MAUD from Hugging Face and build clause triples.
    Skips records with no answer text.

    splits: list of splits to include, e.g. ["train", "validation", "test"]
            defaults to all three
    """
    from datasets import load_dataset

    if splits is None:
        splits = ["train", "validation", "test"]

    print(f"\n  Loading MAUD dataset from HuggingFace...")
    ds = load_dataset("theatticusproject/maud")

    triples = []
    skipped = 0

    for split in splits:
        data = ds[split]
        print(f"  Processing {split}: {len(data):,} records")

        for record in data:
            text_type     = record.get("text_type", "") or ""
            contract_name = record.get("contract_name", "") or ""
            text          = record.get("text", "") or ""
            answer        = record.get("answer", "") or ""
            category      = record.get("category", "") or ""
            record_id     = str(record.get("id", ""))

            # Skip if no clause text or no answer
            if not text.strip() or not answer.strip():
                skipped += 1
                continue

            # Skip None answers
            if answer is None or str(answer).strip().lower() in ("", "none", "null"):
                skipped += 1
                continue

            clause_text = clean_text(text)
            if not clause_text:
                skipped += 1
                continue

            unique_id = f"{contract_name}__{text_type.replace(' ', '_')}__{record_id}"

            triples.append({
                "record_id":     unique_id,
                "subject":       contract_name,
                "predicate":     text_type,
                "object":        clause_text[:2000],
                "answer":        str(answer).strip(),
                "question_type": text_type,
                "category":      category,
                "split":         split,
                "contract_name": contract_name,
            })

    print(f"\n  Built {len(triples):,} triples ({skipped:,} skipped)")
    return triples


# ── Precompute embeddings ─────────────────────────────────────────────────────

def precompute_embeddings(
    triples:    list[dict],
    cache_path: str = CACHE_PATH,
) -> dict[str, list[float]]:
    """
    Batch-encode all unique subject, predicate, and object strings.
    """
    from sentence_transformers import SentenceTransformer

    cache: dict[str, list[float]] = {}
    if Path(cache_path).exists():
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
        print(f"  ✓ Loaded cache: {len(cache):,} embeddings")

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

    print(f"  Loading {EMBED_MODEL}...")
    model      = SentenceTransformer(EMBED_MODEL)
    batch_size = 64
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
    import pandas as pd

    rows    = []
    missing = 0

    for t in triples:
        vec = cache.get(t["object"])
        if vec is None:
            missing += 1
            vec = [0.0] * EMBED_DIM

        rows.append({
            "record_id":     t["record_id"],
            "subject":       t["subject"],
            "predicate":     t["predicate"],
            "object":        t["object"],
            "answer":        t["answer"],
            "question_type": t["question_type"],
            "category":      t["category"],
            "split":         t["split"],
            "contract_name": t["contract_name"],
            VECTOR_COL:      json.dumps(vec),
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
    splits:     list[str] = None,
    batch_size: int       = BATCH_SIZE,
    wipe:       bool      = False,
    cache_path: str       = CACHE_PATH,
) -> None:
    if splits is None:
        splits = ["train", "validation", "test"]

    print("\n" + "=" * 65)
    print("  maud_ingest.py — MAUD Merger Agreement Ingest")
    print(f"  Server    : {HB_SERVER_URL}")
    print(f"  DB        : {HB_DB_NAME} / {NAMESPACE}")
    print(f"  Splits    : {splits}")
    print(f"  Model     : {EMBED_MODEL}")
    print(f"  Wipe      : {wipe}")
    print("=" * 65)

    triples = build_maud_triples(splits=splits)
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

    missing  = sum(1 for t in triples if t["object"] not in cache)
    coverage = (1 - missing / max(len(triples), 1)) * 100
    print(f"  Cache coverage: {coverage:.1f}%")

    if wipe:
        wipe_namespace()

    total_added   = 0
    total_missing = 0
    total_batches = (len(triples) + batch_size - 1) // batch_size
    t0            = time.time()

    print(f"\n  Ingesting {len(triples):,} triples in {total_batches} batch(es)...\n")

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
            f"+{added:4d} triples  "
            f"{rate:5.0f}/s  "
            f"ETA {remaining/60:.1f}m"
        )

    elapsed     = time.time() - t0
    final_count = get_namespace_count()

    print(f"\n{'='*65}")
    print(f"  ✓ INGEST COMPLETE")
    print(f"  Triples ingested : {total_added:,}")
    print(f"  Cache misses     : {total_missing:,}")
    print(f"  Total time       : {elapsed:.1f}s")
    if final_count >= 0:
        print(f"  Namespace count  : {final_count:,}")
    print(f"{'='*65}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ingest MAUD merger agreement clauses into HyperBinder"
    )
    parser.add_argument("--split",      type=str, default=None,
                        help="Ingest one split only: train, validation, or test "
                             "(default: all three)")
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

    splits = [args.split] if args.split else ["train", "validation", "test"]

    if args.precompute:
        print("\n" + "=" * 65)
        print("  maud_ingest.py — precompute embeddings only")
        print("=" * 65)
        triples = build_maud_triples(splits=splits)
        precompute_embeddings(triples, args.cache)
    else:
        run(
            splits     = splits,
            batch_size = args.batch_size,
            wipe       = args.wipe,
            cache_path = args.cache,
        )