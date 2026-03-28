"""
legal_cached.py -- Legal Clause Classification and Risk Detection with Semantic Cache
=====================================================================================

Uses hybi.SemanticCache (local, disk-backed) in front of the remote HyperBinder
legal index. Cache runs in-process with Legal-BERT embeddings. Index queries go
over the network only on cache miss.

Architecture:
    Your script
        |
        +-- hybi SemanticCache (LOCAL, disk-backed at ./legal_cache_db/)
        |       Legal-BERT embeddings, context = EXACT domain isolation
        |       cache miss -> falls through to HyperBinder server
        |
        +-- HyperBinder server (REMOTE, your existing 510-contract index)
                queried only on cache miss

Usage:
    # First run  -- cold cache, all queries hit HyperBinder server
    python legal_cached.py --risk jpmorgan

    # Second run -- warm cache, results served locally in milliseconds
    python legal_cached.py --risk jpmorgan

    # Classify a clause
    python legal_cached.py --classify "JPMC may terminate for convenience with 90 days notice"

    # Clear the cache
    python legal_cached.py --clear-cache

Install:
    pip install hybi sentence-transformers
"""

from __future__ import annotations

import os
import json
import time
import argparse
import shutil
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path

import requests
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from hybi import HyperBinder, SemanticCache

load_dotenv()

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["CUDA_VISIBLE_DEVICES"]  = ""

# ── Config ────────────────────────────────────────────────────────────────────

SERVER_URL = os.environ.get("HB_SERVER_URL", os.environ.get("SERVER_URL", ""))
API_KEY    = os.environ.get("HB_API_KEY",    os.environ.get("API_KEY", ""))
DB_NAME    = os.environ.get("HB_DB_NAME",    "fraud_db")
NAMESPACE  = "cuad_clauses"

EMBED_MODEL     = "nlpaueb/legal-bert-base-uncased"
CACHE_DB_PATH   = "./legal_cache_db"
CACHE_THRESHOLD = 0.92
CACHE_TTL       = timedelta(days=7)

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

RISK_THRESHOLDS = {
    "Uncapped Liability":               (0.65, 0.82),
    "Cap On Liability":                 (0.50, 0.70),
    "Liquidated Damages":               (0.55, 0.72),
    "Non-Compete":                      (0.50, 0.68),
    "Anti-Assignment":                  (0.50, 0.65),
    "Change Of Control":                (0.50, 0.68),
    "Termination For Convenience":      (0.55, 0.70),
    "Ip Ownership Assignment":          (0.50, 0.68),
    "Irrevocable Or Perpetual License": (0.60, 0.80),
    "Covenant Not To Sue":              (0.50, 0.65),
}

# ── JPMorgan contract clauses ─────────────────────────────────────────────────

JPMORGAN_CLAUSES = {
    "Termination For Convenience": (
        "JPMC may terminate this Agreement or any Schedule(s) for convenience at any "
        "time by giving Supplier at least 90 days prior written notice of the termination "
        "date or such other period set forth in the applicable Schedule. JPMC will remain "
        "liable for fees and expenses incurred for all Deliverables Accepted by JPMC "
        "pursuant to the terminated Schedule(s) up to the effective date of termination "
        "for convenience."
    ),
    "Anti-Assignment": (
        "Neither party may assign any rights or delegate any obligations under this "
        "Agreement without the prior written consent of the other party. Notwithstanding "
        "any such consent, any assignment by Supplier that is not accompanied by an "
        "assumption agreement executed by the assignee shall be null and void. However, "
        "JPMC may assign this Agreement, any Schedule, or any of its rights hereunder or "
        "thereunder, in whole or in part, without Supplier's consent: (a) to any existing "
        "or future JPMC Entity, or (b) to a surviving entity in the case of a JPMC merger, "
        "acquisition, divestiture, consolidation or corporate reorganization (whether or "
        "not JPMC is the surviving entity). Any assignment or attempted assignment contrary "
        "to this Section 20.2 will be a material breach of this Agreement and null and void. "
        "For purposes of this Section 20.2, any merger, change of control or other "
        "combination by operation of law constitutes an assignment."
    ),
    "Ip Ownership Assignment": (
        "JPMC will own exclusively all Works (excluding Outside Materials) developed, in "
        "whole or in part, by or on behalf of Supplier for JPMC or Recipient pursuant to a "
        "Schedule together with all related Intellectual Property Rights throughout the world. "
        "Supplier will and hereby does, without further consideration, assign to JPMC any "
        "and all right, title or interest that Supplier may now or hereafter possess in or "
        "to the Developed Works. Supplier irrevocably designates and appoints JPMC its agent "
        "and attorney-in-fact to act for and on its behalf to execute, register and file any "
        "applications, and to do all other lawfully permitted acts, to further the "
        "registration, prosecution, issuance and enforcements of the Intellectual Property "
        "Rights in the Developed Works with the same legal force and effect as if executed, "
        "registered and filed by Supplier."
    ),
    "Cap On Liability": (
        "EXCEPT AS PROVIDED IN SECTIONS 16.2 AND 16.3 BELOW OR OTHERWISE SET FORTH IN "
        "THIS AGREEMENT, NEITHER PARTY WILL BE LIABLE TO THE OTHER PARTY FOR INDIRECT, "
        "INCIDENTAL, CONSEQUENTIAL, EXEMPLARY, PUNITIVE OR SPECIAL DAMAGES, INCLUDING "
        "LOST PROFITS, REGARDLESS OF THE FORM OF THE ACTION OR THE THEORY OF RECOVERY, "
        "EVEN IF THAT PARTY HAS BEEN ADVISED OF THE POSSIBILITY OF THOSE DAMAGES. "
        "Notwithstanding the foregoing, the limitations of liability set forth in this "
        "Section 16 will not apply to Losses in connection with: (a) death, personal "
        "injury or property damage caused by either party; (b) fraud, a party's gross "
        "negligence or the willful or reckless misconduct of a party; (c) either party's "
        "repudiation of their obligations under this Agreement; (d) Supplier's breach of "
        "the confidentiality provisions; (e) Supplier's breach of the privacy provisions; "
        "(f) any Security Breach; (g) claims pursuant to the indemnification provisions."
    ),
    "Change Of Control": (
        "For purposes of this Section 20.2, any merger, change of control or other "
        "combination by operation of law constitutes an assignment. If any merger or "
        "acquisition results in a JPMC Entity and Supplier having in effect other "
        "agreement(s) with the same general subject matter as this Agreement, JPMC may, "
        "at its option: (a) terminate this Agreement or the other agreement(s) in whole "
        "or in part (and without any JPMC Entity having any liability or incurring any "
        "additional charges to Supplier), and (b) require Supplier to enter into Schedules "
        "or other appropriate documents to move Supplier's delivery, license and services "
        "obligations from any terminated agreement to any continuing agreement."
    ),
    "Liquidated Damages": (
        "If Supplier fails to do so, JPMC may, at its option: (a) extend the required "
        "date and receive from Supplier, as liquidated damages and not as a penalty, the "
        "late delivery discount set forth in the applicable Schedule, or (b) terminate "
        "the applicable Schedule(s), in whole or in part, and receive from Supplier a "
        "refund of all amounts paid to Supplier relative to the late Deliverable."
    ),
    "Uncapped Liability": (
        "Supplier will indemnify, defend and hold harmless JPMorgan Chase & Co. from any "
        "and all losses, liabilities, damages (including taxes), and all related costs and "
        "expenses, including reasonable legal fees and disbursements and costs of "
        "investigation, litigation, settlement, judgment, interest and penalties incurred "
        "by itself or any of its direct or indirect officers, directors, employees, agents, "
        "successors or assigns, and threatened Losses due to, arising from or relating to "
        "third party claims arising from or relating to: (a) Supplier's actual or alleged "
        "breach of any warranties; (b) any actual or alleged infringement of Intellectual "
        "Property Rights; (c) Supplier's actual or alleged breach of confidentiality or "
        "privacy provisions; (d) any Security Breach; (e) fraud, negligent, willful or "
        "reckless acts or omissions of Supplier or any Supplier Personnel."
    ),
}

KNOWN_CONTRACTS = {
    "jpmorgan": ("JPMorganChase_Cardlytics_MSA_2018", JPMORGAN_CLAUSES),
}


# ── Cache setup ───────────────────────────────────────────────────────────────

def build_cache(threshold: float = CACHE_THRESHOLD) -> SemanticCache:
    """
    Build a local hybi SemanticCache backed by Legal-BERT embeddings.

    db_path persists the cache to disk between runs via LocalHyperBinder.
    encode_fn uses Legal-BERT to match the remote index embeddings exactly.

    Context keys use EXACT encoding for domain isolation:
        "classify:all"                     -- all classification queries
        "risk:Termination For Convenience" -- per-clause-type risk queries
    This means a Termination clause can never return a cached
    Anti-Assignment result, by construction.
    """
    print(f"  [cache] Loading {EMBED_MODEL}...")
    model = SentenceTransformer(EMBED_MODEL)

    hb = HyperBinder(
        local     = True,
        db_path   = CACHE_DB_PATH,
        encode_fn = model.encode,
    )

    cache = SemanticCache(
        hb,
        collection  = "legal_clause_cache",
        threshold   = threshold,
        default_ttl = CACHE_TTL,
    )

    print(f"  [cache] Ready  (db: {CACHE_DB_PATH}, threshold: {threshold})")
    return cache


# ── HyperBinder server queries ────────────────────────────────────────────────

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
                time.sleep(2 ** attempt)
                continue
            resp.raise_for_status()
            return resp.json().get("results", [])
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"  Query failed: {e}")
    return []


# ── Classification ────────────────────────────────────────────────────────────

def _classify_live(clause_text: str, top_k: int = 3) -> dict:
    all_scores = {}

    def query_one(clause_type):
        hits  = search_slots({
            "clause_type": {"query": clause_type,       "weight": 0.01, "encoding": "exact"},
            "object":      {"query": clause_text[:500], "weight": 1.0,  "encoding": "semantic"},
        }, top_k=top_k)
        score = hits[0].get("_score", 0.0)                          if hits else 0.0
        match = hits[0].get("data", {}).get("contract", "?")[:50]   if hits else "?"
        return clause_type, score, match

    with ThreadPoolExecutor(max_workers=3) as executor:
        for ct, score, match in executor.map(query_one, RISK_CLAUSE_TYPES):
            all_scores[ct] = {"score": score, "match": match}

    ranked    = sorted(all_scores.items(), key=lambda x: -x[1]["score"])
    best      = ranked[0]

    return {
        "predicted_type": best[0],
        "top_score":      round(best[1]["score"], 4),
        "matched_from":   best[1]["match"],
        "all_scores":     {ct: round(v["score"], 4) for ct, v in all_scores.items()},
    }


def classify_clause(clause_text: str, cache: SemanticCache, top_k: int = 3) -> dict:
    """Classify a clause, using hybi SemanticCache if available."""
    context = "classify:all"

    hit = cache.get(clause_text, context=context)
    if hit:
        result = json.loads(hit.response)
        result["cache_hit"] = True
        return result

    result = _classify_live(clause_text, top_k)
    cache.put(clause_text, context=context, response=json.dumps(result))
    result["cache_hit"] = False
    return result


# ── Risk detection ────────────────────────────────────────────────────────────

def _detect_risk_live(clause_type: str, clause_text: str) -> dict:
    hits = search_slots({
        "clause_type": {"query": clause_type,       "weight": 0.01, "encoding": "exact"},
        "object":      {"query": clause_text[:500], "weight": 1.0,  "encoding": "semantic"},
    }, top_k=20)

    other_hits = [
        h for h in hits
        if "jpmorgan"    not in h.get("data", {}).get("contract", "").lower()
        and "cardlytics" not in h.get("data", {}).get("contract", "").lower()
    ]

    top_score    = other_hits[0].get("_score", 0.0)                        if other_hits else 0.0
    top_contract = other_hits[0].get("data", {}).get("contract", "?")[:50] if other_hits else "?"
    top_text     = other_hits[0].get("data", {}).get("object",   "")[:120] if other_hits else ""

    unusual_t, review_t = RISK_THRESHOLDS.get(clause_type, (0.5, 0.7))
    risk_level = (
        "UNUSUAL" if top_score < unusual_t else
        "REVIEW"  if top_score < review_t  else
        "NORMAL"
    )

    return {
        "clause_type":    clause_type,
        "risk_level":     risk_level,
        "peer_score":     round(top_score, 4),
        "peer_contract":  top_contract,
        "peer_text":      top_text,
        "unusual_thresh": unusual_t,
        "review_thresh":  review_t,
    }


def detect_risk(clause_type: str, clause_text: str, cache: SemanticCache) -> dict:
    """
    Detect risk for a clause, using hybi SemanticCache if available.
    Context key is "risk:<clause_type>" -- EXACT encoded, so
    Termination clauses never collide with Anti-Assignment clauses.
    """
    context = f"risk:{clause_type}"

    hit = cache.get(clause_text, context=context)
    if hit:
        result = json.loads(hit.response)
        result["cache_hit"] = True
        return result

    result = _detect_risk_live(clause_type, clause_text)
    cache.put(clause_text, context=context, response=json.dumps(result))
    result["cache_hit"] = False
    return result


# ── CLI output ────────────────────────────────────────────────────────────────

def cmd_classify(clause_text: str, cache: SemanticCache):
    print(f"\n  -- Clause Classification")
    print(f"  -- Input: \"{clause_text[:80]}...\"")
    print(f"  {'─'*65}")

    t0      = time.time()
    result  = classify_clause(clause_text, cache)
    elapsed = time.time() - t0

    status = "CACHE HIT" if result.get("cache_hit") else "live query"
    print(f"\n  Predicted type : {result['predicted_type']}")
    print(f"  Top score      : {result['top_score']}")
    print(f"  Matched from   : {result['matched_from']}")
    print(f"  Status         : {status}  [{elapsed*1000:.0f}ms]")
    print(f"\n  Score per clause type (top 5):")
    for ct, sc in sorted(result["all_scores"].items(), key=lambda x: -x[1])[:5]:
        marker = " <" if ct == result["predicted_type"] else ""
        print(f"    {ct:<40} {sc:.4f}{marker}")


def cmd_risk(contract_key: str, cache: SemanticCache):
    if contract_key.lower() not in KNOWN_CONTRACTS:
        print(f"\n  Unknown contract: {contract_key}")
        print(f"  Known contracts : {', '.join(KNOWN_CONTRACTS.keys())}")
        return

    contract_name, clauses = KNOWN_CONTRACTS[contract_key.lower()]

    print(f"\n{'='*70}")
    print(f"  Risk Analysis: {contract_name}")
    print(f"  Comparing against 510 CUAD peer contracts")
    print(f"{'='*70}")

    results = []
    for clause_type, text in clauses.items():
        print(f"\n  Querying: {clause_type}...", end=" ", flush=True)
        t0      = time.time()
        result  = detect_risk(clause_type, text, cache)
        elapsed = time.time() - t0

        status = "CACHED" if result.get("cache_hit") else "live"
        print(f"{result['risk_level']:<8} (peer similarity: {result['peer_score']:.4f})  [{status} {elapsed*1000:.0f}ms]")
        print(f"    Nearest peer : {result['peer_contract']}")
        print(f"    Peer text    : {result['peer_text']}...")
        print(f"    Thresholds   : unusual < {result['unusual_thresh']:.2f}  |  review < {result['review_thresh']:.2f}")
        results.append(result)

    unusual = [r for r in results if r["risk_level"] == "UNUSUAL"]
    review  = [r for r in results if r["risk_level"] == "REVIEW"]
    normal  = [r for r in results if r["risk_level"] == "NORMAL"]
    cached  = sum(1 for r in results if r.get("cache_hit"))

    print(f"\n{'='*70}")
    print(f"  RISK SUMMARY")
    print(f"{'='*70}")
    print(f"  Contract : {contract_name}")
    print(f"  Clauses  : {len(results)} analyzed  ({cached} from cache, {len(results)-cached} live)")
    print()
    if unusual:
        print(f"  UNUSUAL ({len(unusual)}) -- flagged for attorney review:")
        for r in unusual:
            print(f"    {r['clause_type']:<35}  peer score: {r['peer_score']:.4f}")
    if review:
        print(f"\n  REVIEW ({len(review)}) -- warrants attention:")
        for r in review:
            print(f"    {r['clause_type']:<35}  peer score: {r['peer_score']:.4f}")
    if normal:
        print(f"\n  NORMAL ({len(normal)}):")
        for r in normal:
            print(f"    {r['clause_type']:<35}  peer score: {r['peer_score']:.4f}")
    print(f"\n{'='*70}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Legal clause analysis with hybi SemanticCache"
    )
    parser.add_argument("--classify",    type=str, default=None)
    parser.add_argument("--risk",        type=str, default=None,
                        help="Contract key, e.g. jpmorgan")
    parser.add_argument("--clear-cache", action="store_true")
    parser.add_argument("--threshold",   type=float, default=CACHE_THRESHOLD)
    args = parser.parse_args()

    if args.clear_cache:
        p = Path(CACHE_DB_PATH)
        if p.exists():
            shutil.rmtree(p)
            print(f"  [cache] Cleared {CACHE_DB_PATH}")
        else:
            print(f"  [cache] Nothing to clear")
        return

    cache = build_cache(threshold=args.threshold)

    if args.classify:
        cmd_classify(args.classify, cache)
    elif args.risk:
        cmd_risk(args.risk, cache)
    else:
        print("\nExamples:")
        print('  python legal_cached.py --classify "JPMC may terminate for convenience with 90 days notice"')
        print('  python legal_cached.py --risk jpmorgan')
        print('  python legal_cached.py --clear-cache')


if __name__ == "__main__":
    main()