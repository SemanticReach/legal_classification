"""
legal_query.py — Contract Clause Retrieval, Risk Detection, Q&A, Classification
================================================================================

Four capabilities on top of the ingested CUAD clause index:

1. RETRIEVAL  — find similar clauses across all contracts (or within one)
2. RISK       — flag unusual clauses with no close matches
3. QA         — answer questions about a contract
4. CLASSIFY   — predict clause type from raw text

Usage:
    # Find similar clauses to a given text (cross-contract)
    python legal_query.py --retrieve "termination without cause 30 days notice"

    # Find similar clauses within a specific contract
    python legal_query.py --retrieve "termination without cause 30 days notice" \
        --contract "CybergyHoldingsInc_20140520"

    # Ask a question about a specific contract
    python legal_query.py --qa "Does this contract have a non-compete clause?" \
        --contract "CybergyHoldingsInc_20140520"

    # Classify a raw clause
    python legal_query.py --classify "Either party may terminate with 60 days written notice"

    # Flag risky clauses in a contract
    python legal_query.py --risk --contract "CybergyHoldingsInc_20140520"

    # Interactive mode
    python legal_query.py --interactive
"""

import os
import argparse
import requests
import pandas as pd
from dotenv import load_dotenv

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────

SERVER_URL = os.environ.get("HB_SERVER_URL", os.environ.get("SERVER_URL", ""))
API_KEY    = os.environ.get("HB_API_KEY",    os.environ.get("API_KEY",    ""))
DB_NAME    = os.environ.get("HB_DB_NAME",    "fraud_db")
NAMESPACE  = "cuad_clauses"

# Must match the model used in legal_ingest.py — if you change the ingest
# model you must re-ingest AND update this constant so query embeddings match.
# Options:
#   "nlpaueb/legal-bert-base-uncased"  — legal domain, 768-dim (default)
#   "all-MiniLM-L6-v2"                 — general purpose, 384-dim
EMBED_MODEL = "nlpaueb/legal-bert-base-uncased"

# Per-clause-type presence thresholds — calibrated against CUAD eval results.
# Higher threshold = more conservative (fewer false positives).
# Uncapped Liability and Irrevocable Or Perpetual License need high thresholds
# because their language overlaps heavily with adjacent clause types.
PRESENCE_THRESHOLDS = {
    "Uncapped Liability":             0.82,  # overlaps with Cap On Liability
    "Cap On Liability":               0.65,
    "Liquidated Damages":             0.72,
    "Non-Compete":                    0.68,
    "Anti-Assignment":                0.65,
    "Change Of Control":              0.68,
    "Termination For Convenience":    0.70,
    "Ip Ownership Assignment":        0.68,
    "Irrevocable Or Perpetual License": 0.80,  # overlaps with License Grant
    "Covenant Not To Sue":            0.65,
}

# Risk level thresholds for detect_risks() peer similarity scoring.
# Per-clause-type — same calibration logic as PRESENCE_THRESHOLDS.
# Similarity below UNUSUAL threshold  -> flagged as UNUSUAL (🚨)
# Similarity below REVIEW threshold   -> flagged as REVIEW  (⚠️)
# Similarity above REVIEW threshold   -> NORMAL (✓)
RISK_THRESHOLDS = {
    # clause_type:                     (unusual, review)
    "Uncapped Liability":             (0.65, 0.82),
    "Cap On Liability":               (0.50, 0.70),
    "Liquidated Damages":             (0.55, 0.72),
    "Non-Compete":                    (0.50, 0.68),
    "Anti-Assignment":                (0.50, 0.65),
    "Change Of Control":              (0.50, 0.68),
    "Termination For Convenience":    (0.55, 0.70),
    "Ip Ownership Assignment":        (0.50, 0.68),
    "Irrevocable Or Perpetual License": (0.60, 0.80),
    "Covenant Not To Sue":            (0.50, 0.65),
}

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

# Keyword → clause type inference map
QUERY_TO_CLAUSE_TYPE = {
    "terminat":         "Termination For Convenience",
    "non-compete":      "Non-Compete",
    "noncompete":       "Non-Compete",
    "governing law":    "Governing Law",
    "jurisdiction":     "Governing Law",
    "ip ownership":     "Ip Ownership Assignment",
    "intellectual prop":"Ip Ownership Assignment",
    "liability":        "Cap On Liability",
    "liquidated":       "Liquidated Damages",
    "assign":           "Anti-Assignment",
    "change of control":"Change Of Control",
    "non-solicit":      "No-Solicit Of Employees",
    "solicit":          "No-Solicit Of Employees",
    "exclusiv":         "Exclusivity",
    "audit":            "Audit Rights",
    "warranty":         "Warranty Duration",
    "insurance":        "Insurance",
    "indemnif":         "Uncapped Liability",
    "covenant not to sue": "Covenant Not To Sue",
    "license":          "License Grant",
    "source code":      "Source Code Escrow",
    "renewal":          "Renewal Term",
    "notice period":    "Notice Period To Terminate Renewal",
    "profit sharing":   "Revenue/Profit Sharing",
    "price restrict":   "Price Restrictions",
    "minimum commit":   "Minimum Commitment",
    "volume":           "Volume Restriction",
    "third party":      "Third Party Beneficiary",
}


def infer_clause_type(query: str) -> str | None:
    """Infer the most likely clause type from free-text query keywords."""
    q = query.lower()
    for keyword, ctype in QUERY_TO_CLAUSE_TYPE.items():
        if keyword in q:
            return ctype
    return None


# ── Core search ───────────────────────────────────────────────────────────────

def search_slots(slot_queries: dict, top_k: int = 5, retries: int = 3) -> list:
    """Query HyperBinder using /compose/search_slots/ with retry on 429."""
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


def filter_by_contract(hits: list, contract_name: str, top_k: int) -> list:
    """
    Post-filter results to a specific contract using substring match
    against the stored 'contract' field. Works regardless of whether
    subject was ingested with exact or semantic encoding.
    """
    prefix = contract_name[:20].lower()
    return [
        h for h in hits
        if prefix in h.get("data", {}).get("contract", "").lower()
    ][:top_k]


# ── 1. Clause Retrieval ───────────────────────────────────────────────────────

def retrieve_similar_clauses(
    query_text:    str,
    clause_type:   str = None,
    contract_name: str = None,
    top_k:         int = 5,
    verbose:       bool = True,
) -> list:
    """
    Find the most similar clauses across all 510 contracts,
    or within a specific contract if contract_name is provided.

    Uses inferred clause type to anchor the predicate slot so that
    date/party fields don't dominate retrieval.
    """
    clause_type = clause_type or infer_clause_type(query_text)

    slot_queries = {
        "object":    {"query": query_text,                "weight": 0.6, "encoding": "semantic"},
        "predicate": {"query": clause_type or query_text, "weight": 0.4, "encoding": "semantic"},
    }

    # Fetch a larger pool when post-filtering to a single contract
    fetch_k = top_k if not contract_name else top_k * 20

    hits = search_slots(slot_queries, top_k=fetch_k)

    if contract_name:
        hits = filter_by_contract(hits, contract_name, top_k)

    if verbose:
        print(f"\n  ── Similar clauses to: \"{query_text[:60]}...\"")
        print(f"  ── Clause type filter : {clause_type or 'none (inferred)'}")
        print(f"  ── Contract filter    : {contract_name or 'all contracts'}")
        print(f"  {'─'*65}")
        for i, h in enumerate(hits, 1):
            data     = h.get("data", {})
            score    = h.get("_score", 0.0)
            contract = data.get("contract", "?")[:40]
            ctype    = data.get("clause_type", "?")
            answer   = data.get("answer", "?")
            text     = data.get("object", "")[:120]
            print(f"\n  #{i}  score={score:.4f}  [{answer}]  {ctype}")
            print(f"       Contract : {contract}")
            print(f"       Text     : {text}...")

    return hits


# ── 2. Risk Detection ─────────────────────────────────────────────────────────

def detect_risks(
    contract_name: str,
    csv_path:      str  = "CUAD_v1/master_clauses.csv",
    verbose:       bool = True,
) -> list:
    """
    Flag high-risk clauses in a contract by:
    1. Finding what risk clauses exist in the contract (from CSV)
    2. Checking if they are unusual compared to similar contracts
    3. Flagging clauses where the contract's version is an outlier
    """
    import ast

    df  = pd.read_csv(csv_path)
    row = df[df["Filename"].str.contains(contract_name, na=False)]

    if row.empty:
        print(f"  ✗ Contract not found: {contract_name}")
        return []

    row   = row.iloc[0]
    risks = []

    if verbose:
        print(f"\n  ── Risk Analysis: {contract_name[:60]}")
        print(f"  {'─'*65}")

    # Build a short lowercase key for self-exclusion that matches
    # what was stored at ingest time (Path(filename).stem[:60])
    contract_key = contract_name.lower()

    for clause_type in RISK_CLAUSE_TYPES:
        answer_col = f"{clause_type}-Answer"  # "Yes" / "No" presence flag
        text_col   = clause_type              # actual extracted clause text

        # Check presence flag first
        flag_raw = row.get(answer_col, "")
        flag     = str(flag_raw).strip().lower() if not pd.isna(flag_raw) else ""

        if flag != "yes":
            if verbose:
                print(f"\n  ⚠️  {clause_type}")
                print(f"       Status : MISSING — not found in contract")
            risks.append({
                "clause_type": clause_type,
                "risk":        "missing",
                "text":        "",
            })
            continue

        # Clause is present — read actual text from the non-Answer column
        text_raw = row.get(text_col, "")
        raw_str  = str(text_raw).strip() if not pd.isna(text_raw) else ""

        if not raw_str or raw_str in ("[]", ""):
            if verbose:
                print(f"\n  ⚠️  {clause_type}")
                print(f"       Status : MISSING — no clause text extracted")
            risks.append({
                "clause_type": clause_type,
                "risk":        "missing",
                "text":        "",
            })
            continue

        # Parse list-string format e.g. "['text here', ...]"
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

        # Skip if nothing meaningful survived parsing
        if not text.strip():
            if verbose:
                print(f"\n  ⚠️  {clause_type}")
                print(f"       Status : MISSING — no clause text extracted")
            risks.append({
                "clause_type": clause_type,
                "risk":        "missing",
                "text":        "",
            })
            continue

        # Find similar clauses in other contracts using exact symbolic filter.
        # clause_type weight is near-zero so it acts as a hard filter without
        # inflating the combined score — object semantic score dominates.
        hits = search_slots({
            "clause_type": {"query": clause_type, "weight": 0.01, "encoding": "exact"},
            "object":      {"query": text[:500],  "weight": 1.0,  "encoding": "semantic"},
        }, top_k=20)

        # Exclude hits from this contract — case-insensitive substring match
        other_hits = [
            h for h in hits
            if contract_key[:20] not in h.get("data", {}).get("contract", "").lower()
        ]

        top_score = other_hits[0].get("_score", 0.0) if other_hits else 0.0

        # Use per-clause-type thresholds calibrated against CUAD eval results
        unusual_t, review_t = RISK_THRESHOLDS.get(clause_type, (0.5, 0.7))
        risk_level = "normal"
        if top_score < unusual_t:
            risk_level = "unusual"
        elif top_score < review_t:
            risk_level = "review"

        if verbose:
            sym = "🚨" if risk_level == "unusual" else "⚠️ " if risk_level == "review" else "✓ "
            print(f"\n  {sym} {clause_type}")
            print(f"       Status    : {risk_level.upper()}")
            print(f"       Similarity: {top_score:.4f} to nearest peer contract")
            print(f"       Text      : {text[:100]}...")

        risks.append({
            "clause_type": clause_type,
            "risk":        risk_level,
            "similarity":  top_score,
            "text":        text[:200],
        })

    if verbose:
        unusual = [r for r in risks if r["risk"] == "unusual"]
        review  = [r for r in risks if r["risk"] == "review"]
        missing = [r for r in risks if r["risk"] == "missing"]
        print(f"\n  {'─'*65}")
        print(f"  Risk Summary:")
        print(f"    🚨 Unusual clauses : {len(unusual)}")
        print(f"    ⚠️  Review clauses  : {len(review)}")
        print(f"    Missing clauses   : {len(missing)}")

    return risks


# ── 3. Contract Q&A ───────────────────────────────────────────────────────────

def answer_question(
    question:      str,
    contract_name: str  = None,
    top_k:         int  = 3,
    verbose:       bool = True,
) -> list:
    """
    Answer a natural language question by retrieving the most relevant clauses.
    If contract_name is provided, results are post-filtered to that contract.

    Examples:
        "Does this contract have a non-compete clause?"
        "What is the governing law?"
        "Can the contract be terminated without cause?"
        "What happens to IP ownership?"
    """
    # Use inferred clause type as predicate anchor; fall back to full question
    clause_hint = infer_clause_type(question) or question

    slot_queries = {
        "predicate": {"query": clause_hint, "weight": 0.25, "encoding": "semantic"},
        "object":    {"query": question,    "weight": 0.75, "encoding": "semantic"},
    }

    # Fetch larger pool when filtering to a single contract
    fetch_k = top_k if not contract_name else top_k * 20

    hits = search_slots(slot_queries, top_k=fetch_k)

    if contract_name:
        hits = filter_by_contract(hits, contract_name, top_k)

    if verbose:
        print(f"\n  ── Q&A: \"{question}\"")
        if contract_name:
            print(f"  ── Contract: {contract_name}")
        print(f"  {'─'*65}")

        if not hits:
            print("  No relevant clauses found.")
        else:
            for i, h in enumerate(hits, 1):
                data     = h.get("data", {})
                score    = h.get("_score", 0.0)
                contract = data.get("contract", "?")[:40]
                ctype    = data.get("clause_type", "?")
                answer   = data.get("answer", "?")
                text     = data.get("object", "")

                print(f"\n  #{i}  score={score:.4f}  [{answer}]  {ctype}")
                print(f"       Contract : {contract}")
                print(f"       Clause   : {text[:300]}...")

    return hits


# ── 4. Clause Classification ──────────────────────────────────────────────────

def classify_clause(
    clause_text: str,
    top_k:       int  = 3,
    verbose:     bool = True,
) -> str:
    """
    Predict the clause type from raw text using exact symbolic predicate filtering.

    Queries each known clause type separately with an exact predicate constraint
    so results are restricted to that clause type only — no cross-type
    contamination. Picks whichever clause type returns the highest scoring
    object match.

    This is more accurate than voting across a single mixed-pool query because
    liability clauses cannot bleed into termination results etc.
    """
    from concurrent.futures import ThreadPoolExecutor

    all_hits   = {}
    all_scores = {}

    def query_one(clause_type):
        hits = search_slots({
            "clause_type": {"query": clause_type,       "weight": 0.01, "encoding": "exact"},
            "object":      {"query": clause_text[:500], "weight": 1.0,  "encoding": "semantic"},
        }, top_k=top_k)
        score = hits[0].get("_score", 0.0) if hits else 0.0
        return clause_type, score, hits[0] if hits else None

    with ThreadPoolExecutor(max_workers=3) as executor:
        for clause_type, score, hit in executor.map(
            lambda ct: query_one(ct), RISK_CLAUSE_TYPES
        ):
            all_scores[clause_type] = score
            all_hits[clause_type]   = hit

    best_type  = max(all_scores, key=all_scores.get)
    best_score = all_scores[best_type]
    best_hit   = all_hits[best_type]

    if verbose:
        print(f"\n  ── Clause Classification (exact symbolic predicate)")
        print(f"  ── Input: \"{clause_text[:80]}...\"")
        print(f"  {chr(9472)*65}")
        print(f"\n  Score per clause type (top 5):")
        for ct, sc in sorted(all_scores.items(), key=lambda x: -x[1])[:5]:
            marker = " ◀" if ct == best_type else ""
            print(f"    {ct:<40} {sc:.4f}{marker}")
        if best_hit:
            contract = best_hit.get("data", {}).get("contract", "?")[:40]
            print(f"\n  Predicted type : {best_type}")
            print(f"  Top score      : {best_score:.4f}")
            print(f"  Matched from   : {contract}")

    return best_type


# ── Interactive mode ──────────────────────────────────────────────────────────

def interactive_mode():
    print("\n" + "=" * 65)
    print("  CUAD Legal Assistant — Interactive Mode")
    print("=" * 65)
    print("\n  Commands:")
    print("    retrieve <text>              — find similar clauses")
    print("    retrieve <text> @<contract>  — find clauses in one contract")
    print("    qa <question>                — answer a question")
    print("    qa <question> @<contract>    — answer within one contract")
    print("    classify <text>              — predict clause type")
    print("    risk <contract>              — flag risky clauses")
    print("    quit                         — exit")
    print()

    while True:
        try:
            user_input = input("  > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Goodbye.")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "q"):
            print("  Goodbye.")
            break

        # Parse optional @contract suffix
        contract_name = None
        if " @" in user_input:
            user_input, contract_name = user_input.rsplit(" @", 1)
            contract_name = contract_name.strip()

        parts = user_input.split(" ", 1)
        cmd   = parts[0].lower()
        args  = parts[1] if len(parts) > 1 else ""

        if cmd == "retrieve":
            retrieve_similar_clauses(args, contract_name=contract_name)
        elif cmd == "qa":
            answer_question(args, contract_name=contract_name)
        elif cmd == "classify":
            classify_clause(args)
        elif cmd == "risk":
            detect_risks(args)
        else:
            # Treat as a Q&A question
            answer_question(user_input, contract_name=contract_name)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    global NAMESPACE, DB_NAME

    parser = argparse.ArgumentParser(
        description="CUAD contract clause retrieval, risk detection, Q&A, classification"
    )
    parser.add_argument("--retrieve",    type=str,  default=None,
                        help="Find similar clauses for this text")
    parser.add_argument("--qa",          type=str,  default=None,
                        help="Answer a question about contracts")
    parser.add_argument("--classify",    type=str,  default=None,
                        help="Classify a raw clause text")
    parser.add_argument("--risk",        action="store_true",
                        help="Run risk analysis on a contract")
    parser.add_argument("--contract",    type=str,  default=None,
                        help="Contract name filter (partial match, used by --retrieve, --qa, --risk)")
    parser.add_argument("--clause-type", type=str,  default=None,
                        help="Override inferred clause type for --retrieve")
    parser.add_argument("--csv",         default="CUAD_v1/master_clauses.csv")
    parser.add_argument("--top-k",       type=int,  default=5)
    parser.add_argument("--namespace",   default="cuad_clauses")
    parser.add_argument("--db",          default="fraud_db")
    parser.add_argument("--interactive", action="store_true",
                        help="Launch interactive mode")
    args = parser.parse_args()

    NAMESPACE = args.namespace
    DB_NAME   = args.db

    if args.interactive:
        interactive_mode()

    elif args.retrieve:
        retrieve_similar_clauses(
            args.retrieve,
            clause_type=args.clause_type,
            contract_name=args.contract,
            top_k=args.top_k,
        )

    elif args.qa:
        answer_question(
            args.qa,
            contract_name=args.contract,
            top_k=args.top_k,
        )

    elif args.classify:
        classify_clause(args.classify, top_k=args.top_k)

    elif args.risk:
        if not args.contract:
            print("  --risk requires --contract <name>")
            print("  Example: python legal_query.py --risk --contract CybergyHoldings")
        else:
            detect_risks(args.contract, csv_path=args.csv)

    else:
        print("\nNo action specified. Examples:")
        print('  python legal_query.py --retrieve "termination without cause 30 days"')
        print('  python legal_query.py --retrieve "termination without cause" --contract CybergyHoldings')
        print('  python legal_query.py --qa "Does this contract have a non-compete?"')
        print('  python legal_query.py --qa "termination" --contract CybergyHoldingsInc_20140520')
        print('  python legal_query.py --classify "Either party may terminate with 60 days notice"')
        print('  python legal_query.py --risk --contract CybergyHoldings')
        print('  python legal_query.py --interactive')


if __name__ == "__main__":
    main()