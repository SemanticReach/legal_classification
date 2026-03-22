"""
maud_query.py — MAUD Merger Agreement Retrieval, Risk Detection, Q&A, Classification
======================================================================================

Four capabilities on top of the ingested MAUD clause index:

1. RETRIEVAL  — find similar clauses across all merger agreements
2. RISK       — flag unusual answer labels compared to peer deal distribution
3. QA         — answer questions about a specific merger agreement
4. CLASSIFY   — predict question type from raw clause text

Usage:
    # Find similar MAE definition clauses across all deals
    python maud_query.py --retrieve "material adverse effect carve-out pandemic"

    # Find clauses within one specific deal
    python maud_query.py --retrieve "termination fee acquirer" --contract "contract_41"

    # Risk analysis on a specific deal
    python maud_query.py --risk --contract "contract_41"

    # Ask a question about a deal
    python maud_query.py --qa "What type of consideration is used?" --contract "contract_41"

    # Classify a raw clause
    python maud_query.py --classify "Either party may terminate if the merger is not consummated by the outside date"

    # Interactive mode
    python maud_query.py --interactive
"""

import os
import re
import argparse
import requests
from collections import Counter
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────

SERVER_URL = os.environ.get("HB_SERVER_URL", os.environ.get("SERVER_URL", ""))
API_KEY    = os.environ.get("HB_API_KEY",    os.environ.get("API_KEY",    ""))
DB_NAME    = os.environ.get("HB_DB_NAME",    "fraud_db")
NAMESPACE  = "maud_clauses"

EMBED_MODEL = "nlpaueb/legal-bert-base-uncased"

# All 22 MAUD question types
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

# High-risk question types to analyze in detect_risks()
RISK_QUESTION_TYPES = [
    "Type of Consideration",
    "MAE Definition",
    "No-Shop",
    "Specific Performance",
    "General Antitrust Efforts Standard",
    "Superior Offer Definition",
    "Fiduciary exception to COR covenant",
    "Fiduciary exception:  Board determination (no-shop)",
    "FTR Triggers",
    "Tail Period & Acquisition Proposal Details",
]

# Keyword → question type inference map
QUERY_TO_QUESTION_TYPE = {
    "consideration":      "Type of Consideration",
    "cash":               "Type of Consideration",
    "stock":              "Type of Consideration",
    "mae":                "MAE Definition",
    "material adverse":   "MAE Definition",
    "no-shop":            "No-Shop",
    "no shop":            "No-Shop",
    "specific performance": "Specific Performance",
    "antitrust":          "General Antitrust Efforts Standard",
    "efforts standard":   "General Antitrust Efforts Standard",
    "superior offer":     "Superior Offer Definition",
    "fiduciary":          "Fiduciary exception to COR covenant",
    "ftr":                "FTR Triggers",
    "tail period":        "Tail Period & Acquisition Proposal Details",
    "termination fee":    "Tail Period & Acquisition Proposal Details",
    "matching right":     "Agreement provides for matching rights in connection with FTR",
    "knowledge":          "Knowledge Definition",
    "litigation":         "Absence of Litigation Closing Condition",
    "ordinary course":    "Ordinary course covenant",
    "interim":            "Negative interim operating covenant",
    "closing condition":  "Compliance with Covenant Closing Condition",
    "breach":             "Breach of No Shop",
}

# Risk threshold — answer labels used in fewer than this % of peer deals
# are flagged as unusual
UNUSUAL_THRESHOLD = 0.10   # bottom 10%
REVIEW_THRESHOLD  = 0.25   # bottom 25%


def infer_question_type(query: str) -> str | None:
    """Infer the most likely question type from free-text query keywords."""
    q = query.lower()
    for keyword, qtype in QUERY_TO_QUESTION_TYPE.items():
        if keyword in q:
            return qtype
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
    """Post-filter results to a specific contract using substring match."""
    prefix = contract_name[:20].lower()
    return [
        h for h in hits
        if prefix in h.get("data", {}).get("contract_name", "").lower()
    ][:top_k]


# ── Answer label distribution (loaded from HuggingFace) ──────────────────────

_answer_dist_cache: dict = {}

def get_answer_distribution(question_type: str) -> dict[str, float]:
    """
    Load answer label distribution for a given question type from MAUD.
    Returns dict of {answer_label: fraction_of_deals} using train split.
    Results are cached after first load.
    """
    global _answer_dist_cache

    if question_type in _answer_dist_cache:
        return _answer_dist_cache[question_type]

    if not _answer_dist_cache:
        print("  Loading MAUD answer distributions from HuggingFace...")
        from datasets import load_dataset
        ds = load_dataset("theatticusproject/maud")

        # Build distribution for all question types from train split
        # (train only — don't leak test labels into risk scoring)
        for qt in QUESTION_TYPES:
            records = [
                r for r in ds["train"]
                if r.get("text_type") == qt and r.get("answer")
            ]
            if not records:
                _answer_dist_cache[qt] = {}
                continue

            counts = Counter(r["answer"] for r in records if r["answer"])
            total  = sum(counts.values())
            _answer_dist_cache[qt] = {
                ans: count / total
                for ans, count in counts.most_common()
            }

        print(f"  ✓ Loaded distributions for {len(_answer_dist_cache)} question types")

    return _answer_dist_cache.get(question_type, {})


# ── 1. Clause Retrieval ───────────────────────────────────────────────────────

def retrieve_similar_clauses(
    query_text:    str,
    question_type: str  = None,
    contract_name: str  = None,
    top_k:         int  = 5,
    verbose:       bool = True,
) -> list:
    """
    Find the most similar clauses across all merger agreements,
    or within a specific deal if contract_name is provided.
    """
    question_type = question_type or infer_question_type(query_text)

    slot_queries = {
        "object":    {"query": query_text,                   "weight": 0.6, "encoding": "semantic"},
        "predicate": {"query": question_type or query_text,  "weight": 0.4, "encoding": "semantic"},
    }

    fetch_k = top_k if not contract_name else top_k * 20
    hits    = search_slots(slot_queries, top_k=fetch_k)

    if contract_name:
        hits = filter_by_contract(hits, contract_name, top_k)

    if verbose:
        print(f"\n  ── Similar clauses to: \"{query_text[:60]}...\"")
        print(f"  ── Question type filter : {question_type or 'none (inferred)'}")
        print(f"  ── Contract filter      : {contract_name or 'all deals'}")
        print(f"  {'─'*65}")
        for i, h in enumerate(hits, 1):
            data     = h.get("data", {})
            score    = h.get("_score", 0.0)
            contract = data.get("contract_name", "?")
            qtype    = data.get("question_type", "?")
            answer   = data.get("answer", "?")
            text     = data.get("object", "")[:120]
            print(f"\n  #{i}  score={score:.4f}  [{answer}]  {qtype}")
            print(f"       Deal : {contract}")
            print(f"       Text : {text}...")

    return hits


# ── 2. Risk Detection ─────────────────────────────────────────────────────────

def detect_risks(
    contract_name: str,
    verbose:       bool = True,
) -> list:
    """
    Flag unusual deal terms by comparing this deal's answer labels against
    the distribution of answer labels across all peer deals in the MAUD
    training set.

    A deal term is flagged as UNUSUAL if its answer label appears in fewer
    than 10% of peer deals for that question type. REVIEW if fewer than 25%.

    This is richer than CUAD risk detection because the answer label itself
    encodes the legal characterization — not just whether language is present
    but HOW it is structured compared to market standard.
    """
    if verbose:
        print(f"\n  ── Risk Analysis: {contract_name}")
        print(f"  {'─'*65}")

    # Load answer distributions (cached after first call)
    _ = get_answer_distribution(RISK_QUESTION_TYPES[0])

    # Retrieve this deal's clauses for each risk question type
    risks = []

    for question_type in RISK_QUESTION_TYPES:
        # Query index for this contract + question type
        hits = search_slots({
            "question_type": {"query": question_type,  "weight": 1.0, "encoding": "exact"},
            "subject":       {"query": contract_name,  "weight": 0.5, "encoding": "semantic"},
        }, top_k=20)

        # Filter to this contract
        contract_hits = [
            h for h in hits
            if contract_name[:20].lower() in
               h.get("data", {}).get("contract_name", "").lower()
        ]

        if not contract_hits:
            if verbose:
                print(f"\n  ⚠️  {question_type}")
                print(f"       Status : MISSING — not found in this deal")
            risks.append({
                "question_type": question_type,
                "risk":          "missing",
                "answer":        "",
                "peer_pct":      0.0,
                "text":          "",
            })
            continue

        # Get this deal's answer label and clause text
        deal_data  = contract_hits[0].get("data", {})
        deal_answer = deal_data.get("answer", "")
        deal_text   = deal_data.get("object", "")[:150]

        # Get peer distribution
        dist = get_answer_distribution(question_type)
        peer_pct = dist.get(deal_answer, 0.0)

        # Flag risk level based on how common this answer is among peers
        if peer_pct == 0.0:
            risk_level = "unusual"   # answer not seen in any peer deal
        elif peer_pct < UNUSUAL_THRESHOLD:
            risk_level = "unusual"
        elif peer_pct < REVIEW_THRESHOLD:
            risk_level = "review"
        else:
            risk_level = "normal"

        if verbose:
            sym = "🚨" if risk_level == "unusual" else "⚠️ " if risk_level == "review" else "✓ "
            print(f"\n  {sym} {question_type}")
            print(f"       Status     : {risk_level.upper()}")
            print(f"       Answer     : {deal_answer}")
            print(f"       Peer freq  : {peer_pct:.1%} of deals use this label")
            print(f"       Text       : {deal_text}...")

        risks.append({
            "question_type": question_type,
            "risk":          risk_level,
            "answer":        deal_answer,
            "peer_pct":      peer_pct,
            "text":          deal_text,
        })

    if verbose:
        unusual = [r for r in risks if r["risk"] == "unusual"]
        review  = [r for r in risks if r["risk"] == "review"]
        missing = [r for r in risks if r["risk"] == "missing"]
        print(f"\n  {'─'*65}")
        print(f"  Risk Summary for {contract_name}:")
        print(f"    🚨 Unusual terms : {len(unusual)}")
        print(f"    ⚠️  Review terms  : {len(review)}")
        print(f"    Missing terms   : {len(missing)}")

        if unusual:
            print(f"\n  Unusual deal terms (rare answer labels):")
            for r in unusual:
                print(f"    • {r['question_type'][:50]:<50}  [{r['answer'][:40]}]  "
                      f"{r['peer_pct']:.1%} of peers")

    return risks


# ── 3. Contract Q&A ───────────────────────────────────────────────────────────

def answer_question(
    question:      str,
    contract_name: str  = None,
    top_k:         int  = 3,
    verbose:       bool = True,
) -> list:
    """
    Answer a natural language question about a merger agreement by
    retrieving the most relevant clauses.

    Examples:
        "What type of consideration is used in this deal?"
        "What is the MAE definition?"
        "Does this deal have a no-shop provision?"
        "What are the FTR triggers?"
    """
    question_hint = infer_question_type(question) or question

    slot_queries = {
        "predicate": {"query": question_hint, "weight": 0.25, "encoding": "semantic"},
        "object":    {"query": question,      "weight": 0.75, "encoding": "semantic"},
    }

    fetch_k = top_k if not contract_name else top_k * 20
    hits    = search_slots(slot_queries, top_k=fetch_k)

    if contract_name:
        hits = filter_by_contract(hits, contract_name, top_k)

    if verbose:
        print(f"\n  ── Q&A: \"{question}\"")
        if contract_name:
            print(f"  ── Deal: {contract_name}")
        print(f"  {'─'*65}")

        if not hits:
            print("  No relevant clauses found.")
        else:
            for i, h in enumerate(hits, 1):
                data     = h.get("data", {})
                score    = h.get("_score", 0.0)
                contract = data.get("contract_name", "?")
                qtype    = data.get("question_type", "?")
                answer   = data.get("answer", "?")
                text     = data.get("object", "")

                # Show peer frequency for context
                dist     = get_answer_distribution(qtype)
                peer_pct = dist.get(answer, 0.0)
                freq_str = f" ({peer_pct:.0%} of deals)" if peer_pct > 0 else ""

                print(f"\n  #{i}  score={score:.4f}  [{answer}]{freq_str}")
                print(f"       Deal  : {contract}  |  {qtype}")
                print(f"       Clause: {text[:300]}...")

    return hits


# ── 4. Clause Classification ──────────────────────────────────────────────────

def classify_clause(
    clause_text: str,
    top_k:       int  = 3,
    verbose:     bool = True,
) -> str:
    """
    Predict the question type from raw clause text using exact symbolic
    filtering — queries each question type separately, picks the highest
    scoring object match.
    """
    from concurrent.futures import ThreadPoolExecutor

    all_scores = {}
    all_hits   = {}

    def query_one(question_type):
        hits = search_slots({
            "question_type": {"query": question_type,       "weight": 0.01, "encoding": "exact"},
            "object":        {"query": clause_text[:500],   "weight": 1.0,  "encoding": "semantic"},
        }, top_k=top_k)
        score = hits[0].get("_score", 0.0) if hits else 0.0
        return question_type, score, hits[0] if hits else None

    with ThreadPoolExecutor(max_workers=3) as executor:
        for qt, score, hit in executor.map(lambda qt: query_one(qt), QUESTION_TYPES):
            all_scores[qt] = score
            all_hits[qt]   = hit

    best_type  = max(all_scores, key=all_scores.get)
    best_score = all_scores[best_type]
    best_hit   = all_hits[best_type]

    if verbose:
        print(f"\n  ── Clause Classification (exact symbolic predicate)")
        print(f"  ── Input: \"{clause_text[:80]}...\"")
        print(f"  {'─'*65}")
        print(f"\n  Score per question type (top 5):")
        for qt, sc in sorted(all_scores.items(), key=lambda x: -x[1])[:5]:
            marker = " ◀" if qt == best_type else ""
            print(f"    {qt:<55} {sc:.4f}{marker}")
        if best_hit:
            contract = best_hit.get("data", {}).get("contract_name", "?")
            answer   = best_hit.get("data", {}).get("answer", "?")
            print(f"\n  Predicted type : {best_type}")
            print(f"  Top score      : {best_score:.4f}")
            print(f"  Matched from   : {contract}  [{answer}]")

    return best_type


# ── Interactive mode ──────────────────────────────────────────────────────────

def interactive_mode():
    print("\n" + "=" * 65)
    print("  MAUD M&A Assistant — Interactive Mode")
    print("=" * 65)
    print("\n  Commands:")
    print("    retrieve <text>              — find similar clauses")
    print("    retrieve <text> @<contract>  — find clauses in one deal")
    print("    qa <question>                — answer a question")
    print("    qa <question> @<contract>    — answer within one deal")
    print("    classify <text>              — predict question type")
    print("    risk <contract>              — flag unusual deal terms")
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

        contract_name = None
        if " @" in user_input:
            user_input, contract_name = user_input.rsplit(" @", 1)
            contract_name = contract_name.strip()

        parts = user_input.split(" ", 1)
        cmd   = parts[0].lower()
        args  = parts[1] if len(parts) > 1 else ""

        if not args.strip():
            print(f"  Please provide text after '{cmd}'.")
            continue

        if cmd == "retrieve":
            retrieve_similar_clauses(args, contract_name=contract_name)
        elif cmd == "qa":
            answer_question(args, contract_name=contract_name)
        elif cmd == "classify":
            classify_clause(args)
        elif cmd == "risk":
            detect_risks(args)
        else:
            answer_question(user_input, contract_name=contract_name)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    global NAMESPACE, DB_NAME

    parser = argparse.ArgumentParser(
        description="MAUD merger agreement retrieval, risk detection, Q&A, classification"
    )
    parser.add_argument("--retrieve",      type=str, default=None,
                        help="Find similar clauses for this text")
    parser.add_argument("--qa",            type=str, default=None,
                        help="Answer a question about a merger agreement")
    parser.add_argument("--classify",      type=str, default=None,
                        help="Classify a raw clause text")
    parser.add_argument("--risk",          action="store_true",
                        help="Run risk analysis on a deal")
    parser.add_argument("--contract",      type=str, default=None,
                        help="Contract name filter (e.g. contract_41)")
    parser.add_argument("--question-type", type=str, default=None,
                        help="Override inferred question type for --retrieve")
    parser.add_argument("--top-k",         type=int, default=5)
    parser.add_argument("--namespace",     default=NAMESPACE)
    parser.add_argument("--db",            default=DB_NAME)
    parser.add_argument("--interactive",   action="store_true",
                        help="Launch interactive mode")
    args = parser.parse_args()

    NAMESPACE = args.namespace
    DB_NAME   = args.db

    if args.interactive:
        interactive_mode()

    elif args.retrieve:
        retrieve_similar_clauses(
            args.retrieve,
            question_type=args.question_type,
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
            print("  Example: python maud_query.py --risk --contract contract_41")
        else:
            detect_risks(args.contract)

    else:
        print("\nNo action specified. Examples:")
        print('  python maud_query.py --retrieve "material adverse effect pandemic carve-out"')
        print('  python maud_query.py --retrieve "termination fee" --contract contract_41')
        print('  python maud_query.py --qa "What type of consideration is used?" --contract contract_41')
        print('  python maud_query.py --classify "Either party may terminate if merger not consummated by outside date"')
        print('  python maud_query.py --risk --contract contract_41')
        print('  python maud_query.py --interactive')


if __name__ == "__main__":
    main()