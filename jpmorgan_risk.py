"""
jpmorgan_risk.py
================
Runs risk detection on the JPMorgan/Cardlytics MSA clauses
without needing the contract in the CUAD CSV or index.

The clause texts are embedded directly in this script.
The index (510 CUAD contracts) is queried for peer similarity.

Run from your legal folder:
    python jpmorgan_risk.py
"""

import os
import time
import requests
from dotenv import load_dotenv

load_dotenv()

SERVER_URL = os.environ.get("HB_SERVER_URL", os.environ.get("SERVER_URL", ""))
API_KEY    = os.environ.get("HB_API_KEY",    os.environ.get("API_KEY", ""))
DB_NAME    = os.environ.get("HB_DB_NAME",    "fraud_db")
NAMESPACE  = "cuad_clauses"

CONTRACT   = "JPMorganChase_Cardlytics_MSA_2018"

# ── Clause texts from the public EDGAR filing ─────────────────────────────────

CLAUSES = {
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

# Per-clause risk thresholds (from legal_query.py)
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


def search_slots(slot_queries: dict, top_k: int = 20, retries: int = 3) -> list:
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


def run_risk(clause_type: str, text: str) -> dict:
    # Query index for peer contracts of the same clause type
    hits = search_slots({
        "clause_type": {"query": clause_type, "weight": 0.01, "encoding": "exact"},
        "object":      {"query": text[:500],  "weight": 1.0,  "encoding": "semantic"},
    }, top_k=20)

    # Exclude any hits that mention this contract (none will, it's not ingested)
    other_hits = [
        h for h in hits
        if "jpmorgan" not in h.get("data", {}).get("contract", "").lower()
        and "cardlytics" not in h.get("data", {}).get("contract", "").lower()
    ]

    top_score    = other_hits[0].get("_score", 0.0) if other_hits else 0.0
    top_contract = other_hits[0].get("data", {}).get("contract", "?")[:50] if other_hits else "?"
    top_text     = other_hits[0].get("data", {}).get("object", "")[:120] if other_hits else ""

    unusual_t, review_t = RISK_THRESHOLDS.get(clause_type, (0.5, 0.7))
    if top_score < unusual_t:
        risk_level = "UNUSUAL"
        symbol     = "UNUSUAL"
    elif top_score < review_t:
        risk_level = "REVIEW"
        symbol     = "REVIEW "
    else:
        risk_level = "NORMAL"
        symbol     = "NORMAL "

    return {
        "clause_type":    clause_type,
        "risk_level":     risk_level,
        "symbol":         symbol,
        "peer_score":     top_score,
        "peer_contract":  top_contract,
        "peer_text":      top_text,
        "unusual_thresh": unusual_t,
        "review_thresh":  review_t,
    }


def main():
    print(f"\n{'='*70}")
    print(f"  Risk Analysis: {CONTRACT}")
    print(f"  Comparing against 510 CUAD peer contracts")
    print(f"{'='*70}")

    results = []
    for clause_type, text in CLAUSES.items():
        print(f"\n  Querying: {clause_type}...", end=" ", flush=True)
        result = run_risk(clause_type, text)
        results.append(result)

        sym = result["symbol"]
        score = result["peer_score"]
        print(f"{sym}  (peer similarity: {score:.4f})")
        print(f"    Nearest peer : {result['peer_contract']}")
        print(f"    Peer text    : {result['peer_text']}...")
        print(f"    Thresholds   : unusual < {result['unusual_thresh']:.2f}  |  review < {result['review_thresh']:.2f}")

    # Summary
    unusual = [r for r in results if r["risk_level"] == "UNUSUAL"]
    review  = [r for r in results if r["risk_level"] == "REVIEW"]
    normal  = [r for r in results if r["risk_level"] == "NORMAL"]

    print(f"\n{'='*70}")
    print(f"  RISK SUMMARY")
    print(f"{'='*70}")
    print(f"  Contract  : {CONTRACT}")
    print(f"  Source    : SEC EDGAR Exhibit 10.1, CIK 1666071 (filed August 2018)")
    print(f"  Clauses   : {len(results)} analyzed")
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


if __name__ == "__main__":
    main()