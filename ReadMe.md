# CUAD Legal Intelligence System

A semantic contract analysis system built on the CUAD (Contract Understanding Atticus Dataset) that can retrieve, classify, risk-score, and answer questions about legal clauses across 510 commercial contracts.

---

## Background

Commercial contracts are dense, repetitive, and expensive to review. A single contract review by a lawyer can cost hundreds to thousands of dollars, and enterprises routinely manage portfolios of hundreds of contracts across vendors, partners, and customers. The risk is not just cost — it is that unusual or one-sided clauses go unnoticed because reviewers are comparing against memory rather than against a structured baseline of what peer contracts actually say.

CUAD is a dataset of 510 real commercial contracts annotated by legal experts across 41 clause types. It was released by the Atticus Project as a benchmark for contract understanding. Each contract has been labeled for the presence or absence of each clause type, with the exact extracted text where present.

This project uses CUAD as both the knowledge base and the ground truth for a retrieval-based legal intelligence system.

---

## What We Are Building

A semantic search and risk detection system that treats every contract clause as a structured triple:

- **Subject** — the contract identity (which contract this clause comes from)
- **Predicate** — the clause type (e.g. "Termination For Convenience", "Anti-Assignment")
- **Object** — the actual clause text extracted from the contract

These triples are embedded and indexed in HyperBinder, a vector database that supports multi-slot weighted queries. This allows the system to search across all three dimensions simultaneously — finding clauses that are semantically similar in content, anchored to the right clause type, and optionally scoped to a specific contract.

The system exposes four capabilities:

**1. Clause Retrieval**
Given a plain English description of a clause (e.g. "termination without cause 30 days notice"), find the most semantically similar clauses across all 510 contracts. Optionally scope the search to a single contract.

**2. Risk Detection**
Given a contract name, analyze the 10 highest-risk clause types (liability caps, non-competes, IP assignment, anti-assignment, etc.) and flag each as NORMAL, REVIEW, or UNUSUAL based on how closely the contract's clause text matches peer contracts. A clause with no close matches in other contracts is flagged as unusual — it may be one-sided, overly broad, or simply non-standard language that warrants attorney review.

**3. Contract Q&A**
Answer natural language questions about a contract by retrieving the most relevant clause. Questions like "What is the governing law?" or "Can either party terminate without cause?" are answered by finding the clause whose content best matches the question.

**4. Clause Classification**
Given raw clause text with no context, predict which of the 41 CUAD clause types it belongs to using nearest-neighbor retrieval.

---

## Architecture

```
CUAD CSV (510 contracts × 41 clause types)
        │
        ▼
legal_ingest.py
  ├── Reads clause text from each (contract, clause_type) pair
  ├── Embeds subject, predicate, and object using all-MiniLM-L6-v2
  └── Ingests ~5,000 clause triples into HyperBinder (cuad_clauses namespace)
        │
        ▼
HyperBinder Vector Index
  └── Each record: clause_id, subject, predicate, object, answer, clause_type, contract
        │
        ▼
legal_query.py
  ├── retrieve_similar_clauses()  — weighted multi-slot semantic search
  ├── detect_risks()              — CSV presence check + peer similarity scoring
  ├── answer_question()           — natural language Q&A via retrieval
  └── classify_clause()           — nearest-neighbor clause type prediction
```

### Key design decisions

**Why triples?** Legal clauses have two independent dimensions of meaning — what type of clause it is (the predicate) and what it actually says (the object). A single embedding vector conflates these. Separating them into weighted slots lets the system search for "termination clauses that say X" rather than just "text similar to X."

**Why retrieval instead of generation?** The system returns actual clause text from real contracts rather than generating synthetic answers. This matters for legal use — every result is traceable to a specific contract and can be verified. There is no hallucination risk on the clause content itself.

**Why peer comparison for risk?** A clause is only unusual in context. "Either party may terminate with 30 days notice" is standard. "Licensor may terminate immediately upon written notice for any reason" is not. The system detects this by comparing each clause against the full distribution of how other contracts handle the same clause type. Low peer similarity is a signal that the clause is worded differently from market standard — which may or may not be a problem, but warrants review.

---

## Dataset

**CUAD v1** — 510 commercial contracts, 41 clause types, ~13,000 labeled clause annotations.

Available at: [https://www.atticusprojectai.org/cuad](https://www.atticusprojectai.org/cuad)

The 10 clause types currently tracked for risk analysis:

| Clause Type | Contracts with clause | Base rate |
|---|---|---|
| Anti-Assignment | 374 / 510 | 73% |
| Cap On Liability | 275 / 510 | 54% |
| Termination For Convenience | 183 / 510 | 36% |
| Change Of Control | 121 / 510 | 24% |
| Ip Ownership Assignment | 124 / 510 | 24% |
| Non-Compete | 119 / 510 | 23% |
| Uncapped Liability | 111 / 510 | 22% |
| Covenant Not To Sue | 100 / 510 | 20% |
| Irrevocable Or Perpetual License | 70 / 510 | 14% |
| Liquidated Damages | 61 / 510 | 12% |

---

## Current Limitations

**Embedding model.** The system uses `all-MiniLM-L6-v2`, a general-purpose 384-dimensional sentence transformer. This model was not trained on legal text and underrepresents domain-specific language. Upgrading to a legal-domain model such as `nlpaueb/legal-bert-base-uncased` would meaningfully improve retrieval quality.

**Risk thresholds are uncalibrated.** The NORMAL / REVIEW / UNUSUAL boundaries (similarity < 0.7 = REVIEW, < 0.5 = UNUSUAL) were set heuristically. They have not been validated against labeled ground truth.

**No batch evaluation.** The system currently analyzes one contract at a time. A batch pipeline that runs all 510 contracts and outputs a ranked risk report does not yet exist.

**Single-vector ingest.** HyperBinder receives one precomputed vector per record (the object embedding). The subject and predicate slots rely on the model re-encoding at query time. A full multi-vector ingest would improve index quality.

---

## Evaluation Plan

The system will be evaluated against CUAD ground truth on the following metrics:

- **Presence detection** — for each of the 10 risk clause types, does the system correctly identify whether the clause is present or absent? Measured as precision, recall, and F1 across all 510 contracts.
- **Retrieval relevance** — for `--retrieve` queries, does the top-ranked result belong to the correct clause type? Measured as precision@1 and precision@5.
- **Baseline comparison** — results will be compared against (1) a TF-IDF keyword baseline, (2) always-MISSING majority class baseline, and (3) GPT-4 zero-shot clause identification.

The goal is to demonstrate that semantic retrieval over structured clause triples outperforms keyword search on clause type recall, particularly for clause types with low lexical overlap across contracts.

---

## Usage

```bash
# Risk analysis on a specific contract
python legal_query.py --risk --contract "BELLICUMPHARMACEUTICALS"

# Find similar termination clauses across all contracts
python legal_query.py --retrieve "termination without cause 30 days notice"

# Find termination clauses within one contract
python legal_query.py --retrieve "termination without cause" --contract "EuromediaHoldings"

# Answer a question about a contract
python legal_query.py --qa "What is the governing law?" --contract "CybergyHoldings"

# Classify a raw clause
python legal_query.py --classify "Either party may terminate with 60 days written notice"

# Interactive mode
python legal_query.py --interactive
```

---

## Setup

```bash
# Install dependencies
pip install sentence-transformers pandas requests python-dotenv

# Configure environment
cp .env.example .env
# Set HB_SERVER_URL, HB_API_KEY, HB_DB_NAME in .env

# Precompute embeddings (run once, ~2 minutes)
python legal_ingest.py --precompute

# Ingest into HyperBinder (~5-10 minutes)
python legal_ingest.py --wipe
```

---

## Files

| File | Purpose |
|---|---|
| `legal_ingest.py` | Builds clause triples from CUAD CSV, precomputes embeddings, ingests into HyperBinder |
| `legal_query.py` | CLI for retrieval, risk detection, Q&A, and clause classification |
| `CUAD_v1/master_clauses.csv` | CUAD ground truth — 510 contracts × 41 clause types |
| `legal_embeddings_cache.pkl` | Cached sentence embeddings (generated by `--precompute`) |