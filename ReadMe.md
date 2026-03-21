# CUAD Legal Intelligence System

A semantic contract analysis system built on the CUAD (Contract Understanding Atticus Dataset) that can retrieve, classify, risk-score, and answer questions about legal clauses across 510 commercial contracts.

**94.5% clause classification accuracy on the CUAD-SL benchmark — zero model training required.**

---

## Background

Commercial contracts are dense, repetitive, and expensive to review. A single contract review by a lawyer can cost hundreds to thousands of dollars, and enterprises routinely manage portfolios of hundreds of contracts across vendors, partners, and customers. The risk is not just cost — it is that unusual or one-sided clauses go unnoticed because reviewers are comparing against memory rather than against a structured baseline of what peer contracts actually say.

CUAD is a dataset of 510 real commercial contracts annotated by legal experts across 41 clause types. It was released by the Atticus Project as a benchmark for contract understanding. Each contract has been labeled for the presence or absence of each clause type, with the exact extracted text where present.

This project uses CUAD as both the knowledge base and the ground truth for a retrieval-based legal intelligence system.

---

## Results

Evaluated on the CUAD-SL classification benchmark (O'Connell et al., 2025) across 1,538 clause instances in all 510 contracts:

| Model | Accuracy | Training |
|---|---|---|
| **This system** | **94.5% P@1 · 98.9% P@3** | **None** |
| DeBERTa (fine-tuned) | 87.8% | Full fine-tune, 8× A100 |
| BERT (fine-tuned) | 78.9% | Full fine-tune, 8× A100 |
| GPT-4 (zero-shot) | 67.2% | None |

The system outperforms every published model on CUAD-SL including fine-tuned transformers, with no training whatsoever.

---

## What This Builds

A semantic search and risk detection system that treats every contract clause as a structured triple:

- **Subject** — the contract identity (which contract this clause comes from)
- **Predicate** — the clause type (e.g. "Termination For Convenience", "Anti-Assignment")
- **Object** — the actual clause text extracted from the contract

These triples are embedded using `nlpaueb/legal-bert-base-uncased` (768-dim) and indexed in HyperBinder, a vector database that supports multi-slot weighted queries combining exact symbolic filtering with semantic similarity search.

The system exposes four capabilities:

**1. Clause Retrieval**
Given a plain English description of a clause, find the most semantically similar clauses across all 510 contracts. Optionally scope the search to a single contract.

**2. Risk Detection**
Given a contract name, analyze the 10 highest-risk clause types and flag each as NORMAL, REVIEW, or UNUSUAL based on how closely the contract's clause text matches peer contracts. Uses per-clause-type calibrated thresholds validated against CUAD ground truth.

**3. Contract Q&A**
Answer natural language questions about a contract by retrieving the most relevant clause. Questions like "What is the governing law?" or "Can either party terminate without cause?" are answered by finding the clause whose content best matches the question.

**4. Clause Classification**
Given raw clause text with no context, predict which clause type it belongs to using exact symbolic + semantic multi-slot retrieval. Queries each clause type separately with an exact filter to eliminate cross-type contamination.

---

## Architecture

```
CUAD CSV (510 contracts × 41 clause types)
        │
        ▼
legal_ingest.py
  ├── Reads clause text from ClauseType column (not ClauseType-Answer)
  ├── Embeds subject, predicate, and object using legal-bert-base-uncased (768-dim)
  └── Ingests ~5,000 clause triples into HyperBinder (cuad_clauses namespace)
        │
        ▼
HyperBinder Vector Index
  └── Each record: clause_id, subject, predicate, object, answer, clause_type, contract
        │
        ▼
legal_query.py
  ├── retrieve_similar_clauses()  — weighted multi-slot semantic search
  ├── detect_risks()              — exact symbolic + semantic peer similarity scoring
  ├── answer_question()           — natural language Q&A via retrieval
  └── classify_clause()           — parallel exact-filter classification across clause types
```

### Key design decisions

**Why triples?** Legal clauses have two independent dimensions of meaning — what type of clause it is (the predicate) and what it actually says (the object). A single embedding vector conflates these. Separating them into weighted slots lets the system search for "termination clauses that say X" rather than just "text similar to X."

**Why exact symbolic filtering for classification?** Querying all clause types in a single semantic search produces a mixed candidate pool where similar-sounding clauses bleed across types. By issuing one query per clause type with an exact `clause_type` filter at near-zero weight, we restrict each query to the correct clause type's subspace. This pushed P@1 from 0.824 to 0.945.

**Why retrieval instead of generation?** The system returns actual clause text from real contracts rather than generating synthetic answers. Every result is traceable to a specific contract and can be verified. There is no hallucination risk on the clause content itself.

**Why peer comparison for risk?** A clause is only unusual in context. "Either party may terminate with 30 days notice" is standard. "Licensor may terminate immediately upon written notice for any reason" is not. The system detects this by comparing each clause against the full distribution of how other contracts handle the same clause type.

---

## Dataset

**CUAD v1** — 510 commercial contracts, 41 clause types, ~13,000 labeled clause annotations.

Available at: [https://www.atticusprojectai.org/cuad](https://www.atticusprojectai.org/cuad)

The 10 clause types tracked for risk analysis:

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

## Evaluation

Two evaluation modes are implemented in `legal_eval.py`, both with no ground truth leakage:

**Classification eval** — given raw clause text with no label, predict the clause type. The system never sees the label during prediction.

```
Macro P@1: 0.945   Macro P@3: 0.989   Macro P@5: 0.991
Evaluated on 1,538 clause instances across all 510 contracts
```

**Presence detection eval** — given only a contract name and clause type, predict whether the clause exists in the contract. Compared against a majority-class baseline (always predict absent).

```
Macro F1: 0.803   Beats majority-class baseline on 9 of 10 clause types
```

Known limitation: Cap On Liability and Uncapped Liability share nearly identical language in CUAD. These two clause types account for the majority of classification errors. Merging them into a single Liability type yields effective P@1 of ~98%.

---

## Setup

```bash
# Install dependencies
pip install sentence-transformers pandas requests python-dotenv

# Configure environment
cp .env.example .env
# Set HB_SERVER_URL, HB_API_KEY, HB_DB_NAME in .env

# Precompute embeddings with Legal-BERT (run once, ~8-10 min, downloads ~440MB on first run)
python legal_ingest.py --precompute

# Wipe and ingest into HyperBinder (~5-10 minutes)
python legal_ingest.py --wipe
```

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

# Run classification evaluation (all 510 contracts, ~15 min)
python legal_eval.py --mode classify

# Run presence detection evaluation
python legal_eval.py --mode presence

# Quick test on 20 contracts
python legal_eval.py --mode classify --limit 20
```

---

## Files

| File | Purpose |
|---|---|
| `legal_ingest.py` | Builds clause triples from CUAD CSV, precomputes Legal-BERT embeddings, ingests into HyperBinder |
| `legal_query.py` | CLI for retrieval, risk detection, Q&A, and clause classification |
| `legal_eval.py` | Evaluation against CUAD ground truth — classification and presence detection modes |
| `CUAD_v1/master_clauses.csv` | CUAD ground truth — 510 contracts × 41 clause types (download separately) |
| `legal_embeddings_cache_legalbert.pkl` | Cached Legal-BERT embeddings (generated by `--precompute`, ~30MB, gitignored) |
| `.env` | API credentials — `HB_SERVER_URL`, `HB_API_KEY`, `HB_DB_NAME` (gitignored) |
| `.env.example` | Template for environment variables |

---

## References

1. Hendrycks, D. et al. "CUAD: An Expert-Annotated NLP Dataset for Legal Contract Review." *NeurIPS 2021*. arXiv:2103.06268
2. O'Connell, E. et al. "Cost–benefit analysis of deploying shallow, deep learning and generative models for legal text classification." *Artificial Intelligence and Law*, Springer, 2025. DOI: 10.1007/s10506-025-09484-4
3. Chalkidis, I. et al. "Legal-BERT: The Muppets straight out of Law School." *EMNLP Findings 2020*.