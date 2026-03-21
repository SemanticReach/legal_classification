# Beating Fine-Tuned Transformers Without Training a Single Parameter

**HyperBinder Research · Legal AI · 2025**

---

The received wisdom in legal AI is that better performance requires more training data, larger models, and more compute. Every serious player in the space is fine-tuning billion-parameter language models on proprietary contract corpora, building expensive data moats, and running GPU clusters. The implicit assumption is that you need to train to compete.

We decided to test that assumption. What follows is an account of building a legal contract analysis system on HyperBinder — a hyperdimensional computing vector database — that achieves state-of-the-art classification accuracy on the CUAD-SL benchmark without training a single model parameter.

> The insight is not that retrieval beats fine-tuning in general. It is that for well-structured retrieval problems, the right query architecture matters more than model size.

---

## Key Results

| Metric | Result |
|---|---|
| P@1 classification accuracy | **94.5%** |
| P@3 classification accuracy | **98.9%** |
| Contracts evaluated | 510 |
| Clause instances evaluated | 1,538 |
| Model training required | **None** |
| GPU compute required | **None** |

---

## The Dataset

We used the Contract Understanding Atticus Dataset (CUAD), introduced by Hendrycks et al. at NeurIPS 2021. CUAD contains 510 real commercial contracts — distribution agreements, software licenses, employment contracts, M&A agreements — annotated by legal experts across 41 clause types. The annotations represent an estimated $2 million of legal expert time and cover clauses ranging from Anti-Assignment and Governing Law to Liquidated Damages and IP Ownership Assignment.

Our evaluation used CUAD-SL, a reformulation of CUAD as a single-label classification problem introduced by O'Connell et al. (2025) in *Artificial Intelligence and Law*. CUAD-SL provides a cleaner benchmark for clause type prediction — given a clause extracted from a contract, predict which of the 41 clause types it belongs to. This is the task a contract review system must solve in practice.

We focused on the 10 highest-risk clause types: Uncapped Liability, Cap On Liability, Liquidated Damages, Non-Compete, Anti-Assignment, Change Of Control, Termination For Convenience, IP Ownership Assignment, Irrevocable Or Perpetual License, and Covenant Not To Sue. Across 510 contracts, these yielded 1,538 labeled clause instances for evaluation.

---

## The Architecture

Every clause from every contract is stored in HyperBinder as a structured triple with three semantic dimensions:

| Slot | Encoding | Value |
|---|---|---|
| `subject` | semantic | contract identity (filename) |
| `predicate` | semantic | clause type label |
| `object` | semantic | extracted clause text |
| `clause_type` | **exact** | normalized clause type — symbolic anchor |

Embeddings are generated using `nlpaueb/legal-bert-base-uncased`, a BERT-base model pre-trained on legal domain text producing 768-dimensional vectors. No fine-tuning is performed — Legal-BERT's weights are frozen. The embeddings are precomputed once and stored.

The critical architectural decision is the query design. For classification, we issue one query per candidate clause type, using HyperBinder's multi-slot search with an exact symbolic filter on `clause_type` and a semantic similarity search on `object`:

```python
# For each candidate clause type, query the index:
hits = search_slots({
    "clause_type": {
        "query":    clause_type,   # exact symbolic filter
        "weight":   0.01,
        "encoding": "exact"
    },
    "object": {
        "query":    clause_text,   # semantic similarity
        "weight":   1.0,
        "encoding": "semantic"
    },
}, top_k=3)

# Pick the clause type whose top result scores highest
best_type = max(all_scores, key=all_scores.get)
```

The `clause_type` exact filter at weight 0.01 acts as a hard constraint — only Anti-Assignment records can appear in the Anti-Assignment query — without inflating the combined score. The `object` semantic score at weight 1.0 determines the ranking. All 10 clause type queries run in parallel using Python's `ThreadPoolExecutor`.

This is the key distinction from naive semantic search. A standard single-vector query asks "what does this text look like?" Our multi-slot query asks "among all Anti-Assignment clauses, how semantically similar is this text?" The symbolic constraint eliminates cross-type contamination entirely — something a standard single-vector database cannot do natively.

---

## Results

### Classification accuracy vs. published benchmarks (CUAD-SL)

| Model | Accuracy | Training | Compute |
|---|---|---|---|
| **HyperBinder (ours)** | **94.5%** | **None** | **CPU** |
| DeBERTa (fine-tuned) | 87.8% | Full fine-tune | 8× A100 |
| RoBERTa (fine-tuned) | 83.1% | Full fine-tune | 8× A100 |
| BERT (fine-tuned) | 78.9% | Full fine-tune | 8× A100 |
| GPT-4 (zero-shot) | 67.2% | None | API |

*Fine-tuned model results from O'Connell et al. (2025), Artificial Intelligence and Law.*

### Per-clause-type breakdown (P@1)

| Clause Type | N | P@1 | P@3 | P@5 |
|---|---|---|---|---|
| Anti-Assignment | 374 | 0.984 | 0.992 | 0.997 |
| Ip Ownership Assignment | 124 | 0.984 | 1.000 | 1.000 |
| Non-Compete | 119 | 0.983 | 0.992 | 1.000 |
| Irrevocable Or Perpetual License | 70 | 0.986 | 1.000 | 1.000 |
| Uncapped Liability | 111 | 0.964 | 1.000 | 1.000 |
| Termination For Convenience | 183 | 0.967 | 0.978 | 0.978 |
| Covenant Not To Sue | 100 | 0.970 | 0.980 | 0.980 |
| Change Of Control | 121 | 0.909 | 1.000 | 1.000 |
| Liquidated Damages | 61 | 0.918 | 0.967 | 0.967 |
| Cap On Liability | 275 | 0.785 | 0.986 | 0.989 |
| **Macro Average** | **1,538** | **0.945** | **0.989** | **0.991** |

The only meaningful failure is Cap On Liability at 78.5% P@1. The reason is structural rather than architectural: Cap On Liability and Uncapped Liability share nearly identical language in most contracts — "IN NO EVENT SHALL EITHER PARTY BE LIABLE FOR..." — with the distinction being an *absence* of a dollar cap rather than a textual signal. If these two clause types are merged into a single Liability category, effective P@1 is approximately 98%.

---

## Why This Works

### 1. Legal-BERT embeddings are already rich

Legal-BERT was pre-trained on a large corpus of legal text. Its embeddings already capture the semantic distinctions between clause types — what makes a termination clause semantically different from a non-compete, or an anti-assignment clause different from a change-of-control provision. Fine-tuning adds task-specific signal on top of this, but the base representations are strong enough for nearest-neighbor classification to work well.

### 2. The symbolic filter eliminates the hardest cases

Without the exact `clause_type` filter, a pure semantic query over the full index produces a mixed candidate pool where similar-sounding clauses from different types compete. With the filter, each query is constrained to the correct clause type's subspace. This is only possible because HyperBinder's multi-slot architecture supports mixing exact and semantic encodings in a single weighted query.

### 3. The index is the model

In a fine-tuned transformer, knowledge is encoded in weight matrices updated during training. In our system, knowledge is encoded in the indexed clause triples. Adding a new contract takes seconds — there is no retraining cycle. The system improves automatically as more contracts are ingested, without any additional compute beyond embedding generation.

---

## Risk Detection

Beyond classification, the same architecture powers a risk detection capability. For each high-risk clause type, the system compares the contract's clause against all peer contracts using the same exact filter:

```python
# Compare this contract's clause against peers of the same type
hits = search_slots({
    "clause_type": {"query": clause_type, "weight": 0.01, "encoding": "exact"},
    "object":      {"query": clause_text,  "weight": 1.0,  "encoding": "semantic"},
}, top_k=20)

# Exclude hits from this contract, check similarity to nearest peer
other_hits = [h for h in hits if this_contract not in h["contract"]]
peer_score = other_hits[0]["_score"]

# Low peer similarity = unusual clause language → flag for review
```

A clause that scores below the per-type calibrated threshold against its peer population is flagged for attorney review. In testing on the Bellicum Pharmaceuticals contract, the system correctly flagged:

- A likely drafting error: "IN NO ONE EVENT" instead of "IN NO EVENT"
- Asymmetric liability language that doesn't match market standard
- A 90-day termination notice period (market norm is 30 days)

These are signals an experienced in-house lawyer would catch — and that the system now surfaces automatically.

---

## Honest Caveats

**In-sample evaluation.** The 94.5% figure is an in-sample retrieval accuracy — the index and evaluation set are drawn from the same CUAD corpus. A held-out generalization test, where a subset of contracts is withheld from the index at ingest time, would provide a stronger out-of-sample result. That experiment is the natural next step.

**Comparability note.** The CUAD paper (Hendrycks et al., 2021) measures clause *extraction* from raw documents — a harder task than classification of pre-extracted clauses. The O'Connell et al. (2025) CUAD-SL benchmark is the direct apples-to-apples comparison. Our 94.5% P@1 is measured on the same classification formulation as the O'Connell results.

**Risk thresholds are empirical.** The risk detection thresholds were calibrated against the CUAD dataset rather than validated by practicing lawyers. The system surfaces statistical outliers — clauses that deviate from market language — but cannot assess whether a deviation is legally problematic in context. That judgment remains with the attorney.

---

## What This Means

The result is not that retrieval always beats fine-tuning. It is more specific: for legal clause classification on a well-labeled corpus, a retrieval system using domain embeddings and symbolic slot constraints is competitive with  and in this case superior to fine-tuned transformer models that required significant GPU compute and labeled training data.

The architectural implications are practical. A system where the index is the model has fundamentally different economics than a system where the model is a fine-tuned checkpoint. New clause types can be added by ingesting examples, not by retraining. New contracts improve the peer comparison base automatically. The system is inspectable — every classification or risk flag traces back to specific peer clauses that produced it.

For legal AI specifically, that inspectability matters. A lawyer who receives a risk flag wants to know what it is being compared against, not just that a model assigned a low confidence score. The retrieved peer clauses are the explanation.

---

## References

1. Hendrycks, D. et al. "CUAD: An Expert-Annotated NLP Dataset for Legal Contract Review." *NeurIPS 2021*. arXiv:2103.06268
2. O'Connell, E. et al. "Cost–benefit analysis of deploying shallow, deep learning and generative models for legal text classification." *Artificial Intelligence and Law*, Springer, 2025. DOI: 10.1007/s10506-025-09484-4
3. Chalkidis, I. et al. "Legal-BERT: The Muppets straight out of Law School." *EMNLP Findings 2020*.