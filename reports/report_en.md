# Enhancing RAG with Adaptive Chunking: ColBERTv2 vs Bi-Encoder Retrieval Benchmark

**Course**: ECE1508 — Advanced Topics in Machine Learning Systems  
**Sub-experiment**: Retrieval Model Benchmarking on KILT NaturalQuestions  
**Date**: April 2026

---

## 1. Introduction

Retrieval-Augmented Generation (RAG) pipelines combine a retrieval component with a generative language model to ground responses in factual, up-to-date document corpora. The quality of the retrieval step is a primary bottleneck: if the relevant passage is not retrieved, no amount of generation sophistication can compensate.

This project, "Enhancing RAG with Adaptive Chunking", investigates how chunking strategy interacts with retrieval model architecture to affect end-to-end RAG performance. Before exploring adaptive chunking strategies, it is necessary to establish a baseline understanding of retrieval model behavior across query types.

This report covers a targeted sub-experiment: a head-to-head benchmark of two retrieval paradigms on the KILT NaturalQuestions (NQ) development set.

1. **Bi-encoder baseline** — `all-MiniLM-L6-v2` with FAISS nearest-neighbor search. Both query and passage are independently encoded into a single dense vector; retrieval is a simple inner-product search.
2. **Late-interaction model** — `ColBERTv2`, retrieved via the RAGatouille library using PLAID indexing. Each query token attends over every passage token via a MaxSim operation, preserving fine-grained token-level interactions at retrieval time.

The central motivation is to determine whether the richer token-level matching in ColBERTv2 provides a measurable recall advantage, particularly on queries that mention multiple distinct named entities, which are queries where the single-vector bottleneck of bi-encoders is expected to be most damaging.

The findings from this benchmark will directly inform later design decisions about chunking granularity and retrieval model selection within the broader adaptive RAG system.

---

## 2. Hypothesis

**Primary hypothesis**: ColBERTv2's MaxSim token-level matching will outperform the bi-encoder baseline in recall across most values of k, because late interaction preserves independent query-token signals rather than collapsing them into a single embedding.

**Entity-stratified hypothesis**: The recall gap between ColBERTv2 and the bi-encoder will be larger for multi-entity queries (those containing two or more distinct named entities) than for single-entity queries (zero or one named entity). The rationale is as follows:

- A bi-encoder must represent all query semantics in a fixed-dimensional vector. When a query references multiple named entities (e.g., "Who directed the film that stars both X and Y?"), the encoder must trade off representational fidelity between entities, potentially under-weighting one.
- ColBERTv2 retains a separate contextual embedding for each query token. The MaxSim score over a passage is the sum of per-token maximum similarities, so each entity token contributes independently to the final score. This architecture is structurally better suited to multi-entity constraint satisfaction.

**Null hypothesis**: No statistically significant difference in Recall@k exists between the two models for either query group.

---

## 3. Methodology

### 3.1 Data Pipeline

**Dataset**: KILT NaturalQuestions (NQ) development split. KILT reformats NQ so that each question is paired with a gold Wikipedia page identifier and a specific passage provenance annotation, enabling precise retrieval evaluation against a fixed Wikipedia snapshot.

**Corpus construction**: Using the full KILT Wikipedia dump would be computationally prohibitive for a local benchmark. A reduced corpus of approximately 10,000 Wikipedia pages was constructed as follows:

1. All gold provenance pages referenced in the NQ dev split were included unconditionally.
2. A set of distractor pages was sampled uniformly at random from the remainder of the KILT Wikipedia dump to reach the target corpus size.
3. Each page was split into paragraph-level chunks. Paragraphs shorter than 20 tokens were merged with adjacent paragraphs to avoid degenerate chunks.

**Chunking**: Paragraph-level splitting was applied uniformly to both models in this benchmark. Adaptive chunking is the subject of later experiments and is intentionally excluded here to isolate the effect of retrieval model architecture.

### 3.2 Query Classification

Named entity recognition was performed on every NQ dev query using the `en_core_web_sm` spaCy pipeline. Each query was labeled with the count of unique named entities detected (entity types: PERSON, ORG, GPE, LOC, WORK_OF_ART, EVENT, FAC).

Queries were partitioned into two groups:

| Group | Criterion | Target size |
|---|---|---|
| Single-entity | 0 or 1 detected named entity | ~500 queries |
| Multi-entity | 2 or more detected named entities | ~500 queries |

To obtain balanced groups of approximately 500 queries each, the smaller group was used at full size and the larger group was randomly downsampled to match.

### 3.3 Retrieval Models

**Bi-encoder (baseline)**

- Encoder: `sentence-transformers/all-MiniLM-L6-v2` (22M parameters, 384-dimensional embeddings)
- Index: FAISS `IndexFlatIP` (exact inner-product search; no approximation error)
- Query encoding: single forward pass per query; corpus passages encoded offline
- Retrieval: top-k nearest neighbors by cosine similarity (L2-normalized embeddings)

**ColBERTv2 (late interaction)**

- Model: `colbert-ir/colbertv2.0` accessed via the RAGatouille wrapper library
- Indexing: PLAID (Practical Late-Interaction Approximate Dense retrieval) index, which uses centroid-based compression for scalable MaxSim retrieval
- Query encoding: token-level contextualized embeddings (128-dimensional per token)
- Retrieval: approximate MaxSim over compressed passage token matrices

### 3.4 Evaluation Metric

**Recall@k** is the primary metric. A query is considered a hit at rank k if at least one of the top-k retrieved passages matches a gold provenance passage for that query (passage-level match on KILT provenance identifiers).

Recall@k was computed at k ∈ {1, 5, 10, 20} for both models, across three query groups: all queries, single-entity queries, and multi-entity queries.

---

## 4. Experimental Setup

### 4.1 Hardware

| Resource | Specification |
|---|---|
| CPU | [To be filled after benchmark run] |
| RAM | [To be filled after benchmark run] |
| GPU | [To be filled after benchmark run] |
| VRAM | [To be filled after benchmark run] |
| Storage | [To be filled after benchmark run] |

### 4.2 Corpus Configuration

| Parameter | Value |
|---|---|
| Total Wikipedia pages | ~10,000 |
| Gold provenance pages | [To be filled after benchmark run] |
| Distractor pages | [To be filled after benchmark run] |
| Total paragraph chunks | [To be filled after benchmark run] |
| Average chunk length (tokens) | [To be filled after benchmark run] |

### 4.3 Software Versions

| Package | Version |
|---|---|
| Python | [To be filled after benchmark run] |
| PyTorch | [To be filled after benchmark run] |
| sentence-transformers | [To be filled after benchmark run] |
| faiss-cpu / faiss-gpu | [To be filled after benchmark run] |
| RAGatouille | [To be filled after benchmark run] |
| ColBERT (via RAGatouille) | [To be filled after benchmark run] |
| spaCy | [To be filled after benchmark run] |
| spaCy model (en_core_web_sm) | [To be filled after benchmark run] |

---

## 5. Results

### 5.1 Overall Recall@k

The table below reports Recall@k across all evaluated queries, combining both entity groups.

| k | Bi-encoder (MiniLM) | ColBERTv2 | Delta |
|---|---|---|---|
| 1 | [TBF] | [TBF] | [TBF] |
| 5 | [TBF] | [TBF] | [TBF] |
| 10 | [TBF] | [TBF] | [TBF] |
| 20 | [TBF] | [TBF] | [TBF] |

*[To be filled after benchmark run]*

![Overall Recall@k bar chart](../results/charts/recall_bar_chart.png)

### 5.2 Recall@k by Entity Group

**Single-entity queries** (0–1 named entities):

| k | Bi-encoder (MiniLM) | ColBERTv2 | Delta |
|---|---|---|---|
| 1 | [TBF] | [TBF] | [TBF] |
| 5 | [TBF] | [TBF] | [TBF] |
| 10 | [TBF] | [TBF] | [TBF] |
| 20 | [TBF] | [TBF] | [TBF] |

**Multi-entity queries** (2+ named entities):

| k | Bi-encoder (MiniLM) | ColBERTv2 | Delta |
|---|---|---|---|
| 1 | [TBF] | [TBF] | [TBF] |
| 5 | [TBF] | [TBF] | [TBF] |
| 10 | [TBF] | [TBF] | [TBF] |
| 20 | [TBF] | [TBF] | [TBF] |

*[To be filled after benchmark run]*

The distribution of per-query Recall@10 scores across the two models is visualized below.

![Recall@10 histogram](../results/charts/recall_histogram_k10.png)

---

## 6. Analysis

### 6.1 Overall Retrieval Performance

[To be filled after benchmark run]

Based on expected behavior from the literature, ColBERTv2 should demonstrate higher Recall@1 and Recall@5 compared to the bi-encoder, with the gap likely narrowing at larger k values. At k=20, both models can retrieve a wide net of candidates, so architecture differences matter less.

The bi-encoder's advantage is speed and simplicity: a single embedding per passage allows for highly efficient FAISS lookup. If ColBERTv2's recall advantage is modest at larger k values, the bi-encoder may remain preferable for latency-sensitive production systems, while ColBERTv2 is better suited as a reranker in a retrieve-then-rerank pipeline.

### 6.2 Entity Group Breakdown

[To be filled after benchmark run]

The hypothesis predicts that the gap between ColBERTv2 and the bi-encoder will be more pronounced in the multi-entity group. Several outcomes are possible:

- **Hypothesis confirmed**: ColBERTv2 outperforms the bi-encoder by a larger margin on multi-entity queries, providing strong motivation for using late interaction in multi-hop or compositional retrieval tasks.
- **Hypothesis partially confirmed**: ColBERTv2 outperforms overall but the entity-stratified gap is not statistically significant, suggesting that the MaxSim advantage is more general than entity-count-specific.
- **Hypothesis rejected**: The bi-encoder matches or exceeds ColBERTv2 on multi-entity queries, which would suggest that MiniLM's training data and architecture generalizes well to multi-entity queries despite the single-vector bottleneck.

The entity classification itself introduces noise: spaCy's `en_core_web_sm` is a lightweight model and may misclassify entities, particularly for less common proper nouns. This could blur the single/multi-entity boundary and attenuate any real performance difference.

### 6.3 Failure Mode Analysis

[To be filled after benchmark run]

Qualitative analysis of queries where the bi-encoder succeeds but ColBERTv2 fails (and vice versa) is planned. Expected patterns:

- **Bi-encoder advantage cases**: Short, unambiguous queries with a single well-represented entity, where MiniLM's compressed representation is sufficient and PLAID's approximation introduces more noise than benefit.
- **ColBERTv2 advantage cases**: Long, multi-constraint queries or queries where a specific rare entity must be jointly satisfied with another condition.

---

## 7. Profiling Summary

### 7.1 Indexing Time and Resource Usage

| Metric | Bi-encoder (MiniLM + FAISS) | ColBERTv2 (PLAID) |
|---|---|---|
| Index build time | [TBF] | [TBF] |
| Index size on disk | [TBF] | [TBF] |
| Peak VRAM during indexing | [TBF] | [TBF] |
| Peak RAM during indexing | [TBF] | [TBF] |

### 7.2 Query Latency

| Metric | Bi-encoder (MiniLM + FAISS) | ColBERTv2 (PLAID) |
|---|---|---|
| Mean latency per query | [TBF] | [TBF] |
| Median latency per query | [TBF] | [TBF] |
| p95 latency per query | [TBF] | [TBF] |
| Total time for ~1000 queries | [TBF] | [TBF] |

*[To be filled after benchmark run]*

![Latency comparison chart](../results/charts/latency_comparison.png)

![Index size comparison chart](../results/charts/index_size_comparison.png)

### 7.3 Interpretation

[To be filled after benchmark run]

The profiling data will quantify the compute-recall tradeoff. PLAID indexing uses centroid-based approximation to make ColBERTv2 practical at corpus scale, but it still stores per-token embeddings and requires more storage than a single-vector FAISS index. The latency comparison will determine whether the recall gain (if any) justifies the additional overhead in the context of a real-time RAG pipeline.

---

## 8. Discussion

### 8.1 Implications for Adaptive Chunking

This benchmark uses uniform paragraph-level chunking for both models. The results will establish a retrieval quality ceiling under the current chunking scheme and motivate the adaptive chunking experiments that follow.

If ColBERTv2 shows meaningful recall improvements over the bi-encoder, this suggests that retrieval quality is sensitive to the granularity of information alignment between query tokens and passage tokens. Adaptive chunking, which adjusts chunk boundaries based on semantic density, entity distribution, or query-specific signals, could further amplify this alignment advantage.

Conversely, if both models plateau at similar Recall@20, the primary bottleneck may be at the chunking level: even a powerful retrieval model cannot retrieve a passage that does not exist as a chunk. This would prioritize chunking strategy research over model selection.

### 8.2 Limitations of the Reduced Corpus

The benchmark uses a corpus of approximately 10,000 Wikipedia pages rather than the full KILT Wikipedia snapshot (~5.9M pages). This has several consequences:

- **Reduced competition**: With fewer distractors, retrieval is an easier task for both models. Recall numbers will be inflated compared to a realistic production deployment. The relative ranking of models should be directionally correct, but absolute numbers should not be extrapolated.
- **Coverage**: A small fraction of NQ dev queries may reference gold passages that were not included in the reduced corpus (i.e., the gold page is a distractor-excluded page). These queries are unanswerable by construction and are excluded from evaluation, but this introduces a selection bias favoring queries with more prominent Wikipedia pages.
- **Distractor quality**: Random sampling of distractor pages may produce an atypically easy or hard distractor set depending on the random seed. Future work should consider stratified distractor sampling or multiple random seeds for robustness.

### 8.3 Entity Classification Quality

The spaCy `en_core_web_sm` pipeline is a small, fast model suitable for development-time NER but may not be accurate enough for rigorous entity-stratified analysis. Misclassified entities shift queries between the single-entity and multi-entity groups, attenuating any real performance difference. A more robust analysis could use a larger spaCy model (`en_core_web_trf`) or cross-validate entity counts with a second NER system.

### 8.4 Generalization Beyond NaturalQuestions

NaturalQuestions originates from Google Search logs, which skews toward entity-lookup and factoid queries. The relative performance of ColBERTv2 vs. bi-encoders may differ on other query distributions (e.g., abstractive questions, multi-hop questions, or domain-specific technical queries). The broader ECE1508 project should validate findings on at least one additional dataset before drawing architecture conclusions.

---

## 9. Conclusion

This sub-experiment benchmarks ColBERTv2 late interaction retrieval against a MiniLM bi-encoder baseline on the KILT NaturalQuestions development set, using a reduced Wikipedia corpus of approximately 10,000 pages with paragraph-level chunking.

**Key findings** [to be finalized after benchmark run]:

- [TBF] ColBERTv2 achieved a Recall@k improvement of [X] percentage points over the bi-encoder at k=10 on all queries.
- [TBF] The recall gap was [larger / smaller / similar] for multi-entity queries compared to single-entity queries.
- [TBF] ColBERTv2 indexing required [X]x more storage and [X]x more query latency than the FAISS bi-encoder.
- [TBF] The latency-recall tradeoff suggests ColBERTv2 is best deployed as a [first-stage retriever / reranker] in this corpus size regime.

**Next steps toward adaptive chunking**:

1. Use the recall numbers from this benchmark as the fixed-chunking baseline to compare against adaptive chunking variants.
2. Analyze the queries where both models fail: these are likely cases where the relevant information spans multiple chunks or is split at a paragraph boundary — a strong signal for adaptive chunking.
3. Implement sentence-level and sliding-window chunking and re-run the same benchmark to disentangle chunking effects from model architecture effects.
4. Explore whether per-query chunk size selection (based on query entity count or query length) can close the gap between single-entity and multi-entity recall without changing the retrieval model.
5. Evaluate the full pipeline (adaptive chunking + best retrieval model) on a held-out test set and a second domain to assess generalization.

---

*Report template generated April 2026. Numerical results and hardware specifications to be filled in after benchmark execution.*
