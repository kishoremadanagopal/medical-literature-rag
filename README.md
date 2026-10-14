# 🧬 Medical Literature Retrieval System (RAG + LLMs)

End-to-end retrieval-augmented generation (RAG) pipeline for biomedical literature. Ingests PubMed abstracts (configurable, default 50K), and answers natural-language questions with cited sources using **biomedical sentence embeddings** (PubMedBERT), **hybrid retrieval** (dense vectors in Chroma + FAISS, fused with **BM25** keyword search), **LangChain** orchestration, and **Claude** for generation.

> **Status:** Pipeline complete, 112 tests passing. Evaluation harness compares BM25, dense and hybrid retrieval; corpus-scale results are produced by `scripts/run_evaluation.py` and saved to `results/eval_results.json`.

---

## 🎯 Evaluation

Run `python scripts/run_evaluation.py` after building the indices. It writes every number below to `results/eval_results.json`, together with the corpus size and embedding model used.

| What | How it's measured |
|---|---|
| **Retrieval quality** — NDCG@10, Recall@10, MRR | Known-item search on 200 sampled articles: each article's title is the query, and that article is the correct answer. Compared across **BM25** (keyword baseline), **dense** (FAISS) and **hybrid** (dense + BM25 via Reciprocal Rank Fusion). |
| **Domain consistency** | For each query article, does the nearest *other* article share its broad domain (oncology, cardiology, …, derived from MeSH headings)? Reported with a majority-class baseline for context. A sanity check on the embedding space, not a trained classifier. |
| **Vector search latency** | FAISS (IVF) vs Chroma (HNSW) on the same real query embeddings, mean and p95. Excludes query embedding time, which dominates end-to-end latency. |

**Limitations.** There is no human-labelled query set. Title-as-query evaluation is cheap and objective, but titles share exact words with their own abstracts, so it favours keyword search (BM25). A stronger evaluation would use a labelled biomedical benchmark such as BEIR's TREC-COVID, NFCorpus or SciFact.

---

## 🏗️ Architecture

```
┌─────────────────┐   ┌──────────────────┐   ┌─────────────────────┐
│  PubMed (NCBI)  │──▶│  Ingestion       │──▶│  Raw JSON           │
│  Entrez API     │   │  (Biopython)     │   │  data/raw/          │
└─────────────────┘   └──────────────────┘   └──────────┬──────────┘
                                                        │
                                              ┌─────────▼──────────┐
                                              │  Chunking +        │
                                              │  Cleaning          │
                                              └─────────┬──────────┘
                                                        │
                                              ┌─────────▼──────────┐
                                              │  PubMedBERT        │
                                              │  sentence embedder │
                                              └─────────┬──────────┘
                                                        │
                          ┌─────────────────────────────┼─────────────────────────────┐
                          │                                                            │
                ┌─────────▼──────────┐                                       ┌────────▼─────────┐
                │  Chroma            │                                       │  FAISS (IVF)     │
                │  (persistent +     │                                       │  (fast ANN       │
                │  metadata filter)  │                                       │  similarity)     │
                └─────────┬──────────┘                                       └────────┬─────────┘
                          │                                                            │
                          └─────────────────────────────┬──────────────────────────────┘
                                                        │
                                              ┌─────────▼──────────┐
                                              │  Hybrid Retriever  │
                                              │  (LangChain)       │
                                              └─────────┬──────────┘
                                                        │
                                              ┌─────────▼──────────┐
                                              │  Claude (Anthropic)│
                                              │  Answer + Citations│
                                              └─────────┬──────────┘
                                                        │
                                              ┌─────────▼──────────┐
                                              │  Streamlit Demo    │
                                              └────────────────────┘
```

**Why both Chroma and FAISS?**
Chroma gives persistence and rich metadata filtering (journal, year, MeSH terms). FAISS gives very fast approximate nearest-neighbour search. Queries with a filter go to Chroma; plain similarity search goes to FAISS. Both hold the same vectors, so they are a routing choice, not two different opinions.

**Why add BM25?**
Dense embeddings match meaning ("heart attack" ≈ "myocardial infarction") but can miss exact tokens such as gene names (BRCA1), drug names and abbreviations. BM25 keyword search catches those. Hybrid mode fuses the dense and BM25 rankings with Reciprocal Rank Fusion (RRF), which combines ranks rather than raw scores, so the two very different score scales never need normalising. BM25 is also the standard baseline any embedding retriever should beat.

**Why PubMedBERT (S-PubMedBert-MS-MARCO) rather than raw BioBERT?**
BioBERT knows biomedical vocabulary but was trained to predict masked words, not to compare sentences, so its pooled vectors rank search results poorly. `pritamdeka/S-PubMedBert-MS-MARCO` starts from PubMedBERT and is fine-tuned for search on MS MARCO. Any sentence-transformers model can be swapped in via `EMBEDDING_MODEL` and compared with the eval script. (Note: that model's licence is CC BY-NC 2.0 — non-commercial.)

**Ingesting more than 10,000 articles.**
PubMed's search API only returns the first 10,000 IDs of any query, so the fetcher splits the query by publication year (newest first) and takes up to 9,999 IDs per year until the target is reached.

---

## 📁 Project Structure

```
medical-rag/
├── app.py                 # Streamlit demo UI (Step 7)
├── config/                # Pydantic-validated settings, logging
│   ├── config.py
│   └── logging_config.py
├── src/
│   ├── ingestion/         # Step 2 — PubMed fetcher + XML parser
│   ├── processing/        # Step 3 — chunker + sentence embedder
│   ├── retrieval/         # Step 4 — Chroma + FAISS + BM25 + hybrid retriever
│   ├── generation/        # Step 5 — prompts + RAG chain with Claude
│   └── evaluation/        # Step 6 — metrics + eval harness
├── scripts/               # CLI entry points for each step
├── data/                  # (gitignored) raw, processed, vector stores
├── notebooks/             # Exploration + demos
├── tests/                 # 112 pytest tests
├── results/               # eval_results.json written by run_evaluation.py
├── docs/                  # Architecture notes, design decisions
├── requirements.txt
├── .env.example
└── README.md
```

---

## 🚀 Quickstart

### Prerequisites
- Python 3.10, 3.11 or 3.12 (`numpy<2.0` has no wheels for 3.13+)
- A few GB free disk for 50K embedded abstracts + indices, plus ~0.5 GB for the embedding model
- NCBI contact email (free, required by NCBI's terms of use)
- Anthropic API key ([console.anthropic.com](https://console.anthropic.com/)) — only needed to generate answers; ingestion, indexing and evaluation run without it

### Setup

```bash
# 1. Clone and enter
git clone <your-repo-url> medical-rag && cd medical-rag

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env — add NCBI_EMAIL (and ANTHROPIC_API_KEY to generate answers)

# 5. Verify setup
python scripts/verify_setup.py
```

If `verify_setup.py` prints all green checks, you're ready for the pipeline.

### Full Pipeline

```bash
# Step 2: Ingest 50K PubMed articles (~2-4 hours)
python scripts/run_ingestion.py
python scripts/run_ingestion.py --resume    # if interrupted

# Step 3: Chunk articles + embed (~30-60 min on CPU for 50K)
python scripts/run_processing.py

# Step 4: Build vector store indices (~5 min)
python scripts/run_indexing.py

# Step 5: Query the system
python scripts/run_query.py "What are CRISPR off-target effects?"
python scripts/run_query.py "BRCA1 breast cancer risk" --ensemble   # hybrid
python scripts/run_query.py --interactive

# Step 6: Run evaluation suite (writes results/eval_results.json)
python scripts/run_evaluation.py

# Step 7: Launch the demo UI
streamlit run app.py
```

### Quick Test (skip the 4-hour wait)

```bash
# Ingest just 1K articles (~5 min), then run the full pipeline
python scripts/run_ingestion.py --count 1000
python scripts/run_processing.py
python scripts/run_indexing.py
python scripts/run_evaluation.py --n-queries 100
streamlit run app.py
```

---

## 🗺️ Roadmap

- [x] **Step 1** — Project scaffolding, config, logging
- [x] **Step 2** — PubMed ingestion (50K articles via Entrez)
- [x] **Step 3** — Chunking + biomedical embedding pipeline
- [x] **Step 4** — Chroma (persistent) + FAISS (fast ANN) vector stores + BM25 keyword index
- [x] **Step 5** — LangChain RAG chain with Claude + citation tracking
- [x] **Step 6** — Evaluation harness (BM25 vs dense vs hybrid, domain consistency, latency)
- [ ] **Next** — Evaluate on a labelled benchmark (BEIR TREC-COVID / NFCorpus), add a cross-encoder re-ranker, and score answer faithfulness with RAGAs
- [x] **Step 7** — Streamlit demo UI

---

## 🛠️ Tech Stack

**Orchestration:** LangChain
**Embeddings:** `pritamdeka/S-PubMedBert-MS-MARCO` via sentence-transformers (BioBERT optional)
**Vector Stores:** Chroma (persistent) + FAISS (ANN)
**Keyword search:** BM25 (`rank-bm25`), fused with dense results via RRF
**LLM:** Claude (Anthropic API)
**Data Source:** PubMed via Biopython/Entrez
**Demo:** Streamlit
**Evaluation:** custom retrieval metrics (NDCG, Recall, MRR) + scikit-learn
**Language:** Python 3.10+

---

## 🧪 Testing

```bash
# Run the full test suite (112 tests)
pytest tests/ -v

# Run tests for a specific module
pytest tests/test_parser.py -v       # Step 2: XML parsing
pytest tests/test_chunker.py -v      # Step 3: chunking logic
pytest tests/test_retrieval.py -v    # Step 4: vector stores
pytest tests/test_rag_chain.py -v    # Step 5: RAG chain + prompts
pytest tests/test_evaluation.py -v   # Step 6: metrics
pytest tests/test_fixes.py -v        # BM25, hybrid fusion, eval logic, PubMed slicing
```

All tests run without API keys or data — the Claude API is mocked and vector stores use synthetic data.

---

## 📄 License

MIT — see [LICENSE](LICENSE).

## ✍️ Author

Built as a portfolio project to demonstrate production-grade RAG architecture on a domain-specific corpus.
