"""
🧬 Medical Literature RAG — Streamlit Demo

Run:
    streamlit run app.py

This is the portfolio demo interface. It shows:
    1. Natural language search over 50K+ PubMed articles
    2. Claude-generated answers with [PMID:xxxxx] citations
    3. Metadata filtering (journal, date range, domain)
    4. Source documents with relevance scores
    5. System metrics (retrieval time, generation time)
"""

import time

import streamlit as st

# ── Page config (must be first Streamlit call) ──────────────────────
st.set_page_config(
    page_title="Medical Literature RAG",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Custom CSS ──────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Clean typography */
    .main .block-container { max-width: 1100px; padding-top: 2rem; }

    /* Source cards */
    .source-card {
        background: #f8f9fa;
        border-left: 3px solid #4CAF50;
        padding: 12px 16px;
        margin: 8px 0;
        border-radius: 0 6px 6px 0;
        font-size: 14px;
    }
    .source-card-uncited {
        background: #f8f9fa;
        border-left: 3px solid #ccc;
        padding: 12px 16px;
        margin: 8px 0;
        border-radius: 0 6px 6px 0;
        font-size: 14px;
    }

    /* Metric badges */
    .metric-badge {
        display: inline-block;
        background: #e8f5e9;
        color: #2e7d32;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 13px;
        font-weight: 500;
        margin-right: 8px;
    }

    /* PMID links */
    .pmid-link { color: #1565c0; text-decoration: none; font-weight: 500; }
    .pmid-link:hover { text-decoration: underline; }

    /* Hide Streamlit branding */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Initialization ──────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedder():
    from src.processing.embedder import BioBERTEmbedder
    embedder = BioBERTEmbedder()
    _ = embedder.model  # Force load
    return embedder


@st.cache_resource(show_spinner="Loading vector stores...")
def load_stores():
    from src.retrieval.chroma_store import ChromaStore
    from src.retrieval.faiss_store import FAISSStore
    from src.retrieval.hybrid_retriever import HybridRetriever
    from src.retrieval.bm25_store import BM25Store

    chroma = ChromaStore()
    faiss_store = FAISSStore()
    faiss_store.load()
    bm25 = BM25Store.from_faiss_store(faiss_store)
    retriever = HybridRetriever(chroma, faiss_store, bm25)
    return chroma, faiss_store, retriever


@st.cache_resource(show_spinner="Initializing RAG chain...")
def load_chain(_retriever, _embedder):
    from src.generation.rag_chain import MedicalRAGChain
    return MedicalRAGChain(_retriever, _embedder)


def format_pmid_link(pmid: str) -> str:
    """Create a clickable PubMed link."""
    url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
    return f'<a href="{url}" target="_blank" class="pmid-link">PMID:{pmid}</a>'


# ── Sidebar ─────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🧬 Medical RAG")
    st.markdown("Search 50K+ PubMed articles with AI-powered Q&A")

    st.divider()

    # Search settings
    st.markdown("### Settings")
    top_k = st.slider("Documents to retrieve", 3, 20, 8)
    use_ensemble = st.toggle("Hybrid retrieval (embeddings + BM25, RRF)", value=False,
                              help="Merge results from both FAISS and Chroma using Reciprocal Rank Fusion")

    st.divider()

    # Metadata filters
    st.markdown("### Filters")
    journal_filter = st.text_input("Journal name", placeholder="e.g. Nature Medicine",
                                    help="Filter results to a specific journal")

    st.divider()

    # System info
    st.markdown("### System")
    try:
        _, _, _ = load_stores()
        st.success("Stores loaded")
    except Exception:
        st.error("Stores not loaded — run Steps 2-4 first")

    st.markdown("""
    **Tech Stack**
    - BioBERT embeddings
    - Chroma + FAISS stores
    - Claude (Anthropic)
    - LangChain orchestration
    """)

    st.divider()

    # Example queries
    st.markdown("### Try these")
    examples = [
        "What are the mechanisms of CRISPR off-target effects?",
        "How does immunotherapy work for lung cancer?",
        "What is the role of gut microbiome in depression?",
        "Latest advances in mRNA vaccine technology",
        "How do statins affect cardiovascular outcomes?",
    ]
    for ex in examples:
        if st.button(ex, key=f"ex_{hash(ex)}", use_container_width=True):
            st.session_state["query_input"] = ex


# ── Main Area ───────────────────────────────────────────────────────

st.markdown("# 🧬 Medical Literature Q&A")
st.markdown("Ask a biomedical research question and get an evidence-based answer with citations.")

# Query input
query = st.text_input(
    "Your question",
    value=st.session_state.get("query_input", ""),
    placeholder="e.g. What are the latest treatments for glioblastoma?",
    key="main_query",
    label_visibility="collapsed",
)

col_search, col_summarize = st.columns([1, 1])
search_clicked = col_search.button("🔍 Search & Answer", type="primary", use_container_width=True)
summarize_clicked = col_summarize.button("📋 Summarize Topic", use_container_width=True)

if (search_clicked or summarize_clicked) and query:
    try:
        embedder = load_embedder()
        chroma, faiss_store, retriever = load_stores()
        chain = load_chain(retriever, embedder)
    except Exception as e:
        st.error(f"Failed to initialize: {e}")
        st.info("Make sure you've completed Steps 2-4 (ingestion → embedding → indexing)")
        st.stop()

    # Build metadata filter
    where = None
    if journal_filter.strip():
        where = {"journal": journal_filter.strip()}

    if search_clicked:
        # ── Streaming answer ────────────────────────────────────────
        with st.status("Searching medical literature...", expanded=True) as status:
            st.write("Embedding query...")
            retrieval_start = time.time()
            query_vec = embedder.embed_single(query)

            st.write(f"Retrieving top {top_k} documents...")
            if use_ensemble:
                docs = retriever.retrieve_ensemble(
                    query_vec, top_k=top_k, where=where, query_text=query
                )
            else:
                docs = retriever.retrieve(query_vec, top_k=top_k, where=where)
            retrieval_ms = (time.time() - retrieval_start) * 1000

            st.write(f"Found {len(docs)} relevant passages ({retrieval_ms:.0f}ms)")
            status.update(label=f"Retrieved {len(docs)} sources in {retrieval_ms:.0f}ms", state="complete")

        # Metrics bar
        col1, col2, col3 = st.columns(3)
        col1.metric("Retrieved", f"{len(docs)} docs")
        col2.metric("Retrieval", f"{retrieval_ms:.0f} ms")
        col3.metric("Mode", "Hybrid" if use_ensemble else "Dense (FAISS)")

        st.divider()

        # Stream the answer
        st.markdown("### Answer")
        from src.generation.prompts import format_context, QUERY_TEMPLATE, SYSTEM_PROMPT

        context = format_context(docs)
        user_message = QUERY_TEMPLATE.format(context=context, question=query)

        gen_start = time.time()
        answer_placeholder = st.empty()
        full_answer = ""

        try:
            from config.config import settings
            import anthropic

            client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
            with client.messages.stream(
                model=settings.claude_model,
                max_tokens=settings.claude_max_tokens,
                temperature=settings.claude_temperature,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}],
            ) as stream:
                for text in stream.text_stream:
                    full_answer += text
                    answer_placeholder.markdown(full_answer + "▌")

            answer_placeholder.markdown(full_answer)
            gen_ms = (time.time() - gen_start) * 1000

            st.caption(f"Generated in {gen_ms:.0f}ms using {settings.claude_model}")

        except Exception as e:
            st.error(f"Generation failed: {e}")
            st.info("Check your ANTHROPIC_API_KEY in .env")
            full_answer = ""

        # ── Source documents ────────────────────────────────────────
        st.divider()
        st.markdown("### Sources")

        # Extract cited PMIDs
        import re
        cited_pmids = set(re.findall(r'PMID:\s*(\d+)', full_answer))

        cited_tab, all_tab = st.tabs([f"Cited ({len(cited_pmids)})", f"All Retrieved ({len(docs)})"])

        with cited_tab:
            if not cited_pmids:
                st.info("No citations found in the answer.")
            for doc in docs:
                pmid = doc.get("metadata", {}).get("pmid", "")
                if pmid in cited_pmids:
                    title = doc.get("metadata", {}).get("title", "Untitled")
                    journal = doc.get("metadata", {}).get("journal", "")
                    date = doc.get("metadata", {}).get("pub_date", "")
                    score = doc.get("score", 0)
                    text_preview = doc.get("text", "")[:300]

                    st.markdown(f"""
                    <div class="source-card">
                        <strong>{format_pmid_link(pmid)}</strong> — {title}<br>
                        <em>{journal}</em> · {date} · Relevance: {score:.3f}<br>
                        <small style="color: #666;">{text_preview}...</small>
                    </div>
                    """, unsafe_allow_html=True)

        with all_tab:
            for doc in docs:
                pmid = doc.get("metadata", {}).get("pmid", "")
                title = doc.get("metadata", {}).get("title", "Untitled")
                journal = doc.get("metadata", {}).get("journal", "")
                score = doc.get("score", 0)
                is_cited = pmid in cited_pmids
                card_class = "source-card" if is_cited else "source-card-uncited"
                badge = " ✓ cited" if is_cited else ""

                st.markdown(f"""
                <div class="{card_class}">
                    <strong>{format_pmid_link(pmid)}</strong> — {title}{badge}<br>
                    <em>{journal}</em> · Relevance: {score:.3f}
                </div>
                """, unsafe_allow_html=True)

    elif summarize_clicked:
        # ── Topic summary mode ──────────────────────────────────────
        with st.spinner("Generating literature summary..."):
            try:
                result = chain.summarize_topic(query, top_k=top_k, where=where)
                st.markdown("### Literature Summary")
                st.markdown(result["summary"])
                st.caption(
                    f"Cited {len(result['cited_pmids'])} papers from "
                    f"{result['total_sources']} retrieved"
                )
            except Exception as e:
                st.error(f"Summary failed: {e}")

elif search_clicked or summarize_clicked:
    st.warning("Please enter a question first.")


# ── Footer ──────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<div style='text-align: center; color: #999; font-size: 13px;'>"
    "Medical Literature RAG System — Portfolio Project · "
    "BioBERT · Chroma · FAISS · LangChain · Claude"
    "</div>",
    unsafe_allow_html=True,
)
