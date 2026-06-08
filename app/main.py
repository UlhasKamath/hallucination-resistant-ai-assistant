import streamlit as st
import time
import hashlib

from app.agent_builder import build_agent
from app.agents.router import route_and_run
from app.guardrails.safety import check_guardrails
from app.ingestion.file_ingest import ingest_file
from app.retrieval.dense import get_chroma_stats
from app.retrieval.sparse import get_sparse_chunk_count
from app.storage.redis_cache import (
    get_cache_stats,
    redis_client,
)
from app.logging.logger import logger
from app.logging.evaluator import evaluate_response
from app.utils.filename import normalize_filename

# Page config
st.set_page_config(
    page_title="Multi-Agent RAG Chatbot",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Session state
for key, default in [
    ("messages", []),
    ("cache_hits", 0),
    ("cache_misses", 0),
    ("query_times", []),
    ("last_cache_hit", False),
    ("last_agent", ""),
    ("uploaded_file_context", {}),   # {filename: text_content}
    ("processed_uploads", {}),       # {file_hash: filename}
]:
    if key not in st.session_state:
        st.session_state[key] = default

# Agent (cached resource)
@st.cache_resource
def get_agent():
    return build_agent()

agent = get_agent()

# System prompt
SYSTEM_PROMPT = {
    "role": "system",
    "content": (
        "You are a retrieval-augmented AI assistant with access to multiple specialised agents.\n\n"
        "CRITICAL RULES:\n"
        "- Never reveal tool calls or internal routing\n"
        "- Never say 'I will search' or 'Using the function'\n"
        "- Answer naturally and directly\n\n"
        "For factual questions: use retrieved context silently.\n"
        "For live/current events: use fresh web data.\n"
        "For code questions: write clean, commented code.\n"
        "For SQL questions: write well-structured SQL.\n"
        "If information is insufficient, say: 'I could not find enough reliable information.'"
    )
}

# Sidebar
with st.sidebar:

    st.title("Multi-Agent RAG")
    st.caption("RAG · Web Search · General Knowledge · Coding · SQL")

    # Agent legend
    with st.expander("🤖 Agent Priority", expanded=True):
        st.markdown(
            "1. **🌐 Web Search** — live / recent info\n"
            "2. **🧠 General Knowledge** — factual / conceptual Q&A\n"
            "3. **📚 RAG Agent** — indexed documents (default)\n"
            "4. **💻 Coding Agent** — code & debugging\n"
            "5. **🗄️ SQL Agent** — database queries\n"
            "6. **📄 File Context** — fallback when RAG finds nothing\n"
            
        )

    st.divider()

    # FILE UPLOAD
    st.subheader("Upload Documents")
    st.caption("Supported: PDF, TXT, MD, DOCX")

    uploaded_files = st.file_uploader(
        "Upload files to index",
        type=["pdf", "txt", "md", "docx"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    # ───────────────────────────────────────────────────────────────────────────
    # CURRENTLY ACTIVE UPLOADS
    # ───────────────────────────────────────────────────────────────────────────
    current_file_hashes = set()
    current_file_names  = set()

    if uploaded_files:

        for uploaded_file in uploaded_files:

            file_bytes = uploaded_file.getvalue()

            # Stable content hash
            file_hash = hashlib.sha256(file_bytes).hexdigest()

            current_file_hashes.add(file_hash)
            current_file_names.add(uploaded_file.name)

            # Skip already-processed uploads during reruns
            if file_hash in st.session_state.processed_uploads:

                # Ensure session context still exists
                if uploaded_file.name not in st.session_state.uploaded_file_context:

                    from app.ingestion.file_ingest import _extract_text

                    try:
                        extracted_text = _extract_text(
                            uploaded_file.name,
                            file_bytes
                        )

                        st.session_state.uploaded_file_context[
                            uploaded_file.name
                        ] = extracted_text

                    except Exception:
                        pass

                continue

            # NEW FILE DETECTED
            with st.spinner(f"Indexing {uploaded_file.name}..."):

                result = ingest_file(
                    uploaded_file.name,
                    file_bytes
                )

            # Mark processed
            st.session_state.processed_uploads[file_hash] = uploaded_file.name

            if result["status"] == "success":

                st.session_state.uploaded_file_context[
                    uploaded_file.name
                ] = result["text"]

                st.success(
                    f"{uploaded_file.name}\n\n"
                    f"{result['chunk_count']} chunks indexed"
                )

            elif result["status"] == "duplicate":

                from app.ingestion.file_ingest import _extract_text

                try:
                    extracted_text = _extract_text(
                        uploaded_file.name,
                        file_bytes
                    )

                    st.session_state.uploaded_file_context[
                        uploaded_file.name
                    ] = extracted_text

                except Exception:

                    st.session_state.uploaded_file_context[
                        uploaded_file.name
                    ] = file_bytes.decode(
                        "utf-8",
                        errors="ignore"
                    )

                st.info(f"{uploaded_file.name} already indexed")

            else:
                st.error(result["message"])

    # ───────────────────────────────────────────────────────────────────────────
    # REMOVE DELETED FILES FROM SESSION CONTEXT
    # ───────────────────────────────────────────────────────────────────────────
    stale_files = [
        k for k in st.session_state.uploaded_file_context
        if k not in current_file_names
    ]

    for k in stale_files:

        del st.session_state.uploaded_file_context[k]

        logger.info(f"FILE CONTEXT REMOVED -> {k}")

    # Remove stale processed upload hashes
    stale_hashes = [
        h for h, fname in st.session_state.processed_uploads.items()
        if fname not in current_file_names
    ]

    for h in stale_hashes:
        del st.session_state.processed_uploads[h]

    # Flush classifier cache after removals
    if stale_files:

        try:
            clf_keys = redis_client.keys("cache:clf:*")

            if clf_keys:
                redis_client.delete(*clf_keys)

                logger.info(
                    f"CLASSIFIER CACHE FLUSHED -> "
                    f"{len(clf_keys)} entries cleared after file removal"
                )

        except Exception:
            pass

    st.divider()

    # KNOWLEDGE BASE
    st.subheader("Knowledge Base")

    if st.button("Refresh", use_container_width=True):
        st.cache_data.clear()

    @st.cache_data(ttl=10)
    def fetch_chroma_stats():
        return get_chroma_stats()

    chroma = fetch_chroma_stats()

    if "error" in chroma:
        st.error(f"ChromaDB error: {chroma['error']}")
    else:
        col1, col2 = st.columns(2)
        col1.metric("Chunks", chroma["total_chunks"])
        col2.metric("Files", chroma["unique_sources"])

        if chroma["sources"]:
            session_files = set(
                st.session_state.get(
                    "uploaded_file_context",
                    {}
                ).keys()
            )

            st.caption("Indexed files:")

            for source in sorted(chroma["sources"]):

                is_web = not source.startswith("file://")

                icon = "🌐" if is_web else "📄"

                stem, _ = normalize_filename(source)

                label = stem if not is_web else source

                active = any(
                    normalize_filename(f)[0] == stem
                    for f in session_files
                )

                suffix = " 🟢" if active else ""

                st.caption(f"{icon} {label}{suffix}")

        sparse_count = get_sparse_chunk_count()

        st.caption(
            f"BM25: {sparse_count} chunks  |  "
            f"🟢 = active this session"
        )

    st.divider()

    # REDIS CACHE STATS
    st.subheader("Redis Cache")

    try:
        redis_client.ping()
        st.success("Connected", icon="🟢")
    except Exception:
        st.error("Redis not reachable", icon="🔴")

    cache_stats = get_cache_stats()

    cached_queries_placeholder = st.empty()

    cached_queries_placeholder.metric(
        "Cached Queries",
        cache_stats["cached_queries"]
    )

    col1, col2 = st.columns(2)

    hits_placeholder = col1.empty()
    miss_placeholder = col2.empty()

    hits_placeholder.metric(
        "Hits",
        st.session_state.cache_hits
    )

    miss_placeholder.metric(
        "Misses",
        st.session_state.cache_misses
    )

    hitrate_placeholder = st.empty()

    total = (
        st.session_state.cache_hits
        + st.session_state.cache_misses
    )

    if total > 0:

        hit_rate = round(
            st.session_state.cache_hits / total * 100,
            1
        )

        hitrate_placeholder.progress(
            hit_rate / 100,
            text=f"Hit rate: {hit_rate}%"
        )

    avg_time_placeholder = st.empty()

    if st.session_state.query_times:

        avg_time = round(
            sum(st.session_state.query_times)
            / len(st.session_state.query_times),
            2
        )

        avg_time_placeholder.caption(
            f"Avg response time: {avg_time}s"
        )

    if cache_stats["keys"]:

        with st.expander("Cached Queries", expanded=False):

            for k in cache_stats["keys"]:

                st.caption(
                    f"• {k[:60]}"
                    f"{'...' if len(k) > 60 else ''}"
                )

    if st.button(
        "Clear Cache",
        use_container_width=True,
        type="secondary"
    ):

        keys = redis_client.keys("cache:*")

        if keys:
            redis_client.delete(*keys)

        st.session_state.cache_hits = 0
        st.session_state.cache_misses = 0
        st.session_state.query_times = []

        cached_queries_placeholder.metric(
            "Cached Queries",
            0
        )

        hits_placeholder.metric("Hits", 0)
        miss_placeholder.metric("Misses", 0)

        st.success("Cache cleared")

        st.rerun()

    st.divider()

    if st.button(
        "Clear Chat History",
        use_container_width=True,
        type="secondary"
    ):

        st.session_state.messages = []

        st.rerun()

# Main chat
st.title("AI Search Assistant")

st.caption(
    "Multi-Agent: RAG · Web Search · Coding · SQL  |  "
    "Hybrid Retrieval · Query Rewriting · Self-Critiquing · Redis Cache"
)

# Last agent badge
# if st.session_state.last_agent:
#     st.info(
#         f"Last response by: "
#         f"**{st.session_state.last_agent}**",
#         icon="🤖"
#     )

if st.session_state.last_cache_hit:
    st.info(
        "Last response served from Redis cache",
        icon="⚡"
    )

# Chat history
for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):

        st.write(msg["content"])

        if msg.get("meta"):
            st.caption(msg["meta"])

# User input
user_input = st.chat_input(
    "Ask anything - documents, web, code, SQL, math..."
)

# Main query flow
if user_input:

    start_time = time.time()

    logger.info(f"USER QUERY -> {user_input}")

    with st.chat_message("user"):
        st.write(user_input)

    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    # Guardrails
    allowed, block_msg = check_guardrails(user_input)

    if not allowed:

        logger.warning(
            f"GUARDRAIL BLOCKED -> {user_input}"
        )

        with st.chat_message("assistant"):
            st.warning(block_msg)

        st.session_state.messages.append({
            "role": "assistant",
            "content": block_msg
        })

    else:

        try:

            with st.chat_message("assistant"):

                thinking = st.empty()

                thinking.markdown("Thinking...")

                answer, agent_label, retrieved_docs = route_and_run(
                    query=user_input,
                    rag_agent=agent,
                    session_messages=st.session_state.messages,
                    system_prompt=SYSTEM_PROMPT,
                    file_ctx=st.session_state.get(
                        "uploaded_file_context",
                        {}
                    ),
                )

                latency = round(
                    time.time() - start_time,
                    2
                )

                cache_hit = agent_label.startswith(
                    "⚡ Cache"
                )

                # Update cache metrics
                if cache_hit:
                    st.session_state.cache_hits += 1
                    st.session_state.last_cache_hit = True
                else:
                    st.session_state.cache_misses += 1
                    st.session_state.last_cache_hit = False

                st.session_state.last_agent = agent_label

                st.session_state.query_times.append(latency)

                # Live sidebar metric update
                cache_stats = get_cache_stats()

                cached_queries_placeholder.metric(
                    "Cached Queries",
                    cache_stats["cached_queries"]
                )

                hits_placeholder.metric(
                    "Hits",
                    st.session_state.cache_hits
                )

                miss_placeholder.metric(
                    "Misses",
                    st.session_state.cache_misses
                )

                total = (
                    st.session_state.cache_hits
                    + st.session_state.cache_misses
                )

                if total > 0:

                    hit_rate = round(
                        st.session_state.cache_hits
                        / total * 100,
                        1
                    )

                    hitrate_placeholder.progress(
                        hit_rate / 100,
                        text=f"Hit rate: {hit_rate}%"
                    )

                avg_time = round(
                    sum(st.session_state.query_times)
                    / len(st.session_state.query_times),
                    2
                )

                avg_time_placeholder.caption(
                    f"Avg response time: {avg_time}s"
                )

                logger.info(
                    f"ASSISTANT RESPONSE -> {answer}"
                )

                # Render response
                thinking.markdown(answer)

                meta_text = (
                    f"{'⚡ Cache hit' if cache_hit else agent_label} "
                    f"| {latency}s"
                )

                st.caption(meta_text)

            # Evaluation
            evaluate_response(
                query=user_input,
                response=answer,
                start_time=start_time,
                retrieved_docs=retrieved_docs,
                agent_label=agent_label
            )

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "meta": meta_text
            })

        except Exception as e:

            logger.error(f"PIPELINE ERROR -> {str(e)}")

            error_msg = (
                "Something went wrong while processing your request."
            )

            with st.chat_message("assistant"):
                st.error(error_msg)

            st.session_state.messages.append({
                "role": "assistant",
                "content": error_msg
            })