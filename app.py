"""
Streamlit Frontend for Local Companies Act 2013 Legal AI Assistant.

Features:
- 100% Local Execution (No external paid APIs, no OpenAI/Anthropic keys).
- Conversational chat UI with persistent session memory.
- Expandable source citation drawers displaying Section numbers, Chapter titles,
  PDF pages, and exact retrieved statutory snippets.
- Ollama health check and model selector.
- Ingestion status banner and one-click re-indexing.
"""

import os
import sys
import time
import requests
from pathlib import Path
from typing import List, Dict, Any

import streamlit as st

import config
from rag_chain import LegalRAGPipeline, get_vectorstore
import ingest

# Page configuration
st.set_page_config(
    page_title="Companies Act 2013 - Legal AI Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling for Legal Tech Aesthetics
st.markdown("""
<style>
    /* Main container styling */
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .badge-local {
        background-color: #DEF7EC;
        color: #03543F;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
        margin-left: 10px;
        border: 1px solid #BCF0DA;
    }
    .citation-card {
        background-color: #F8FAFC;
        border-left: 4px solid #2563EB;
        padding: 12px 16px;
        margin-bottom: 12px;
        border-radius: 4px;
        font-size: 0.92rem;
    }
    .citation-meta {
        font-weight: 600;
        color: #1E293B;
        margin-bottom: 6px;
    }
    .citation-badge {
        background-color: #E2E8F0;
        color: #0F172A;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-right: 6px;
    }
    .citation-text {
        color: #334155;
        font-family: monospace;
        font-size: 0.85rem;
        white-space: pre-wrap;
        background-color: #FFFFFF;
        padding: 8px;
        border-radius: 4px;
        border: 1px solid #E2E8F0;
    }
    .sidebar-section {
        margin-top: 1.2rem;
        margin-bottom: 0.6rem;
        font-weight: 700;
        font-size: 0.95rem;
        color: #1E293B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
</style>
""", unsafe_allow_html=True)


def check_ollama_status(base_url: str) -> Dict[str, Any]:
    """Check if the local Ollama server is running and fetch downloaded models."""
    try:
        resp = requests.get(f"{base_url}/api/tags", timeout=2.0)
        if resp.status_code == 200:
            models_data = resp.json().get("models", [])
            model_names = [m.get("name", "") for m in models_data]
            return {"online": True, "models": model_names}
    except Exception:
        pass
    return {"online": False, "models": []}


def get_collection_chunk_count() -> int:
    """Return the number of items stored in ChromaDB."""
    try:
        if config.CHROMA_DIR.exists():
            vs = get_vectorstore()
            return vs._collection.count()
    except Exception:
        pass
    return 0


# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

if "sample_prompt_clicked" not in st.session_state:
    st.session_state.sample_prompt_clicked = None


# --- SIDEBAR CONFIGURATION ---
with st.sidebar:
    st.markdown("### ⚙️ System Configuration")
    
    # Ollama Health Check
    ollama_status = check_ollama_status(config.DEFAULT_OLLAMA_BASE_URL)
    if ollama_status["online"]:
        st.success("🟢 Local Ollama Server: **Connected**")
        available_models = ollama_status["models"]
    else:
        st.error("🔴 Local Ollama Server: **Offline**")
        st.caption("Ollama is not responding at `http://localhost:11434`.")
        available_models = []

    st.markdown("<div class='sidebar-section'>LLM Runner Settings</div>", unsafe_allow_html=True)
    
    # Model selection
    default_models = ["llama3.2", "mistral", "qwen2.5:7b"]
    all_choices = list(dict.fromkeys(default_models + available_models))
    selected_model = st.selectbox(
        "Select Local LLM:",
        options=all_choices,
        index=0 if "llama3.2" in all_choices else 0,
        help="Model running locally via Ollama. Make sure to run 'ollama pull <model>'."
    )

    ollama_url = st.text_input(
        "Ollama Server URL:",
        value=config.DEFAULT_OLLAMA_BASE_URL,
        help="Default is http://localhost:11434"
    )

    temperature = st.slider(
        "Temperature (Strictness):",
        min_value=0.0,
        max_value=0.7,
        value=config.DEFAULT_TEMPERATURE,
        step=0.05,
        help="Lower values (0.0 - 0.1) ensure deterministic legal citations and avoid creative liberties."
    )

    top_k = st.slider(
        "Top-k Chunks Retrieved:",
        min_value=2,
        max_value=8,
        value=config.TOP_K_CHUNKS,
        step=1,
        help="Number of statutory chunks provided to the LLM context."
    )

    st.markdown("<div class='sidebar-section'>Vector Store & Document Status</div>", unsafe_allow_html=True)
    chunk_count = get_collection_chunk_count()
    if chunk_count > 0:
        st.info(f"📚 Indexed Chunks: **{chunk_count:,}** chunks in `./chroma_db`")
    else:
        st.warning("⚠️ Vector store not detected or empty!")

    if st.button("🔄 Re-index Companies Act PDF", use_container_width=True):
        with st.spinner("Processing & embedding Companies Act 2013 PDF..."):
            try:
                ingest.run_ingestion()
                st.success("Vector store rebuilt successfully!")
                time.sleep(1)
                st.rerun()
            except Exception as e:
                st.error(f"Ingestion failed: {e}")

    st.markdown("<div class='sidebar-section'>Session Management</div>", unsafe_allow_html=True)
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    with st.expander("🛠️ Ollama Quick Setup Guide"):
        st.markdown("""
        **1. Install Ollama:**
        - Windows / Mac / Linux: [ollama.com](https://ollama.com)
        
        **2. Start Ollama Server:**
        ```powershell
        ollama serve
        ```
        
        **3. Pull Required Model:**
        ```powershell
        ollama pull llama3.2
        # Or alternative:
        ollama pull mistral
        ollama pull qwen2.5:7b
        ```
        """)


# --- MAIN CHAT INTERFACE ---
col_title, col_badge = st.columns([4, 1])
with col_title:
    st.markdown("""
    <div class="main-header">
        ⚖️ Companies Act 2013 - Legal AI Assistant
        <span class="badge-local">100% Local & Offline</span>
    </div>
    <div class="sub-header">
        Strict statutory retrieval grounded exclusively in the Indian Companies Act, 2013 with verified Section, Chapter & Page citations.
    </div>
    """, unsafe_allow_html=True)

# Sample Questions for Quick Evaluation
st.markdown("**💡 Quick Legal Questions:**")
sample_questions = [
    "What are the CSR expenditure and committee requirements under Section 135?",
    "What is the requirement for Independent Directors and women directors under Section 149?",
    "What constitutes fraud and what is the punishment under Section 447?",
    "What is the definition and formation criteria of a One Person Company (OPC)?"
]

cols = st.columns(len(sample_questions))
for i, q in enumerate(sample_questions):
    if cols[i].button(q, key=f"sample_q_{i}", use_container_width=True):
        st.session_state.sample_prompt_clicked = q


# Display Past Chat Messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Display citations drawer if source documents exist
        if message.get("sources"):
            with st.expander(f"📚 View Legal Citations & Statutory Chunks ({len(message['sources'])})"):
                for idx, src in enumerate(message["sources"], start=1):
                    sec = src.get("section", "N/A")
                    chap = src.get("chapter", "N/A")
                    page = src.get("page", "N/A")
                    content = src.get("content", "")
                    
                    st.markdown(f"""
                    <div class="citation-card">
                        <div class="citation-meta">
                            <span class="citation-badge">Chunk #{idx}</span>
                            <span class="citation-badge">Page {page}</span>
                            <b>{chap}</b> | <b>{sec}</b>
                        </div>
                        <div class="citation-text">{content}</div>
                    </div>
                    """, unsafe_allow_html=True)


# Process User Input
user_query = st.chat_input("Ask any legal question on the Companies Act 2013...")

# Handle sample button clicks
if st.session_state.sample_prompt_clicked:
    user_query = st.session_state.sample_prompt_clicked
    st.session_state.sample_prompt_clicked = None

if user_query:
    # 1. Append user message to state and display
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # 2. Check Vector DB Readiness
    if get_collection_chunk_count() == 0:
        with st.chat_message("assistant"):
            st.error("Vector store is empty! Please run `python ingest.py` or click 'Re-index Companies Act PDF' in the sidebar.")
    elif not ollama_status["online"]:
        with st.chat_message("assistant"):
            st.error("""
            **Local Ollama Server is offline!**
            
            Please start Ollama in your terminal:
            ```powershell
            ollama serve
            ```
            And make sure you have pulled your selected model:
            ```powershell
            ollama pull llama3.2
            ```
            Once running, refresh this page and ask again.
            """)
    else:
        # 3. Execute RAG Retrieval & Generation
        with st.chat_message("assistant"):
            with st.spinner("Searching Companies Act 2013 and synthesizing verified response..."):
                try:
                    pipeline = LegalRAGPipeline(
                        model_name=selected_model,
                        base_url=ollama_url,
                        temperature=temperature,
                        top_k=top_k
                    )

                    # Build history tuples for conversational context
                    history = [
                        (m["role"], m["content"])
                        for m in st.session_state.messages[:-1]
                    ]

                    # Retrieve and stream response
                    token_stream, source_docs = pipeline.stream_query(user_query, chat_history=history)
                    
                    response_container = st.empty()
                    full_response = ""
                    for token in token_stream:
                        full_response += token
                        response_container.markdown(full_response + "▌")
                    response_container.markdown(full_response)

                    # Serialize sources for session state
                    sources_data = [
                        {
                            "page": d.metadata.get("page", "N/A"),
                            "chapter": d.metadata.get("chapter", "N/A"),
                            "section": d.metadata.get("section", "N/A"),
                            "content": d.page_content.strip()
                        }
                        for d in source_docs
                    ]

                    # Save to chat history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": full_response,
                        "sources": sources_data
                    })

                    # Display citations drawer
                    with st.expander(f"📚 View Legal Citations & Statutory Chunks ({len(sources_data)})"):
                        for idx, src in enumerate(sources_data, start=1):
                            sec = src.get("section", "N/A")
                            chap = src.get("chapter", "N/A")
                            page = src.get("page", "N/A")
                            content = src.get("content", "")
                            
                            st.markdown(f"""
                            <div class="citation-card">
                                <div class="citation-meta">
                                    <span class="citation-badge">Chunk #{idx}</span>
                                    <span class="citation-badge">Page {page}</span>
                                    <b>{chap}</b> | <b>{sec}</b>
                                </div>
                                <div class="citation-text">{content}</div>
                            </div>
                            """, unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"Error during RAG execution: {e}")
                    st.info("Tip: Verify that the selected model is pulled in Ollama (`ollama pull llama3.2`).")
