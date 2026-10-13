# ⚖️ Companies Act 2013 - Local RAG AI Assistant

A 100% local, privacy-preserving Retrieval-Augmented Generation (RAG) assistant for researching and navigating the Indian **Companies Act, 2013**.

This system runs **completely on local hardware** without any external paid APIs (no OpenAI, no Anthropic API keys required).

---

## 🌟 Architecture Overview

```
                      +-----------------------------+
                      |   Companies Act 2013.pdf    |
                      +-----------------------------+
                                     |
                                     v
                       [ingest.py Legal Parser]
                      - Slices text at Section boundaries
                      - Tracks Chapter titles & Page numbers
                      - RecursiveCharacterTextSplitter (1000/150)
                                     |
                                     v
                     [HuggingFace MiniLM Embeddings]
                   sentence-transformers/all-MiniLM-L6-v2
                                     |
                                     v
                     +-------------------------------+
                     | ChromaDB Vector Store (Disk)  |
                     |         (./chroma_db)         |
                     +-------------------------------+
                                     |
                                     | Top-k (k=4) Chunks
                                     v
                      [rag_chain.py LCEL Pipeline]
                     - Strict Anti-Hallucination Prompt
                     - Mandatory Section & Chapter Citations
                     - Hybrid Section-Aware Routing
                                     |
                  +------------------+------------------+
                  |                                     |
                  v                                     v
       [ChatOllama Local LLM]               [Streamlit Frontend]
      (llama3.2 / mistral / qwen2.5)       - Conversational Chat Memory
                                           - Expandable Citations Drawer
                                           - Model & Diagnostic Controls
```

---

## 📁 Repository Structure

```
Companies Act 2013 - AI Assistant/
│
├── Companies Act 2013.pdf     # Primary source document (288 pages)
├── config.py                   # Centralized configuration (paths, models, chunk sizes)
├── ingest.py                   # Legal parser, chunker, metadata extractor & ChromaDB builder
├── rag_chain.py                # LCEL RAG chain with strict prompt & Ollama integration
├── app.py                      # Streamlit interactive chat UI with expandable citations
├── query_cli.py                # Command-line testing & verification tool
├── requirements.txt            # Locked, production-ready dependencies
├── chroma_db/                  # Persistent local ChromaDB vector store
└── README.md                   # Documentation & setup guide
```

---

## 🚀 Quickstart Guide

### 1. Prerequisites
- **Python 3.10+** (Tested on Python 3.11, 3.12, 3.13)
- **Ollama** installed on your system ([Download from ollama.com](https://ollama.com))

---

### 2. Install & Start Ollama

#### Step 2.1: Start the Ollama local daemon
Open a terminal (PowerShell or Bash) and run:
```bash
ollama serve
```

#### Step 2.2: Pull the LLM weights
In another terminal, pull the recommended model (`llama3.2` 3B parameter model, fast & lightweight):
```bash
ollama pull llama3.2
```
*(Optional alternatives: `ollama pull mistral` or `ollama pull qwen2.5:7b`)*

---

### 3. Setup Python Environment & Dependencies

```bash
# Clone or navigate to the project directory
cd "Companies Act 2013 - AI Assistant"

# Create a virtual environment (recommended)
python -m venv venv

# Activate the virtual environment
# On Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# On macOS / Linux:
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

---

### 4. Build the Vector Index (Ingestion)

Run the legal ingestion script to parse the PDF, extract Section and Chapter metadata, and generate embeddings:

```bash
python ingest.py
```

**Expected output:**
- Extracted 774 legal segments across 288 pages.
- Generated 1,527 finely tuned chunks with legal metadata.
- Embedded and indexed into `./chroma_db` in under 60 seconds.
- Automatically executes verification similarity queries.

---

### 5. Launch the Web Application

Launch the Streamlit interactive conversational application:

```bash
streamlit run app.py
```

The app will open automatically in your browser at:
👉 **`http://localhost:8501`**

---

### 6. (Optional) Test Directly via Command Line (CLI)

You can query the RAG engine directly from your terminal:

```bash
# Full generation using Ollama:
python query_cli.py -q "What are the CSR expenditure requirements under Section 135?"

# Retrieve-only mode (inspect retrieved chunks without running Ollama):
python query_cli.py -q "What is the punishment for fraud under Section 447?" --retrieve-only
```

---

## 🛡️ Anti-Hallucination & Legal Citation Guarantee

The RAG pipeline enforces strict rules in `rag_chain.py`:
1. **Strict Context Grounding**: The LLM is instructed to answer strictly based on the provided statutory text.
2. **Explicit Citations**: Every response must cite the Section number, Chapter title, and PDF Page.
3. **No Hallucination Fallback**: If a provision is not present in the retrieved text, the assistant explicitly states:
   > *"Based on the provided context of the Companies Act, 2013, the answer cannot be found in the retrieved statutory text."*
4. **Transparent Verification**: Every answer in the Streamlit UI contains an expandable **"📚 View Legal Citations & Statutory Chunks"** drawer showing the exact text excerpts, page numbers, and section labels used to generate the answer.
