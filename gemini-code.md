Act as a Senior AI/ML Engineer. Write complete, production-ready Python code to build a fully local Retrieval-Augmented Generation (RAG) chatbot using the attached "Companies Act 2013" document.

The application must run 100% locally on a laptop without relying on external paid APIs (e.g., no OpenAI, no Anthropic API keys).

### Tech Stack Requirements:
1. LLM Runner: Ollama using `llama3.2` (or `mistral` / `qwen2.5:7b`).
2. Embedding Model: HuggingFace embeddings running locally via `sentence-transformers/all-MiniLM-L6-v2` or `BAAI/bge-small-en-v1.5`.
3. Vector Store: ChromaDB or FAISS stored on local disk.
4. Framework: LangChain or LlamaIndex.
5. User Interface: Streamlit or Chainlit.

---

### Project Architecture & Requirements:

1. Document Processing & Ingestion (`ingest.py`):
   - Ingest the PDF document: `Companies Act 2013.pdf`.
   - Use `RecursiveCharacterTextSplitter` with chunk parameters tuned for legal text (`chunk_size=1000`, `chunk_overlap=150`).
   - Extract and preserve metadata for every chunk, specifically capturing the Section numbers, Chapter titles, and page numbers where available.
   - Embed and store the chunks in a persistent local directory (`./chroma_db`).

2. Retrieval & RAG Pipeline (`rag_chain.py`):
   - Configure a similarity retriever fetching top-k chunks ($k=4$).
   - Create a strict legal-assistant prompt template:
     * Ground answers strictly in the provided retrieved context.
     * Always cite specific Section numbers and Chapter references from the retrieved context.
     * Explicitly state if the answer cannot be found in the provided text rather than hallucinating legal provisions.
   - Wire the pipeline using LangChain Expression Language (LCEL) or RetrievalQA chain connecting `ChatOllama` to the vector retriever.

3. Frontend App (`app.py`):
   - Provide a clean conversational UI with message history memory.
   - Display the source citations (Section, Chapter, and chunk snippets) in an expandable drawer under each response.

4. Deliverables:
   - Provide the complete folder structure.
   - Supply `requirements.txt` with exact version-compatible packages.
   - Provide step-by-step setup and run instructions (including the exact `ollama pull` command and startup commands).