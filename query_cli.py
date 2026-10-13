"""
CLI Verification & Testing Tool for Companies Act 2013 RAG.
Useful for rapid testing of retrieval and generation directly from the terminal.
"""

import argparse
import sys

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from rag_chain import LegalRAGPipeline, get_vectorstore, format_retrieved_context
import config


def main():
    parser = argparse.ArgumentParser(description="Query the Companies Act 2013 Local RAG")
    parser.add_argument(
        "--query", "-q",
        type=str,
        default="What are the Corporate Social Responsibility obligations and spending thresholds under Section 135?",
        help="Legal query to evaluate"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=config.DEFAULT_OLLAMA_MODEL,
        help="Ollama model name (e.g. llama3.2, mistral, qwen2.5:7b)"
    )
    parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=config.TOP_K_CHUNKS,
        help="Number of chunks to retrieve"
    )
    parser.add_argument(
        "--retrieve-only", "-r",
        action="store_true",
        help="Only display retrieved context and metadata without invoking LLM"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("🏛️ Companies Act 2013 - Local Legal RAG CLI")
    print("=" * 70)
    print(f"Query: '{args.query}'\n")

    vs = get_vectorstore()
    print(f"[*] Total chunks in Chroma vector store: {vs._collection.count():,}")
    print(f"[*] Retrieving top {args.top_k} statutory chunks...\n")
    
    pipeline = LegalRAGPipeline(
        model_name=args.model,
        base_url=config.DEFAULT_OLLAMA_BASE_URL,
        temperature=config.DEFAULT_TEMPERATURE,
        top_k=args.top_k
    )
    docs = pipeline.retrieve_documents(args.query)

    for i, doc in enumerate(docs, start=1):
        sec = doc.metadata.get("section", "N/A")
        chap = doc.metadata.get("chapter", "N/A")
        page = doc.metadata.get("page", "N/A")
        snippet = doc.page_content.strip().replace("\n", " ")[:200]
        print(f"  [{i}] Page {page} | {chap} | {sec}")
        print(f"      Snippet: {snippet}...\n")

    if args.retrieve_only:
        print("[*] Retrieve-only mode selected. Exiting.")
        return

    print("-" * 70)
    print(f"[*] Invoking Ollama model '{args.model}'...")
    print("-" * 70)
    try:
        pipeline = LegalRAGPipeline(
            model_name=args.model,
            base_url=config.DEFAULT_OLLAMA_BASE_URL,
            temperature=config.DEFAULT_TEMPERATURE,
            top_k=args.top_k
        )
        stream, _ = pipeline.stream_query(args.query)
        print("Generated Legal Response:\n")
        for chunk in stream:
            sys.stdout.write(chunk)
            sys.stdout.flush()
        print("\n")
    except Exception as e:
        print(f"\n[!] Note: Ollama generation error: {e}")
        print("    If Ollama is not running, start it using: 'ollama serve' and 'ollama pull llama3.2'.")


if __name__ == "__main__":
    main()
