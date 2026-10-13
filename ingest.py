"""
Document Processing & Ingestion Module for Companies Act 2013.

Extracts text from the PDF, detects legal structural hierarchy (Chapters,
Sections, Schedules), slices legal text cleanly at section boundaries,
chunks text using RecursiveCharacterTextSplitter, enriches each chunk
with legal metadata, and indexes into a persistent local ChromaDB.
"""

import os
import re
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import pypdf
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

import config

# Regex patterns tailored for the Indian Companies Act 2013 layout
CHAPTER_PATTERN = re.compile(
    r'(?:^|\n)\s*(CHAPTER\s+[IVXLCDM]+)\s*\n\s*([A-Z\s,–—\-]{3,})'
)
SCHEDULE_PATTERN = re.compile(
    r'(?:^|\n)\s*(SCHEDULE\s+[IVXLCDM]+)\b(?:\s*\n\s*([^\n]+))?'
)
SECTION_HEADING_PATTERN = re.compile(
    r'(?:^|\n)\s*(\d+[A-Z]?)\.\s+([A-Z][A-Za-z0-9 ,()/\'\"–—\-]+?)\s*[.—–\-]+'
)
FOOTNOTE_KEYWORDS = ["subs. by", "omitted by", "ins. by", "w.e.f.", "the words", "vide notif"]


def clean_pdf_page_text(text: str) -> str:
    """Clean redundant spaces and header page numbers from raw PDF page text."""
    if not text:
        return ""
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    lines = text.split('\n')
    cleaned_lines = []
    for i, line in enumerate(lines):
        # Skip leading standalone page number lines (e.g. "14 \n")
        if i == 0 and line.strip().isdigit():
            continue
        cleaned_lines.append(line)
    return '\n'.join(cleaned_lines)


def is_valid_section(num_str: str, title: str) -> bool:
    """Verify if a detected match is an authentic legislative section rather than a footnote."""
    title_lower = title.lower()
    if any(keyword in title_lower for keyword in FOOTNOTE_KEYWORDS):
        return False
    digits = re.sub(r'\D', '', num_str)
    if not digits:
        return False
    val = int(digits)
    # The Companies Act 2013 has sections 1 to 470
    return 1 <= val <= 470


def parse_pdf_into_legal_segments(pdf_path: Path) -> List[Document]:
    """
    Parse PDF page by page, tracking Chapters and Schedules,
    and slicing text precisely at Section boundaries to preserve 100% legal integrity.
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF document not found at: {pdf_path}")

    print(f"[*] Reading PDF: {pdf_path.name}")
    reader = pypdf.PdfReader(str(pdf_path))
    total_pages = len(reader.pages)
    print(f"[*] Total PDF pages: {total_pages}")

    legal_segments: List[Document] = []
    current_chapter = "ARRANGEMENT OF SECTIONS"
    current_section = "Introductory Provisions"
    current_sec_num = "N/A"

    for page_idx in range(total_pages):
        page_num = page_idx + 1
        raw_text = reader.pages[page_idx].extract_text() or ""
        cleaned_text = clean_pdf_page_text(raw_text)
        if not cleaned_text.strip():
            continue

        # Check for chapter transition
        chap_match = CHAPTER_PATTERN.search(cleaned_text)
        if chap_match:
            chap_id = chap_match.group(1).strip()
            chap_title = chap_match.group(2).strip().splitlines()[0].strip()
            current_chapter = f"{chap_id}: {chap_title}"

        # Check for schedule transition
        sched_match = SCHEDULE_PATTERN.search(cleaned_text)
        if sched_match:
            sched_id = sched_match.group(1).strip()
            sched_desc = (sched_match.group(2) or "").strip()
            current_chapter = f"{sched_id}" + (f": {sched_desc}" if sched_desc else "")
            current_section = sched_id
            current_sec_num = sched_id

        # Find section headings within this page
        sec_matches: List[Tuple[int, str, str]] = []
        # Exclude TOC pages (pages 1-13) from section body slicing
        if page_num >= 14:
            for sm in SECTION_HEADING_PATTERN.finditer(cleaned_text):
                sec_num = sm.group(1).strip()
                sec_title = sm.group(2).strip()
                if is_valid_section(sec_num, sec_title):
                    sec_matches.append((sm.start(), sec_num, sec_title))

        if not sec_matches:
            # Whole page text continues the active section
            legal_segments.append(Document(
                page_content=cleaned_text,
                metadata={
                    "source": pdf_path.name,
                    "page": page_num,
                    "chapter": current_chapter,
                    "section": current_section,
                    "section_number": current_sec_num
                }
            ))
        else:
            # Slice page across section boundaries
            last_pos = 0
            for pos, sec_num, sec_title in sec_matches:
                segment_text = cleaned_text[last_pos:pos].strip()
                if segment_text:
                    legal_segments.append(Document(
                        page_content=segment_text,
                        metadata={
                            "source": pdf_path.name,
                            "page": page_num,
                            "chapter": current_chapter,
                            "section": current_section,
                            "section_number": current_sec_num
                        }
                    ))
                current_sec_num = sec_num
                current_section = f"Section {sec_num}: {sec_title}"
                last_pos = pos

            tail_text = cleaned_text[last_pos:].strip()
            if tail_text:
                legal_segments.append(Document(
                    page_content=tail_text,
                    metadata={
                        "source": pdf_path.name,
                        "page": page_num,
                        "chapter": current_chapter,
                        "section": current_section,
                        "section_number": current_sec_num
                    }
                ))

    print(f"[*] Extracted {len(legal_segments)} legal segments across {total_pages} pages.")
    return legal_segments


def chunk_segments_with_legal_tuning(segments: List[Document]) -> List[Document]:
    """
    Split legal segments with RecursiveCharacterTextSplitter tuned for statutes.
    chunk_size=1000, chunk_overlap=150.
    """
    print(f"[*] Splitting text using chunk_size={config.CHUNK_SIZE}, chunk_overlap={config.CHUNK_OVERLAP}...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", "—", "  ", " ", ""],
        length_function=len,
    )

    chunks = text_splitter.split_documents(segments)
    print(f"[*] Generated {len(chunks)} finely tuned chunks with legal metadata.")
    return chunks


def build_vector_store(chunks: List[Document], persist_dir: Path, collection_name: str, reset: bool = True) -> Chroma:
    """Embed chunks using local HuggingFace embeddings and persist to ChromaDB."""
    persist_dir.mkdir(parents=True, exist_ok=True)
    print(f"[*] Loading local HuggingFace embedding model: '{config.EMBEDDING_MODEL_NAME}'...")
    embeddings = HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    # If resetting, remove previous collection entries to prevent duplicate accumulations
    if reset:
        try:
            temp_vs = Chroma(
                persist_directory=str(persist_dir),
                embedding_function=embeddings,
                collection_name=collection_name
            )
            temp_vs.delete_collection()
            print(f"[*] Cleared previous '{collection_name}' collection for fresh indexing.")
        except Exception:
            pass

    print(f"[*] Indexing {len(chunks)} chunks into ChromaDB at '{persist_dir}'...")
    t0 = time.time()
    
    batch_size = 200
    vectorstore: Chroma = None

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        print(f"    - Indexing chunks {i + 1} to {min(i + batch_size, len(chunks))} of {len(chunks)}...", flush=True)
        if vectorstore is None:
            vectorstore = Chroma.from_documents(
                documents=batch,
                embedding=embeddings,
                persist_directory=str(persist_dir),
                collection_name=collection_name
            )
        else:
            vectorstore.add_documents(batch)

    elapsed = time.time() - t0
    print(f"[+] Ingestion complete in {elapsed:.2f}s! Vector store saved at: {persist_dir}")
    return vectorstore


def verify_ingestion(persist_dir: Path, collection_name: str):
    """Run verification search on the newly indexed ChromaDB vector store."""
    print("\n" + "=" * 60)
    print("[*] Running Verification Similarity Search...")
    embeddings = HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    vectorstore = Chroma(
        persist_directory=str(persist_dir),
        embedding_function=embeddings,
        collection_name=collection_name
    )

    test_queries = [
        "Corporate Social Responsibility CSR obligations under Section 135",
        "Company to have Board of Directors and independent directors under Section 149",
        "Punishment for fraud under Section 447"
    ]

    for q in test_queries:
        print(f"\nQuery: '{q}'")
        results = vectorstore.similarity_search(q, k=2)
        for rank, doc in enumerate(results, start=1):
            sec = doc.metadata.get('section', 'N/A')
            chap = doc.metadata.get('chapter', 'N/A')
            page = doc.metadata.get('page', 'N/A')
            snippet = doc.page_content[:180].replace('\n', ' ')
            print(f"  [{rank}] Page {page} | Chapter: {chap} | Section: {sec}")
            print(f"      Snippet: {snippet}...")
    print("=" * 60 + "\n")


def run_ingestion():
    """Main pipeline execution for ingestion."""
    print("=" * 60)
    print("Companies Act 2013 - Local RAG Ingestion Pipeline")
    print("=" * 60)
    
    # Remove existing collection if rebuilding
    segments = parse_pdf_into_legal_segments(config.PDF_PATH)
    chunks = chunk_segments_with_legal_tuning(segments)
    build_vector_store(chunks, config.CHROMA_DIR, config.COLLECTION_NAME)
    verify_ingestion(config.CHROMA_DIR, config.COLLECTION_NAME)


if __name__ == "__main__":
    run_ingestion()
