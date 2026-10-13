"""
Retrieval & RAG Pipeline Module for Companies Act 2013.

Implements a local RAG chain connecting ChromaDB vector retriever to
ChatOllama (llama3.2, mistral, qwen2.5:7b) via LangChain Expression Language (LCEL).
Enforces strict legal grounding, explicit section/chapter citations,
and no hallucination rules.
"""

from typing import List, Dict, Any, Generator, Tuple
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama

import config

# Strict Legal Prompt Template
SYSTEM_PROMPT = """You are an authoritative, senior legal AI research assistant specializing in the Indian Companies Act, 2013.

You must answer the user's inquiry based STRICTLY and ONLY on the statutory context provided below.

Strict Operational Guidelines:
1. Grounding: Rely exclusively on the retrieved statutory provisions, definitions, schedules, and explanations provided in the context.
2. Citation Requirement: ALWAYS explicitly cite the specific Section numbers (including sub-sections/clauses if available), Chapter titles, and Page numbers from the context.
3. No Hallucination: If the provided text does not contain sufficient information to answer the question, or if the provision is not present in the retrieved context, you MUST explicitly state:
   "Based on the provided context of the Companies Act, 2013, the answer cannot be found in the retrieved statutory text."
   Do NOT extrapolate, speculate, or fabricate legal sections, penalties, or provisions.
4. Structure & Clarity:
   - Present your analysis clearly and professionally.
   - Use structured bullet points where appropriate (e.g., eligibility criteria, thresholds, procedural requirements, penalties).
   - End your response with a concise "Statutory References" summary table or bullet list specifying: Section Number, Chapter, and Page.

Retrieved Statutory Context:
----------------------------------------
{context}
----------------------------------------
"""


def get_vectorstore(persist_dir: Path = config.CHROMA_DIR, collection_name: str = config.COLLECTION_NAME) -> Chroma:
    """Load the existing persistent Chroma vector store."""
    if not persist_dir.exists():
        raise FileNotFoundError(
            f"ChromaDB directory not found at '{persist_dir}'. "
            "Please run 'python ingest.py' first to build the vector index."
        )

    embeddings = HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    return Chroma(
        persist_directory=str(persist_dir),
        embedding_function=embeddings,
        collection_name=collection_name
    )


def format_retrieved_context(docs: List[Document]) -> str:
    """
    Format retrieved document chunks into structured legal blocks
    clearly presenting Section, Chapter, Page, and Content.
    """
    formatted_chunks = []
    for idx, doc in enumerate(docs, start=1):
        meta = doc.metadata
        sec = meta.get("section", "N/A")
        chap = meta.get("chapter", "N/A")
        page = meta.get("page", "N/A")
        chunk_header = f"[EXCERPT {idx}] | Chapter: {chap} | Section: {sec} | Page: {page}"
        formatted_chunks.append(f"{chunk_header}\n{doc.page_content.strip()}")

    return "\n\n" + ("=" * 50) + "\n\n".join(formatted_chunks) + "\n\n" + ("=" * 50)


def create_legal_rag_chain(
    model_name: str = config.DEFAULT_OLLAMA_MODEL,
    base_url: str = config.DEFAULT_OLLAMA_BASE_URL,
    temperature: float = config.DEFAULT_TEMPERATURE,
    top_k: int = config.TOP_K_CHUNKS,
):
    """
    Build and return the LangChain LCEL RAG chain.
    """
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k}
    )

    llm = ChatOllama(
        model=model_name,
        base_url=base_url,
        temperature=temperature,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{question}")
    ])

    # LCEL pipeline: returns generated text response
    rag_chain = prompt | llm | StrOutputParser()

    return retriever, rag_chain


class LegalRAGPipeline:
    """High-level wrapper for retrieval, generation, streaming, and citation inspection."""

    def __init__(
        self,
        model_name: str = config.DEFAULT_OLLAMA_MODEL,
        base_url: str = config.DEFAULT_OLLAMA_BASE_URL,
        temperature: float = config.DEFAULT_TEMPERATURE,
        top_k: int = config.TOP_K_CHUNKS,
    ):
        self.model_name = model_name
        self.base_url = base_url
        self.temperature = temperature
        self.top_k = top_k
        self.retriever, self.rag_chain = create_legal_rag_chain(
            model_name=model_name,
            base_url=base_url,
            temperature=temperature,
            top_k=top_k
        )

    def retrieve_documents(self, query: str) -> List[Document]:
        """
        Fetch top-k chunks relevant to query using hybrid section-aware routing:
        1. If query references a specific section (e.g. 'Section 135', 'Sec 149'),
           prioritize statutory chunks matching that section number.
        2. Supplement with semantic similarity search to fill top-k.
        """
        import re
        sec_match = re.search(r'\b(?:section|sec\.?)\s*(\d+[A-Z]?)\b', query, re.I)
        combined_docs: List[Document] = []
        seen_contents = set()

        # Step 1: If an explicit section is requested, fetch exact section chunks
        if sec_match:
            sec_num = sec_match.group(1).upper()
            try:
                sec_docs = self.retriever.vectorstore.similarity_search(
                    query,
                    k=self.top_k,
                    filter={"section_number": sec_num}
                )
                for d in sec_docs:
                    if d.page_content not in seen_contents:
                        seen_contents.add(d.page_content)
                        combined_docs.append(d)
            except Exception:
                pass

        # Step 2: Fetch general semantic similarity chunks
        semantic_docs = self.retriever.invoke(query)
        for d in semantic_docs:
            if d.page_content not in seen_contents:
                seen_contents.add(d.page_content)
                combined_docs.append(d)
            if len(combined_docs) >= self.top_k:
                break

        return combined_docs[: self.top_k]

    def query(self, question: str, chat_history: List[Tuple[str, str]] = None) -> Dict[str, Any]:
        """
        Execute synchronous RAG query.
        Returns:
            {
                "answer": str,
                "source_documents": List[Document]
            }
        """
        docs = self.retrieve_documents(question)
        context_str = format_retrieved_context(docs)

        # Format history as LangChain messages
        history_messages = []
        if chat_history:
            for role, content in chat_history:
                history_messages.append((role, content))

        answer = self.rag_chain.invoke({
            "context": context_str,
            "question": question,
            "chat_history": history_messages
        })

        return {
            "answer": answer,
            "source_documents": docs
        }

    def stream_query(
        self,
        question: str,
        chat_history: List[Tuple[str, str]] = None
    ) -> Tuple[Generator[str, None, None], List[Document]]:
        """
        Execute streaming RAG query.
        Returns:
            (token_generator, source_documents)
        """
        docs = self.retrieve_documents(question)
        context_str = format_retrieved_context(docs)

        history_messages = []
        if chat_history:
            for role, content in chat_history:
                history_messages.append((role, content))

        stream = self.rag_chain.stream({
            "context": context_str,
            "question": question,
            "chat_history": history_messages
        })

        return stream, docs
