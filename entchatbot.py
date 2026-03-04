#!/usr/bin/env python3
"""
Lightweight RAG Web App (ENT) — Gradio + LangChain + Groq + FAISS

Change in this version:
- The UI launches immediately.
- FAISS indexing + RAG chain setup happens in a background thread.
- While building, the chat responds with a “still loading” message.
- A status banner auto-refreshes to show build progress / errors.

Deploy (Hugging Face Spaces):
- Put app.py + requirements.txt + your docs/ folder in the Space repo
- Add GROQ_API_KEY in Space → Settings → Secrets
"""

import os
import logging
import threading
import time
import traceback
from typing import List, Tuple, Optional, Dict, Any

# Optional: load .env if present
try:
    from dotenv import load_dotenv  # pip install python-dotenv (optional)
    load_dotenv()
except Exception:
    pass

from bs4 import BeautifulSoup
import gradio as gr

# LangChain
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Groq (cloud LLM)
from langchain_groq import ChatGroq


# ------------------------------
# Config
# ------------------------------
DOCS_DIR = os.environ.get("DOCS_DIR", "docs")

# Embeddings
HF_EMBEDDING_MODEL = os.environ.get(
    "HF_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)

# LLM (Groq)
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.1"))

# Retrieval
TOP_K = int(os.environ.get("TOP_K", "3"))

# Chunking
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "200"))

# Status refresh interval (seconds)
STATUS_REFRESH_EVERY = float(os.environ.get("STATUS_REFRESH_EVERY", "2.0"))


# ------------------------------
# Logging
# ------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger("ent-rag-web")


# ------------------------------
# Global App State (thread-safe)
# ------------------------------
QA_CHAIN: Optional[RetrievalQA] = None
STATE_LOCK = threading.Lock()

APP_STATE: Dict[str, Any] = {
    "state": "starting",  # starting | building | ready | error
    "message": "Starting…",
    "docs_loaded": 0,
    "chunks_built": 0,
    "last_error": "",
    "started_at": time.time(),
}


def set_state(**kwargs):
    with STATE_LOCK:
        APP_STATE.update(kwargs)


def get_state() -> Dict[str, Any]:
    with STATE_LOCK:
        return dict(APP_STATE)


# ------------------------------
# Document loading
# ------------------------------
def load_documents(directory: str = DOCS_DIR) -> List[Document]:
    """Recursively load PDFs and HTML files from a directory."""
    documents: List[Document] = []

    if not os.path.isdir(directory):
        logger.warning("Docs directory %s does not exist.", directory)
        return documents

    for root, _, files in os.walk(directory):
        for filename in files:
            path = os.path.join(root, filename)
            try:
                fl = filename.lower()
                if fl.endswith(".pdf"):
                    loader = PyPDFLoader(path)
                    docs = loader.load()
                    for d in docs:
                        meta = dict(d.metadata or {})
                        meta["source"] = path
                        d.metadata = meta
                    documents.extend(docs)

                elif fl.endswith(".html") or fl.endswith(".htm"):
                    with open(path, "r", encoding="utf-8", errors="ignore") as f:
                        soup = BeautifulSoup(f, "lxml")
                        for el in soup(["script", "style", "nav", "header", "footer"]):
                            el.decompose()
                        text = soup.get_text(separator=" ", strip=True)
                        if text:
                            documents.append(
                                Document(
                                    page_content=text,
                                    metadata={"source": path}
                                )
                            )
            except Exception as e:
                logger.error("Error loading %s: %s", path, e)

    logger.info("Loaded %d documents from %s", len(documents), directory)
    return documents


def split_documents(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(docs)
    logger.info("Split into %d chunks.", len(chunks))
    return chunks


def get_embeddings():
    logger.info("Using HuggingFaceEmbeddings model=%s", HF_EMBEDDING_MODEL)
    return HuggingFaceEmbeddings(model_name=HF_EMBEDDING_MODEL)


def build_vector_store(chunks: List[Document]) -> FAISS:
    embeddings = get_embeddings()
    logger.info("Building FAISS index in-memory ...")
    vs = FAISS.from_documents(chunks, embeddings)
    logger.info("FAISS index ready.")
    return vs


def setup_rag_chain(vector_store: FAISS) -> RetrievalQA:
    """
    Build a RetrievalQA chain with a custom prompt.
    Uses Groq cloud LLM via ChatGroq.
    """
    if not os.environ.get("GROQ_API_KEY"):
        raise RuntimeError(
            "Missing GROQ_API_KEY. Set it as an environment variable (or as a Space Secret)."
        )

    llm = ChatGroq(
        model=GROQ_MODEL,
        temperature=TEMPERATURE,
        max_retries=2,
    )

    prompt_template = (
        "You are a legal assistant specializing in ENT study and treatment.\n"
        "Answer the user's question using ONLY the context provided.\n"
        "If the answer is not in the context, say you don't have enough information.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer (concise, accurate):"
    )
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": TOP_K}),
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,
    )
    logger.info("RAG chain ready (Groq model=%s).", GROQ_MODEL)
    return qa_chain


# ------------------------------
# Background initialization
# ------------------------------
def init_app_background():
    global QA_CHAIN
    try:
        set_state(state="building",
                  message=f"Loading documents from '{DOCS_DIR}'…")
        logger.info(
            "Initializing app (background)… loading docs from %s", DOCS_DIR)

        docs = load_documents(DOCS_DIR)
        set_state(docs_loaded=len(docs))

        if not docs:
            raise RuntimeError(
                f"No documents found in '{DOCS_DIR}'. Add PDFs/HTML files under docs/ and restart."
            )

        set_state(message="Splitting documents into chunks…")
        chunks = split_documents(docs)
        set_state(chunks_built=len(chunks))

        set_state(message="Building FAISS index (in-memory)…")
        vector_store = build_vector_store(chunks)

        set_state(message="Setting up Groq LLM + RetrievalQA chain…")
        chain = setup_rag_chain(vector_store)

        with STATE_LOCK:
            QA_CHAIN = chain
            APP_STATE["state"] = "ready"
            APP_STATE["message"] = "Ready ✅ Ask your question."
            APP_STATE["last_error"] = ""

        logger.info("Background init finished: READY.")

    except Exception as e:
        err = f"{e}\n\n{traceback.format_exc()}"
        logger.exception("Background init failed:")
        set_state(state="error", message="Failed to initialize ❌", last_error=err)


def start_background_init_once():
    st = get_state()
    if st["state"] in ("building", "ready"):
        return
    t = threading.Thread(target=init_app_background, daemon=True)
    t.start()


# ------------------------------
# UI helpers
# ------------------------------
def format_sources(source_docs: List[Document]) -> str:
    if not source_docs:
        return "No sources returned."
    seen = set()
    sources = []
    for d in source_docs:
        src = (d.metadata or {}).get("source", "unknown")
        if src not in seen:
            seen.add(src)
            sources.append(src)
    return "\n".join(f"- {s}" for s in sources)


def status_markdown() -> str:
    st = get_state()
    elapsed = int(time.time() - st["started_at"])
    header = f"**Status:** `{st['state']}`  |  **Elapsed:** {elapsed}s  \n"
    details = (
        f"**Message:** {st['message']}  \n"
        f"**Docs loaded:** {st.get('docs_loaded', 0)}  \n"
        f"**Chunks built:** {st.get('chunks_built', 0)}  \n"
    )
    if st["state"] == "error" and st.get("last_error"):
        details += "\n<details><summary><b>Error details</b></summary>\n\n```text\n"
        details += st["last_error"][:8000]
        details += "\n```\n</details>\n"
    return header + details


def answer_question(message: str, history: List[Tuple[str, str]]):
    message = (message or "").strip()
    if not message:
        return "Please enter a question."

    st = get_state()
    if st["state"] != "ready" or QA_CHAIN is None:
        if st["state"] == "error":
            return (
                "The app failed to initialize, so I can't answer yet.\n\n"
                "Check the **Status** panel above for error details."
            )
        return (
            f"Index is still building…\n\n"
            f"- {st['message']}\n"
            f"- Docs loaded: {st.get('docs_loaded', 0)}\n"
            f"- Chunks built: {st.get('chunks_built', 0)}\n\n"
            "Try again in a moment."
        )

    try:
        result = QA_CHAIN.invoke({"query": message})
        answer = result.get("result", "").strip() if isinstance(
            result, dict) else str(result).strip()
        src_docs = result.get("source_documents", []
                              ) if isinstance(result, dict) else []
        sources_md = format_sources(src_docs)

        if not answer:
            return "Sorry, I couldn't produce an answer from the available context."

        return f"{answer}\n\n---\n**Sources:**\n{sources_md}"

    except Exception as e:
        logger.exception("Error during generation:")
        return f"Error: {str(e)}"


# ------------------------------
# Main (Gradio)
# ------------------------------
def build_ui():
    title = "ENT RAG Chatbot"
    description = (
        "Ask questions about the ENT documents you placed in the **docs/** folder.\n\n"
        "**Note:** The UI loads immediately; indexing happens in the background."
    )

    with gr.Blocks(title=title) as demo:
        gr.Markdown(f"# {title}\n\n{description}")
        gr.Markdown(
            "**Model:** " + GROQ_MODEL + "  \n"
            "**Retriever:** FAISS (in-memory)  \n"
            "**Embeddings:** " + HF_EMBEDDING_MODEL
        )

        status = gr.Markdown(status_markdown())

        # ✅ Gradio 5+ replacement for `every=`: use a Timer + tick event
        timer = gr.Timer(value=STATUS_REFRESH_EVERY, active=True)
        timer.tick(fn=status_markdown, inputs=[], outputs=status, queue=False)

        gr.Markdown("---")

        gr.ChatInterface(
            fn=answer_question,
            chatbot=gr.Chatbot(height=420),
            textbox=gr.Textbox(
                placeholder="Type your question about ENT …", scale=7),
            examples=[
                "What does Article 40 cover?",
                "Summarise the main points relevant to personal rights.",
                "What does the context say about due process?"
            ],
        )

    return demo


if __name__ == "__main__":
    start_background_init_once()
    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", "7860"))
    )
