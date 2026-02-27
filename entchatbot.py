#!/usr/bin/env python3
"""
Lightweight RAG Web App (ENT) — Gradio + LangChain + Groq + FAISS

Free hosting-friendly:
- UI: Gradio (works great on Hugging Face Spaces)
- LLM: Groq cloud via ChatGroq (set GROQ_API_KEY)
- Embeddings: sentence-transformers via HuggingFaceEmbeddings (no Ollama needed)
- Vector DB: FAISS (in-memory; rebuilt at startup from docs/)

Quick start (local):
  pip install -r requirements.txt
  export GROQ_API_KEY="your_groq_key"
  python app.py
  # open the printed Gradio URL

Deploy (Hugging Face Spaces):
- Put app.py + requirements.txt + your docs/ folder in the Space repo
- Add GROQ_API_KEY in Space → Settings → Secrets
"""

import os
import logging
from typing import List, Tuple

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

# ------------------------------
# Logging
# ------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger("ent-rag-web")


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
                                Document(page_content=text,
                                         metadata={"source": path})
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
    prompt = PromptTemplate(template=prompt_template,
                            input_variables=["context", "question"])

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": TOP_K}),
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,  # we will show sources in the UI
    )
    logger.info("RAG chain ready (Groq model=%s).", GROQ_MODEL)
    return qa_chain


# ------------------------------
# App state (build once at startup)
# ------------------------------
QA_CHAIN: RetrievalQA = None


def init_app():
    global QA_CHAIN
    logger.info("Initializing app... loading docs from %s", DOCS_DIR)
    docs = load_documents(DOCS_DIR)
    if not docs:
        raise RuntimeError(
            f"No documents found in '{DOCS_DIR}'. Add PDFs/HTML files under docs/ and restart."
        )

    chunks = split_documents(docs)
    vector_store = build_vector_store(chunks)
    QA_CHAIN = setup_rag_chain(vector_store)


def format_sources(source_docs: List[Document]) -> str:
    if not source_docs:
        return "No sources returned."
    # Deduplicate by source
    seen = set()
    sources = []
    for d in source_docs:
        src = (d.metadata or {}).get("source", "unknown")
        if src not in seen:
            seen.add(src)
            sources.append(src)
    return "\n".join(f"- {s}" for s in sources)


def answer_question(message: str, history: List[Tuple[str, str]]):
    """
    Gradio ChatInterface callback.
    """
    message = (message or "").strip()
    if not message:
        return "Please enter a question."

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
    title = "Irish Law RAG Chatbot"
    description = (
        "Ask questions about the ENT documents you placed in the **docs/** folder.\n\n"
        "**Tip:** Ask specific questions (e.g., “What does Article 40 cover?”)."
    )

    with gr.Blocks(title=title) as demo:
        gr.Markdown(f"# {title}\n\n{description}")
        gr.Markdown(
            "**Model:** " + GROQ_MODEL + "  \n"
            "**Retriever:** FAISS (in-memory)  \n"
            "**Embeddings:** " + HF_EMBEDDING_MODEL
        )
        chat = gr.ChatInterface(
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
    init_app()
    demo = build_ui()
    # server_name=0.0.0.0 helps in containers/Spaces
    demo.launch(server_name="0.0.0.0", server_port=int(
        os.environ.get("PORT", "7860")))
