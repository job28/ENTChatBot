# ENT Chatbot — RAG-Powered Medical Document Assistant

A Retrieval-Augmented Generation (RAG) chatbot designed to answer questions from ENT-related documents using semantic search and an LLM-powered response pipeline.

This project demonstrates how large document collections can be transformed into an interactive question-answering assistant by combining document ingestion, embedding-based retrieval, vector search, and generative AI.

## Overview

The ENT Chatbot allows users to ask natural-language questions about ENT-focused medical documents and receive context-aware answers grounded in the indexed source material.

Instead of relying only on a language model’s general knowledge, the application retrieves relevant chunks from the document corpus first, then uses those retrieved passages to generate a more accurate and relevant response.

## Demo

Live demo: [Hugging Face Spaces](https://huggingface.co/spaces/jobalexander2801/ENTChatBot2)

## Features

* RAG-based question answering over ENT documents
* Semantic search using vector embeddings
* FAISS-based similarity retrieval
* LLM-generated answers grounded in retrieved context
* Web-based chat interface for interactive use
* Supports large document collections through offline indexing
* Deployable on Hugging Face Spaces

## How It Works

1. **Document Loading**
   Source documents are collected and loaded into the pipeline.

2. **Text Splitting**
   Documents are split into smaller chunks to improve retrieval quality and fit within model context limits.

3. **Embedding Generation**
   Each chunk is converted into vector embeddings using a sentence embedding model.

4. **Vector Indexing**
   The embeddings are stored in a FAISS vector database for efficient similarity search.

5. **User Query Processing**
   When a user asks a question, the query is embedded and matched against the indexed document chunks.

6. **Context Retrieval**
   The most relevant chunks are retrieved from FAISS.

7. **Answer Generation**
   The retrieved context is passed to the LLM, which generates a grounded response.

## Tech Stack

### Languages

* Python

### AI / ML / RAG

* LangChain
* FAISS
* Hugging Face Embeddings
* Groq LLM API
* Retrieval-Augmented Generation (RAG)

### UI / Deployment

* Gradio
* Hugging Face Spaces

### Supporting Tools

* GitHub
* GitHub Releases (for FAISS index distribution)
* Environment variables / secret management for API keys

## Project Structure

```bash
.
├── app.py
├── requirements.txt
├── docs/
├── faiss_index/
├── utils/
└── README.md
```

> Adjust the structure above if your repository uses different file or folder names.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set environment variables

Create a `.env` file or configure secrets in your deployment platform:

```env
GROQ_API_KEY=your_groq_api_key
```

## Running the App Locally

```bash
python app.py
```

Then open the local URL shown in the terminal.

## Deployment

This project is deployed on **Hugging Face Spaces** for public demonstration.

Typical deployment steps:

1. Push the application code to a Hugging Face Space repository
2. Add required secrets such as `GROQ_API_KEY`
3. Upload or download the FAISS index during startup
4. Launch the Gradio application

## FAISS Index Handling

Because vector indexes can become large, the FAISS index can be stored separately and downloaded during app startup.

The full 10 GB document corpus is **not uploaded to this GitHub repository**. Large source documents and related index assets are managed separately to keep the repository lightweight and practical for demo deployment.

This approach helps keep the demo application lightweight while still supporting large-scale retrieval.

## Use Cases

* Question answering over ENT medical documents
* Clinical knowledge exploration demos
* RAG proof-of-concept for healthcare document retrieval
* Demonstration of semantic search and LLM grounding
* Portfolio project for AI / ML / GenAI / NLP roles

## Why RAG?

Traditional LLM applications may hallucinate or answer from general training knowledge. RAG improves trustworthiness by retrieving relevant source content first and generating answers using that retrieved context.

This makes the chatbot more suitable for document-based question answering, internal knowledge assistants, and domain-specific search applications.

## Skills Demonstrated

This project showcases practical skills in:

* Building end-to-end RAG pipelines
* Working with embeddings and vector databases
* Prompting LLMs with retrieved context
* Deploying AI apps to Hugging Face Spaces
* Handling large document collections efficiently
* Structuring production-style AI demos for portfolio use

## Limitations

* Responses depend on the quality of retrieved chunks
* The chatbot is limited to the knowledge present in the indexed corpus
* It is intended as a demo / educational project and should not be used as a substitute for professional medical advice

## Future Improvements

* Add citations for retrieved source passages
* Improve chunking and retrieval strategy
* Add reranking for better answer quality
* Support multiple document domains
* Add conversation memory
* Add evaluation metrics for retrieval and answer relevance
* Add authentication and usage analytics

## Disclaimer

This project is intended for educational, research, and demonstration purposes only. It does not provide medical diagnosis or professional healthcare advice.

## License

Copyright © 2026 Job Alexander. All rights reserved.

This repository is shared for portfolio and demonstration purposes only. No permission is granted to use, copy, modify, distribute, or create derivative works from this code without prior written consent.

## Author

**Job Alexander**
AI / ML / DevOps / Cloud Enthusiast
LinkedIn: *Add your LinkedIn profile here*
GitHub: *Add your GitHub profile here*
