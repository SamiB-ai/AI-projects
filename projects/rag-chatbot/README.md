# 🧠 AI Document Intelligence

A local RAG (Retrieval-Augmented Generation) application that lets you upload PDF documents and interact with them through an AI-powered chat interface — entirely on your own machine, no API key required.

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red?logo=streamlit)
![LangChain](https://img.shields.io/badge/LangChain-Orchestration-green)
![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM-black)
![FAISS](https://img.shields.io/badge/FAISS-Vector%20Store-orange)

---

## 🎯 What I Built

I designed a modular and scalable RAG system that allows users to chat with their own documents locally, without sending sensitive data to external APIs.

The project follows a clean architecture inspired by production AI systems, separating core logic, services, utilities, and UI.

**What you can do with it:**

- Upload one or multiple PDFs
- Automatically build a semantic search database
- Generate document summaries
- Extract key insights
- Ask questions with grounded answers and sources
- Run everything locally (privacy-friendly)

---

## 📸 Screenshots

### 📄 Document Summary
<img width="1910" src="https://github.com/user-attachments/assets/eebd97ce-3e9b-49cf-a085-6749a8142f42" />

### 💡 Key Insights
<img width="1915" src="https://github.com/user-attachments/assets/8e81bbd8-aba1-40cf-98b2-0b237e9f3305" />

### 💬 Ask Questions
<img width="1899" src="https://github.com/user-attachments/assets/a3e3f9c2-23ec-471c-91d0-a1044d4c58b9" />

---

## 🛠️ Tech Stack

| Layer | Technology | Role |
|---|---|---|
| UI | Streamlit | Web interface |
| LLM | Ollama — Gemma3 12B | Answer generation |
| Embeddings | all-MiniLM-L6-v2 | Semantic search |
| Vector Store | FAISS | Similarity search |
| Orchestration | LangChain | RAG pipeline |
| PDF Parsing | PyPDFLoader | Document loading |

---

## 🗂️ Architecture

```
rag-chatbot/
├── app/
│   └── streamlit_app.py     # UI layer (Streamlit)
│   └── ui.py                # Css layer 
│
├── core/                    # Core AI logic
│   ├── llm.py               # LLM loader (Ollama)
│   ├── rag.py               # RetrievalQA pipeline
│   ├── ingest.py            # PDF ingestion pipeline
│   └── vectorstore.py       # FAISS loading
│
├── services/                # Business logic
│   ├── summarizer.py        # Document summarization
│   └── insights.py          # Insight extraction
│
├── utils/
│   └── file_handler.py      # File upload management
│
├── data/                    # Uploaded PDFs (ignored)
└── vectorstore/             # FAISS index (ignored)
```

---

## ⚙️ How It Works

```
PDF Upload
│
▼
Documents loaded & split into chunks
│
▼
Chunks embedded using MiniLM
│
▼
Stored in FAISS vector database
│
▼
User query → embedding → similarity search
│
▼
Relevant chunks retrieved
│
▼
Sent to LLM (Gemma3 via Ollama)
│
▼
Answer generated locally ✅
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- Ollama installed
- 8 GB RAM minimum (16–32 GB recommended)

### Installation

```bash
git clone https://github.com/SamiB-ai/AI-projects.git
cd projects/rag-chatbot

python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate # Mac/Linux

pip install -r requirements.txt
```

### Pull a Local Model

```bash
ollama pull gemma3:12b
```

> You can change the model in `core/llm.py`.

### Run the App

```bash
streamlit run app/streamlit_app.py
```

Open: [http://localhost:8501](http://localhost:8501)

---

## 🔁 Rebuilding the Knowledge Base

After uploading PDFs in the UI, click **"Rebuild Knowledge Base"**.

This runs `core/ingest.py`, which:

1. Loads PDFs
2. Splits them into chunks
3. Generates embeddings
4. Stores them in FAISS

---

## 🌐 Sharing the App

```bash
ngrok http 8501
```

---

## 💡 What I Learned

- Designing a modular RAG architecture
- Separation of concerns (core vs services vs UI)
- Vector search and embeddings
- Running local LLMs with Ollama
- Building AI apps with Streamlit
- Managing large document pipelines
