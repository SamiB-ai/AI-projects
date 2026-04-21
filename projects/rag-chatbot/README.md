# 🧠 AI Document Intelligence

A local RAG (Retrieval-Augmented Generation) application I built that lets you upload PDF documents and interact with them through an AI-powered chat interface — entirely on your own machine, no API key required.

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red?logo=streamlit)
![LangChain](https://img.shields.io/badge/LangChain-Orchestration-green)
![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM-black)
![FAISS](https://img.shields.io/badge/FAISS-Vector%20Store-orange)

---

## 🎯 What I Built

I wanted to create a tool that allows anyone to chat with their own documents without sending sensitive data to external APIs. The result is a fully local RAG pipeline with a clean Streamlit interface, powered by Ollama and FAISS.

**What you can do with it:**
- Upload one or multiple PDFs
- Get an automatic summary of your documents
- Extract key insights with one click
- Ask any question and get a grounded answer with source references
- Share the app temporarily via ngrok

---

## 📸 Screenshots

### 📄 Document Summary
<img width="1910" alt="Document Summary" src="https://github.com/user-attachments/assets/eebd97ce-3e9b-49cf-a085-6749a8142f42" />

### 💡 Key Insights
<img width="1915" alt="Key Insights" src="https://github.com/user-attachments/assets/8e81bbd8-aba1-40cf-98b2-0b237e9f3305" />

### 💬 Ask Questions
<img width="1899" alt="Ask Questions" src="https://github.com/user-attachments/assets/a3e3f9c2-23ec-471c-91d0-a1044d4c58b9" />

---

## 🛠️ Tech Stack

| Layer | Technology | Role |
|-------|-----------|------|
| UI | Streamlit | Web interface |
| LLM | Ollama — Gemma3 12B | Answer generation |
| Embeddings | `all-MiniLM-L6-v2` | Semantic search |
| Vector Store | FAISS | Similarity search |
| PDF Parsing | LangChain + PyPDF | Document loading |
| Tunneling | ngrok | Public URL sharing |

---

## ⚙️ How It Works

```
PDF Upload
    │
    ▼
Split into chunks (500 tokens, 100 overlap)
    │
    ▼
Each chunk converted to embedding vector (MiniLM)
    │
    ▼
Vectors indexed in FAISS
    │
    ▼
User question → embedding → top 5 similar chunks retrieved
    │
    ▼
Chunks + question sent to Gemma3 via Ollama
    │
    ▼
Answer generated locally ✅
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) installed
- 32GB RAM recommended (minimum 8GB)

### Installation

```bash
# Clone the repository
git clone https://github.com/SamiB-ai/AI-projects.git
cd projects/rag-chatbot

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt
```

### Pull a local LLM

```bash
ollama pull gemma3:12b
```

> You can swap this for `mistral` or `llama3` by changing the model name in `app.py`.

### Run

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

---

## 📁 Project Structure

```
rag-chatbot/
├── app.py              # Main Streamlit application
├── ingest.py           # PDF ingestion & vector store builder
├── requirements.txt    # Python dependencies
├── .gitignore
├── data/               # PDF storage (gitignored)
└── vectorstore/        # FAISS index (gitignored)
```

---

## 🌐 Sharing the App

To expose the app publicly using ngrok:

```bash
# In a separate terminal
ngrok http 8501
```

This generates a temporary public URL that anyone can access as long as your machine is running.

---

## 💡 What I Learned

Building this project gave me hands-on experience with:
- Designing a full RAG pipeline from scratch
- Working with vector embeddings and similarity search
- Integrating local LLMs via Ollama into a LangChain pipeline
- Managing LangChain versioning and dependency conflicts
- Building and deploying a Streamlit app

