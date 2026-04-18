import streamlit as st
import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from langchain_classic.chains import RetrievalQA
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from langchain_core.language_models.llms import LLM
from typing import Optional, List

st.set_page_config(page_title="AI Document Intelligence")
st.title("AI Document Intelligence")
st.write("Upload PDFs and analyze them with AI")

# Upload PDFs
uploaded_files = st.file_uploader(
    "Upload your PDFs",
    type="pdf",
    accept_multiple_files=True
)

if uploaded_files:
    os.makedirs("data", exist_ok=True)
    
    for old_file in os.listdir("data"):
        os.remove(f"data/{old_file}")
        print(f"[UPLOAD] Deleted old file: {old_file}")
    
    for file in uploaded_files:
        path = f"data/{file.name}"
        with open(path, "wb") as f:
            f.write(file.getbuffer())
        print(f"[UPLOAD] Saved: {path}")
    
    st.success("Files uploaded successfully")

# Rebuild vector database
if st.button("Rebuild Knowledge Base"):
    import shutil
    # Supprime l'ancienne vectorstore
    if os.path.exists("vectorstore"):
        shutil.rmtree("vectorstore")
        print("[INGEST] Old vectorstore deleted")
    
    print("[INGEST] Starting ingest.py...")
    result = os.system("python ingest.py")
    print(f"[INGEST] ingest.py exited with code: {result}")
    
    if result == 0:
        st.success("Vector database updated")
        st.cache_resource.clear() 
    else:
        st.error("ingest.py failed! Check the terminal.")

# Load DB
@st.cache_resource
def load_db():
    print("[DB] Loading vectorstore...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.load_local("vectorstore", embeddings, allow_dangerous_deserialization=True)
    print(f"[DB] Vectorstore loaded OK")
    return db

# Load LLM
@st.cache_resource
def load_llm():
    from langchain_ollama import OllamaLLM
    print("[LLM] Loading Gemma via Ollama...")
    llm = OllamaLLM(model="gemma3:12b")  
    print("[LLM] Ready")
    return llm

# Check DB existence
if not os.path.exists("vectorstore"):
    st.warning("Upload PDFs and rebuild database first")
    st.stop()

print("[APP] Loading db and llm...")
db = load_db()
llm = load_llm()
print("[APP] db and llm ready")

# Chat history
if "history" not in st.session_state:
    st.session_state.history = []

# Summary
st.header("Document Summary")
if st.button("Generate Summary"):
    docs = db.similarity_search("summary", k=5)
    context = "\n".join([doc.page_content for doc in docs])[:1500]
    
    prompt = f"What is this document about? Give a detailed summary:\n\n{context[:2000]}"
    
    result = llm.invoke(prompt)
    
    print(f"[SUMMARY] Result: '{result}'")
    if result and result.strip():
        st.write(result)
    else:
        st.warning("Empty response. Check terminal.")

# Insights
if st.button("Extract Insights"):
    docs = db.similarity_search("key points requirements", k=5)
    context = "\n".join([doc.page_content for doc in docs])[:1500]
    
    prompt = f"List the main requirements and rules from this text:\n\n{context[:2000]}"
    
    result = llm.invoke(prompt)

    print(f"[INSIGHTS] Result: '{result}'")
    if result and result.strip():
        st.write(result)
    else:
        st.warning("Empty response. Check terminal.")

# Q&A
st.header("Ask Questions")
query = st.text_input("Ask something about your documents")

if query:
    print(f"[QA] Query: '{query}'")
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=db.as_retriever(),
        return_source_documents=True
    )
    result = qa.invoke({"query": query})
    print(f"[QA] Result: '{result['result']}'")

    st.subheader("Answer")
    st.write(result["result"])
    st.session_state.history.append((query, result["result"]))

    st.subheader("Sources")
    for doc in result["source_documents"]:
        st.markdown(
            f"""<div style="background-color:#f5f5f5; padding:10px; border-radius:5px; color:black;">
            {doc.page_content[:300]}
            </div>""",
            unsafe_allow_html=True
        )

# History
st.subheader("Chat History")
for q, r in st.session_state.history:
    st.write(f"**You:** {q}")
    st.write(f"**AI:** {r}")