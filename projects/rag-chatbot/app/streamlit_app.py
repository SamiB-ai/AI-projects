import streamlit as st
import os
import sys

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# UI
from ui import load_css

# Core
from core.llm import load_llm
from core.vectorstore import load_vectorstore
from core.rag import build_qa_chain, ask_question

# Services
from services.summarizer import generate_summary
from services.insights import extract_insights

# Utils
from utils.file_handler import save_uploaded_files


# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="AI Document Intelligence",
    layout="wide",
    page_icon="🧠"
)

# ---------------- CSS ----------------
load_css()

# ---------------- HEADER ----------------
st.markdown("""
# 🧠 AI Document Intelligence  
### Analyze your PDFs with AI — fast, smart, powerful
""")

st.markdown("<br>", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.title("📂 Documents")
    st.write("Upload and manage your PDFs")

    uploaded_files = st.file_uploader(
        "Upload PDFs",
        type="pdf",
        accept_multiple_files=True
    )

    if uploaded_files:
        save_uploaded_files(uploaded_files)
        st.success("Files uploaded")

    if st.button("Rebuild Knowledge Base", use_container_width=True):
        import shutil

        if os.path.exists("vectorstore"):
            shutil.rmtree("vectorstore")

        result = os.system("python core/ingest.py")

        if result == 0:
            st.success("Database ready")
            st.cache_resource.clear()
        else:
            st.error("Ingest failed")

# ---------------- CHECK DB ----------------
if not os.path.exists("vectorstore"):
    st.warning("Upload PDFs and rebuild database first")
    st.stop()

# ---------------- LOAD MODELS ----------------
db = load_vectorstore()
llm = load_llm()
qa_chain = build_qa_chain(llm, db)

# ---------------- SESSION STATE ----------------
if "history" not in st.session_state:
    st.session_state.history = []

if "summary" not in st.session_state:
    st.session_state.summary = None

if "insights" not in st.session_state:
    st.session_state.insights = None

# ---------------- SUMMARY ----------------
st.markdown("## 📄 Document Summary")

col1, col2 = st.columns(2)

with col1:
    if st.button("Generate Summary", use_container_width=True):
        with st.spinner("Generating summary..."):
            st.session_state.summary = generate_summary(llm, db)

with col2:
    if st.button("Extract Insights", use_container_width=True):
        with st.spinner("Extracting insights..."):
            st.session_state.insights = extract_insights(llm, db)

# Display summary
if st.session_state.summary:
    st.markdown("### 🧾 Summary")
    st.write(st.session_state.summary)

# Display insights
if st.session_state.insights:
    st.markdown("### 💡 Insights")
    st.write(st.session_state.insights)

# ---------------- QA ----------------
st.markdown("## 💬 Ask Questions")

query = st.text_input("", placeholder="Ask something about your documents...")

if query:
    with st.spinner("Thinking..."):
        result = ask_question(qa_chain, query)

    answer = result["result"]
    st.write(answer)

    st.session_state.history.append((query, answer))

    # Sources
    st.markdown("### 📌 Sources")
    for doc in result["source_documents"]:
        st.markdown(
            f"""<div class="source-box">
            {doc.page_content[:300]}
            </div>""",
            unsafe_allow_html=True
        )

# ---------------- CHAT HISTORY ----------------
st.markdown("## 🧾 Chat History")

for q, r in st.session_state.history:
    with st.chat_message("user"):
        st.write(q)
    with st.chat_message("assistant"):
        st.write(r)