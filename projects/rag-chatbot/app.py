import streamlit as st
import os

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import RetrievalQA

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="AI Document Intelligence",
    layout="wide",
    page_icon="🧠"
)

# ---------------- CSS ----------------
st.markdown("""
<style>

/* 🌍 Global */
.stApp {
    background-color: #f7f9fc;
}

/* 🧠 Text */
html, body, [class*="css"] {
    color: #1f2937 !important;
}

/* 📂 Sidebar */
section[data-testid="stSidebar"] {
    background-color: white;
    border-right: 1px solid #e5e7eb;
}



/* 🔘 Buttons */
.stButton > button {
    background: linear-gradient(135deg, #4f8cff, #6ea8fe);
    color: white;
    border-radius: 10px;
    border: none;
    padding: 10px;
    font-weight: 600;
}

/* 📤 Upload fix */
.stFileUploader > div {
    background-color: white !important;
    border: 1px solid #e5e7eb !important;
    border-radius: 10px !important;
}

/* 💬 Chat */
[data-testid="stChatMessage"] {
    background-color: white;
    border-radius: 10px;
    padding: 10px;
    margin-bottom: 10px;
    border: 1px solid #e5e7eb;
}

/* 📌 Sources */
.source-box {
    background-color: #ffffff;
    padding: 12px;
    border-radius: 10px;
    border: 1px solid #e5e7eb;
    margin-bottom: 10px;
}

/* 🧾 Input */
input {
    border-radius: 10px !important;
    border: 1px solid #e5e7eb !important;
    padding: 10px !important;
    background-color: white !important;
    color: black !important;
}

</style>
""", unsafe_allow_html=True)



# ---------------- HEADER ----------------
st.markdown("""
# 🧠 AI Document Intelligence  
### Analyze your PDFs with AI — fast, smart, powerful
""")

st.markdown("<br>", unsafe_allow_html=True)

with st.sidebar:
    st.title("📂 Documents")
    st.write("Upload and manage your PDFs")

    st.markdown("### Upload PDFs")

    uploaded_files = st.file_uploader(
        "",
        type="pdf",
        accept_multiple_files=True
    )

    if uploaded_files:
        os.makedirs("data", exist_ok=True)

        for old_file in os.listdir("data"):
            os.remove(f"data/{old_file}")

        for file in uploaded_files:
            path = f"data/{file.name}"
            with open(path, "wb") as f:
                f.write(file.getbuffer())

        st.success("Files uploaded")

    if st.button("Rebuild Knowledge Base", use_container_width=True):
        import shutil

        if os.path.exists("vectorstore"):
            shutil.rmtree("vectorstore")

        result = os.system("python ingest.py")

        if result == 0:
            st.success("Database ready")
            st.cache_resource.clear()
        else:
            st.error("ingest failed")


if not os.path.exists("vectorstore"):
    st.warning("Upload PDFs and rebuild database first")
    st.stop()

@st.cache_resource
def load_db():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.load_local("vectorstore", embeddings, allow_dangerous_deserialization=True)

@st.cache_resource
def load_llm():
    from langchain_ollama import OllamaLLM
    return OllamaLLM(model="gemma3:12b")

db = load_db()
llm = load_llm()

if "history" not in st.session_state:
    st.session_state.history = []

if "summary" not in st.session_state:
    st.session_state.summary = None

if "insights" not in st.session_state:
    st.session_state.insights = None


st.markdown("## 📄 Document Summary")

if st.button("Generate Summary", use_container_width=True):
    docs = db.similarity_search("summary", k=5)
    context = "\n".join([doc.page_content for doc in docs])[:1500]

    prompt = f"What is this document about?\n\n{context}"

    with st.spinner("Generating summary..."):
        result = llm.invoke(prompt)

    st.session_state.summary = result    


if st.button("Extract Insights", use_container_width=True):
    docs = db.similarity_search("key points", k=5)
    context = "\n".join([doc.page_content for doc in docs])[:1500]

    with st.spinner("Extracting insights..."):
        result = llm.invoke(f"Extract key insights:\n\n{context}")

    st.session_state.insights = result

st.markdown('</div>', unsafe_allow_html=True)
# Display summary
if st.session_state.summary is not None:
    st.markdown("### 🧾 Summary")
    st.write(st.session_state.summary)

# Display insights
if st.session_state.insights is not None:
    st.markdown("### 💡 Insights")
    st.write(st.session_state.insights)

st.markdown("## 💬 Ask Questions")

query = st.text_input("", placeholder="Ask something about your documents...")

if query:
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=db.as_retriever(),
        return_source_documents=True
    )

    with st.spinner("Thinking..."):
        result = qa.invoke({"query": query})

    st.write(result["result"])
    st.session_state.history.append((query, result["result"]))

    st.markdown("### 📌 Sources")

    for doc in result["source_documents"]:
        st.markdown(
            f"""<div class="source-box">
            {doc.page_content[:300]}
            </div>""",
            unsafe_allow_html=True
        )

st.markdown('</div>', unsafe_allow_html=True)

# -------- CHAT --------
st.markdown("## 🧾 Chat History")

for q, r in st.session_state.history:
    with st.chat_message("user"):
        st.write(q)
    with st.chat_message("assistant"):
        st.write(r)
        
