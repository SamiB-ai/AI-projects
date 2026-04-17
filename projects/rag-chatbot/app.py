import streamlit as st
import os
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

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

    for file in uploaded_files:
        with open(f"data/{file.name}", "wb") as f:
            f.write(file.getbuffer())

    st.success("Files uploaded successfully")

# Rebuild vector database
if st.button("Rebuild Knowledge Base"):
    os.system("python ingest.py")
    st.success("Vector database updated")

# Load DB
@st.cache_resource
def load_db():
    embeddings = OpenAIEmbeddings()
    return FAISS.load_local("vectorstore", embeddings)

# LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# Check DB existence
if not os.path.exists("vectorstore"):
    st.warning("Upload PDFs and rebuild database first")
    st.stop()

db = load_db()

# Chat history
if "history" not in st.session_state:
    st.session_state.history = []

# Summary
st.header("Document Summary")

if st.button("Generate Summary"):
    docs = db.similarity_search("summarize this document", k=5)
    context = "\n".join([doc.page_content for doc in docs])

    summary = llm.predict(f"Summarize this:\n{context}")
    st.write(summary)

# Insights
st.header("Key Insights")

if st.button("Extract Insights"):
    docs = db.similarity_search("main ideas", k=5)
    context = "\n".join([doc.page_content for doc in docs])

    insights = llm.predict(f"Give key insights:\n{context}")
    st.write(insights)

# Q&A
st.header("Ask Questions")

query = st.text_input("Ask something about your documents")

if query:
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=db.as_retriever(),
        return_source_documents=True
    )

    result = qa({"query": query})

    st.subheader("Answer")
    st.write(result["result"])

    st.session_state.history.append((query, result["result"]))

    st.subheader("Sources")

    for doc in result["source_documents"]:
        st.markdown(
            f"""
            <div style="background-color:#f5f5f5; padding:10px; border-radius:5px;">
            {doc.page_content[:300]}
            </div>
            """,
            unsafe_allow_html=True
        )

# History
st.subheader("Chat History")

for q, r in st.session_state.history:
    st.write(f"You: {q}")
    st.write(f"AI: {r}")
