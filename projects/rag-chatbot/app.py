import streamlit as st
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import os

st.set_page_config(page_title="AI Document Intelligence", page_icon="🧠")

st.title("AI Document Intelligence")
st.write("Analyze your documents with AI")

# Load DB
@st.cache_resource
def load_db():
    embeddings = OpenAIEmbeddings()
    return FAISS.load_local("vectorstore", embeddings)

# LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# Check DB
if not os.path.exists("vectorstore"):
    st.error("Run ingest.py first")
    st.stop()

db = load_db()

# -------------------
# 1. SUMMARY
# -------------------
st.header("Document Summary")

if st.button("Generate Summary"):
    docs = db.similarity_search("summarize this document", k=5)
    context = "\n".join([doc.page_content for doc in docs])

    prompt = f"""
    Summarize the following document clearly:

    {context}
    """

    summary = llm.predict(prompt)

    st.write(summary)

# -------------------
# 2. INSIGHTS
# -------------------
st.header("Key Insights")

if st.button("Extract Insights"):
    docs = db.similarity_search("main ideas", k=5)
    context = "\n".join([doc.page_content for doc in docs])

    prompt = f"""
    Extract key insights from this document as bullet points:

    {context}
    """

    insights = llm.predict(prompt)

    st.write(insights)

# -------------------
# 3. Q&A
# -------------------
st.header("Ask Questions")

query = st.text_input("Ask something about the document")

if query:
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=db.as_retriever(),
        return_source_documents=True
    )

    result = qa({"query": query})

    st.subheader("Answer")
    st.write(result["result"])

    st.subheader("Sources")
    for doc in result["source_documents"]:
        st.info(doc.page_content[:300])
