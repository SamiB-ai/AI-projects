import streamlit as st
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import os

# Load vector DB

@st.cache_resource
def load_db():
    embeddings = OpenAIEmbeddings()
    return FAISS.load_local("vectorstore", embeddings)

# Create RAG chain
@st.cache_resource
def create_chain(_db):
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=_db.as_retriever(),
        return_source_documents=True
    )

# UI
st.set_page_config(page_title="RAG Chatbot", page_icon="🤖")

st.title("RAG Chatbot")
st.write("Ask questions about your PDF documents")

# Load system
if not os.path.exists("vectorstore"):
    st.error("Run ingest.py first to create vector database")
    st.stop()

db = load_db()
qa = create_chain(db)

# Input
query = st.text_input("Your question:")

if query:
    with st.spinner("Thinking... "):
        result = qa({"query": query})

    st.subheader("Answer")
    st.write(result["result"])

    st.subheader("Sources")

    for doc in result["source_documents"]:
        st.info(doc.page_content[:300])
