from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import os

# 1. Load PDF
def load_pdf(file_path):
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    return pages

# 2. Split text into chunks
def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    return splitter.split_documents(docs)

# 3. Create vector store
def create_vectorstore(chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore

# 4. Main pipeline
if __name__ == "__main__":
    file_path = "data/sample.pdf"  

    if not os.path.exists(file_path):
        print("❌ Ajoute un PDF dans data/sample.pdf")
        exit()

    docs = load_pdf(file_path)
    chunks = split_documents(docs)
    db = create_vectorstore(chunks)

    db.save_local("vectorstore")

    print("Vector database created successfully")
