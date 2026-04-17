from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import os

DATA_PATH = "data"

def load_all_pdfs():
    documents = []

    if not os.path.exists(DATA_PATH):
        print("No data folder found")
        return documents

    for file in os.listdir(DATA_PATH):
        if file.endswith(".pdf"):
            path = os.path.join(DATA_PATH, file)
            loader = PyPDFLoader(path)
            documents.extend(loader.load())

    return documents

def split_docs(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    return splitter.split_documents(docs)

if __name__ == "__main__":
    docs = load_all_pdfs()

    if not docs:
        print("No PDFs found in data/")
        exit()

    chunks = split_docs(docs)

    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(chunks, embeddings)

    db.save_local("vectorstore")

    print("Vector database created from all PDFs")
