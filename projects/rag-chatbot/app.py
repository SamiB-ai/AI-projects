from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

import os

# 1. Load vector store
def load_db():
    embeddings = OpenAIEmbeddings()
    db = FAISS.load_local("vectorstore", embeddings)
    return db

# 2. Create chatbot chain
def create_qa_chain(db):
    llm = ChatOpenAI(model="gpt-3.5-turbo")

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=db.as_retriever()
    )

    return qa_chain

# 3. Chat loop
if __name__ == "__main__":
    if not os.path.exists("vectorstore"):
        print("Lance d'abord ingest.py")
        exit()

    db = load_db()
    qa = create_qa_chain(db)

    print("RAG Chatbot prêt ! (pose tes questions)")

    while True:
        query = input("\nToi: ")

        if query.lower() in ["exit", "quit"]:
            break

        result = qa.run(query)
        print("\nIA:", result)
