from langchain_classic.chains import RetrievalQA

def build_qa_chain(llm, db):
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=db.as_retriever(),
        return_source_documents=True
    )

def ask_question(chain, query):
    return chain.invoke({"query": query})