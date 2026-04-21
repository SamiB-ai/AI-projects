def generate_summary(llm, db):
    docs = db.similarity_search("summary", k=5)
    context = "\n".join([doc.page_content for doc in docs])[:1500]

    prompt = f"What is this document about?\n\n{context}"

    return llm.invoke(prompt)