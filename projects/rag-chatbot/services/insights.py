def extract_insights(llm, db):
    docs = db.similarity_search("key points", k=5)
    context = "\n".join([doc.page_content for doc in docs])[:1500]

    prompt = f"Extract key insights:\n\n{context}"

    return llm.invoke(prompt)