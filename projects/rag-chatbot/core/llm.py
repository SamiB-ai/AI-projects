import streamlit as st

@st.cache_resource
def load_llm():
    from langchain_ollama import OllamaLLM
    return OllamaLLM(model="gemma3:12b")