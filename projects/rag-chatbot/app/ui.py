import streamlit as st

def load_css():
    st.markdown("""
    <style>

    /* 🌍 Global */
    .stApp {
        background-color: #f7f9fc;
    }

    /* 🧠 Text */
    html, body {
        color: #1f2937 !important;
    }

    /* 📂 Sidebar */
    section[data-testid="stSidebar"] {
        background-color: white;
        border-right: 1px solid #e5e7eb;
    }

    /* 🔘 Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #4f8cff, #6ea8fe);
        color: white;
        border-radius: 10px;
        border: none;
        padding: 10px;
        font-weight: 600;
    }

    /* 📤 Upload */
    .stFileUploader > div {
        background-color: white !important;
        border: 1px solid #e5e7eb !important;
        border-radius: 10px !important;
    }

    /* 💬 Chat */
    [data-testid="stChatMessage"] {
        background-color: white;
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 10px;
        border: 1px solid #e5e7eb;
    }

    /* 📌 Sources */
    .source-box {
        background-color: #ffffff;
        padding: 12px;
        border-radius: 10px;
        border: 1px solid #e5e7eb;
        margin-bottom: 10px;
    }

    /* 🧾 Input */
    input {
        border-radius: 10px !important;
        border: 1px solid #e5e7eb !important;
        padding: 10px !important;
        background-color: white !important;
        color: black !important;
    }

    </style>
    """, unsafe_allow_html=True)