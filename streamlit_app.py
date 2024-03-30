import streamlit as st
import asyncio
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core import ServiceContext, set_global_service_context
from llama_index.llms.gradient import GradientBaseModelLLM
from llama_index.embeddings.gradient import GradientEmbedding
import os
import textwrap 
import difflib

asyncio.set_event_loop(asyncio.new_event_loop())

def perform_document_comparison(uploaded_files):
    if len(uploaded_files) != 2:
        st.error("Please upload exactly two PDF files.")
        return

    directory = "uploaded_documents"
    os.makedirs(directory, exist_ok=True)

    filenames = []
    for i, uploaded_file in enumerate(uploaded_files):
        filename = f"document_{i+1}.pdf"
        filenames.append(filename)
        with open(os.path.join(directory, filename), "wb") as f:
            f.write(uploaded_file.getbuffer())

    texts = []
    for filename in filenames:
        with open(os.path.join(directory, filename), "rb") as f:
            texts.append(f.read().decode("utf-8"))

    diff = '\n'.join(difflib.ndiff(texts[0].splitlines(), texts[1].splitlines()))
    return diff

def main():
    st.set_page_config(page_title="Document Q&A Chatbot", page_icon="🤖", layout="wide", initial_sidebar_state="collapsed", menu_items={"Get Help": None, "Report a Bug": None})
    
    st.title("Document Q&A Chatbot")

    page_bg_img = '''
    <style>
    body {
    background-image: url("https://example.com/background.jpg");
    background-size: cover;
    }
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)

    st.title("Upload PDF Documents")
    uploaded_files = st.file_uploader("Upload two PDF files", accept_multiple_files=True, type=["pdf"])

    if uploaded_files:
        with st.spinner("Comparing documents..."):
            # Perform document comparison
            diff = perform_document_comparison(uploaded_files)
            if diff:
                st.subheader("Differences between the two versions:")
                st.text(diff)
            else:
                st.warning("Please upload two PDF files to compare.")

if __name__ == "__main__":
    main()
