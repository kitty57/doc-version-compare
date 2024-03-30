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

def perform_question_answering(uploaded_files, question):
    if uploaded_files:
        directory = "uploaded_documents"
        os.makedirs(directory, exist_ok=True)
        for i, uploaded_file in enumerate(uploaded_files):
            with open(os.path.join(directory, f"document_{i}.pdf"), "wb") as f:
                f.write(uploaded_file.getbuffer())

        llm = GradientBaseModelLLM(
            base_model_slug="llama2-7b-chat",
            max_tokens=400,
        )
        embed_model = GradientEmbedding(
            gradient_access_token='VqgYGFvkpiYCc00NlKoCUDPxFGTQPYwN',
            gradient_workspace_id='01cded86-e9ad-481e-8809-dc29d22725cd_workspace',
            gradient_model_slug="bge-large",
        )
        service_context = ServiceContext.from_defaults(
            llm=llm,
            embed_model=embed_model,
            chunk_size=256,
        )
        set_global_service_context(service_context)

        documents_reader = SimpleDirectoryReader(directory).load_data()
        vector_store_index = VectorStoreIndex.from_documents(documents_reader, service_context=service_context)
        query_engine = vector_store_index.as_query_engine()

        response = query_engine.query(question)

        return response
        
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
    question = "Explain the changes introduced in the new version of the document."

    if uploaded_files:
        with st.spinner("Comparing documents..."):
            response = perform_question_answering(uploaded_files, question)
            if response:
                wrapped_text = textwrap.fill(response.response, width=70)
                st.text("Bot: " + wrapped_text) 
                question = ""  
            else:
                st.text("Bot: Sorry, I couldn't find an answer.")

if __name__ == "__main__":
    main()
