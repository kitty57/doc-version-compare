import streamlit as st
import asyncio
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core import ServiceContext, set_global_service_context
from llama_index.llms.gradient import GradientBaseModelLLM
from llama_index.embeddings.gradient import GradientEmbedding
import os
import textwrap

asyncio.set_event_loop(asyncio.new_event_loop())


async def perform_question_answering(uploaded_files, question):
    if not uploaded_files:
        return None

    try:
        # Create directory with proper error handling
        directory = "uploaded_documents"
        os.makedirs(directory, exist_ok=True)

        # Save files asynchronously (improves performance)
        tasks = [asyncio.create_task(save_file(os.path.join(directory, f"document_{i}.pdf"), uploaded_file))
                 for i, uploaded_file in enumerate(uploaded_files)]
        await asyncio.gather(*tasks)

        # Initialize models with proper error handling (wrap in try-except)
        try:
            llm = GradientBaseModelLLM(
                base_model_slug="llama2-7b-chat",
                max_tokens=400,
            )
            embed_model = GradientEmbedding(
                gradient_access_token=st.secrets["GRADIENT_ACCESS_TOKEN"],
                gradient_workspace_id=st.secrets["GRADIENT_WORKSPACE_ID"],
                gradient_model_slug="bge-large",
            )
        except Exception as e:
            print(f"Error initializing models: {e}")
            st.error("Error: Couldn't initialize the models. Check configurations.")
            return None

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
    except Exception as e:
        print(f"Error processing documents: {e}")
        st.error("Error: An unexpected error occurred during processing.")
        return None

async def save_file(filepath, uploaded_file):
    with open(filepath, "wb") as f:
        await f.write(uploaded_file.getbuffer())


def main():
    st.set_page_config(page_title="Document Q&A Chatbot", page_icon="", layout="wide", initial_sidebar_state="collapsed", menu_items={"Get Help": None, "Report a Bug": None})

    st.title("Upload PDF Documents")

    col1, col2 = st.columns(2)

    with col1:
        uploaded_files_1 = st.file_uploader("Upload first version", accept_multiple_files=False, type=["pdf"])

    with col2:
        uploaded_files_2 = st.file_uploader("Upload second version", accept_multiple_files=False, type=["pdf"])

    question = "I've given 2 documents doc1 and doc2 that are 2 versions of the same product. Explain the new changes introduced in the new version of the document."

    if uploaded_files_1 is not None and uploaded_files_2 is not None:
        with st.spinner("Processing..."):
            uploaded_files = [uploaded_files_1, uploaded_files_2]
            response = asyncio.run(perform_question_answering(uploaded_files, question))
            if response:
                wrapped_text = textwrap.fill(response.response, width=70)
                st.text("Bot: " + wrapped_text)
            else:
                st.text("Bot: Sorry, I couldn't find an answer in the documents.")

