import streamlit as st
import asyncio
import difflib

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from PyPDF2 import PdfReader

asyncio.set_event_loop(asyncio.new_event_loop())

async def extract_text_async(pdf_file):
    with open(pdf_file, "rb") as pdf_reader:
      pdf = PdfReader(pdf_reader)
      text = ""
      for page in pdf.pages:
        text += page.extract_text()
    return text

def generate_summary(text):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LsaSummarizer()
    summary = summarizer(parser.document, sentences_count=3)  # Adjust the number of sentences as needed
    return " ".join(str(sentence) for sentence in summary)

def main():
    st.set_page_config(
        page_title="Document Comparison",
        page_icon="",
        layout="wide",
        initial_sidebar_state="collapsed",
        menu_items={"Get Help": None, "Report a Bug": None},
    )

    st.title("Upload PDF Documents")

    col1, col2 = st.columns(2)

    with col1:
        uploaded_file_1 = st.file_uploader("Upload first version", accept_multiple_files=False, type=["pdf"])

    with col2:
        uploaded_file_2 = st.file_uploader("Upload second version", accept_multiple_files=False, type=["pdf"])

    if uploaded_file_1 is not None and uploaded_file_2 is not None:
        with st.spinner("Processing..."):
            text_1 = asyncio.run(extract_text_async(uploaded_file_1.name))
            text_2 = asyncio.run(extract_text_async(uploaded_file_2.name))

            # Perform document comparison
            differences = list(difflib.unified_diff(text_1.splitlines(), text_2.splitlines(), lineterm=""))

            # Generate summary of differences
            summary = generate_summary("\n".join(differences))

            # Display summary
            if summary:
                st.subheader("Summary of Changes:")
                st.write(summary)
            else:
                st.write("No significant changes found.")

if __name__ == "__main__":
    main()
