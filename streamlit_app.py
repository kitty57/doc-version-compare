import streamlit as st
import difflib
import textwrap

def perform_document_comparison(doc1, doc2):
    content1 = doc1.read().decode('utf-8')
    content2 = doc2.read().decode('utf-8')

    differ = difflib.Differ()
    diff = list(differ.compare(content1.splitlines(), content2.splitlines()))
    diff_text = '\n'.join(diff)
    
    return diff_text

def main():
    st.set_page_config(page_title="Document Comparison", page_icon="📄", layout="wide", initial_sidebar_state="expanded", menu_items={"Get Help": None, "Report a Bug": None})
    
    st.title("Document Comparison")

    uploaded_file1 = st.file_uploader("Upload the first PDF document", type=["pdf"])
    uploaded_file2 = st.file_uploader("Upload the second PDF document", type=["pdf"])

    if st.button("Compare Documents"):
        if uploaded_file1 and uploaded_file2:
            with st.spinner("Comparing documents..."):
                diff_text = perform_document_comparison(uploaded_file1, uploaded_file2)
                if diff_text:
                    st.text("The new lines added in the second version:")
                    # Wrap the text to ensure it fits within the display width
                    wrapped_text = textwrap.fill(diff_text, width=70)
                    st.info(wrapped_text)
                else:
                    st.text("No differences found between the documents.")
        else:
            st.warning("Please upload two PDF documents.")

if __name__ == "__main__":
    main()
