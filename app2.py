
from dotenv import load_dotenv
import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

def main():
    load_dotenv()
    st.set_page_config(page_title="ASK YOUR PDF")
    st.header("Ask your PDF")

    pdf = st.file_uploader("Upload your PDF", type="pdf")
    text = ""
    document = None  # Initialize document variable

    if pdf is not None:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
                else:
                    text += "Page contains no text.\n"
            if not text:
                st.write("No text could be extracted from the uploaded PDF.")
        except PdfReadError as e:
            st.error("The PDF file appears to be corrupted or incomplete (EOF marker not found). Please check the file and try again.")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=200,
        chunk_overlap=100,
        length_function=len
    )
    chunks = text_splitter.split_text(text)

    if chunks:
        embeddings = OpenAIEmbeddings()
        document = FAISS.from_texts(chunks, embeddings)
        st.write(chunks)
    else:
        st.error("No text chunks available for embedding.")

    # Input Box
    user_question = st.text_input("Ask a Question")
    if user_question and document:
        docs = document.similarity_search(user_question)  # Pass user_question to similarity_search
        llm = OpenAI()
        chain = load_qa_chain(llm, chain_type="stuff")
        answer = chain.run(input_documents=docs, question = user_question)

        st.write(answer)

    
        

if __name__ == '__main__':
    main()