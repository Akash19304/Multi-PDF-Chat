import streamlit as st 
from dotenv import load_dotenv
import os



def main():
    load_dotenv()

    st.set_page_config(
        page_title="Multi-PDF-Chat",
        page_icon=":books:"
    )
    
    st.header("Multi-PDF-Chat")
    st.text_input("Ask a question from your PDFs")

    with st.sidebar:
        st.subheader("Your Documents")
        st.file_uploader(label="Upload your PDFs here")
        st.button("Upload")



if __name__ == "__main__":
    main()