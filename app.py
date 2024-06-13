import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
import os
from langchain_groq import ChatGroq

load_dotenv()
os.environ['HUGGINGFACEHUB_API_TOKEN']=os.getenv('HUGGINGFACEHUB_API_TOKEN')
os.environ['GROQ_API_KEY']=os.getenv('GROQ_API_KEY')


def get_pdf_text(pdf_docs):
    """
    Retrieves text from a list of PDF documents.

    Args:
        pdf_docs (list): List of PDF documents to extract text from.

    Returns:
        str: Text extracted from the PDF documents.
    """
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    """
    Splits the given text into chunks of a specified size, with a specified overlap.
    
    Args:
        text (str): The text to be split into chunks.
        
    Returns:
        List[str]: A list of text chunks, each of which is a substring of the input text.
    """

    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    """
    Generate a vector store from a list of text chunks.

    Args:
        text_chunks (List[str]): A list of text chunks.

    Returns:
        vectorstore (FAISS): A vector store created from the text chunks using the HuggingFaceEmbeddings model.

    """

    embeddings = HuggingFaceEmbeddings(
        model_name="thenlper/gte-small",
        multi_process=True,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    """
    Generates a conversational retrieval chain from the provided vector store.

    Args:
        vectorstore (FAISS): A vector store used for retrieval.

    Returns:
        conversation_chain (ConversationalRetrievalChain): A conversational retrieval chain created from the provided vector store, language model, and memory.
    """

    llm = ChatGroq(
        temperature=0,
        model="llama3-70b-8192",
    )

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def handle_userinput(user_question):
    """
    Handles user input by initiating a conversation based on the user's question, updating the chat history in the session state, and displaying messages alternatively from user and bot templates.

    Args:
        user_question (str): The question entered by the user.

    Returns:
        None
    """

    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)


def main():
    """
    Renders the main page of the application and handles user input for chat with multiple PDFs.

    This function sets the page configuration for the application and writes the CSS code to the page.
    It initializes the session state variables for the conversation and chat history if they are not already present.
    It displays the header "Chat with multiple PDFs :books:" and prompts the user to enter a question about their documents.
    If a user question is provided, the function calls the `handle_userinput` function to handle the user input.
    The function also renders a sidebar with a subheader "Your documents" and a file uploader for uploading PDF documents.
    If the "Process" button is clicked, the function processes the uploaded PDF documents by extracting text, splitting the text into chunks, creating a vector store, and creating a conversation chain.
    The conversation chain is stored in the session state variable `conversation`.

    Parameters:
    None

    Returns:
    None
    """

    st.set_page_config(page_title="Chat with multiple PDFs",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)

                # create vector store
                vectorstore = get_vectorstore(text_chunks)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(
                    vectorstore)


if __name__ == '__main__':
    main()