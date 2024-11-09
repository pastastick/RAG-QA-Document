import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os
from dotenv import load_dotenv

load_dotenv()
os.environ['Hugging_Token'] = os.getenv('Hugging_Token')
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


# Set up Streamlit
st.title("Conversational RAG with PDF uploads and chat history")
st.write("Upload your PDF and chat about the content")

# Input the groq API KEY
api_key = st.text_input("Enter your Groq API Key:", type="password")

# check if groq api key is provided
if api_key:
    llm=ChatGroq(api_key=api_key, model="gemma2-9b-it")
    session_id = st.text_input("Session ID", value="default_session")

    # manage chat history
    if 'store' not in st.session_state:
        st.session_state.store={}

    # upload file pdf
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf", accept_multiple_files=False)

    if uploaded_file:
        documents = []
        tempPDF = "./temp.pdf"
        
        # Menulis konten file yang diunggah ke file sementara
        with open(tempPDF, 'wb') as f:
            f.write(uploaded_file.read())  # Gunakan read() untuk membaca konten file
            file_name = uploaded_file.name

        # Memuat dokumen PDF menggunakan PyPDFLoader
        loader = PyPDFLoader(tempPDF)
        docs = loader.load()
        documents.extend(docs)
    
    # split and create embeddings for the documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
        splits = text_splitter.split_documents(documents)
        vectorStore = Chroma.from_documents(documents=splits, embedding=embeddings)
        retriever = vectorStore.as_retriever()

        # Prompt
        contextual_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT aswer the question, "
            "just reformulate it id needed and otherwise return it as is"
        )

        prompt_template = ChatPromptTemplate.from_messages(
            [
                ("ai", contextual_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        # chat history
        history_aware_retriever = create_history_aware_retriever(llm, retriever, prompt_template)

        # answer question
        system_prompt = (
            "You are an assistant for question-answering tasks."
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that "
            "you don't know. Use six sentences maximum and keep "
            "the answer concise"
            "\n\n"
            "{context}"
        )

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("ai", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        # handle session history
        def get_session_history(session:str)->BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id]=ChatMessageHistory()
            return st.session_state.store[session_id]

        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain, get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

        user_input = st.text_input("Your Question:")
        if user_input:
            session_history=get_session_history(session_id)
            response = conversational_rag_chain.invoke(
                {"input": user_input},
                config={
                    "configurable": {"session_id":session_id}
                },
            )
            st.write(st.session_state.store)
            st.write("Assistant:", response['answer'])
            st.write("Chat History:", session_history.messages)
else:
    st.warning("Please enter the groq API KEY")

