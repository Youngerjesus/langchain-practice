from typing import Dict, Any, List, Optional, Union
from uuid import UUID

import streamlit as st
from dotenv import load_dotenv
import openai
from langchain.document_loaders import PyPDFLoader
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema.runnable import RunnableLambda
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.storage import LocalFileStore
from langchain.callbacks.base import BaseCallbackHandler
from langchain_core.outputs import LLMResult, GenerationChunk, ChatGenerationChunk


@st.cache_resource(show_spinner="Embedding file...")
def embed_file(file):
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)

    loader = PyPDFLoader(file_path)

    spliter = RecursiveCharacterTextSplitter()

    docs = spliter.split_documents(loader.load())

    embedder = OpenAIEmbeddings()

    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")

    cached_embedding = CacheBackedEmbeddings.from_bytes_store(embedder, cache_dir)

    vectorstore = Chroma.from_documents(docs, cached_embedding)

    retriever = vectorstore.as_retriever()

    return retriever

if "messages" not in st.session_state:
    st.session_state["messages"] = []

def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)

    if save:
        st.session_state["messages"].append({"message": message, "role": role})


def paint_history():
    for message in st.session_state["messages"]:
        send_message(message["message"], message["role"], save=False)

def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)

class ChatCallbackHandler(BaseCallbackHandler):

    message = ""

    def on_llm_start(
        self,
        *args,
        **kwargs: Any,
    ) -> Any:
        self.message_box = st.empty()

    def on_llm_end(
        self,
        *args,
        **kwargs: Any,
    ) -> Any:
        with st.sidebar:
            st.write("llm ended")
            st.session_state["messages"].append({"message": self.message, "role": "ai"})
    def on_llm_new_token(
        self,
        token,
        *args,
        **kwargs: Any,
    ) -> Any:
        self.message += token
        self.message_box.markdown(self.message)


load_dotenv()

llm = ChatOpenAI(
    temperature=0.1,
    streaming=True,
    callbacks=[
        ChatCallbackHandler()
    ]
)


st.title("DocumentGPT")

prompt_template = ChatPromptTemplate.from_messages([
    ("system", """
        Answer the question using Only the following context. If you don't know the answer just say you don't know. DON'T make anything up.
        Context: {context} 
    """),
    ("human", "{question}")
])

st.markdown(""" 
    Welcome! 
    
    Use this chatbot to ask questions to an AI about your files!
    
    Upload your files on the sidebar. 
""")

with st.sidebar:
    file = st.file_uploader("Upload a .txt .pdf of .docs file", type=["pdf", "txt", "docs"])

if file:
    retriever = embed_file(file)

    send_message("I'm ready! Ask away!", "ai", save=False)

    paint_history()

    message = st.chat_input("Ask anything about your file...")

    if message:
        send_message(message, "human")

        chain = {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough()
        } | prompt_template | llm

        with st.chat_message("ai"):
            response = chain.invoke(message)

else:
    st.session_state["messages"] = []