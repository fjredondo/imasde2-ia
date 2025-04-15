import streamlit as st
import os
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

template = """
Eres un asistente especializado en procesar y responder preguntas sobre documentos PDF:
1. Analizar el contexto proporcionado en español 
2. Entender la pregunta en español 
3. Generar una respuesta clara y concisa en español 

Si no encuentras la respuesta en el contexto, simplemente indica que no tienes información suficiente.
Limita tu respuesta a tres oraciones máximo

Pregunta : {question}
Contexto : {context}
Respuesta (en español): 
"""
pdfs_directory = 'pdfs/'
db_directory = 'vectordb'

# Asegurar que el directorio de la base de datos existe
os.makedirs(db_directory, exist_ok=True)

embeddings = OllamaEmbeddings(model="deepseek-r1:14b")
vector_store = Chroma(persist_directory=db_directory, embedding_function=embeddings)
model = OllamaLLM(model="deepseek-r1:14b", temperature=0.0, max_tokens=512)

def upload_pdf(file):
    with open(pdfs_directory + file.name, "wb") as f:
        f.write(file.getbuffer())

def load_pdf(file_path):
    loader = PDFPlumberLoader(file_path)
    documents = loader.load()
    return documents

def split_text(documents):
    # Cargar documentos PDF
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=0,
        add_start_index=True
    )

    return text_splitter.split_documents(documents)

def index_docs(documents):
    vector_store.add_documents(documents)
    vector_store.persist()

def retrieve_docs(query):
    return vector_store.similarity_search(query)

def answer_question(question, documents):
    context = "\n\n".join([doc.page_content for doc in documents])
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model

    return chain.invoke({"question": question, "context": context})

uploaded_file = st.file_uploader(
    "Subir PDF",
    type=["pdf"],
    accept_multiple_files=False
)

if uploaded_file:
    upload_pdf(uploaded_file)
    documents = load_pdf(pdfs_directory + uploaded_file.name)
    chunked_documents = split_text(documents)
    index_docs(chunked_documents)

    question = st.chat_input("Escribe tu pregunta aquí...")

    if question:
        st.chat_message("user").write(question)
        related_documents = retrieve_docs(question)
        answer = answer_question(question, related_documents)
        st.chat_message("assistant").write(answer)
