import os
from dotenv import load_dotenv

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from groq import Groq

load_dotenv()

# Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Embeddings
embeddings = HuggingFaceEmbeddings()


def load_and_process_urls(urls):
    docs = []
    for url in urls:
        loader = WebBaseLoader(url)
        docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    split_docs = splitter.split_documents(docs)

    vectorstore = FAISS.from_documents(split_docs, embeddings)
    return vectorstore


def retrieve_docs(vectorstore, query):
    retriever = vectorstore.as_retriever()
    docs = retriever.get_relevant_documents(query)
    return docs


def ask_llm(context, question):
    prompt = f"""
    You are an AI assistant.

    Answer ONLY using the context below.
    If answer is not found, say "Answer not found".

    Context:
    {context}

    Question:
    {question}
    """

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content