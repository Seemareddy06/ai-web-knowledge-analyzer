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


# =========================
# 🚀 LOAD + PROCESS URLS (ROBUST)
# =========================
def load_and_process_urls(urls):
    docs = []

    for url in urls:
        try:
            loader = WebBaseLoader(url)
            data = loader.load()

            if data:
                docs.extend(data)

        except Exception as e:
            print(f"Error loading {url}: {e}")

    if not docs:
        raise ValueError("No valid content could be loaded from URLs")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    split_docs = splitter.split_documents(docs)

    vectorstore = FAISS.from_documents(split_docs, embeddings)
    return vectorstore


# =========================
# 🔍 RETRIEVE DOCUMENTS
# =========================
def retrieve_docs(vectorstore, query):
    retriever = vectorstore.as_retriever()
    docs = retriever.get_relevant_documents(query)
    return docs


# =========================
# 🤖 LLM CALL
# =========================
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


# =========================
# ⚖️ NEW: GET CONTENT PER URL (FOR ACCURATE COMPARISON)
# =========================
def get_content_per_url(urls):
    url_contents = {}

    for url in urls:
        try:
            loader = WebBaseLoader(url)
            docs = loader.load()

            text = " ".join([doc.page_content for doc in docs])

            # Limit text for LLM (important)
            url_contents[url] = text[:3000]

        except Exception as e:
            url_contents[url] = f"Error loading content: {e}"

    return url_contents