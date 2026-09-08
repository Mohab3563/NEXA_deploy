import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

# Uses local CPU embedding model—bypasses all Google API rate limits and quotas
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)

_vector_store = None

def add_documents_to_store(documents):
    global _vector_store
    
    # Process documents directly into FAISS locally without API calls or sleeps
    if _vector_store is None:
        _vector_store = FAISS.from_documents(documents, embeddings)
    else:
        _vector_store.add_documents(documents)
        
    return _vector_store

def get_vector_store():
    return _vector_store