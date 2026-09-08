# Splits documents into manageable text chunks.
import os
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_and_split_document(file_path: str):
    ext = os.path.splitext(file_path)[-1].lower()
    
    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext in [".docx", ".doc"]:
        loader = Docx2txtLoader(file_path)
    elif ext in [".txt", ".md"]:
        loader = TextLoader(file_path, encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file format: {ext}")

    docs = loader.load()
    filename = os.path.basename(file_path)
    
    for doc in docs:
        doc.metadata["source"] = filename

    # Increased chunk size to reduce the total number of API embedding requests
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500, 
        chunk_overlap=200
    )
    return splitter.split_documents(docs)