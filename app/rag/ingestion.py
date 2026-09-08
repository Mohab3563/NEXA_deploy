# Handles multi-format file uploads (PDF, DOCX, TXT).
import os
import shutil
from fastapi import UploadFile
from app.rag.chunking import load_and_split_document
from app.rag.vector_store import add_documents_to_store

UPLOAD_DIR = "./uploaded_files"
os.makedirs(UPLOAD_DIR, exist_ok=True)

async def process_file_upload(file: UploadFile):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    chunks = load_and_split_document(file_path)
    add_documents_to_store(chunks)
    return {"filename": file.filename, "total_chunks": len(chunks)}