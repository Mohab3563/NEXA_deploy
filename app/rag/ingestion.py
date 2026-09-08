# Handles multi-format file uploads (PDF, DOCX, TXT) with user isolation.
import os
import shutil
from fastapi import UploadFile
from app.rag.chunking import load_and_split_document
from app.rag.vector_store import add_documents_to_store

BASE_UPLOAD_DIR = "./uploaded_files"
os.makedirs(BASE_UPLOAD_DIR, exist_ok=True)

async def process_file_upload(file: UploadFile, user_id: str = "default_user"):
    # Sanitize user_id to prevent path traversal issues
    clean_user_id = "".join(c for c in user_id if c.isalnum() or c in ("_", "-")) or "default_user"
    
    # Create a user-specific folder for their uploaded files
    user_upload_dir = os.path.join(BASE_UPLOAD_DIR, clean_user_id)
    os.makedirs(user_upload_dir, exist_ok=True)
    
    file_path = os.path.join(user_upload_dir, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    chunks = load_and_split_document(file_path)
    
    # Pass user_id to the vector store function so it saves to the user's specific index
    add_documents_to_store(chunks, user_id=clean_user_id)
    
    return {"filename": file.filename, "total_chunks": len(chunks)}