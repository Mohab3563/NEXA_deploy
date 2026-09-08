import json
import os
import shutil
import traceback

from fastapi import FastAPI, File, HTTPException, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

from app.rag.ingestion import process_file_upload
from app.rag.retriever import retrieve_context

load_dotenv()

if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY is missing! Check your .env file.")

app = FastAPI(title="Nexa RAG Backend API")

# Configure CORS Middleware for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Robust path resolution for JSON profiles
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KNOWLEDGE_DIR = os.path.join(BASE_DIR, "app", "knowledge")

# Fallback path if knowledge directory sits directly in /app
if not os.path.exists(KNOWLEDGE_DIR):
    KNOWLEDGE_DIR = os.path.join(BASE_DIR, "knowledge")

with open(os.path.join(KNOWLEDGE_DIR, "assistant_profile.json"), "r", encoding="utf-8") as f:
    assistant_profile = json.load(f)

with open(os.path.join(KNOWLEDGE_DIR, "developer_profile.json"), "r", encoding="utf-8") as f:
    developer_profile = json.load(f)

# Format JSON string representations safely for LangChain prompt parsing
assistant_json_str = json.dumps(assistant_profile, indent=2).replace("{", "{{").replace("}", "}}")
developer_json_str = json.dumps(developer_profile, indent=2).replace("{", "{{").replace("}", "}}")

# System Prompt Definition
SYSTEM_PROMPT = f"""
You are {assistant_profile['name']}, a warm, helpful, empathetic, and human-like personal knowledge AI assistant.
Creator: {developer_profile['name']} ({developer_profile['professional_title'][0]})

Instructions:
1. Speak conversationally as a friendly study buddy or mentor. Avoid sounding like a rigid robot.
2. Answer the student's question clearly using the context provided from uploaded documents or built-in profile knowledge.
3. Explicitly reference which file or page you retrieved the answer from when relying on context.
4. If the information isn't present in the uploaded docs, kindly mention it while offering whatever assistance you can.

System Metadata Context:
Assistant Profile:
{assistant_json_str}

Developer Profile:
{developer_json_str}
"""

# Initialize Gemini Model
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")

# Create Prompt Template
prompt_template = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "Uploaded Documents Context:\n{context}\n\nStudent Question: {query}")
])

class QueryRequest(BaseModel):
    query: str
    user_id: str = "default_user"  # Session/User isolation identifier
    mode: str = "ask"
    selected_document: str = "ALL"

def get_user_vector_dir(user_id: str):
    """Helper to resolve and create user-specific vector directories."""
    clean_user_id = "".join(c for c in user_id if c.isalnum() or c in ("_", "-")) or "default_user"
    user_dir = os.path.join(BASE_DIR, "app", "knowledge", "vectors", clean_user_id)
    os.makedirs(user_dir, exist_ok=True)
    return user_dir

@app.get("/")
def read_root():
    return {"message": "Nexa RAG API is live!"}

@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    user_id: str = Form("default_user")
):
    try:
        # 1. Enforce File Size Limit (Max 10 MB per file)
        contents = await file.read()
        file_size_mb = len(contents) / (1024 * 1024)
        MAX_FILE_SIZE_MB = 10
        
        if file_size_mb > MAX_FILE_SIZE_MB:
            raise HTTPException(
                status_code=413, 
                detail=f"File too large ({file_size_mb:.1f}MB). Maximum allowed size is {MAX_FILE_SIZE_MB}MB."
            )
        
        # Reset file pointer after reading contents
        await file.seek(0)

        # 2. Enforce File Count Limit per User (Max 5 files stored)
        user_dir = get_user_vector_dir(user_id)
        existing_items = os.listdir(user_dir)
        MAX_FILES_PER_USER = 5
        
        if len(existing_items) >= MAX_FILES_PER_USER:
            raise HTTPException(
                status_code=400, 
                detail=f"Storage limit reached. You can only store up to {MAX_FILES_PER_USER} documents."
            )

        # 3. Process upload targeting the specific user directory
        res = await process_file_upload(file, user_id=user_id)
        
        return {
            "status": "success",
            "file": res["filename"],
            "chunks": res["total_chunks"]
        }
    except HTTPException as he:
        raise he
    except Exception as e:
        print("\n--- ERROR IN UPLOAD ENDPOINT ---")
        traceback.print_exc()
        print("--------------------------------\n")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
def query_rag(request: QueryRequest):
    try:
        # 1. Retrieve vector context isolated to the specific user's directory
        retrieved_docs = retrieve_context(request.query, user_id=request.user_id, k=4)
        
        # 2. Extract content & reference metadata
        context_str = ""
        references = []
        
        if retrieved_docs:
            for idx, doc in enumerate(retrieved_docs, start=1):
                source_file = doc.metadata.get("source", "Uploaded Document")
                page_num = doc.metadata.get("page", None)
                
                ref = f"{source_file} (Page {page_num + 1})" if page_num is not None else source_file
                references.append({"doc_id": idx, "source": ref})
                
                context_str += f"\n--- Context Source [{idx}]: {ref} ---\n{doc.page_content}\n"
        else:
            context_str = "No document context available."

        # 3. Invoke LLM Chain
        chain = prompt_template | llm
        response = chain.invoke({"context": context_str, "query": request.query})

        # 4. Extract Text String Cleanly
        if isinstance(response.content, list):
            answer_text = "".join([part.get("text", "") for part in response.content if isinstance(part, dict)])
        else:
            answer_text = str(response.content)

        return {
            "answer": answer_text,
            "references": references,
            "mode": request.mode,
            "selected_document": request.selected_document
        }
    except Exception as e:
        print("\n--- ERROR IN QUERY ENDPOINT ---")
        traceback.print_exc()
        print("-------------------------------\n")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/cleanup/{user_id}")
def cleanup_user_data(user_id: str):
    """Endpoint triggered when the user closes the website to wipe their data."""
    clean_user_id = "".join(c for c in user_id if c.isalnum() or c in ("_", "-"))
    if not clean_user_id:
        raise HTTPException(status_code=400, detail="Invalid user ID")
        
    user_vector_dir = os.path.join(BASE_DIR, "app", "knowledge", "vectors", clean_user_id)
    user_upload_dir = os.path.join(BASE_DIR, "uploaded_files", clean_user_id)
    
    if os.path.exists(user_vector_dir):
        shutil.rmtree(user_vector_dir)
    if os.path.exists(user_upload_dir):
        shutil.rmtree(user_upload_dir)
        
    return {"status": "success", "message": f"Data wiped for {clean_user_id}"}