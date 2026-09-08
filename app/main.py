import json
import os
import traceback

from fastapi import FastAPI, File, HTTPException, UploadFile
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
    mode: str = "ask"
    selected_document: str = "ALL"

@app.get("/")
def read_root():
    return {"message": "Nexa RAG API is live!"}

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    try:
        res = await process_file_upload(file)
        return {
            "status": "success",
            "file": res["filename"],
            "chunks": res["total_chunks"]
        }
    except Exception as e:
        print("\n--- ERROR IN UPLOAD ENDPOINT ---")
        traceback.print_exc()
        print("--------------------------------\n")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
def query_rag(request: QueryRequest):
    try:
        # 1. Retrieve vector context
        retrieved_docs = retrieve_context(request.query, k=4)
        
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