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

def _get_user_dir(user_id: str) -> str:
    """Resolves and creates the isolated directory path for a specific user's vectors."""
    clean_user_id = "".join(c for c in user_id if c.isalnum() or c in ("_", "-")) or "default_user"
    
    # Resolves project root dynamically (assuming file is in app/rag/)
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    user_dir = os.path.join(base_dir, "app", "knowledge", "vectors", clean_user_id)
    os.makedirs(user_dir, exist_ok=True)
    return user_dir

def add_documents_to_store(documents, user_id: str = "default_user"):
    user_dir = _get_user_dir(user_id)
    index_file = os.path.join(user_dir, "index.faiss")
    
    # If the user already has an existing index on disk, load it and add new docs
    if os.path.exists(index_file):
        vector_store = FAISS.load_local(user_dir, embeddings, allow_dangerous_deserialization=True)
        vector_store.add_documents(documents)
    else:
        # Otherwise, create a brand-new vector store index for this user
        vector_store = FAISS.from_documents(documents, embeddings)
        
    # Save the updated index back to the user's isolated folder
    vector_store.save_local(user_dir)
    return vector_store

def get_vector_store(user_id: str = "default_user"):
    user_dir = _get_user_dir(user_id)
    index_file = os.path.join(user_dir, "index.faiss")
    
    # Load the user's specific store if it exists
    if os.path.exists(index_file):
        return FAISS.load_local(user_dir, embeddings, allow_dangerous_deserialization=True)
        
    return None