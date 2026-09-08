from app.rag.vector_store import get_vector_store

def retrieve_context(query: str, user_id: str = "default_user", k: int = 4, selected_document: str = "ALL"):
    # Pass the user_id to load the correct user's vector index
    store = get_vector_store(user_id=user_id)
    if store is None:
        return []
    
    # Return matches across all documents for this user
    if selected_document == "ALL":
        return store.similarity_search(query, k=k)
    
    # Filter matches to a specific document only within this user's index
    return store.similarity_search(
        query, 
        k=k, 
        filter={"source": selected_document}
    )