from app.rag.vector_store import get_vector_store

def retrieve_context(query: str, k: int = 4, selected_document: str = "ALL"):
    store = get_vector_store()
    if store is None:
        return []
    
    # Return matches across all documents
    if selected_document == "ALL":
        return store.similarity_search(query, k=k)
    
    # Filter matches to a specific document only
    return store.similarity_search(
        query, 
        k=k, 
        filter={"source": selected_document}
    )