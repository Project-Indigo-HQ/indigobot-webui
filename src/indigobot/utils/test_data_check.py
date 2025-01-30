from langchain_chroma import Chroma

from indigobot.config import RAG_DIR, vectorstore
import numpy as np

def quary_chroma(text : str):
    """
    Queries the Chroma database for a given text.

    :param text: Text to query in Chroma database
    :type text: str
    :return: Chroma response for the given text
    :rtype: dict
    """
    retriever = vectorstore.as_retriever(
        search_type="mmr", search_kwargs={"k": 1, "fetch_k": 5}
    )
    retriever.invoke(text, filter={"source": "news"})

def check_duplicate(vectorstore, new_item_vectore, similarity_threshold=0.9):
    """
    Checks if a new item is a duplicate of any item in the vectorstore.

    :param vectorstore: Vectorstore to check for duplicates
    :type vectorstore: Vectorstore
    :param new_item_vectore: Vector of the new item to check for duplicates
    :type new_item_vectore: np.array
    :param similarity_threshold: Threshold for similarity between vectors
    :type similarity_threshold: float
    :return: True if the new item is a duplicate, False otherwise
    :rtype: bool
    """
    #FIXME: Chorma is not iterable
    for item in vectorstore:
        similarity = np.dot(item["vector"], new_item_vectore)
        if similarity > similarity_threshold:
            return True
    return False

if __name__ == "__main__":
    chroma = Chroma()

    # Test data for checking duplicates-----------------------
    #existing_item_vector = np.array([0.1, 0.2, 0.3])  # Example vector representation
    #existing_item_metadata = {"source": "news", "title": "Housing Market Update"}
    
    # Add the existing item to the vector store for testing
    #vectorstore.add_item(existing_item_vector, existing_item_metadata)
    
    # New item that is exactly the same as the existing item
    new_item_vector = np.array([0.1, 0.2, 0.3])  # Same vector as existing item
    
    # Check for duplicates
    is_duplicate = check_duplicate(vectorstore, new_item_vector)
    print("Is duplicate:", is_duplicate)
    #---------------------------------------------------------

    results = quary_chroma("Housing")
    print(results)

