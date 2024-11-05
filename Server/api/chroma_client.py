import chromadb
from chromadb.config import Settings
import os
from sentence_transformers import SentenceTransformer


def get_chroma_client():
    # Create persist directory if it doesn't exist
    persist_directory = os.path.join(os.getcwd(), "chroma_db")
    os.makedirs(persist_directory, exist_ok=True)

    # Initialize the client with the new configuration
    client = chromadb.PersistentClient(
        path=persist_directory
    )

    return client


def get_embedding_model():
    """
    Returns a sentence transformer model that produces 384-dimensional embeddings
    compatible with ChromaDB's default configuration
    """
    return SentenceTransformer('all-MiniLM-L6-v2')  # produces 384-dimensional embeddings


def get_or_create_collection(client, name="documents"):
    try:
        # Try to get existing collection
        collection = client.get_collection(name=name)
    except ValueError:
        # Create new collection if it doesn't exist
        collection = client.create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
            embedding_dim=384  # Match the dimension of all-MiniLM-L6-v2 embeddings
        )
    return collection
