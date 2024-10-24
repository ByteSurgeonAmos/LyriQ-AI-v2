import chromadb
from chromadb.config import Settings
import os


def get_chroma_client():
    persist_directory = os.path.join(os.getcwd(), "chroma_db")

    # Ensure the directory exists
    os.makedirs(persist_directory, exist_ok=True)

    # Initialize the client with persistence
    client = chromadb.Client(Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory=persist_directory
    ))

    return client


def get_or_create_collection(client, name="documents"):
    try:
        collection = client.get_collection(name)
    except ValueError:
        collection = client.create_collection(name)
    return collection
