import chromadb
from chromadb.config import Settings


def get_chroma_client():
    client = chromadb.Client(Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory="./chroma_db"  # Directory where Chroma will store data
    ))
    return client
