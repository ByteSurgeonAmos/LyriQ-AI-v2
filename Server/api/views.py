from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .serializers import UploadedDocumentSerializer
from transformers import pipeline
from .chroma_client import get_chroma_client

# Initialize an embedding model (e.g., from Hugging Face)
embedding_model = pipeline(
    'feature-extraction', model="sentence-transformers/all-MiniLM-L6-v2")


class DocumentUploadView(APIView):
    def post(self, request, *args, **kwargs):
        serializer = UploadedDocumentSerializer(data=request.data)
        if serializer.is_valid():
            document = serializer.save()
            # Handle file reading and embeddings
            with open(document.file.path, 'r') as file:
                content = file.read()
            embeddings = embedding_model(content)[0]

            # Store in ChromaDB
            chroma_client = get_chroma_client()
            collection = chroma_client.get_or_create_collection(
                name="documents")
            collection.add(
                embeddings=[embeddings],
                metadatas=[{"file_name": document.file.name}],
                ids=[str(document.id)]
            )
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class SearchDocumentsView(APIView):
    def post(self, request, *args, **kwargs):
        query = request.data.get('query')
        if not query:
            return Response({"error": "Query is required"}, status=status.HTTP_400_BAD_REQUEST)
        query_embedding = embedding_model(query)[0]
        chroma_client = get_chroma_client()
        collection = chroma_client.get_collection(name="documents")
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=5
        )
        return Response(results, status=status.HTTP_200_OK)
