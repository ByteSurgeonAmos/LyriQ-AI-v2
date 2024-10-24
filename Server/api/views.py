from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .models import UploadedDocument, ChatSession, ChatMessage
from .serializers import UploadedDocumentSerializer, ChatSessionSerializer
from .document_processor import DocumentProcessor
from .chroma_client import get_chroma_client, get_or_create_collection
import uuid
from transformers import pipeline
import torch


class DocumentUploadView(APIView):
    def post(self, request, *args, **kwargs):
        serializer = UploadedDocumentSerializer(data=request.data)
        if serializer.is_valid():
            document = serializer.save()

            try:
                # Read and process document
                with open(document.file.path, 'r', encoding='utf-8') as file:
                    content = file.read()

                processor = DocumentProcessor()
                result = processor.process_document(content)

                # Store in ChromaDB
                chroma_client = get_chroma_client()
                collection = get_or_create_collection(chroma_client)

                # Store each chunk with its embedding
                for i, (chunk, embedding) in enumerate(zip(result['chunks'], result['embeddings'])):
                    collection.add(
                        embeddings=[embedding],
                        metadatas=[{
                            "file_name": document.file.name,
                            "chunk_index": i,
                            "sentiment": result['detailed_sentiments'][i]
                        }],
                        documents=[chunk],
                        ids=[f"{document.id}-chunk-{i}"]
                    )

                # Update document model
                document.content = content
                document.processed = True
                document.language = result.get('language', 'en')
                document.average_sentiment = result['sentiment']
                document.save()

                return Response({
                    "message": "Document processed successfully",
                    "document_id": document.id,
                    "sentiment": result['sentiment'],
                    "language": result.get('language', 'en')
                }, status=status.HTTP_201_CREATED)

            except Exception as e:
                return Response({
                    "error": str(e)
                }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class ChatView(APIView):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize text generation pipeline
        self.generator = pipeline(
            "text2text-generation",
            model="google/flan-t5-base",
            device=0 if torch.cuda.is_available() else -1,
        )

    def generate_response(self, context: str, question: str) -> str:
        prompt = f"""Context: {context}\n\nQuestion: {question}\n\nAnswer:"""
        response = self.generator(
            prompt, max_length=150, num_return_sequences=1)
        return response[0]['generated_text']

    def post(self, request, *args, **kwargs):
        session_id = request.data.get('session_id')
        message = request.data.get('message')

        if not message:
            return Response({"error": "Message is required"}, status=status.HTTP_400_BAD_REQUEST)

        # Get or create session
        if not session_id:
            session_id = str(uuid.uuid4())
            session = ChatSession.objects.create(session_id=session_id)
        else:
            try:
                session = ChatSession.objects.get(session_id=session_id)
            except ChatSession.DoesNotExist:
                return Response({"error": "Invalid session ID"}, status=status.HTTP_404_NOT_FOUND)

        # Process user message
        processor = DocumentProcessor()

        # Analyze sentiment of user message
        sentiment_result = processor.analyze_sentiment(message)

        # Save user message
        user_message = ChatMessage.objects.create(
            session=session,
            content=message,
            is_user=True,
            sentiment_score=sentiment_result['score']
        )

        try:
            # Get relevant context from ChromaDB
            query_embedding = processor.generate_embeddings(message)
            chroma_client = get_chroma_client()
            collection = get_or_create_collection(chroma_client)

            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=3
            )

            # Generate response
            if results['documents']:
                context = " ".join(results['documents'][0])
                response = self.generate_response(context, message)
            else:
                response = "I don't have enough information to answer that question."

            # Save bot response
            bot_message = ChatMessage.objects.create(
                session=session,
                content=response,
                is_user=False,
                sentiment_score=processor.analyze_sentiment(response)['score']
            )

            return Response({
                "session_id": session_id,
                "response": response,
                "user_sentiment": sentiment_result,
                "response_sentiment": bot_message.sentiment_score
            })

        except Exception as e:
            return Response({
                "error": str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class ChatHistoryView(APIView):
    def get(self, request, session_id=None):
        if session_id:
            try:
                session = ChatSession.objects.get(session_id=session_id)
                serializer = ChatSessionSerializer(session)
                return Response(serializer.data)
            except ChatSession.DoesNotExist:
                return Response({"error": "Session not found"}, status=status.HTTP_404_NOT_FOUND)
        else:
            sessions = ChatSession.objects.all().order_by('-last_interaction')
            serializer = ChatSessionSerializer(sessions, many=True)
            return Response(serializer.data)
