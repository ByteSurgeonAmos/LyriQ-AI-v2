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
        self.generator = pipeline(
            "text2text-generation",
            model="google/flan-t5-large",
            device=0 if torch.cuda.is_available() else -1,
        )

    def generate_response(self, context: str, question: str) -> str:
        prompt = f"""Context: {context}\n\nQuestion: {question}\n\nAnswer:"""
        response = self.generator(
            prompt,
            max_length=300,  # Increased max length
            min_length=50,   # Added minimum length
            num_return_sequences=1,
            temperature=0.7,  # Added temperature for more natural responses
            do_sample=True   # Enable sampling for more diverse responses
        )
        return response[0]['generated_text']


    def post(self, request, *args, **kwargs):
        session_id = request.data.get('session_id')
        message = request.data.get('message')
        document_id = request.data.get('document_id')

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

        # Handle document selection
        if document_id:
            try:
                document = UploadedDocument.objects.get(id=document_id)
                session.current_document = document
                session.save()
            except UploadedDocument.DoesNotExist:
                return Response({"error": "Document not found"}, status=status.HTTP_404_NOT_FOUND)
        else:
            if not session.current_document:
                return Response({"error": "Document ID is required for querying."}, status=status.HTTP_400_BAD_REQUEST)
            document = session.current_document

        # Process user message
        processor = DocumentProcessor()
        sentiment_result = processor.analyze_sentiment(message)

        # Save user message
        user_message = ChatMessage.objects.create(
            session=session,
            content=message,
            is_user=True,
            sentiment_score=sentiment_result['score']
        )

        try:
            # Get relevant context from ChromaDB for the current document
            query_embedding = processor.generate_embeddings(message)
            chroma_client = get_chroma_client()
            collection = get_or_create_collection(chroma_client)

            # Query with document-specific filter
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=3,
                where={"file_name": {"$eq": document.file.name}}  # Filter by current document
            )

            # Fallback if no results found for the current document
            if not results['documents'] or not results['documents'][0]:
                return Response({
                    "session_id": session_id,
                    "response": "I don't have any information from the current document to answer that question.",
                    "user_sentiment": sentiment_result,
                    "response_sentiment": 0.0
                })

            context = " ".join(results['documents'][0])
            response = self.generate_response(context, message)

            # Save bot response
            bot_message = ChatMessage.objects.create(
                session=session,
                content=response,
                is_user=False,
                sentiment_score=processor.analyze_sentiment(response)['score'],
                relevant_document=document
            )

            return Response({
                "session_id": session_id,
                "response": response,
                "user_sentiment": sentiment_result,
                "response_sentiment": bot_message.sentiment_score,
                "document_id": document.id  # Add document_id to response for clarity
            })

        except Exception as e:
            print(f"Error in chat processing: {str(e)}")  # Add logging for debugging
            return Response({
                "error": str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class ChatHistoryView(APIView):
    def get(self, request, session_id=None):
        if session_id:
            try:
                session = ChatSession.objects.get(session_id=session_id)
                # Get associated document information
                document_info = None
                if session.current_document:
                    document_info = {
                        'id': session.current_document.id,
                        'title': session.current_document.title or session.current_document.file.name,
                        'uploaded_at': session.current_document.uploaded_at
                    }

                # Get all messages for this session
                messages = ChatMessage.objects.filter(
                    session=session).order_by('timestamp')
                messages_data = [{
                    'content': msg.content,
                    'is_user': msg.is_user,
                    'sentiment_score': msg.sentiment_score,
                    'timestamp': msg.timestamp
                } for msg in messages]

                return Response({
                    'session_id': session.session_id,
                    'created_at': session.created_at,
                    'last_interaction': session.last_interaction,
                    'current_document': document_info,
                    'messages': messages_data
                })
            except ChatSession.DoesNotExist:
                return Response(
                    {"error": "Session not found"},
                    status=status.HTTP_404_NOT_FOUND
                )
        else:
            # Return list of all chat sessions
            sessions = ChatSession.objects.all().order_by('-last_interaction')
            sessions_data = [{
                'session_id': session.session_id,
                'created_at': session.created_at,
                'last_interaction': session.last_interaction,
                'document': session.current_document.title if session.current_document else None,
                'message_count': session.messages.count()
            } for session in sessions]

            return Response(sessions_data)

    def delete(self, request, session_id):
        try:
            session = ChatSession.objects.get(session_id=session_id)
            session.delete()
            return Response(
                {"message": "Chat session deleted successfully"},
                status=status.HTTP_200_OK
            )
        except ChatSession.DoesNotExist:
            return Response(
                {"error": "Session not found"},
                status=status.HTTP_404_NOT_FOUND
            )
