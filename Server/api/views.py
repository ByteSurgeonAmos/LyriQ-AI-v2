from transformers import pipeline, AutoTokenizer, AutoModel, AutoModelForCausalLM
from typing import List, Dict, Any, Optional, Union
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.parsers import MultiPartParser, FormParser, JSONParser
from django.core.files.base import ContentFile
from django.db import transaction
from django.utils import timezone
from .models import UploadedDocument, ChatSession, ChatMessage
from .serializers import UploadedDocumentSerializer, ChatSessionSerializer
from .document_processor import DocumentProcessor
from .chroma_client import get_chroma_client, get_or_create_collection, get_embedding_model
import uuid
from transformers import pipeline, AutoTokenizer
import torch
from .chat_model.chat_model import ChatModel
from uuid import uuid4


import logging
from pathlib import Path
from django.shortcuts import get_object_or_404
import io
import re
from functools import lru_cache

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentUploadView(APIView):
    parser_classes = (MultiPartParser, FormParser, JSONParser)

    def post(self, request, *args, **kwargs):
        try:
            # Handle direct text input
            if 'text' in request.data:
                content = request.data['text']
                title = request.data.get('title', 'Direct Text Input')

                # Create document record
                document = UploadedDocument.objects.create(
                    title=title,
                    content=content
                )

            # Handle file upload
            elif 'file' in request.FILES:
                serializer = UploadedDocumentSerializer(data=request.data)
                if not serializer.is_valid():
                    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

                document = serializer.save()
                content = document.file.read()
                if isinstance(content, bytes):
                    content = content.decode('utf-8')

            else:
                return Response({
                    "error": "No content provided. Please provide either 'text' or 'file'."
                }, status=status.HTTP_400_BAD_REQUEST)

            # Process the content
            with transaction.atomic():
                processor = DocumentProcessor()
                result = processor.process_document(content)

                # Initialize ChromaDB
                chroma_client = get_chroma_client()
                collection = get_or_create_collection(chroma_client)

                # Store chunks with embeddings
                for i, (chunk, embedding) in enumerate(zip(result['chunks'], result['embeddings'])):
                    collection.add(
                        embeddings=[embedding],
                        documents=[chunk],
                        ids=[f"{document.id}-chunk-{i}"],
                        metadatas=[{
                            "document_id": str(document.id),
                            "chunk_index": i,
                            "sentiment": result['detailed_sentiments'][i]
                        }]
                    )

                # Update document
                document.content = content
                document.processed = True
                document.language = result.get('language', 'en')
                document.average_sentiment = result['sentiment']
                document.save()

                return Response({
                    "message": "Content processed successfully",
                    "document_id": document.id,
                    "sentiment": result['sentiment'],
                    "language": result.get('language', 'en'),
                    "chunk_count": len(result['chunks'])
                }, status=status.HTTP_201_CREATED)

        except Exception as e:
            logger.error(f"Error processing content: {str(e)}", exc_info=True)
            if 'document' in locals():
                document.delete()  # Cleanup if document was created
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


logger = logging.getLogger(__name__)


logger = logging.getLogger(__name__)


class ChatView(APIView):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.chat_model = ChatModel()

    def post(self, request, *args, **kwargs):
        try:
            # Extract request data
            session_id = request.data.get('session_id')
            message = request.data.get('message')
            document_id = request.data.get('document_id')

            if not message:
                return Response({"error": "Message is required"}, status=status.HTTP_400_BAD_REQUEST)

            # Get or create chat session
            session = self.get_or_create_session(session_id, document_id)
            document = session.current_document

            if not document:
                return Response({"error": "No document selected"}, status=status.HTTP_400_BAD_REQUEST)

            # Generate response using the ChatModel
            context = self.format_context(document.content, document.language)
            response = self.chat_model.generate_response(context, message)

            # Analyze sentiments
            user_sentiment = self.chat_model.sentiment_analyzer(message)
            response_sentiment = self.chat_model.sentiment_analyzer(response)

            # Ensure score fields are floats
            user_sentiment_score = float(user_sentiment.get('score', 0)) if isinstance(
                user_sentiment.get('score'), str) else user_sentiment.get('score', 0)
            response_sentiment_score = float(response_sentiment.get('score', 0)) if isinstance(
                response_sentiment.get('score'), str) else response_sentiment.get('score', 0)

            # Analyze emotions (if needed)
            user_emotions = self.chat_model.analyze_emotions(message)
            response_emotions = self.chat_model.analyze_emotions(response)

            # Create chat message
            ChatMessage.objects.create(
                session=session,
                content=response,
                is_user=False,
                sentiment_score=response_sentiment_score,
                relevant_document=document,
                user_sentiment=user_sentiment,
                response_sentiment=response_sentiment_score,
                user_emotions=user_emotions,
                response_emotions=response_emotions
            )

            return Response({
                "session_id": session.session_id,
                "response": response,
                "user_sentiment": user_sentiment,
                "response_sentiment": response_sentiment_score,
                "user_emotions": user_emotions,
                "response_emotions": response_emotions,
                "document_id": document.id
            })

        except Exception as e:
            logger.error(f"Error in chat processing: {str(e)}")
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    def get_or_create_session(self, session_id, document_id=None):
        """Get existing chat session or create a new one."""
        if session_id:
            session = ChatSession.objects.filter(session_id=session_id).first()
            if session:
                if document_id:
                    document = get_object_or_404(
                        UploadedDocument, id=document_id)
                    session.current_document = document
                    session.save()
                return session

        # Create new session
        document = get_object_or_404(
            UploadedDocument, id=document_id) if document_id else None
        session = ChatSession.objects.create(
            session_id=session_id or str(uuid.uuid4()),
            current_document=document
        )
        return session

    def format_context(self, content, language='en'):
        """Format the document content into labeled sections."""
        def detect_section_type(text, index, total):
            if "chorus" in text.lower() or "refrain" in text.lower():
                return "Chorus"
            return f"Verse {index + 1}"

        formatted_sections = []
        chunks = self.split_content(content)
        for i, chunk in enumerate(chunks):
            section_type = detect_section_type(chunk, i, len(chunks))
            formatted_sections.append(f"{section_type}:\n{chunk.strip()}")
        return "\n\n".join(formatted_sections)

    def split_content(self, content, max_length=500):
        """Split the content into smaller chunks."""
        chunks = []
        for paragraph in content.split('\n'):
            if len(paragraph) <= max_length:
                chunks.append(paragraph)
            else:
                # Split the paragraph into smaller chunks
                while len(paragraph) > max_length:
                    # Find the nearest sentence boundary
                    split_idx = paragraph[:max_length].rfind('.')
                    if split_idx == -1:
                        split_idx = max_length
                    chunk = paragraph[:split_idx + 1]
                    chunks.append(chunk)
                    paragraph = paragraph[split_idx + 1:].strip()
                if paragraph:
                    chunks.append(paragraph)
        return chunks


class ChatHistoryView(APIView):
    """Handle chat history operations"""

    def get(self, request, session_id=None):
        """Get chat history for a session or list all sessions"""
        try:
            if session_id:
                session = ChatSession.objects.get(session_id=session_id)

                # Get document info
                document_info = None
                if session.current_document:
                    document_info = {
                        'id': session.current_document.id,
                        'title': session.current_document.title or session.current_document.file.name,
                        'uploaded_at': session.current_document.uploaded_at,
                        'language': session.current_document.language,
                        'average_sentiment': session.current_document.average_sentiment
                    }

                # Get messages with pagination
                messages = ChatMessage.objects.filter(
                    session=session).order_by('timestamp')
                messages_data = [{
                    'content': msg.content,
                    'is_user': msg.is_user,
                    'sentiment_score': msg.sentiment_score,
                    'timestamp': msg.timestamp,
                    'relevant_document_id': msg.relevant_document.id if msg.relevant_document else None
                } for msg in messages]

                return Response({
                    'session_id': session.session_id,
                    'created_at': session.created_at,
                    'last_interaction': session.last_interaction,
                    'current_document': document_info,
                    'messages': messages_data,
                    'message_count': len(messages_data)
                })

            else:
                # Return list of all chat sessions with pagination
                page = int(request.query_params.get('page', 1))
                page_size = int(request.query_params.get('page_size', 10))
                start = (page - 1) * page_size
                end = start + page_size

                sessions = ChatSession.objects.all().order_by(
                    '-last_interaction')[start:end]

                sessions_data = [{
                    'session_id': session.session_id,
                    'created_at': session.created_at,
                    'last_interaction': session.last_interaction,
                    'document': {
                        'title': session.current_document.title if session.current_document else None,
                        'id': session.current_document.id if session.current_document else None,
                        'language': session.current_document.language if session.current_document else None
                    },
                    'message_count': session.messages.count(),
                    'last_message': session.messages.order_by('-timestamp').first().content if session.messages.exists() else None
                } for session in sessions]

                total_sessions = ChatSession.objects.count()

                return Response({
                    'sessions': sessions_data,
                    'total': total_sessions,
                    'page': page,
                    'page_size': page_size,
                    'total_pages': (total_sessions + page_size - 1) // page_size
                })

        except ChatSession.DoesNotExist:
            return Response(
                {"error": "Session not found"},
                status=status.HTTP_404_NOT_FOUND
            )
        except Exception as e:
            logger.error(
                f"Error retrieving chat history: {str(e)}", exc_info=True)
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    def delete(self, request, session_id):
        """Delete a chat session and its associated messages"""
        try:
            with transaction.atomic():
                session = ChatSession.objects.get(session_id=session_id)
                # Delete all associated messages first
                ChatMessage.objects.filter(session=session).delete()
                # Delete the session
                session.delete()
                return Response(
                    {"message": "Chat session and associated messages deleted successfully"},
                    status=status.HTTP_200_OK
                )
        except ChatSession.DoesNotExist:
            return Response(
                {"error": "Session not found"},
                status=status.HTTP_404_NOT_FOUND
            )
        except Exception as e:
            logger.error(
                f"Error deleting chat session: {str(e)}", exc_info=True)
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    def patch(self, request, session_id):
        """Update chat session properties"""
        try:
            session = ChatSession.objects.get(session_id=session_id)

            # Update allowed fields
            if 'title' in request.data:
                session.title = request.data['title']

            if 'document_id' in request.data:
                try:
                    document = UploadedDocument.objects.get(
                        id=request.data['document_id'])
                    session.current_document = document
                except UploadedDocument.DoesNotExist:
                    return Response({"error": "Document not found"}, status=status.HTTP_404_NOT_FOUND)

            session.save()

            return Response({
                "message": "Session updated successfully",
                "session_id": session.session_id,
                "title": session.title,
                "document_id": session.current_document.id if session.current_document else None
            })

        except ChatSession.DoesNotExist:
            return Response({"error": "Session not found"}, status=status.HTTP_404_NOT_FOUND)
        except Exception as e:
            logger.error(
                f"Error updating chat session: {str(e)}", exc_info=True)
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
