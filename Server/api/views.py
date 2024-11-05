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


class ChatView(APIView):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = 0 if torch.cuda.is_available() else -1

        try:
            # Initialize embedding model
            self.embedding_model = get_embedding_model()

            # Initialize text generation pipeline
            self.generator = pipeline(
                "text2text-generation",
                model="facebook/bart-large-cnn",
                tokenizer="facebook/bart-large-cnn",
                device=self.device,
                model_kwargs={
                    "temperature": 0.7,
                    "top_k": 50,
                    "repetition_penalty": 1.2,
                    "length_penalty": 1.0,
                    "early_stopping": True
                }
            )

            # Initialize sentiment analysis pipeline
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="nlptown/bert-base-multilingual-uncased-sentiment",
                device=self.device
            )

            logger.info("Successfully initialized models for chat processing.")
        except Exception as e:
            logger.error(f"Error initializing models: {str(e)}")
            raise

        self._response_cache = {}
        self.supported_languages = {'en', 'de'}

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
        if document_id:
            document = get_object_or_404(UploadedDocument, id=document_id)
        else:
            document = None

        session = ChatSession.objects.create(
            session_id=session_id or str(uuid.uuid4()),
            current_document=document
        )
        return session

    def get_embeddings(self, text):
        """Generate embeddings for input text using SentenceTransformer."""
        try:
            # Convert text to embedding using the sentence transformer model
            embedding = self.embedding_model.encode(
                text, convert_to_tensor=True)
            # Move to CPU and convert to numpy array
            return embedding.cpu().numpy()
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise

    def get_analysis_prompt(self, question_type: str) -> str:
        """Generate a prompt to guide the analysis based on question type."""
        prompts = {
            'sentiment': "Analyze the emotional content in the provided lyrics based on specific evidence.",
            'theme': "Identify themes in the provided lyrics using direct quotes.",
            'structure': "Analyze the structure of the provided lyrics and describe any patterns.",
            'general': "Provide a general analysis of the provided lyrics."
        }
        return prompts.get(question_type, prompts['general'])

    def determine_question_type(self, question: str) -> str:
        """Identify question type based on keywords."""
        question = question.lower()
        patterns = {
            'sentiment': r'\b(emotion|feel|sentiment|mood|tone)\b',
            'theme': r'\b(theme|meaning|message|subject)\b',
            'structure': r'\b(structure|pattern|organize)\b'
        }
        for qtype, pattern in patterns.items():
            if re.search(pattern, question):
                return qtype
        return 'general'

    def format_context(self, chunks, metadata, language='en'):
        """Format document chunks into labeled sections."""
        def detect_section_type(text, index, total):
            if "chorus" in text.lower() or "refrain" in text.lower():
                return "Chorus"
            return f"Verse {index + 1}"

        formatted_sections = []
        for i, chunk in enumerate(chunks):
            section_type = detect_section_type(chunk, i, len(chunks))
            formatted_sections.append(f"{section_type}:\n{chunk.strip()}")
        return "\n\n".join(formatted_sections)

    def generate_response(self, context, question, document_language='en'):
        """Generate response using BART model."""
        try:
            # Check cache first
            cache_key = f"{hash(context)}-{hash(question)}-{document_language}"
            if cache_key in self._response_cache:
                return self._response_cache[cache_key]

            # Determine question type and get appropriate prompt
            question_type = self.determine_question_type(question)
            analysis_prompt = self.get_analysis_prompt(question_type)

            # Construct full prompt
            full_prompt = f"Question: {question}\n\nContent:\n{context}\n\nAnalysis:\n{analysis_prompt}"

            # Generate response
            response = self.generator(
                full_prompt,
                max_length=750,
                min_length=150,
                num_return_sequences=1,
                do_sample=True,
            )
            generated_text = response[0]['generated_text'].strip()

            if not generated_text or generated_text.isspace():
                return self._get_fallback_response()

            # Cache and return response
            self._response_cache[cache_key] = generated_text
            return generated_text

        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return self._get_fallback_response()

    def _get_fallback_response(self):
        """Fallback response in case of generation errors."""
        return "I apologize, but I couldn't generate a meaningful analysis at this time. Please try rephrasing your question."

    def post(self, request, *args, **kwargs):
        """Handle chat interactions and generate analysis based on uploaded documents."""
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

            # Generate embeddings for the message
            message_embedding = self.get_embeddings(message)

            # Query ChromaDB with embeddings
            chroma_client = get_chroma_client()
            collection = get_or_create_collection(chroma_client)
            results = collection.query(
                query_embeddings=[message_embedding.tolist()],
                n_results=5,
                where={"document_id": str(document.id)}
            )

            if not results['documents'][0]:
                return Response({
                    "session_id": session.session_id,
                    "response": self._get_fallback_response(),
                    "user_sentiment": self.sentiment_analyzer(message)[0]
                })

            # Generate response based on context
            context = self.format_context(
                results['documents'][0], results['metadatas'][0], document.language)
            response = self.generate_response(
                context, message, document.language)

            # Analyze sentiments
            user_sentiment = self.sentiment_analyzer(message)[0]
            response_sentiment = self.sentiment_analyzer(response)[0]

            # Create chat message
            ChatMessage.objects.create(
                session=session,
                content=response,
                is_user=False,
                sentiment_score=float(response_sentiment['score']),
                relevant_document=document
            )

            return Response({
                "session_id": session.session_id,
                "response": response,
                "user_sentiment": user_sentiment,
                "response_sentiment": float(response_sentiment['score']),
                "document_id": document.id
            })

        except Exception as e:
            logger.error(f"Error in chat processing: {str(e)}")
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


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
