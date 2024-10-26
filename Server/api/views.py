from typing import List, Dict, Any
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
                with open(document.file.path, 'r', encoding='utf-8') as file:
                    content = file.read()

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
                    "message": "Document processed successfully",
                    "document_id": document.id,
                    "sentiment": result['sentiment'],
                    "language": result.get('language', 'en')
                }, status=status.HTTP_201_CREATED)

            except Exception as e:
                print(f"Error processing document: {str(e)}")
                return Response({
                    "error": str(e)
                }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class ChatView(APIView):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize model with better defaults
        self.generator = pipeline(
            "text2text-generation",
            model="google/flan-t5-large",
            device=0 if torch.cuda.is_available() else -1,
            model_kwargs={
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 50,
                "repetition_penalty": 1.2,
                "length_penalty": 1.0,
                "early_stopping": True
            }
        )

    def get_analysis_prompt(self, question_type: str) -> str:
        """Return specific prompt based on question type"""
        prompts = {
            'sentiment': """Analyze the emotional tone and sentiment of these lyrics. Consider:
- The overall emotional feel
- Any changes in emotion throughout the lyrics
- Specific words or phrases that convey emotions
- The intensity of the emotions expressed
Provide specific examples from the lyrics to support your analysis.""",

            'theme': """Analyze the main themes and meanings in these lyrics. Consider:
- The central message or themes
- Any metaphors or symbolism used
- How the themes develop throughout the lyrics
- The songwriter's perspective or message
Support your analysis with specific lines from the lyrics.""",

            'structure': """Analyze the structural elements of these lyrics. Consider:
- The organization and flow
- Any patterns or repetition
- The relationship between verses and chorus
- How the structure supports the message
Use specific examples from the lyrics.""",

            'general': """Analyze these lyrics considering:
- The main ideas and messages
- The emotional content and tone
- Any notable techniques or patterns
- The overall impact and meaning
Provide specific examples from the lyrics to support your points."""
        }
        return prompts.get(question_type, prompts['general'])

    def determine_question_type(self, question: str) -> str:
        """Determine the type of analysis needed based on the question"""
        question = question.lower()
        if any(word in question for word in ['emotion', 'feel', 'sentiment', 'mood']):
            return 'sentiment'
        elif any(word in question for word in ['theme', 'meaning', 'message', 'about']):
            return 'theme'
        elif any(word in question for word in ['structure', 'pattern', 'organize', 'form']):
            return 'structure'
        return 'general'

    def format_context(self, chunks: List[str]) -> str:
        """Format the context in a clear, organized way"""
        if not chunks:
            return ""
        # Join chunks with clear separation
        formatted_text = "\n\n".join(chunks)
        # Add section markers if they don't exist
        if not any(marker in formatted_text.lower() for marker in ['verse', 'chorus', 'bridge']):
            lines = formatted_text.split('\n')
            # Group lines into verses (4-6 lines each)
            formatted_sections = []
            current_section = []
            for i, line in enumerate(lines):
                current_section.append(line)
                if len(current_section) >= 4 or i == len(lines) - 1:
                    section_name = f"Verse {len(formatted_sections) + 1}" if len(
                        formatted_sections) % 2 == 0 else "Chorus"
                    formatted_sections.append(
                        f"{section_name}:\n" + "\n".join(current_section))
                    current_section = []
            formatted_text = "\n\n".join(formatted_sections)
        return formatted_text

    def generate_response(self, context: str, question: str) -> str:
        """Generate a response using the language model"""
        # Determine the type of analysis needed
        question_type = self.determine_question_type(question)
        analysis_prompt = self.get_analysis_prompt(question_type)

        # Construct the full prompt
        prompt = f"""Given these lyrics:

{context}

Question: {question}

{analysis_prompt}

Answer:"""

        try:
            response = self.generator(
                prompt,
                max_length=500,  # Allow longer responses
                min_length=100,  # Ensure substantial responses
                num_return_sequences=1,
                do_sample=True,
                no_repeat_ngram_size=3  # Prevent repetition of phrases
            )

            generated_text = response[0]['generated_text'].strip()

            # Post-process the response
            if not generated_text or generated_text.isspace():
                return "I apologize, but I couldn't generate a meaningful analysis. Please try rephrasing your question."

            # Remove any repeated sentences
            sentences = generated_text.split('. ')
            unique_sentences = []
            for sentence in sentences:
                if sentence not in unique_sentences:
                    unique_sentences.append(sentence)

            return '. '.join(unique_sentences)

        except Exception as e:
            print(f"Error generating response: {str(e)}")
            return "I apologize, but I encountered an error while analyzing the lyrics. Please try again."

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
            # Get relevant context from ChromaDB
            query_embedding = processor.generate_embeddings(message)
            chroma_client = get_chroma_client()
            collection = get_or_create_collection(chroma_client)

            # Query with document-specific filter
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=5  # Get more results initially
            )
            relevant_chunks = []
            if results['documents'] and results['documents'][0]:
                for doc, metadata in zip(results['documents'][0], results['metadatas'][0]):
                    if str(metadata.get('document_id')) == str(document.id):
                        relevant_chunks.append(doc)

            if not relevant_chunks:
                return Response({
                    "session_id": session_id,
                    "response": "I don't have enough information from the current document to answer that question.",
                    "user_sentiment": sentiment_result,
                    "response_sentiment": 0.0
                })

            # Format context and generate response
            context = self.format_context(relevant_chunks)
            response = self.generate_response(context, message)

            # Analyze response sentiment
            response_sentiment = processor.analyze_sentiment(response)

            # Save bot response
            bot_message = ChatMessage.objects.create(
                session=session,
                content=response,
                is_user=False,
                sentiment_score=response_sentiment['score'],
                relevant_document=document
            )

            return Response({
                "session_id": session_id,
                "response": response,
                "user_sentiment": sentiment_result,
                "response_sentiment": response_sentiment['score'],
                "document_id": document.id
            })

        except Exception as e:
            print(f"Error in chat processing: {str(e)}")
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
