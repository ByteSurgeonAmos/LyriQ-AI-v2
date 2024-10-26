from transformers import pipeline
import torch
from typing import List, Dict, Any
import numpy as np
from langdetect import detect, LangDetectException

class DocumentProcessor:
    def __init__(self):
        # Initialize sentiment models for different languages
        self.sentiment_models = {
            'en': pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                device=-1
            ),
            'de': pipeline(
                "sentiment-analysis",
                model="oliverguhr/german-sentiment-bert",
                device=-1
            )
        }
        
        # Default to English model if language not supported
        self.default_sentiment_model = self.sentiment_models['en']

        # Initialize embedding model
        self.embedding_model = pipeline(
            'feature-extraction',
            model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",  # Changed to multilingual model
            device=-1
        )

    def detect_language(self, text: str) -> str:
        """
        Detect the language of the input text.
        Returns ISO 639-1 language code (e.g., 'en', 'de', etc.)
        """
        try:
            # Take a sample of the text if it's very long
            sample = text[:1000]  # First 1000 characters should be enough for detection
            return detect(sample)
        except LangDetectException:
            # Return 'en' as fallback if detection fails
            return 'en'

    def chunk_text(self, text: str, max_length: int = 512) -> List[str]:
        """Split text into chunks based on simple sentence splitting."""
        # Simple sentence splitting by common punctuation
        sentences = []
        current = []
        words = text.replace('\n', ' ').split()

        for word in words:
            current.append(word)
            if word.endswith(('.', '!', '?')) and len(current) >= 5:
                sentences.append(' '.join(current))
                current = []

        if current:
            sentences.append(' '.join(current))

        # Combine sentences into chunks
        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            sentence_length = len(sentence.split())
            if current_length + sentence_length > max_length:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length

        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks if chunks else [text]

    def analyze_sentiment(self, text: str, language: str = None) -> Dict[str, Any]:
        """
        Analyze sentiment of text using language-specific models when available.
        """
        try:
            if not language:
                language = self.detect_language(text)

            # Get appropriate sentiment model for the language
            sentiment_model = self.sentiment_models.get(language, self.default_sentiment_model)
            
            result = sentiment_model(text[:512])  # Limit text length
            
            # Handle different model output formats
            if language == 'de':
                # German model specific handling
                score = result[0]['score']
                label = result[0]['label']
                # Convert to consistent format (-1 to 1 scale)
                if label == 'negative':
                    score = -score
            else:
                # English model handling (default)
                score = result[0]['score']
                if result[0]['label'] == 'NEGATIVE':
                    score = -score
                label = result[0]['label']

            return {
                'score': score,
                'label': label,
                'language': language
            }
        except Exception as e:
            print(f"Error in sentiment analysis: {str(e)}")
            return {'score': 0.0, 'label': 'NEUTRAL', 'language': language or 'en'}

    def generate_embeddings(self, text: str) -> List[float]:
        """Generate embeddings for text using multilingual model."""
        try:
            # Limit text length to avoid issues
            embedding = self.embedding_model(text[:512])[0]
            # Take mean of token embeddings
            return np.mean(embedding, axis=0).tolist()
        except Exception as e:
            print(f"Error generating embeddings: {str(e)}")
            # Return zero vector of correct dimension
            return [0.0] * self.embedding_model.model.config.hidden_size

    def process_document(self, text: str, document_id: str = None) -> Dict[str, Any]:
        """Process document text with language detection and document ID tracking."""
        try:
            # Detect language first
            language = self.detect_language(text)

            # Split into manageable chunks
            chunks = self.chunk_text(text)
            results = []

            for chunk in chunks:
                sentiment = self.analyze_sentiment(chunk, language)
                embedding = self.generate_embeddings(chunk)
                results.append({
                    'text': chunk,
                    'embedding': embedding,
                    'sentiment': sentiment['score'],
                    'document_id': document_id  # Track document ID for each chunk
                })

            # Calculate average sentiment
            sentiments = [r['sentiment'] for r in results]
            avg_sentiment = float(np.mean(sentiments)) if sentiments else 0.0

            return {
                'chunks': [r['text'] for r in results],
                'embeddings': [r['embedding'] for r in results],
                'sentiment': avg_sentiment,
                'detailed_sentiments': sentiments,
                'language': language,
                'document_id': document_id,  # Include document ID in result
                'chunk_count': len(chunks)
            }

        except Exception as e:
            print(f"Error processing document: {str(e)}")
            # Return minimal valid response
            return {
                'chunks': [text],
                'embeddings': [self.generate_embeddings(text[:512])],
                'sentiment': 0.0,
                'detailed_sentiments': [0.0],
                'language': 'en',
                'document_id': document_id,
                'chunk_count': 1
            }