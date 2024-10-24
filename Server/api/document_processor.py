from transformers import pipeline
import torch
from typing import List, Dict, Any
import numpy as np


class DocumentProcessor:
    def __init__(self):
        # Use a smaller, faster model for initial testing
        self.sentiment_model = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=-1  # Use CPU for now
        )

        # Initialize embedding model (smaller model for testing)
        self.embedding_model = pipeline(
            'feature-extraction',
            model="sentence-transformers/paraphrase-MiniLM-L3-v2",
            device=-1  # Use CPU for now
        )

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

    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment of text."""
        try:
            result = self.sentiment_model(text[:512])  # Limit text length
            # Convert to score between -1 and 1
            score = result[0]['score']
            if result[0]['label'] == 'NEGATIVE':
                score = -score
            return {
                'score': score,
                'label': result[0]['label']
            }
        except Exception as e:
            print(f"Error in sentiment analysis: {str(e)}")
            return {'score': 0.0, 'label': 'NEUTRAL'}

    def generate_embeddings(self, text: str) -> List[float]:
        """Generate embeddings for text."""
        try:
            # Limit text length to avoid issues
            embedding = self.embedding_model(text[:512])[0]
            # Take mean of token embeddings
            return np.mean(embedding, axis=0).tolist()
        except Exception as e:
            print(f"Error generating embeddings: {str(e)}")
            # Return zero vector of correct dimension
            return [0.0] * self.embedding_model.model.config.hidden_size

    def process_document(self, text: str) -> Dict[str, Any]:
        """Process document text."""
        try:
            # Split into manageable chunks
            chunks = self.chunk_text(text)
            results = []

            for chunk in chunks:
                sentiment = self.analyze_sentiment(chunk)
                embedding = self.generate_embeddings(chunk)
                results.append({
                    'text': chunk,
                    'embedding': embedding,
                    'sentiment': sentiment['score']
                })

            # Calculate average sentiment
            sentiments = [r['sentiment'] for r in results]
            avg_sentiment = float(np.mean(sentiments)) if sentiments else 0.0

            return {
                'chunks': [r['text'] for r in results],
                'embeddings': [r['embedding'] for r in results],
                'sentiment': avg_sentiment,
                'detailed_sentiments': sentiments,
            }

        except Exception as e:
            print(f"Error processing document: {str(e)}")
            # Return minimal valid response
            return {
                'chunks': [text],
                'embeddings': [self.generate_embeddings(text[:512])],
                'sentiment': 0.0,
                'detailed_sentiments': [0.0]
            }
