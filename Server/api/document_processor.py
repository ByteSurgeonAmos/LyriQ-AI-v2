from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
from typing import List, Dict, Any
import textwrap
from langdetect import detect
import numpy as np


class DocumentProcessor:
    def __init__(self):
        # Initialize German sentiment model as per project requirements
        self.german_sentiment = pipeline(
            "sentiment-analysis",
            model="oliverguhr/german-sentiment-bert"
        )

        # Initialize English sentiment model as backup
        self.english_sentiment = pipeline(
            "sentiment-analysis",
            model="nlptown/bert-base-multilingual-uncased-sentiment"
        )

        # Initialize embedding model
        self.embedding_model = pipeline(
            'feature-extraction',
            model="sentence-transformers/all-MiniLM-L6-v2",
            device=0 if torch.cuda.is_available() else -1
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            "sentence-transformers/all-MiniLM-L6-v2")

    def detect_language(self, text: str) -> str:
        """Detect the language of the text."""
        try:
            return detect(text)
        except:
            return 'en'  # Default to English if detection fails

    def chunk_text(self, text: str, max_length: int = 512) -> List[str]:
        """Split text into chunks based on model's max token length."""
        chunks = []
        current_chunk = []
        current_length = 0

        sentences = text.split('.')
        for sentence in sentences:
            sentence = sentence.strip() + '.'
            tokens = self.tokenizer.tokenize(sentence)
            sentence_length = len(tokens)

            if current_length + sentence_length > max_length:
                # Join the current chunk and add it to chunks
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length

        # Add the last chunk if it exists
        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks

    def analyze_sentiment(self, text: str, language: str = None) -> Dict[str, Any]:
        """Analyze sentiment based on detected language."""
        if not language:
            language = self.detect_language(text)

        try:
            if language == 'de':
                result = self.german_sentiment(text)
                # Convert German model output to consistent format
                score = result[0]['score']
                if result[0]['label'] == 'negative':
                    score = -score
            else:
                result = self.english_sentiment(text)
                # Convert 1-5 scale to -1 to 1 scale
                score = (float(result[0]['label'][0]) - 3) / 2

            return {
                'score': score,
                'language': language
            }
        except Exception as e:
            print(f"Error in sentiment analysis: {str(e)}")
            return {'score': 0.0, 'language': language}

    def generate_embeddings(self, text: str) -> List[float]:
        """Generate embeddings for text."""
        try:
            # Generate embeddings and average them
            embedding = self.embedding_model(text)[0]
            # Convert to numpy for easier manipulation
            embedding_np = np.mean(embedding, axis=0)
            return embedding_np.tolist()
        except Exception as e:
            print(f"Error generating embeddings: {str(e)}")
            return [0.0] * 384  # Return zero vector of correct dimension

    def process_document(self, text: str) -> Dict[str, Any]:
        """Process document text fully."""
        language = self.detect_language(text)
        chunks = self.chunk_text(text)
        results = []

        for chunk in chunks:
            chunk_sentiment = self.analyze_sentiment(chunk, language)
            chunk_embedding = self.generate_embeddings(chunk)
            results.append({
                'text': chunk,
                'embedding': chunk_embedding,
                'sentiment': chunk_sentiment['score']
            })

        # Calculate average sentiment
        avg_sentiment = np.mean([r['sentiment'] for r in results])

        return {
            'chunks': [r['text'] for r in results],
            'embeddings': [r['embedding'] for r in results],
            'sentiment': float(avg_sentiment),
            'detailed_sentiments': [r['sentiment'] for r in results],
            'language': language
        }
