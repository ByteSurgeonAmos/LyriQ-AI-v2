from typing import Optional
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM
)
import torch
import logging


class ChatModel:
    def __init__(
        self,
        model_name: str = "gpt2",  # Change to a causal language model like GPT-2
        sentiment_model_name: str = "nlptown/bert-base-multilingual-uncased-sentiment",
        device: Optional[str] = None
    ):
        """
        Initialize the ChatModel with a generative model and sentiment analysis capabilities.

        Args:
            model_name: Name or path of the generative model (e.g., GPT-2)
            sentiment_model_name: Name or path of the sentiment analysis model
            device: Device to run the model on ('cuda' or 'cpu')
        """

        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {self.device}")

        try:
            # Initialize generative model (GPT-2 or another causal LM)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)

            # Initialize sentiment analysis components
            self.sentiment_tokenizer = AutoTokenizer.from_pretrained(
                sentiment_model_name)
            self.sentiment_model = AutoModelForSequenceClassification.from_pretrained(
                sentiment_model_name)

            # Set the model to the desired device
            self.model.to(self.device)
            self.sentiment_model.to(self.device)

            logging.info("Model initialization successful")

        except Exception as e:
            logging.error(f"Error initializing models: {str(e)}")
            raise

    def generate_response(
        self,
        prompt: str,
        max_length: int = 750,
        num_return_sequences: int = 1
    ) -> str:
        """
        Generate a response based on the provided prompt using a causal language model.
        """
        try:
            inputs = self.tokenizer.encode(
                prompt, return_tensors="pt", max_length=512, truncation=True).to(self.device)

            output_ids = self.model.generate(
                inputs,
                max_length=max_length,
                num_return_sequences=num_return_sequences,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                num_beams=2,
                early_stopping=True
            )

            generated_texts = self.tokenizer.decode(
                output_ids[0], skip_special_tokens=True)
            return generated_texts.strip()

        except Exception as e:
            logging.error(f"Error generating response: {str(e)}")
            return f"Error generating response: {str(e)}"

    def sentiment_analyzer(self, text: str) -> str:
        """
        Analyze the sentiment of the provided text using BERT.
        """
        try:
            inputs = self.sentiment_tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)

            with torch.no_grad():
                outputs = self.sentiment_model(**inputs)
                logits = outputs.logits
                predicted_class = torch.argmax(logits, dim=-1).item()

            return self.map_sentiment_to_description(predicted_class)

        except Exception as e:
            logging.error(f"Error analyzing sentiment: {str(e)}")
            return "Error analyzing sentiment"

    @staticmethod
    def map_sentiment_to_description(sentiment_label: int) -> str:
        """
        Map sentiment label to descriptive sentiment.
        """
        sentiment_map = {
            0: "The sentiment in the given text is sadness and despair.",
            1: "The sentiment in the given text is longing and urgency.",
            2: "The sentiment in the given text is joy and excitement.",
            3: "The sentiment in the given text is neutrality and calm.",
            4: "The sentiment in the given text is anger and frustration."
        }
        return sentiment_map.get(sentiment_label, "Sentiment not recognized.")
