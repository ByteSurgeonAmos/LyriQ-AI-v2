from transformers import (
    AutoTokenizer,
    RagTokenForGeneration,
    AutoModelForSequenceClassification,
    RagTokenizer,
    RagRetriever
)
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
import os
from typing import Optional, Dict, List
import logging


class ChatModel:
    def __init__(
        self,
        model_name: str = "facebook/rag-token-nq",
        sentiment_model_name: str = "nlptown/bert-base-multilingual-uncased-sentiment",
        model_dir: Optional[str] = None,
        device: Optional[str] = None
    ):
        """
        Initialize the ChatModel with RAG and sentiment analysis capabilities.

        Args:
            model_name: Name or path of the RAG model
            sentiment_model_name: Name or path of the sentiment analysis model
            model_dir: Directory containing a previously saved model
            device: Device to run the model on ('cuda' or 'cpu')
        """
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {self.device}")

        try:
            # Initialize RAG components
            self.tokenizer = RagTokenizer.from_pretrained(model_name)

            # Configure retriever with custom settings to avoid wiki_dpr issues
            retriever_config = {
                "index_name": "custom",  # Use custom index instead of wiki_dpr
                "use_dummy_dataset": True  # Use dummy dataset for initial setup
            }

            self.retriever = RagRetriever.from_pretrained(
                model_name,
                **retriever_config
            )

            self.model = RagTokenForGeneration.from_pretrained(model_name)

            # Initialize sentiment analysis components
            self.sentiment_tokenizer = AutoTokenizer.from_pretrained(
                sentiment_model_name)
            self.sentiment_model = AutoModelForSequenceClassification.from_pretrained(
                sentiment_model_name)

            # Set up the models
            self.model.set_retriever(self.retriever)
            self.model.to(self.device)
            self.sentiment_model.to(self.device)

            logging.info("Model initialization successful")

        except Exception as e:
            logging.error(f"Error initializing models: {str(e)}")
            raise

    def finetune(
        self,
        train_data: DataLoader,
        val_data: Optional[DataLoader] = None,
        output_dir: Optional[str] = None,
        num_epochs: int = 3,
        learning_rate: float = 2e-5,
        batch_size: int = 4
    ):
        """
        Fine-tune the model on the provided dataset.
        """
        try:
            optimizer = AdamW(self.model.parameters(),
                              lr=learning_rate, no_deprecation_warning=True)
            scheduler = StepLR(optimizer, step_size=1, gamma=0.95)

            for epoch in range(num_epochs):
                self.model.train()
                total_loss = 0

                for i, batch in enumerate(train_data):
                    try:
                        # Move batch to device
                        question_input_ids = batch['input_ids'].to(self.device)
                        question_attention_mask = batch['attention_mask'].to(
                            self.device)
                        labels = batch['labels'].to(self.device)

                        # Get generator inputs
                        generator_inputs = self.retriever(
                            question_input_ids.cpu().numpy(),
                            question_attention_mask.cpu().numpy(),
                            return_tensors="pt"
                        )

                        # Move inputs to device
                        generator_inputs = {
                            k: v.to(self.device) if isinstance(
                                v, torch.Tensor) else v
                            for k, v in generator_inputs.items()
                        }
                        generator_inputs['labels'] = labels

                        # Training step
                        optimizer.zero_grad()
                        outputs = self.model(**generator_inputs)
                        loss = outputs.loss
                        total_loss += loss.item()

                        loss.backward()
                        optimizer.step()

                        if (i + 1) % 10 == 0:
                            logging.info(
                                f"Epoch {epoch+1}/{num_epochs} - Batch {i+1} - Loss: {loss.item():.4f}")

                    except Exception as e:
                        logging.error(f"Error in batch {i}: {str(e)}")
                        continue

                avg_train_loss = total_loss / len(train_data)
                logging.info(
                    f"Epoch {epoch+1}/{num_epochs} - Average Train Loss: {avg_train_loss:.4f}")

                scheduler.step()

                # Save checkpoint
                if output_dir:
                    checkpoint_dir = os.path.join(
                        output_dir, f"checkpoint-epoch-{epoch+1}")
                    self.save_model(checkpoint_dir)

            if output_dir:
                self.save_model(output_dir)

        except Exception as e:
            logging.error(f"Error during fine-tuning: {str(e)}")
            raise

    def generate_response(
        self,
        context: str,
        question: str,
        max_length: int = 750,
        num_return_sequences: int = 1
    ) -> str:
        """
        Generate a response based on the provided context and question.
        """
        try:
            inputs = self.tokenizer.encode_plus(
                f"Question: {question}\n\nContent:\n{context}",
                return_tensors="pt",
                max_length=512,
                truncation=True
            ).to(self.device)

            output_ids = self.model.generate(
                **inputs,
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
            0: "The sentiment in the given lyrics is the one for sadness and despair.",
            1: "The sentiment in the given lyrics is the one for longing and urgency.",
            2: "The sentiment in the given lyrics is the one for joy and excitement.",
            3: "The sentiment in the given lyrics is the one for neutrality and calm.",
            4: "The sentiment in the given lyrics is the one for anger and frustration."
        }
        return sentiment_map.get(sentiment_label, "Sentiment not recognized.")

    def save_model(self, output_dir: str):
        """
        Save the finetuned model to the specified directory.
        """
        try:
            os.makedirs(output_dir, exist_ok=True)
            self.model.save_pretrained(output_dir)
            self.tokenizer.save_pretrained(output_dir)
            logging.info(f"Model saved to {output_dir}")

        except Exception as e:
            logging.error(f"Error saving model: {str(e)}")
            raise

    def load_model(self, model_dir: str):
        """
        Load a finetuned model from the specified directory.
        """
        try:
            self.model = RagTokenForGeneration.from_pretrained(model_dir)
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
            self.model.to(self.device)
            logging.info(f"Model loaded from {model_dir}")

        except Exception as e:
            logging.error(f"Error loading model: {str(e)}")
            raise
