from typing import Optional, Dict, Union
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
import torch
import logging


class ChatModel:
    def __init__(
        self,
        model_name: str = "gpt2",
        sentiment_model_name: str = "nlptown/bert-base-multilingual-uncased-sentiment",
        emotion_model_name: str = "j-hartmann/emotion-english-distilroberta-base",
        device: Optional[str] = None
    ):
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {self.device}")

        # Initialize models with improved error handling
        self._initialize_models(
            model_name, sentiment_model_name, emotion_model_name)

    def _initialize_models(self, model_name: str, sentiment_model_name: str, emotion_model_name: str):
        try:
            # Initialize generative model
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name).to(self.device)

            # Initialize sentiment and emotion analysis models
            self.sentiment_tokenizer = AutoTokenizer.from_pretrained(
                sentiment_model_name)
            self.sentiment_model = AutoModelForSequenceClassification.from_pretrained(
                sentiment_model_name).to(self.device)
            self.emotion_tokenizer = AutoTokenizer.from_pretrained(
                emotion_model_name)
            self.emotion_model = AutoModelForSequenceClassification.from_pretrained(
                emotion_model_name).to(self.device)

            logging.info("Models successfully initialized")

        except Exception as e:
            logging.error(f"Model initialization failed: {str(e)}")
            raise RuntimeError(f"Failed to initialize models: {str(e)}")

    def generate_response(self, prompt: str, message: str, max_length: int = 750, num_return_sequences: int = 1) -> Union[str, list]:
        try:
            # Validate input
            if not prompt or not message:
                raise ValueError("Prompt and message cannot be empty")

            # Construct prompt with context separator
            full_prompt = self._construct_prompt(prompt, message)

            # Tokenize input
            inputs = self._tokenize_input(full_prompt)

            # Generate response with fine-tuned parameters for creativity and diversity
            output_ids = self.model.generate(
                inputs['input_ids'].to(self.device),
                attention_mask=inputs['attention_mask'].to(self.device),
                max_length=min(max_length, 1024),  # Reasonable upper bound
                num_return_sequences=min(num_return_sequences, 3),
                do_sample=True,
                top_k=50,
                top_p=0.85,
                temperature=0.9,
                num_beams=5,
                no_repeat_ngram_size=3,
                repetition_penalty=1.2,
                early_stopping=True,
                pad_token_id=self.tokenizer.pad_token_id
            )

            # Process generated responses
            return self._post_process_generation(output_ids, num_return_sequences)

        except Exception as e:
            logging.error(f"Response generation failed: {str(e)}")
            return f"Error: Unable to generate response due to: {str(e)}"

    def _construct_prompt(self, prompt: str, message: str) -> str:
        # Trim extra whitespace and format prompt with separator for clarity
        return f"{prompt.strip()}\n---\n{message.strip()}"

    def _tokenize_input(self, text: str) -> Dict:
        return self.tokenizer(
            text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True
        )

    def _post_process_generation(self, output_ids: torch.Tensor, num_return_sequences: int) -> Union[str, list]:
        generated_texts = []
        for i in range(num_return_sequences):
            text = self.tokenizer.decode(
                output_ids[i], skip_special_tokens=True)
            cleaned_text = self._clean_generated_text(text)
            generated_texts.append(cleaned_text)

        return generated_texts[0] if num_return_sequences == 1 else generated_texts

    def _clean_generated_text(self, text: str) -> str:
        # Remove repetitive newlines and duplicated phrases
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        unique_lines = []
        for line in lines:
            if line not in unique_lines[-3:]:  # Avoid last 3 repeated lines
                unique_lines.append(line)
        return '\n'.join(unique_lines)

    def sentiment_analyzer(self, text: str) -> Dict[str, str]:
        """Analyze sentiment of input text and return a sentiment statement."""
        try:
            inputs = self.sentiment_tokenizer(
                text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
            with torch.no_grad():
                logits = self.sentiment_model(**inputs).logits
                predicted_class = torch.argmax(logits, dim=-1).item()
                scores = torch.softmax(logits, dim=-1)
                score = scores[0][predicted_class].item()
            sentiment_statement = self.map_sentiment_to_statement(
                predicted_class)
            return {"sentiment": sentiment_statement, "score": float(score)}

        except Exception as e:
            logging.error(f"Error in sentiment analysis: {str(e)}")
            return {"sentiment": "The sentiment of the text is neutral.", "score": 0.0}

    def map_sentiment_to_statement(self, sentiment_label: int) -> str:
        sentiment_map = {
            0: "The sentiment of the text is that of love.",
            1: "The sentiment of the text is negative.",
            2: "The sentiment of the text is that of emotions.",
            3: "The sentiment of the text is positive.",
            4: "The sentiment of the text is very positive."
        }
        return sentiment_map.get(sentiment_label, "The sentiment of the text is not recognized.")

    def analyze_emotions(self, text: str) -> Dict[str, float]:
        """Analyze emotions in input text and return a dictionary of emotions with scores."""
        try:
            if not text:
                return {}

            # Tokenize input for emotion model
            inputs = self.emotion_tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)

            with torch.no_grad():
                outputs = self.emotion_model(**inputs)
                logits = outputs.logits
                scores = torch.softmax(logits, dim=-1)[0]

                # Map scores to emotion labels
                emotions = {self.map_emotion_to_description(i): float(
                    score) for i, score in enumerate(scores)}

            return emotions

        except Exception as e:
            logging.error(f"Error in emotion analysis: {str(e)}")
            return {}

    def map_emotion_to_description(self, emotion_label: int) -> str:
        """Map emotion model labels to human-readable descriptions."""
        emotion_map = {
            0: "Anger",
            1: "Joy",
            2: "Sadness",
            3: "Fear",
            4: "Surprise",
            5: "Disgust"
        }
        return emotion_map.get(emotion_label, "Emotion not recognized")
