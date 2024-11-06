import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from datasets import Dataset, load_from_disk
from transformers import RagConfig, AutoTokenizer, AutoModel
from ..chat_model.chat_model import ChatModel
from ..chat_model.dataset import ChatDataset

# Configurations
BASE_DIR = "./data"
DATA_PATH = os.path.join(BASE_DIR, "train.csv")
DATASET_PATH = os.path.join(BASE_DIR, "processed_dataset")
INDEX_PATH = os.path.join(BASE_DIR, "processed_index/faiss_index.faiss")
MODEL_NAME = "facebook/rag-token-nq"
SENTIMENT_MODEL_NAME = "nlptown/bert-base-multilingual-uncased-sentiment"
OUTPUT_DIR = "results"
EPOCHS = 3
LEARNING_RATE = 2e-5
BATCH_SIZE = 4
MAX_LENGTH = 512


def generate_embeddings(texts, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """Generate embeddings for a list of texts using a pre-trained transformer model."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    # Move model to device if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    embeddings = []
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt",
                               padding=True, truncation=True, max_length=512).to(device)
            outputs = model(**inputs)
            embedding = outputs.last_hidden_state.mean(
                dim=1).cpu().squeeze().numpy()  # Move to CPU for storage
            embeddings.append(embedding)

    return embeddings


def prepare_rag_dataset(data_df, dataset_path, index_path, embedding_model="sentence-transformers/all-MiniLM-L6-v2"):
    """Prepare and save the dataset and index for RAG."""
    # Convert DataFrame to Hugging Face Dataset format
    hf_dataset = Dataset.from_pandas(data_df)

    # Generate embeddings for each 'lyrics' text and add as a new column
    hf_dataset = hf_dataset.map(lambda row: {"embeddings": generate_embeddings([
                                row["lyrics"]], model_name=embedding_model)[0]})

    # Save dataset to disk
    os.makedirs(dataset_path, exist_ok=True)
    hf_dataset.save_to_disk(dataset_path)

    # Reload dataset from disk and add FAISS index
    hf_dataset = load_from_disk(dataset_path)
    hf_dataset.add_faiss_index(column="embeddings")

    # Save index to a file within `index_path`
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    hf_dataset.get_index("embeddings").save(index_path)

    return dataset_path, index_path


def load_data(path):
    """Load and clean the training data."""
    data = pd.read_csv(path)
    data['lyrics'] = data['lyrics'].fillna("")
    data['sentiment'] = data['sentiment'].fillna("unknown")

    # Print dataset info for debugging
    print("\nDataset Info:")
    print(f"Total samples: {len(data)}")
    print("\nColumns:", data.columns.tolist())
    print("\nSample row:")
    print(data.iloc[0])

    return data


def main():
    try:
        # Create necessary directories
        os.makedirs(BASE_DIR, exist_ok=True)
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        # Load training data
        train_data = load_data(DATA_PATH)

        # Prepare RAG dataset and index
        dataset_path, index_path = prepare_rag_dataset(
            train_data, DATASET_PATH, INDEX_PATH)

        # Initialize tokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

        # Create RAG configuration
        rag_config = RagConfig.from_pretrained(MODEL_NAME)
        rag_config.index_name = "custom"
        rag_config.passages_path = dataset_path
        rag_config.index_path = index_path  # Use the returned index path

        # Initialize ChatModel instance with RAG configuration
        chat_model_instance = ChatModel(
            model_name=MODEL_NAME,
            sentiment_model_name=SENTIMENT_MODEL_NAME,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )

        # Initialize ChatDataset
        train_dataset = ChatDataset(
            data=train_data,
            tokenizer=tokenizer,
            max_length=MAX_LENGTH
        )

        # Create DataLoader
        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=0  # Adjusted for debugging
        )

        # Fine-tune the model
        chat_model_instance.finetune(
            train_data=train_loader,
            val_data=None,
            output_dir=OUTPUT_DIR,
            num_epochs=EPOCHS,
            learning_rate=LEARNING_RATE
        )

        print("Training completed. Model saved to:", OUTPUT_DIR)

    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise


if __name__ == "__main__":
    main()
