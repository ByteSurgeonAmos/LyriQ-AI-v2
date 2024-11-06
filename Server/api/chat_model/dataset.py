import pandas as pd
from torch.utils.data import Dataset
import torch
from typing import Dict, Any, List, Optional
import logging


class ChatDataset(Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        tokenizer: Any,
        max_length: int = 512,
        include_title: bool = True
    ):
        """
        Initialize the ChatDataset.

        Args:
            data (pd.DataFrame): DataFrame containing the dataset
            tokenizer: Tokenizer instance to use for text encoding
            max_length (int): Maximum sequence length for tokenization
            include_title (bool): Whether to include title in input text
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.include_title = include_title

        # Create mappings for all categorical columns
        self.label_mappings = {}
        categorical_columns = ['emotions', 'sentiment', 'genre', 'mood']
        for column in categorical_columns:
            if column in self.data.columns:
                self.label_mappings[column] = self._create_label_mapping(
                    column)

        # Store the reverse mappings for later use
        self.id_to_label = {
            column: {idx: label for label, idx in mapping.items()}
            for column, mapping in self.label_mappings.items()
        }

        logging.info(f"Initialized ChatDataset with {len(self.data)} samples")
        for column, mapping in self.label_mappings.items():
            logging.info(
                f"Created mapping for {column} with {len(mapping)} classes")

    def _create_label_mapping(self, column_name: str) -> Dict[str, int]:
        """
        Create a mapping from label strings to numerical IDs.

        Args:
            column_name (str): Name of the column to create mapping for

        Returns:
            Dict[str, int]: Mapping from label strings to numerical IDs
        """
        if column_name in self.data.columns:
            # Handle both single labels and list-like labels
            if isinstance(self.data[column_name].iloc[0], (list, tuple)):
                # Flatten the lists and get unique values
                unique_values = set()
                for labels in self.data[column_name].dropna():
                    unique_values.update(labels)
                unique_values = sorted(unique_values)
            else:
                unique_values = sorted(
                    self.data[column_name].dropna().unique())

            return {label: idx for idx, label in enumerate(unique_values)}
        return {}

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample from the dataset.

        Args:
            idx (int): Index of the sample to get

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing the tokenized input and labels
        """
        item = self.data.iloc[idx]

        # Prepare input text
        text_parts = []
        if self.include_title and 'title' in item:
            text_parts.append(item['title'])
        if 'lyrics' in item:
            text_parts.append(item['lyrics'])
        input_text = ' '.join(text_parts)

        # Tokenize
        try:
            encoding = self.tokenizer(
                input_text,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors="pt"
            )
        except Exception as e:
            logging.error(f"Error tokenizing text at index {idx}: {str(e)}")
            # Return a zero tensor of appropriate size as fallback
            return {
                'input_ids': torch.zeros(self.max_length, dtype=torch.long),
                'attention_mask': torch.zeros(self.max_length, dtype=torch.long),
                'labels': torch.tensor(0, dtype=torch.long)
            }

        # Prepare labels
        labels = {}
        for column, mapping in self.label_mappings.items():
            if column in item:
                if isinstance(item[column], (list, tuple)):
                    # Multi-label case: create one-hot encoding
                    label_tensor = torch.zeros(len(mapping))
                    for label in item[column]:
                        if label in mapping:
                            label_tensor[mapping[label]] = 1
                else:
                    # Single-label case
                    label_id = mapping.get(item[column], 0)
                    label_tensor = torch.tensor(label_id, dtype=torch.long)
                labels[f'{column}_labels'] = label_tensor

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            **labels
        }

    def get_label_mappings(self) -> Dict[str, Dict[str, int]]:
        """Return all label mappings."""
        return self.label_mappings

    def get_reverse_mappings(self) -> Dict[str, Dict[int, str]]:
        """Return all reverse label mappings (ID to label)."""
        return self.id_to_label

    def get_num_labels(self, column: str) -> Optional[int]:
        """
        Get the number of unique labels for a specific column.

        Args:
            column (str): Name of the column

        Returns:
            Optional[int]: Number of unique labels, or None if column doesn't exist
        """
        if column in self.label_mappings:
            return len(self.label_mappings[column])
        return None

    @staticmethod
    def load_data(
        file_path: str,
        required_columns: List[str] = ['title', 'lyrics', 'emotions']
    ) -> pd.DataFrame:
        """
        Load and validate dataset from a file.

        Args:
            file_path (str): Path to the data file
            required_columns (List[str]): List of required columns

        Returns:
            pd.DataFrame: Loaded and validated DataFrame
        """
        try:
            # Determine file type and read accordingly
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith('.json'):
                df = pd.read_json(file_path)
            elif file_path.endswith('.xlsx'):
                df = pd.read_excel(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_path}")

            # Validate required columns
            missing_columns = [
                col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(
                    f"Missing required columns: {missing_columns}")

            # Basic cleaning
            for col in df.columns:
                if df[col].dtype == 'object':
                    df[col] = df[col].fillna('')

            return df

        except Exception as e:
            logging.error(f"Error loading data from {file_path}: {str(e)}")
            raise

    def get_sample_weights(self) -> torch.Tensor:
        """
        Calculate sample weights based on label distribution for balanced training.

        Returns:
            torch.Tensor: Tensor of sample weights
        """
        if 'emotions' in self.label_mappings:
            # Count occurrences of each emotion
            label_counts = self.data['emotions'].value_counts()
            total_samples = len(self.data)

            # Calculate inverse frequency weights
            weights = torch.zeros(len(self.data))
            for idx, row in self.data.iterrows():
                emotion = row['emotions']
                if emotion in label_counts:
                    weights[idx] = total_samples / \
                        (len(label_counts) * label_counts[emotion])

            # Normalize weights
            weights = weights / weights.sum() * len(weights)
            return weights

        return torch.ones(len(self.data))
