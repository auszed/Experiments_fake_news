import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict
import pandas as pd
from transformers import AutoTokenizer


class CustomDataset(Dataset):
    """Custom dataset class to handle tokenization and chunking of text data."""

    def __init__(self, dataframe: pd.DataFrame, comment_str_title: str, tokenizer, target_variable: pd.DataFrame,
                 max_token_len=256):
        """
        Initialize the dataset with the required parameters.

        Args:
            dataframe (pd.DataFrame): Input DataFrame containing the data.
            comment_str_title (str): The column name containing the text to tokenize.
            tokenizer: Tokenizer object to encode the text.
            target_variable (pd.DataFrame): The target variable for the dataset.
            max_token_len (int, optional): Maximum token length for each chunk. Default is 256.
        """
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len
        self.target_variable = target_variable
        self.comment_str_title = comment_str_title

    def chunk_text(self, text, max_length, stride):
        """
        Extract the values in chunks and pass them into tokens.

        Args:
            text (str): The input text to chunk.
            max_length (int): Maximum length for each chunk.
            stride (int): The stride to use when splitting the text into chunks.

        Returns:
            list: A list of chunks, each chunk being a list of token IDs.
        """
        tokens = self.tokenizer.encode_plus(
            text,
            add_special_tokens=False,
            max_length=None,
            truncation=False,
            return_attention_mask=False,
            return_token_type_ids=False
        )['input_ids']

        chunks = []
        for i in range(0, len(tokens), stride):
            chunk = tokens[i:i + max_length]
            if len(chunk) < max_length:
                chunk += [self.tokenizer.pad_token_id] * (max_length - len(chunk))  # Padding
            chunks.append(chunk)
        return chunks

    def __len__(self):
        """
        Get the length of the dataset.

        Returns:
            int: The number of rows in the dataset.
        """
        return len(self.dataframe)

    def __getitem__(self, index):
        """
        Get a single item from the dataset.

        Args:
            index (int): The index of the item to retrieve.

        Returns:
            dict: A dictionary containing input_ids, attention_masks, and labels.
        """
        data_row = self.dataframe.iloc[index]
        comment = str(data_row[self.comment_str_title])
        data_row_target = torch.tensor(self.target_variable.iloc[index].values, dtype=torch.float)

        # Split the comment into chunks
        chunks = self.chunk_text(comment, max_length=self.max_token_len, stride=self.max_token_len // 2)

        input_ids = []
        attention_masks = []

        for chunk in chunks:
            # Here, `chunk` is already tokenized, so we directly create tensors
            tokens = {
                "input_ids": torch.tensor(chunk, dtype=torch.long),
                "attention_mask": torch.tensor([1 if id != self.tokenizer.pad_token_id else 0 for id in chunk],
                                               dtype=torch.long)
            }
            input_ids.append(tokens["input_ids"])
            attention_masks.append(tokens["attention_mask"])

        return {
            "input_ids": torch.stack(input_ids),
            "attention_masks": torch.stack(attention_masks),
            "labels": data_row_target
        }


def custom_collate_fn(batch: List[Dict]) -> Dict:
    """
    Custom collate function to handle a variable number of chunks per sample.

    Args:
        batch (List[Dict]): A batch of data samples.

    Returns:
        dict: A dictionary with padded input_ids, attention_masks, chunk_attention_mask, and labels.
    """
    # Get the max number of chunks in this batch
    max_chunks = max(sample["input_ids"].size(0) for sample in batch)

    # Get batch size and sequence length
    batch_size = len(batch)
    seq_length = batch[0]["input_ids"].size(1)

    # Initialize tensors to hold padded data
    padded_input_ids = torch.zeros((batch_size, max_chunks, seq_length), dtype=torch.long)
    padded_attention_masks = torch.zeros((batch_size, max_chunks, seq_length), dtype=torch.long)

    # Collect labels
    labels = torch.stack([sample["labels"] for sample in batch])

    # Create chunk attention mask to identify valid chunks
    chunk_attention_mask = torch.zeros((batch_size, max_chunks), dtype=torch.long)

    # Fill in the tensors with actual data
    for i, sample in enumerate(batch):
        num_chunks = sample["input_ids"].size(0)
        padded_input_ids[i, :num_chunks, :] = sample["input_ids"]
        padded_attention_masks[i, :num_chunks, :] = sample["attention_masks"]
        chunk_attention_mask[i, :num_chunks] = 1

    return {
        "input_ids": padded_input_ids,
        "attention_masks": padded_attention_masks,
        "chunk_attention_mask": chunk_attention_mask,
        "labels": labels
    }


def pytorch_metrics_calculations(all_labels, all_preds, beta_f1=1):
    """
    Calculate evaluation metrics such as accuracy, F1 score, precision, and recall.

    Args:
        all_labels (list): List of true labels.
        all_preds (list): List of predicted labels.
        beta_f1 (float, optional): The beta value for F1 score calculation. Default is 1.

    Returns:
        tuple: accuracy, f1, precision, recall, true_positives, predicted_positives, actual_positives.
    """
    # convert to bool
    all_preds = torch.tensor(all_preds, dtype=torch.bool)
    all_labels = torch.tensor(all_labels, dtype=torch.bool)

    # predictions
    true_positives = torch.logical_and(all_preds == 1, all_labels == 1).sum().float()
    predicted_positives = (all_preds == 1).sum().float()
    actual_positives = (all_labels == 1).sum().float()

    # Calculate accuracy
    accuracy = (all_preds == all_labels).float().mean()

    # precision and recall
    precision = true_positives / predicted_positives if predicted_positives > 0 else torch.tensor(0.0)
    recall = true_positives / actual_positives if actual_positives > 0 else torch.tensor(0.0)

    # Calculate F1 score
    beta_squared = beta_f1 ** 2
    f1 = (1 + beta_squared) * (precision * recall) / (beta_squared * precision + recall) if (
                                                                                                        precision + recall) > 0 else torch.tensor(
        0.0)

    return accuracy, f1, precision, recall, true_positives, predicted_positives, actual_positives


def train_model(model, dataloader, optimizer, scheduler, device, THRESHOLD_PROBABILITIES_MODEL, beta_f1):
    """
    Train the model for one epoch.

    Args:
        model: The model to train.
        dataloader: DataLoader providing the training data.
        optimizer: Optimizer for model training.
        scheduler: Learning rate scheduler.
        device: Device (CPU or GPU) to train the model on.
        THRESHOLD_PROBABILITIES_MODEL (float): Threshold for probability to make predictions.
        beta_f1 (float): Beta value for F1 score calculation.

    Returns:
        tuple: average loss, accuracy, and F1 score for the epoch.
    """
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []

    for batch in dataloader:
        if len(batch['input_ids']) < dataloader.batch_size:
            break
        input_ids = batch['input_ids'].to(device)
        attention_masks = batch['attention_masks'].to(device)
        chunk_attention_mask = batch['chunk_attention_mask'].to(device)
        labels = batch['labels'].float().to(device)

        optimizer.zero_grad()
        logits = model(input_ids, attention_masks, chunk_attention_mask)
        loss = model.loss_function(logits, labels)
        loss.backward()

        optimizer.step()

        total_loss += loss.item()
        preds = (logits > THRESHOLD_PROBABILITIES_MODEL).float()

        all_preds.extend(preds.squeeze().tolist())
        all_labels.extend(labels.squeeze().tolist())

    avg_loss = total_loss / len(dataloader)
    accuracy, f1, _, _, _, _, _ = pytorch_metrics_calculations(all_labels, all_preds, beta_f1)
    return avg_loss, accuracy, f1


def evaluate_model(model, dataloader, device, THRESHOLD_PROBABILITIES_MODEL, beta_f1=1):
    """
    Evaluate the model on the given dataset.

    Args:
        model: The trained model.
        dataloader: DataLoader providing the evaluation data.
        device: Device (CPU or GPU) for evaluation.
        THRESHOLD_PROBABILITIES_MODEL (float): Threshold for probability to make predictions.
        beta_f1 (float, optional): Beta value for F1 score calculation. Default is 1.

    Returns:
        tuple: accuracy, F1 score, predictions, true labels, predicted probabilities, and input_ids.
    """
    model.eval()
    all_preds, all_labels, all_predict_prob = [], [], []
    inputs_ids = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_masks = batch['attention_masks'].to(device)
            chunk_attention_mask = batch['chunk_attention_mask'].to(device)
            labels = batch['labels'].float().to(device)
            logits = model(input_ids, attention_masks, chunk_attention_mask)
            preds = (logits > THRESHOLD_PROBABILITIES_MODEL).float()
            all_preds.extend(preds.squeeze().tolist())
            all_labels.extend(labels.squeeze().tolist())
            all_predict_prob.extend(logits.squeeze().tolist())
            inputs_ids.extend(input_ids.cpu().tolist())

    accuracy, f1, _, _, _, _, _ = pytorch_metrics_calculations(all_labels, all_preds, beta_f1)
    return accuracy, f1, all_preds, all_labels, all_predict_prob, inputs_ids


class TextChunker:
    """Text chunker for cases where text exceeds the token limit."""

    def __init__(self, model_name):
        """
        Initialize the TextChunker with a tokenizer.

        Args:
            model_name (str): Name of the pre-trained model.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def chunk_text(self, text, max_length, stride):
        """
        Chunk the input text into smaller parts for tokenization.

        Args:
            text (str or pd.DataFrame): The text to chunk.
            max_length (int): Maximum length of each chunk.
            stride (int): The stride for moving through the text.

        Returns:
            list: List of chunks.
        """
        if isinstance(text, pd.DataFrame):
            text = str(text.iloc[0])
        else:
            text = str(text)

        tokens = self.tokenizer.encode_plus(
            text,
            add_special_tokens=False,
            max_length=None,
            truncation=False,
            return_attention_mask=False,
            return_token_type_ids=False
        )['input_ids']

        chunks = []
        for i in range(0, len(tokens), stride):
            chunk = tokens[i:i + max_length]
            if len(chunk) < max_length:
                chunk += [self.tokenizer.pad_token_id] * (max_length - len(chunk))
            chunks.append(chunk)
        return chunks


def tokenize_chunk(chunks, labels_y, pad_token_id):
    """
    Tokenize the chunks and return the padded tensors for input IDs and attention masks.

    Args:
        chunks (list): List of tokenized text chunks.
        labels_y (list): Corresponding labels for the chunks.
        pad_token_id (int): The ID of the padding token.

    Returns:
        dict: A dictionary with padded input_ids, attention_masks, and chunk_attention_mask.
    """
    input_ids = [torch.tensor(chunk, dtype=torch.long) for chunk in chunks]
    attention_masks = [
        torch.tensor([1 if token_id != pad_token_id else 0 for token_id in chunk], dtype=torch.long)
        for chunk in chunks
    ]

    batch_size = len(input_ids)
    max_chunks = len(input_ids)
    seq_length = max(len(chunk) for chunk in chunks)

    padded_input_ids = torch.zeros((batch_size, max_chunks, seq_length), dtype=torch.long)
    padded_attention_masks = torch.zeros((batch_size, max_chunks, seq_length), dtype=torch.long)
    chunk_attention_mask = torch.zeros((batch_size, max_chunks), dtype=torch.long)

    for i, (ids, mask) in enumerate(zip(input_ids, attention_masks)):
        num_chunks = 1
        padded_input_ids[i, :num_chunks, :len(ids)] = ids
        padded_attention_masks[i, :num_chunks, :len(mask)] = mask
        chunk_attention_mask[i, :num_chunks] = 1

    labels = torch.tensor(labels_y, dtype=torch.long)

    return {
        "input_ids": padded_input_ids,
        "attention_mask": padded_attention_masks,
        "labels": labels,
        "chunk_attention_mask": chunk_attention_mask
    }


def predictions_model_value(model, model_token_name, column_data_text, column_target_variable, device, MAX_TOKEN_LEN,
                            pad_token_ids=1, THRESHOLD_PROBABILITIES_MODEL=0.5):
    """
    Make predictions using the trained model.

    Args:
        model: The trained model.
        model_token_name (str): Name of the pre-trained tokenizer.
        column_data_text (str): Column containing text data.
        column_target_variable (str): Column containing target variable data.
        device: Device (CPU or GPU) to run the model.
        MAX_TOKEN_LEN (int): Maximum token length for each chunk.
        pad_token_ids (int, optional): ID for the padding token. Default is 1.
        THRESHOLD_PROBABILITIES_MODEL (float, optional): Threshold for making predictions. Default is 0.5.

    Returns:
        tuple: logits, preds, input_ids, attention_masks, labels.
    """
    text_chunker = TextChunker(model_token_name)
    chunks = text_chunker.chunk_text(column_data_text, max_length=MAX_TOKEN_LEN, stride=MAX_TOKEN_LEN // 2)

    batch_data_prediction = tokenize_chunk(
        chunks=chunks,
        labels_y=column_target_variable,
        pad_token_id=pad_token_ids
    )

    model.eval()
    with torch.no_grad():
        batch_data_prediction = {k: v.to(device) for k, v in batch_data_prediction.items()}

        input_ids = batch_data_prediction["input_ids"]
        attention_masks = batch_data_prediction["attention_mask"]
        chunk_attention_mask = batch_data_prediction["chunk_attention_mask"]
        labels = batch_data_prediction["labels"]

        logits = model(input_ids, attention_masks, chunk_attention_mask)
        preds = (logits > THRESHOLD_PROBABILITIES_MODEL).float()

    return logits, preds, input_ids, attention_masks, labels
