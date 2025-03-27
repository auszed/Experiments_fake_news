# General libraries
import pandas as pd
import numpy as np
import string
from typing import Tuple, List

# Visualization libraries
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import plotly.graph_objects as go

# NLP libraries
from nltk.corpus import stopwords
import nltk
from transformers import PreTrainedTokenizer, BertTokenizer, BertModel, PreTrainedTokenizerFast

# Dimensionality reduction and statistics
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy import stats

# PyTorch
import torch

# NLTK setup
nltk.download('stopwords')



# colorama of the charts
custom_colors = ['#36CE8A', "#7436F5","#3736F4",   "#36AEF5", "#B336F5", "#f8165e", "#36709A",  "#3672F5", "#7ACE5D"]
gradient_colors = [ "#36CE8A", '#7436F5']
color_palette_custom  = sns.set_palette(custom_colors)
theme_color = sns.color_palette(color_palette_custom, 9)
cmap_theme = LinearSegmentedColormap.from_list('custom_colormap', gradient_colors)

# color to manage in the schema
theme_color


def proportion_balance_classes(names: pd.Index, values: np.ndarray) -> None:
    """We plot the proportion of each class in each row."""

    plt.figure(figsize=(15, 4))
    ax = sns.barplot(x=names, y=values, alpha=0.8)
    plt.title("# per class")
    plt.ylabel('# of Occurrences', fontsize=12)
    plt.xlabel('Type', fontsize=12)

    rects = ax.patches
    for rect, label in zip(rects, values):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2, height + 5, f'{label:.2f}', ha='center', va='bottom')

    plt.show()

    return


def histogram_bins(names: pd.Index, values: np.ndarray, bins: int = 10, title: str = "Proportion per Bin") -> None:
    """Plot the proportion of each class in specified bins with integer value ranges on the x-axis and an optional title."""

    # Bin the values and get bin edges for creating range labels
    bin_ranges = pd.cut(values, bins=bins)
    binned_values = bin_ranges.value_counts().sort_index()

    # Generate integer range labels for x-axis based on bin edges
    bin_labels = [f"{int(interval.left)} - {int(interval.right)}" for interval in bin_ranges.categories]
    bin_counts = binned_values.values

    plt.figure(figsize=(15, 4))
    ax = sns.barplot(x=bin_labels, y=bin_counts, alpha=0.8)
    plt.title(title)  # Set the custom or default title
    plt.ylabel('# of Occurrences', fontsize=12)
    plt.xlabel('Value Ranges', fontsize=12)

    # Annotate each bar with integer counts
    rects = ax.patches
    for rect, label in zip(rects, bin_counts):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2, height + 5, f'{int(label)}', ha='center', va='bottom')

    plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for readability
    plt.show()

    return


def sliding_window_tokenize(text, model_tokenizer, max_length=512, stride=256):
    """Helper function: Sliding window tokenization"""
    tokens = model_tokenizer.encode(text, add_special_tokens=True)
    chunks = []
    for i in range(0, len(tokens), stride):
        chunk = tokens[i:i + max_length]
        if len(chunk) < max_length:  # Pad if chunk is smaller than max_length
            chunk += [model_tokenizer.pad_token_id] * (max_length - len(chunk))
        chunks.append(chunk)
    return chunks
def plot_distribution_tokens_per_word(
    model_tokenizer: PreTrainedTokenizer,
    data_series: pd.Series,
    number_words: int = 20,
    max_length: int = 512,
    stride: int = 256
) -> List:
    """
    Tokenize and plot the distributions of token counts, excluding special tokens like [PAD].
    Returns:
        List: A list of all tokens from the input data, excluding special tokens.
    """
    tokens = []

    # Tokenize and collect tokens
    for text in data_series:
        chunks = sliding_window_tokenize(text, model_tokenizer, max_length, stride)
        for chunk in chunks:
            # Convert chunk IDs back to tokens
            chunk_tokens = model_tokenizer.convert_ids_to_tokens(chunk)
            # Exclude special tokens like [PAD]
            chunk_tokens = [token for token in chunk_tokens if token not in model_tokenizer.all_special_tokens]
            tokens.extend(chunk_tokens)

    # Count the distribution of tokens
    token_counts = pd.Series(tokens).value_counts()
    token_counts = token_counts.head(number_words)

    # Plot the token distribution
    plt.figure(figsize=(20, 6))
    sns.barplot(x=token_counts.index, y=token_counts.values)
    plt.title('Token Distribution')
    plt.xlabel('Tokens')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)
    plt.show()

    return tokens


def extractions_text_description(dataset: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """We will be adding more features to the data"""
    # import values

    # Define English stopwords
    eng_stopwords = set(stopwords.words('english'))

    # Create a copy of the original dataset
    df_eda_description = dataset.copy()

    # Word count in each comment:
    df_eda_description[f'{column_name}_count_each_word'] = df_eda_description[column_name].apply(lambda x: len(str(x).split()))
    df_eda_description[f'{column_name}_count_unique_word'] = df_eda_description[column_name].apply(lambda x: len(set(str(x).split())))
    df_eda_description[f'{column_name}_count_punctuations'] = df_eda_description[column_name].apply(
        lambda x: len([c for c in str(x) if c in string.punctuation]))
    df_eda_description[f'{column_name}_count_words_title'] = df_eda_description[column_name].apply(
        lambda x: len([w for w in str(x).split() if w.istitle()]))
    df_eda_description[f'{column_name}_count_stopwords'] = df_eda_description[column_name].apply(
        lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords]))
    df_eda_description[f'{column_name}_mean_word_len'] = df_eda_description[column_name].apply(
        lambda x: np.mean([len(w) for w in str(x).split()]))

    # Additional features from the original dataset
    df_eda_description[f'{column_name}_total_length'] = df_eda_description[column_name].str.len()
    df_eda_description[f'{column_name}_new_line'] = df_eda_description[column_name].str.count('\n' * 1)
    df_eda_description[f'{column_name}_new_small_space'] = df_eda_description[column_name].str.count('\n' * 2)
    df_eda_description[f'{column_name}_new_medium_space'] = df_eda_description[column_name].str.count('\n' * 3)
    df_eda_description[f'{column_name}_new_big_space'] = df_eda_description[column_name].str.count('\n' * 4)

    # Uppercase words count
    df_eda_description[f'{column_name}_uppercase_words'] = df_eda_description[column_name].apply(
        lambda l: sum(map(str.isupper, list(l))))
    df_eda_description[f'{column_name}_question_mark'] = df_eda_description[column_name].str.count('\?')
    df_eda_description[f'{column_name}_exclamation_mark'] = df_eda_description[column_name].str.count('!')

    # Derived features
    df_eda_description[f'{column_name}_word_unique_percent'] = df_eda_description[f'{column_name}_count_unique_word'] * 100 / df_eda_description[
        f'{column_name}_count_each_word']
    df_eda_description[f'{column_name}_punctuations_percent'] = df_eda_description[f'{column_name}_count_punctuations'] * 100 / df_eda_description[
        f'{column_name}_count_each_word']

    return df_eda_description


def components_pca_3d_chart(dataset: pd.DataFrame, tokenizer_model: PreTrainedTokenizerFast, model_import,
                            max_length_pca: int):
    """Dimensionality reduction with PCA to understand how the columns perform."""
    filtered_embeddings = []
    filtered_tokens = []

    # Loop over all texts in the dataset
    for text in dataset:
        # Tokenize the text into chunks
        token_chunks = sliding_window_tokenize(text, tokenizer_model, max_length=max_length_pca)

        # Process each token chunk
        for chunk in token_chunks:
            inputs = torch.tensor([chunk])  # Ensure correct shape for model input
            with torch.no_grad():
                # Pass the chunk through the BERT model
                outputs = model_import(inputs)
                hidden_states = outputs.last_hidden_state

            # Convert tokens and embeddings for the chunk
            tokens = tokenizer_model.convert_ids_to_tokens(chunk)
            token_embeddings = hidden_states.squeeze(0).numpy()

            # Filter out special tokens like [PAD], [CLS], and [SEP]
            for token, embedding in zip(tokens, token_embeddings):
                if token not in ["[PAD]", "[CLS]", "[SEP]"]:
                    filtered_tokens.append(token)
                    filtered_embeddings.append(embedding)

    # Convert filtered embeddings to a numpy array for PCA
    filtered_embeddings = np.array(filtered_embeddings)

    # Perform PCA to reduce dimensions to 3D for visualization
    pca = PCA(n_components=3)
    reduced_embeddings = pca.fit_transform(filtered_embeddings)

    # Apply log scale to the third component (for coloring purposes)
    log_color_scale = np.log(reduced_embeddings[:, 2] - np.min(reduced_embeddings[:, 2]) + 1)

    # Create a 3D scatter plot using Plotly
    fig = go.Figure(data=[go.Scatter3d(
        x=reduced_embeddings[:, 0],
        y=reduced_embeddings[:, 1],
        z=reduced_embeddings[:, 2],
        mode='markers+text',
        text=filtered_tokens,  # Display tokens as text labels
        marker=dict(
            size=5,
            color=log_color_scale,  # Use the log-transformed scale for color
            colorscale='Viridis',
            opacity=0.8
        )
    )])

    # Set plot titles and labels
    fig.update_layout(
        title="3D PCA of Token Embeddings with Log Scale for Color",
        scene=dict(
            xaxis_title="PCA Component 1",
            yaxis_title="PCA Component 2",
            zaxis_title="PCA Component 3"
        )
    )

    # Show the plot
    fig.show()

def dimensionality_reduction_chart_tsne(text_comments: List[str], dataset: pd.DataFrame, df_column_target: str,
                                            tokenizer_model, model_import):
        """Performs dimensionality reduction using t-SNE and visualizes the sentence embeddings."""

        # Tokenize input
        inputs = tokenizer_model(text_comments, return_tensors="pt", padding=True, truncation=True)

        # Extract BERT embeddings
        with torch.no_grad():
            outputs = model_import(**inputs)
            hidden_states = outputs.last_hidden_state

        # Convert to numpy array for t-SNE (averaging token embeddings)
        sentence_embeddings = hidden_states.mean(dim=1).cpu().numpy()

        # Use t-SNE to reduce to 3 dimensions
        tsne = TSNE(n_components=3, random_state=42, perplexity=5)
        embeddings_3d = tsne.fit_transform(sentence_embeddings)

        # Create a color list (blue for real (0), red for fake (1))
        colors = [custom_colors[0] if target == 0 else custom_colors[1] for target in dataset[df_column_target]]

        # Create a 3D scatter plot with Plotly
        fig = go.Figure(data=[go.Scatter3d(
            x=embeddings_3d[:, 0],
            y=embeddings_3d[:, 1],
            z=embeddings_3d[:, 2],
            mode='markers+text',
            marker=dict(size=6, color=colors, opacity=0.8),
            text=[f"Comment {i + 1}" for i in range(len(text_comments))],
            textposition="top center"
        )])

        # Update layout
        fig.update_layout(
            scene=dict(
                xaxis_title='Dimension 1',
                yaxis_title='Dimension 2',
                zaxis_title='Dimension 3'
            ),
            title="3D t-SNE visualization of BERT sentence embeddings"
        )

        # Show the plot
        fig.show()
