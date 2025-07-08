"""
Attention Visualization for Transformer Model

This script provides educational visualization of attention patterns in a trained transformer model.
It helps you understand:
1. How self-attention works in encoder and decoder
2. How cross-attention connects source and target languages  
3. What different attention heads learn to focus on
4. How attention patterns change across layers

The visualizations show attention scores as heatmaps where:
- Rows represent query positions (what is paying attention)
- Columns represent key positions (what is being attended to)
- Color intensity shows attention weight (brighter = more attention)
"""

import torch
import torch.nn as nn
from model import Transformer
from config import get_config, get_weights_file_path
from train import get_model, get_ds, greedy_decode
import altair as alt
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

config = get_config()
train_dataloader, val_dataloader, vocab_src, vocab_tgt = get_ds(config)
model = get_model(config, vocab_src.get_vocab_size(), vocab_tgt.get_vocab_size()).to(device)

# Load the pretrained weights
model_filename = get_weights_file_path(config, f"29")
state = torch.load(model_filename)
model.load_state_dict(state['model_state_dict'])


def load_next_batch():
    """
    Load the next batch from the validation dataloader, process it through the model,
    and return the batch along with the tokenized inputs.

    Returns:
        tuple: Contains the batch, encoder input tokens, and decoder input tokens.

    The function extracts the next batch from the validation dataloader, moves the
    data to the appropriate device (CPU or GPU), and uses the vocabularies to convert
    the token indices to the actual tokens for both the encoder and decoder inputs.
    It ensures the batch size is 1 for validation.

    It also performs a greedy decoding of the model output and returns the processed batch.
    """
    # Load a sample batch from the validation set
    batch = next(iter(val_dataloader))
    encoder_input = batch["encoder_input"].to(device)
    encoder_mask = batch["encoder_mask"].to(device)
    decoder_input = batch["decoder_input"].to(device)
    decoder_mask = batch["decoder_mask"].to(device)

    # Convert indices to tokens using the source and target vocabularies
    encoder_input_tokens = [vocab_src.id_to_token(idx) for idx in encoder_input[0].cpu().numpy()]
    decoder_input_tokens = [vocab_tgt.id_to_token(idx) for idx in decoder_input[0].cpu().numpy()]

    # Ensure that the batch size is 1 for validation
    assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"

    # Perform greedy decoding to obtain model output
    model_out = greedy_decode(
        model, encoder_input, encoder_mask, vocab_src, vocab_tgt, config['seq_len'], device)

    return batch, encoder_input_tokens, decoder_input_tokens


def mtx2df(m, max_row, max_col, row_tokens, col_tokens):
    """
    Convert a matrix into a DataFrame suitable for plotting attention maps.

    Args:
        m (torch.Tensor): The attention score matrix.
        max_row (int): Maximum number of rows to include.
        max_col (int): Maximum number of columns to include.
        row_tokens (list): List of tokens for the rows.
        col_tokens (list): List of tokens for the columns.

    Returns:
        pd.DataFrame: DataFrame containing the attention scores and token labels.

    This function constructs a DataFrame from the attention score matrix, limiting the
    number of rows and columns based on max_row and max_col. It includes additional
    information about the row and column tokens for visualization.
    """
    return pd.DataFrame(
        [
            (
                r,
                c,
                float(m[r, c]),
                "%.3d %s" % (r, row_tokens[r] if len(row_tokens) > r else "<blank>"),
                "%.3d %s" % (c, col_tokens[c] if len(col_tokens) > c else "<blank>"),
            )
            for r in range(m.shape[0])
            for c in range(m.shape[1])
            if r < max_row and c < max_col
        ],
        columns=["row", "column", "value", "row_token", "col_token"],
    )


def get_attn_map(attn_type: str, layer: int, head: int) -> torch.Tensor:
    """
    Retrieve the attention scores for a specific layer and head.

    Args:
        attn_type (str): Type of attention ('encoder', 'decoder', or 'encoder-decoder').
        layer (int): The layer index from which to retrieve attention scores.
        head (int): The attention head index.

    Returns:
        torch.Tensor: The attention scores for the specified layer and head.

    Depending on the attn_type, this function extracts the appropriate attention scores
    from the model's encoder or decoder layers for visualization purposes.
    """
    try:
        if attn_type == "encoder":
            # Encoder self-attention
            attn = model.encoder.layers[layer].self_attention_block.attention_scores
        elif attn_type == "decoder":
            # Decoder self-attention
            attn = model.decoder.layers[layer].self_attention_block.attention_scores
        elif attn_type == "encoder-decoder":
            # Encoder-decoder cross-attention
            attn = model.decoder.layers[layer].cross_attention_block.attention_scores
        else:
            raise ValueError(f"Unknown attention type: {attn_type}")
            
        if attn is None:
            raise ValueError(f"No attention scores found for {attn_type} layer {layer}")
            
        return attn[0, head].data
        
    except Exception as e:
        print(f"Error retrieving attention scores for {attn_type} layer {layer} head {head}: {e}")
        # Return a small dummy matrix to prevent crashes
        return torch.zeros(10, 10)


def attn_map(attn_type: str, layer: int, head: int, row_tokens: list, col_tokens: list, max_sentence_len: int):
    """
    Create an attention map for a specific layer and head using Altair for visualization.

    Args:
        attn_type (str): Type of attention ('encoder', 'decoder', or 'encoder-decoder').
        layer (int): The layer index from which to visualize attention scores.
        head (int): The attention head index.
        row_tokens (list): Tokens for the rows.
        col_tokens (list): Tokens for the columns.
        max_sentence_len (int): Maximum sentence length to visualize.

    Returns:
        alt.Chart: The Altair chart representing the attention map.

    This function generates a heatmap of attention scores for a specific attention head
    and layer, using Altair for interactive visualization.
    """
    df = mtx2df(
        get_attn_map(attn_type, layer, head),
        max_sentence_len,
        max_sentence_len,
        row_tokens,
        col_tokens,
    )
    return (
        alt.Chart(data=df)
        .mark_rect()
        .encode(
            x=alt.X("col_token", axis=alt.Axis(title="")),
            y=alt.Y("row_token", axis=alt.Axis(title="")),
            color="value",
            tooltip=["row", "column", "value", "row_token", "col_token"],
        )
        .properties(height=400, width=400, title=f"Layer {layer} Head {head}")
        .interactive()
    )


def get_all_attention_maps(
    attn_type: str, layers: list[int], heads: list[int], row_tokens: list, col_tokens: list, max_sentence_len: int
):
    """
    Generate all attention maps for specified layers and heads.

    Args:
        attn_type (str): Type of attention ('encoder', 'decoder', or 'encoder-decoder').
        layers (list[int]): List of layers to visualize.
        heads (list[int]): List of heads to visualize.
        row_tokens (list): Tokens for the rows.
        col_tokens (list): Tokens for the columns.
        max_sentence_len (int): Maximum sentence length to visualize.

    Returns:
        alt.VConcatChart: Concatenated Altair charts for all specified attention maps.

    This function loops over the specified layers and heads to create attention maps for each
    combination, using the attn_map function. It combines the individual maps into a single
    visualization for easier comparison.
    """
    charts = []
    for layer in layers:
        rowCharts = []
        for head in heads:
            rowCharts.append(attn_map(attn_type, layer, head, row_tokens, col_tokens, max_sentence_len))
        charts.append(alt.hconcat(*rowCharts))
    return alt.vconcat(*charts)


# Load the next batch for visualization
batch, encoder_input_tokens, decoder_input_tokens = load_next_batch()

# Print the source and target sentences for context
print(f'Source: {batch["src_text"][0]}')
print(f'Target: {batch["tgt_text"][0]}')
print("\n" + "="*80)
print("ATTENTION VISUALIZATION GUIDE:")
print("="*80)
print("1. ENCODER SELF-ATTENTION: Shows how source words relate to each other")
print("   - Look for patterns like attending to nearby words or specific grammatical relationships")
print("\n2. DECODER SELF-ATTENTION: Shows how target words attend to previous target words")
print("   - Should show causal (triangular) patterns due to masking")
print("\n3. ENCODER-DECODER ATTENTION: Shows source-target word alignments")
print("   - This is where translation magic happens! Look for word alignment patterns")
print("="*80 + "\n")

# Determine sentence length for visualization (find where padding starts)
sentence_len = encoder_input_tokens.index("[PAD]") if "[PAD]" in encoder_input_tokens else len(encoder_input_tokens)

# Specify layers and heads to visualize
# Note: Different layers capture different types of linguistic relationships
layers = [0, 1, 2]  # Early, middle, and later layers
heads = [0, 1, 2, 3, 4, 5, 6, 7]  # All 8 attention heads

print("🔍 ENCODER SELF-ATTENTION PATTERNS:")
print("   (How source words attend to other source words)")
get_all_attention_maps("encoder", layers, heads, encoder_input_tokens, encoder_input_tokens, min(20, sentence_len))

print("\n🔍 DECODER SELF-ATTENTION PATTERNS:")
print("   (How target words attend to previous target words - note the triangular mask)")
get_all_attention_maps("decoder", layers, heads, decoder_input_tokens, decoder_input_tokens, min(20, sentence_len))

print("\n🔍 ENCODER-DECODER ATTENTION PATTERNS:")
print("   (Source-target word alignments - the core of translation!)")
get_all_attention_maps("encoder-decoder", layers, heads, encoder_input_tokens, decoder_input_tokens, min(20, sentence_len))

