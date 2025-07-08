"""
Educational Examples for Understanding Transformers

This script provides step-by-step examples to understand how transformer components work.
Run different functions to explore specific concepts interactively.
"""

import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt
from model import MultiHeadAttentionBlock, PositionalEncoding, LayerNormalization

def example_1_attention_mechanism():
    """
    Example 1: Understanding the Attention Mechanism
    
    This example shows how attention works with a simple sentence.
    """
    print("="*60)
    print("EXAMPLE 1: How Attention Works")
    print("="*60)
    
    # Simple example sentence: "The cat sat on the mat"
    sentence = ["The", "cat", "sat", "on", "the", "mat"]
    print(f"Sentence: {' '.join(sentence)}")
    
    # Simplified attention scores (manually created for illustration)
    # Each row shows how much each word attends to every word
    attention_matrix = torch.tensor([
        #    The   cat   sat   on   the   mat
        [0.2, 0.1, 0.1, 0.1, 0.4, 0.1],  # The
        [0.1, 0.6, 0.2, 0.0, 0.0, 0.1],  # cat  
        [0.1, 0.3, 0.4, 0.1, 0.0, 0.1],  # sat
        [0.1, 0.0, 0.2, 0.3, 0.1, 0.3],  # on
        [0.4, 0.1, 0.1, 0.1, 0.2, 0.1],  # the
        [0.1, 0.2, 0.1, 0.2, 0.1, 0.3],  # mat
    ])
    
    print("\nAttention Matrix (how much each word attends to others):")
    print("Rows=Query words, Columns=Key words")
    print("     ", end="")
    for word in sentence:
        print(f"{word:>6}", end="")
    print()
    
    for i, word in enumerate(sentence):
        print(f"{word:>4}:", end="")
        for j in range(len(sentence)):
            print(f"{attention_matrix[i,j].item():6.2f}", end="")
        print()
    
    print("\nInterpretation:")
    print("- 'cat' attends strongly to itself (0.60) - self-attention")
    print("- 'sat' attends to 'cat' (0.30) - verb attending to subject")
    print("- 'The' words attend to each other (0.40) - similar function words")
    print("- 'on' attends to 'mat' (0.30) - preposition attending to object")

def example_2_positional_encoding():
    """
    Example 2: Understanding Positional Encoding
    
    Shows how transformers add position information to word embeddings.
    """
    print("\n" + "="*60)
    print("EXAMPLE 2: Positional Encoding")
    print("="*60)
    
    d_model = 8  # Small model dimension for illustration
    seq_len = 6  # Length of our example sentence
    
    # Create positional encoding
    pos_encoding = PositionalEncoding(d_model, seq_len, dropout=0.0)
    
    # Extract the positional encoding matrix
    pe_matrix = pos_encoding.pe.squeeze(0)  # Remove batch dimension
    
    print(f"Positional encodings for {seq_len} positions with d_model={d_model}")
    print("Each row represents the positional encoding for one position:")
    print("Columns alternate between sin and cos functions")
    
    for pos in range(seq_len):
        print(f"Position {pos}: ", end="")
        for dim in range(d_model):
            print(f"{pe_matrix[pos, dim].item():6.3f}", end=" ")
        print()
    
    print("\nKey insights:")
    print("- Each position gets a unique encoding pattern")
    print("- Sin/cos functions create patterns the model can learn")
    print("- Similar positions have similar (but not identical) encodings")
    
    # Visualize if matplotlib is available
    try:
        plt.figure(figsize=(10, 6))
        plt.imshow(pe_matrix.numpy(), cmap='RdBu', aspect='auto')
        plt.title('Positional Encoding Visualization')
        plt.xlabel('Model Dimension')
        plt.ylabel('Position')
        plt.colorbar()
        plt.show()
        print("- See the visualization above for the sin/cos patterns!")
    except:
        print("- Install matplotlib to see the visualization")

def example_3_multi_head_attention():
    """
    Example 3: Multi-Head Attention in Action
    
    Shows how multiple attention heads can focus on different relationships.
    """
    print("\n" + "="*60)
    print("EXAMPLE 3: Multi-Head Attention")
    print("="*60)
    
    d_model = 8
    num_heads = 2
    seq_len = 4
    
    # Create a small multi-head attention block
    mha = MultiHeadAttentionBlock(d_model, num_heads, dropout=0.0)
    
    # Create dummy input (batch_size=1, seq_len=4, d_model=8)
    x = torch.randn(1, seq_len, d_model)
    
    print(f"Input shape: {x.shape}")
    print(f"Model dimension: {d_model}")
    print(f"Number of heads: {num_heads}")
    print(f"Dimension per head: {d_model // num_heads}")
    
    # Forward pass
    with torch.no_grad():
        output = mha(x, x, x, mask=None)  # Self-attention (Q=K=V=x)
        
    print(f"Output shape: {output.shape}")
    
    # Access attention scores from the last forward pass
    attention_scores = mha.attention_scores  # Shape: (batch, heads, seq_len, seq_len)
    
    print(f"Attention scores shape: {attention_scores.shape}")
    print("\nAttention patterns for each head:")
    
    for head in range(num_heads):
        print(f"\nHead {head} attention matrix:")
        head_attention = attention_scores[0, head]  # Remove batch dimension
        for i in range(seq_len):
            print(f"  Pos {i}: ", end="")
            for j in range(seq_len):
                print(f"{head_attention[i,j].item():6.3f}", end=" ")
            print()
    
    print("\nKey insights:")
    print("- Each head learns different attention patterns")
    print("- Multiple heads capture different types of relationships")
    print("- Final output combines information from all heads")

def example_4_layer_normalization():
    """
    Example 4: Layer Normalization
    
    Shows how layer normalization stabilizes training.
    """
    print("\n" + "="*60)
    print("EXAMPLE 4: Layer Normalization")
    print("="*60)
    
    d_model = 4
    layer_norm = LayerNormalization(d_model)
    
    # Create input with different scales
    x1 = torch.tensor([[1.0, 2.0, 3.0, 4.0]])  # Small values
    x2 = torch.tensor([[100.0, 200.0, 300.0, 400.0]])  # Large values
    x3 = torch.tensor([[1.0, 100.0, 2.0, 200.0]])  # Mixed values
    
    print("Input examples:")
    print(f"Small values:  {x1}")
    print(f"Large values:  {x2}")
    print(f"Mixed values:  {x3}")
    
    with torch.no_grad():
        norm1 = layer_norm(x1)
        norm2 = layer_norm(x2)
        norm3 = layer_norm(x3)
    
    print("\nAfter layer normalization:")
    print(f"Small values:  {norm1}")
    print(f"Large values:  {norm2}")
    print(f"Mixed values:  {norm3}")
    
    print("\nKey insights:")
    print("- All outputs have mean ≈ 0 and standard deviation ≈ 1")
    print("- Relative relationships are preserved")
    print("- This helps with training stability and gradient flow")

def run_all_examples():
    """Run all educational examples in sequence."""
    print("🎓 TRANSFORMER EDUCATIONAL EXAMPLES")
    print("🎓 Understanding Transformers Step by Step\n")
    
    example_1_attention_mechanism()
    example_2_positional_encoding()
    example_3_multi_head_attention()
    example_4_layer_normalization()
    
    print("\n" + "="*60)
    print("🎉 Congratulations! You've explored key transformer concepts!")
    print("="*60)
    print("\nNext steps:")
    print("1. Run attention.py to see real attention patterns")
    print("2. Train your own model with train.py")
    print("3. Experiment with different architectures")
    print("4. Read the 'Attention is All You Need' paper")

if __name__ == "__main__":
    # You can run specific examples or all of them
    run_all_examples()
    
    # Or run individual examples:
    # example_1_attention_mechanism()
    # example_2_positional_encoding() 
    # example_3_multi_head_attention()
    # example_4_layer_normalization() 