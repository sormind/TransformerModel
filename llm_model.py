"""
Large Language Model (LLM) Implementation - GPT-style Architecture

This file contains a decoder-only transformer for language modeling, similar to GPT.
Unlike the translation model, this uses only decoder blocks for autoregressive generation.

Key differences from translation model:
- Decoder-only (no encoder)
- Causal attention masks for autoregressive generation
- Single vocabulary (not bilingual)
- Next-token prediction objective
"""

import torch
import torch.nn as nn
import math
from model import (
    LayerNormalization, 
    FeedForwardBlock, 
    InputEmbeddings, 
    PositionalEncoding,
    MultiHeadAttentionBlock,
    ResidualConnection
)

class LLMDecoderBlock(nn.Module):
    """
    LLM Decoder Block - Simplified from translation decoder
    
    Only has self-attention (no cross-attention like in translation)
    Uses causal masking to prevent looking at future tokens
    """
    
    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock, 
                 feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        # Only 2 residual connections (self-attention + feed-forward)
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(2)])

    def forward(self, x, mask):
        # Self-attention with causal mask
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, mask))
        # Feed-forward
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x

class LLMDecoder(nn.Module):
    """
    Stack of LLM decoder blocks
    """
    
    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class LanguageModel(nn.Module):
    """
    Complete Language Model (GPT-style)
    
    Architecture:
    Input → Embeddings → Positional Encoding → Decoder Stack → Language Head → Logits
    """
    
    def __init__(self, decoder: LLMDecoder, embed: InputEmbeddings, 
                 pos: PositionalEncoding, lm_head: nn.Linear) -> None:
        super().__init__()
        self.decoder = decoder
        self.embed = embed
        self.pos = pos
        self.lm_head = lm_head  # Projects to vocabulary for next-token prediction

    def forward(self, x, mask=None):
        """
        Forward pass for language modeling
        
        Args:
            x: Input token indices (batch, seq_len)
            mask: Causal attention mask (optional, will create if None)
            
        Returns:
            Logits for next token prediction (batch, seq_len, vocab_size)
        """
        # Create causal mask if not provided
        if mask is None:
            mask = causal_mask(x.size(1)).to(x.device)
            
        # Embeddings + positional encoding
        x = self.embed(x)  # (batch, seq_len, d_model)
        x = self.pos(x)    # Add positional information
        
        # Through decoder stack
        x = self.decoder(x, mask)  # (batch, seq_len, d_model)
        
        # Project to vocabulary
        logits = self.lm_head(x)   # (batch, seq_len, vocab_size)
        
        return logits

    def generate(self, input_ids, max_length=100, temperature=1.0, top_k=50):
        """
        Generate text autoregressively
        
        Args:
            input_ids: Starting tokens (batch, seq_len)
            max_length: Maximum sequence length to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Only sample from top-k most likely tokens
            
        Returns:
            Generated token sequence
        """
        self.eval()
        generated = input_ids.clone()
        
        with torch.no_grad():
            for _ in range(max_length - input_ids.size(1)):
                # Get logits for next token
                logits = self.forward(generated)[:, -1, :]  # (batch, vocab_size)
                
                # Apply temperature
                logits = logits / temperature
                
                # Top-k sampling
                if top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(logits, top_k)
                    logits = torch.full_like(logits, float('-inf'))
                    logits.scatter_(1, top_k_indices, top_k_logits)
                
                # Sample next token
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                
                # Append to sequence
                generated = torch.cat([generated, next_token], dim=1)
                
                # Stop if we hit end token (you'd define this)
                # if next_token.item() == eos_token_id:
                #     break
                    
        return generated

def causal_mask(size):
    """
    Create causal attention mask for autoregressive generation
    Prevents attending to future positions
    """
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.bool)
    return ~mask  # Invert so True = can attend, False = cannot attend

def build_language_model(vocab_size: int, seq_len: int, d_model: int = 512, 
                        N: int = 12, h: int = 8, dropout: float = 0.1, 
                        d_ff: int = 2048) -> LanguageModel:
    """
    Build a complete language model (GPT-style)
    
    Args:
        vocab_size: Size of vocabulary
        seq_len: Maximum sequence length
        d_model: Model dimension (embedding size)
        N: Number of decoder layers
        h: Number of attention heads
        dropout: Dropout rate
        d_ff: Feed-forward dimension
        
    Returns:
        Complete LanguageModel ready for training
    """
    # Create embeddings and positional encoding
    embed = InputEmbeddings(d_model, vocab_size)
    pos = PositionalEncoding(d_model, seq_len, dropout)
    
    # Create decoder blocks
    decoder_blocks = []
    for _ in range(N):
        self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = LLMDecoderBlock(d_model, self_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)
    
    # Create decoder stack
    decoder = LLMDecoder(d_model, nn.ModuleList(decoder_blocks))
    
    # Language modeling head (projects to vocabulary)
    lm_head = nn.Linear(d_model, vocab_size)
    
    # Create complete model
    model = LanguageModel(decoder, embed, pos, lm_head)
    
    # Initialize parameters
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return model

# Example usage and comparison
if __name__ == "__main__":
    print("🤖 Language Model (LLM) vs Translation Model")
    print("=" * 60)
    
    # Model parameters
    vocab_size = 10000
    seq_len = 512
    d_model = 512
    
    # Build LLM
    llm = build_language_model(vocab_size, seq_len, d_model, N=6, h=8)
    
    # Example input (batch_size=1, seq_len=10)
    input_ids = torch.randint(0, vocab_size, (1, 10))
    
    print(f"Input shape: {input_ids.shape}")
    
    # Forward pass
    with torch.no_grad():
        logits = llm(input_ids)
        
    print(f"Output logits shape: {logits.shape}")
    print(f"Model parameters: {sum(p.numel() for p in llm.parameters()):,}")
    
    print("\n🔍 Key Differences:")
    print("Translation Model: Encoder-Decoder, Cross-attention, Fixed I/O")
    print("Language Model: Decoder-only, Self-attention, Autoregressive")
    
    print("\n🎯 Next Steps:")
    print("1. Create training script for language modeling")
    print("2. Use text datasets (not translation pairs)")
    print("3. Implement text generation methods")
    print("4. Add training optimizations (gradient checkpointing, etc.)") 