"""
Training Script for Language Model (LLM)

This script trains a GPT-style language model for text generation.
Unlike translation training, this uses:
- Text datasets (not translation pairs)
- Next-token prediction objective
- Causal attention masking
- Autoregressive generation
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from datasets import load_dataset
import math
from tqdm import tqdm
import os
from pathlib import Path

from llm_model import build_language_model, causal_mask

class TextDataset(Dataset):
    """
    Dataset for language modeling
    Converts text into sequences for next-token prediction
    """
    
    def __init__(self, texts, tokenizer, seq_len, stride=None):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.stride = stride or seq_len // 2  # Overlap for more training data
        
        # Tokenize all texts
        self.token_sequences = []
        for text in tqdm(texts, desc="Tokenizing texts"):
            tokens = tokenizer.encode(text).ids
            
            # Split into overlapping sequences
            for i in range(0, len(tokens) - seq_len, self.stride):
                sequence = tokens[i:i + seq_len + 1]  # +1 for target
                if len(sequence) == seq_len + 1:
                    self.token_sequences.append(sequence)
    
    def __len__(self):
        return len(self.token_sequences)
    
    def __getitem__(self, idx):
        sequence = self.token_sequences[idx]
        
        # Input = all tokens except last
        # Target = all tokens except first (shifted by 1)
        input_ids = torch.tensor(sequence[:-1], dtype=torch.long)
        targets = torch.tensor(sequence[1:], dtype=torch.long)
        
        return {
            'input_ids': input_ids,
            'targets': targets
        }

def get_or_build_tokenizer(texts, vocab_size=10000):
    """
    Build a BPE tokenizer for the text data
    """
    tokenizer_path = "llm_tokenizer.json"
    
    if not os.path.exists(tokenizer_path):
        print("Building new tokenizer...")
        
        # Initialize BPE tokenizer
        tokenizer = Tokenizer(BPE(unk_token="<UNK>"))
        tokenizer.pre_tokenizer = Whitespace()
        
        # Train tokenizer
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=["<UNK>", "<PAD>", "<BOS>", "<EOS>"],
            min_frequency=2
        )
        
        tokenizer.train_from_iterator(texts, trainer)
        tokenizer.save(tokenizer_path)
        print(f"Tokenizer saved to {tokenizer_path}")
    else:
        tokenizer = Tokenizer.from_file(tokenizer_path)
        print(f"Loaded existing tokenizer from {tokenizer_path}")
    
    return tokenizer

def get_text_dataset(dataset_name="wikitext", config="wikitext-2-raw-v1"):
    """
    Load text dataset for language modeling
    """
    print(f"Loading dataset: {dataset_name}")
    
    # Load dataset
    dataset = load_dataset(dataset_name, config)
    
    # Extract texts (filter out empty strings)
    train_texts = [text for text in dataset['train']['text'] if text.strip()]
    val_texts = [text for text in dataset['validation']['text'] if text.strip()]
    
    print(f"Loaded {len(train_texts)} training texts, {len(val_texts)} validation texts")
    
    return train_texts, val_texts

def train_llm():
    """
    Main training function for the language model
    """
    # Configuration
    config = {
        'vocab_size': 10000,
        'seq_len': 256,
        'd_model': 512,
        'n_layers': 6,
        'n_heads': 8,
        'dropout': 0.1,
        'd_ff': 2048,
        'batch_size': 8,
        'lr': 1e-4,
        'num_epochs': 5,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    print("🤖 Training Language Model (LLM)")
    print("=" * 50)
    for key, value in config.items():
        print(f"{key}: {value}")
    print("=" * 50)
    
    device = torch.device(config['device'])
    print(f"Using device: {device}")
    
    # Load dataset
    train_texts, val_texts = get_text_dataset()
    
    # Build tokenizer
    all_texts = train_texts + val_texts
    tokenizer = get_or_build_tokenizer(all_texts, config['vocab_size'])
    
    # Update vocab size to actual tokenizer size
    actual_vocab_size = tokenizer.get_vocab_size()
    config['vocab_size'] = actual_vocab_size
    print(f"Actual vocabulary size: {actual_vocab_size}")
    
    # Create datasets
    train_dataset = TextDataset(train_texts, tokenizer, config['seq_len'])
    val_dataset = TextDataset(val_texts, tokenizer, config['seq_len'])
    
    print(f"Training sequences: {len(train_dataset)}")
    print(f"Validation sequences: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True,
        num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False,
        num_workers=2
    )
    
    # Build model
    model = build_language_model(
        vocab_size=config['vocab_size'],
        seq_len=config['seq_len'],
        d_model=config['d_model'],
        N=config['n_layers'],
        h=config['n_heads'],
        dropout=config['dropout'],
        d_ff=config['d_ff']
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.token_to_id("<PAD>") or -100)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=0.01)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['num_epochs'])
    
    # Training loop
    model.train()
    for epoch in range(config['num_epochs']):
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']}")
        
        for batch_idx, batch in enumerate(progress_bar):
            input_ids = batch['input_ids'].to(device)  # (batch, seq_len)
            targets = batch['targets'].to(device)      # (batch, seq_len)
            
            # Forward pass
            logits = model(input_ids)  # (batch, seq_len, vocab_size)
            
            # Compute loss (flatten for cross-entropy)
            loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
            optimizer.step()
            
            total_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss/(batch_idx+1):.4f}',
                'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
            })
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                input_ids = batch['input_ids'].to(device)
                targets = batch['targets'].to(device)
                
                logits = model(input_ids)
                loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        perplexity = math.exp(avg_val_loss)
        
        print(f"Epoch {epoch+1}: Train Loss: {total_loss/len(train_loader):.4f}, "
              f"Val Loss: {avg_val_loss:.4f}, Perplexity: {perplexity:.2f}")
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': config,
            'val_loss': avg_val_loss
        }
        torch.save(checkpoint, f'llm_checkpoint_epoch_{epoch+1}.pt')
        
        scheduler.step()
        model.train()
    
    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'tokenizer_path': 'llm_tokenizer.json'
    }, 'llm_final_model.pt')
    
    print("🎉 Training completed!")
    return model, tokenizer

def generate_text(model, tokenizer, prompt="The", max_length=100, temperature=0.8, top_k=50):
    """
    Generate text using the trained model
    """
    model.eval()
    device = next(model.parameters()).device
    
    # Tokenize prompt
    input_ids = torch.tensor([tokenizer.encode(prompt).ids]).to(device)
    
    print(f"Prompt: '{prompt}'")
    print("Generated text:")
    print("-" * 50)
    
    # Generate
    generated = model.generate(input_ids, max_length, temperature, top_k)
    
    # Decode
    generated_text = tokenizer.decode(generated[0].cpu().tolist())
    print(generated_text)
    print("-" * 50)
    
    return generated_text

if __name__ == "__main__":
    # Train the model
    model, tokenizer = train_llm()
    
    # Generate some text
    print("\n🎯 Testing text generation:")
    generate_text(model, tokenizer, "The quick brown fox", max_length=50)
    generate_text(model, tokenizer, "In the beginning", max_length=50)
    generate_text(model, tokenizer, "Science is", max_length=50) 