"""
LLM Attention Visualization

This script visualizes attention patterns specifically for Language Models (LLMs).
It helps understand:
1. How causal attention masks work (preventing future token access)
2. How different layers attend to different aspects of the prompt
3. How the model builds context and relationships across tokens
4. Comparison between prompt tokens and generated tokens attention

Key differences from translation attention:
- Only self-attention (no cross-attention)
- Causal masking creates triangular patterns
- Focus on prompt-to-generation relationships
"""

import torch
import torch.nn as nn
from llm_model import build_language_model, causal_mask
from train_llm import get_or_build_tokenizer
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
import os

class LLMAttentionVisualizer:
    """Visualize attention patterns in Language Models"""
    
    def __init__(self, model_path=None, tokenizer_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔍 LLM Attention Visualizer - Running on {self.device}")
        
        self.model = None
        self.tokenizer = None
        self.config = None
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path, tokenizer_path)
    
    def load_model(self, model_path, tokenizer_path=None):
        """Load a trained LLM and tokenizer"""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.config = checkpoint['config']
            
            # Build model
            self.model = build_language_model(
                vocab_size=self.config['vocab_size'],
                seq_len=self.config['seq_len'],
                d_model=self.config['d_model'],
                N=self.config['n_layers'],
                h=self.config['n_heads'],
                dropout=0.0,
                d_ff=self.config['d_ff']
            ).to(self.device)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            # Load tokenizer
            if tokenizer_path and os.path.exists(tokenizer_path):
                from tokenizers import Tokenizer
                self.tokenizer = Tokenizer.from_file(tokenizer_path)
            
            print(f"✅ Model loaded: {sum(p.numel() for p in self.model.parameters()):,} parameters")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
    
    def generate_with_attention(self, prompt, max_length=20):
        """Generate text while capturing attention patterns"""
        if not self.model or not self.tokenizer:
            print("❌ No model loaded!")
            return None, None, None
        
        # Tokenize prompt
        input_ids = torch.tensor([self.tokenizer.encode(prompt).ids]).to(self.device)
        generated = input_ids.clone()
        all_attention = []  # Store attention for each generation step
        
        with torch.no_grad():
            for step in range(max_length - input_ids.size(1)):
                # Forward pass to get attention
                logits = self.model(generated)
                
                # Collect attention from all layers
                step_attention = []
                for layer in self.model.decoder.layers:
                    if hasattr(layer.self_attention_block, 'attention_scores'):
                        attn = layer.self_attention_block.attention_scores
                        if attn is not None:
                            step_attention.append(attn.cpu())
                
                if step_attention:
                    all_attention.append(step_attention)
                
                # Generate next token (greedy for reproducibility)
                next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                generated = torch.cat([generated, next_token], dim=1)
                
                # Stop at end token or max length
                if next_token.item() == 0:  # Assuming 0 is end token
                    break
        
        # Decode tokens
        tokens = [self.tokenizer.id_to_token(int(token_id)) for token_id in generated[0]]
        
        return tokens, all_attention, generated
    
    def plot_causal_attention(self, tokens, attention_data, layer_idx=0, head_idx=0, 
                             step_idx=-1, figsize=(12, 8)):
        """
        Plot causal attention pattern for a specific layer and head
        
        Args:
            tokens: List of token strings
            attention_data: Attention scores from generation
            layer_idx: Which layer to visualize
            head_idx: Which attention head to visualize
            step_idx: Which generation step (-1 for final)
            figsize: Figure size
        """
        if not attention_data:
            print("❌ No attention data available")
            return
        
        # Get attention for specified step and layer
        step_attention = attention_data[step_idx][layer_idx]  # (batch, heads, seq_len, seq_len)
        head_attention = step_attention[0, head_idx].numpy()  # (seq_len, seq_len)
        
        # Create the plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Plot 1: Full attention heatmap
        seq_len = len(tokens)
        im1 = ax1.imshow(head_attention[:seq_len, :seq_len], cmap='Blues', aspect='auto')
        ax1.set_xticks(range(seq_len))
        ax1.set_yticks(range(seq_len))
        ax1.set_xticklabels(tokens, rotation=45, ha='right')
        ax1.set_yticklabels(tokens)
        ax1.set_title(f'Causal Attention - Layer {layer_idx}, Head {head_idx}')
        ax1.set_xlabel('Keys (what we attend TO)')
        ax1.set_ylabel('Queries (what we attend FROM)')
        plt.colorbar(im1, ax=ax1, label='Attention Weight')
        
        # Add grid for clarity
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Attention to last token (what the model focuses on for next prediction)
        if seq_len > 1:
            last_token_attention = head_attention[seq_len-1, :seq_len]
            ax2.bar(range(seq_len), last_token_attention, alpha=0.7)
            ax2.set_xticks(range(seq_len))
            ax2.set_xticklabels(tokens, rotation=45, ha='right')
            ax2.set_title(f'What "{tokens[-1]}" Attends To')
            ax2.set_ylabel('Attention Weight')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Print interpretation
        print(f"\n🔍 Attention Analysis:")
        print(f"Layer {layer_idx}, Head {head_idx} - Token: '{tokens[-1]}'")
        print("-" * 50)
        
        if seq_len > 1:
            # Find top attended tokens
            top_indices = np.argsort(last_token_attention)[-3:][::-1]
            for i, idx in enumerate(top_indices):
                weight = last_token_attention[idx]
                print(f"{i+1}. '{tokens[idx]}' (position {idx}): {weight:.3f}")
    
    def compare_layers_attention(self, tokens, attention_data, target_token_idx=-1, 
                                heads_to_show=4, figsize=(15, 10)):
        """Compare attention patterns across different layers"""
        if not attention_data:
            return
        
        step_attention = attention_data[-1]  # Final generation step
        n_layers = len(step_attention)
        seq_len = len(tokens)
        
        # Create subplot grid
        fig, axes = plt.subplots(n_layers, heads_to_show, figsize=figsize)
        if n_layers == 1:
            axes = axes.reshape(1, -1)
        
        for layer_idx in range(n_layers):
            layer_attention = step_attention[layer_idx][0]  # Remove batch dim
            
            for head_idx in range(min(heads_to_show, layer_attention.size(0))):
                ax = axes[layer_idx, head_idx]
                
                # Get attention pattern for target token
                head_attn = layer_attention[head_idx].numpy()
                target_attention = head_attn[target_token_idx, :seq_len]
                
                # Plot as heatmap (1D -> 2D for better visualization)
                im = ax.imshow(target_attention.reshape(1, -1), cmap='Blues', aspect='auto')
                ax.set_xticks(range(seq_len))
                ax.set_xticklabels(tokens, rotation=45, ha='right', fontsize=8)
                ax.set_yticks([])
                ax.set_title(f'L{layer_idx}H{head_idx}', fontsize=10)
                
                if head_idx == 0:
                    ax.set_ylabel(f'Layer {layer_idx}', fontsize=10)
        
        plt.suptitle(f'Attention Patterns Across Layers\nTarget Token: "{tokens[target_token_idx]}"', 
                     fontsize=14)
        plt.tight_layout()
        plt.show()
        
        # Analyze layer differences
        print(f"\n📊 Layer Analysis for '{tokens[target_token_idx]}':")
        print("-" * 50)
        
        for layer_idx in range(min(3, n_layers)):  # Show first 3 layers
            layer_attention = step_attention[layer_idx][0]
            avg_attention = layer_attention.mean(dim=0)[target_token_idx, :seq_len].numpy()
            
            top_idx = np.argmax(avg_attention)
            print(f"Layer {layer_idx}: Most attended to '{tokens[top_idx]}' "
                  f"(weight: {avg_attention[top_idx]:.3f})")
    
    def prompt_vs_generation_analysis(self, prompt, generated_text, max_analysis_length=30):
        """Analyze how attention differs between prompt and generated parts"""
        full_text = prompt + " " + generated_text
        tokens, attention_data, _ = self.generate_with_attention(prompt, max_analysis_length)
        
        if not attention_data:
            return
        
        # Find prompt boundary
        prompt_tokens = self.tokenizer.encode(prompt).tokens
        prompt_length = len(prompt_tokens)
        
        print(f"\n🎯 Prompt vs Generation Analysis")
        print(f"Prompt: '{prompt}' ({prompt_length} tokens)")
        print(f"Generated: '{generated_text}'")
        print("-" * 60)
        
        # Analyze final step attention
        final_attention = attention_data[-1]  # Last generation step
        
        for layer_idx in range(min(3, len(final_attention))):
            layer_attn = final_attention[layer_idx][0]  # (heads, seq_len, seq_len)
            avg_layer_attn = layer_attn.mean(dim=0)  # Average across heads
            
            # Focus on last generated token
            last_token_attn = avg_layer_attn[-1, :len(tokens)].numpy()
            
            # Split attention between prompt and generated parts
            prompt_attention = last_token_attn[:prompt_length].sum()
            generated_attention = last_token_attn[prompt_length:].sum()
            total_attention = prompt_attention + generated_attention
            
            prompt_pct = (prompt_attention / total_attention) * 100
            generated_pct = (generated_attention / total_attention) * 100
            
            print(f"Layer {layer_idx}:")
            print(f"  Attention to prompt: {prompt_pct:.1f}%")
            print(f"  Attention to generated: {generated_pct:.1f}%")
    
    def interactive_attention_explorer(self):
        """Interactive tool to explore attention patterns"""
        print("\n🎮 Interactive Attention Explorer")
        print("Commands:")
        print("  /prompt <text>     - Analyze attention for a prompt")
        print("  /layer <num>       - Set layer to visualize")
        print("  /head <num>        - Set head to visualize") 
        print("  /compare           - Compare layers for last prompt")
        print("  /help              - Show this help")
        print("  /quit              - Exit")
        print("-" * 50)
        
        current_layer = 0
        current_head = 0
        last_tokens = None
        last_attention = None
        
        while True:
            try:
                user_input = input("\n🔍 Enter command: ").strip()
                
                if user_input == "/quit":
                    break
                elif user_input == "/help":
                    continue  # Help already shown above
                elif user_input.startswith("/prompt "):
                    prompt = user_input[8:]
                    print(f"Analyzing: '{prompt}'")
                    last_tokens, last_attention, _ = self.generate_with_attention(prompt)
                    if last_tokens:
                        print(f"Generated tokens: {' '.join(last_tokens)}")
                        self.plot_causal_attention(last_tokens, last_attention, 
                                                  current_layer, current_head)
                elif user_input.startswith("/layer "):
                    current_layer = int(user_input.split()[1])
                    print(f"Layer set to {current_layer}")
                    if last_tokens and last_attention:
                        self.plot_causal_attention(last_tokens, last_attention,
                                                  current_layer, current_head)
                elif user_input.startswith("/head "):
                    current_head = int(user_input.split()[1])
                    print(f"Head set to {current_head}")
                    if last_tokens and last_attention:
                        self.plot_causal_attention(last_tokens, last_attention,
                                                  current_layer, current_head)
                elif user_input == "/compare":
                    if last_tokens and last_attention:
                        self.compare_layers_attention(last_tokens, last_attention)
                    else:
                        print("❌ No prompt analyzed yet. Use /prompt first.")
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

def main():
    """Main visualization interface"""
    print("🔍 LLM Attention Visualization Tool")
    print("=" * 50)
    
    # Try to find trained models
    model_files = [
        'llm_final_model.pt',
        'llm_checkpoint_epoch_5.pt',
        'llm_checkpoint_epoch_4.pt'
    ]
    
    visualizer = None
    for model_file in model_files:
        if os.path.exists(model_file):
            visualizer = LLMAttentionVisualizer(model_file, 'llm_tokenizer.json')
            break
    
    if not visualizer or not visualizer.model:
        print("❌ No trained LLM found. Train one first with:")
        print("   python train_llm.py")
        return
    
    # Quick demo
    print("\n🎯 Quick Demo:")
    demo_prompts = [
        "The cat sat on the",
        "Machine learning is",
        "In the future we will"
    ]
    
    for prompt in demo_prompts:
        print(f"\nAnalyzing: '{prompt}'")
        tokens, attention_data, _ = visualizer.generate_with_attention(prompt, max_length=10)
        if tokens:
            print(f"Generated: {' '.join(tokens)}")
            visualizer.plot_causal_attention(tokens, attention_data, layer_idx=0, head_idx=0)
    
    # Start interactive mode
    visualizer.interactive_attention_explorer()

if __name__ == "__main__":
    main() 