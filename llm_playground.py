"""
Interactive LLM Playground

This script provides an interactive interface to experiment with the trained language model.
You can:
- Try different prompts and see how the model responds
- Experiment with temperature, top-k, and other generation parameters  
- Compare different sampling strategies
- Load different model checkpoints
- See real-time generation token by token

Perfect for understanding how LLMs work in practice!
"""

import torch
import torch.nn.functional as F
from llm_model import build_language_model, LanguageModel
from train_llm import get_or_build_tokenizer
import json
import os
from pathlib import Path

class LLMPlayground:
    """Interactive playground for experimenting with language models"""
    
    def __init__(self, model_path=None, tokenizer_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🚀 LLM Playground - Running on {self.device}")
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path, tokenizer_path)
        else:
            print("No model loaded. You can:")
            print("1. Train a model first with train_llm.py")
            print("2. Load an existing model with load_model()")
            self.model = None
            self.tokenizer = None
            self.config = None
    
    def load_model(self, model_path, tokenizer_path=None):
        """Load a trained model and tokenizer"""
        print(f"Loading model from {model_path}...")
        
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.config = checkpoint['config']
            
            # Build model architecture
            self.model = build_language_model(
                vocab_size=self.config['vocab_size'],
                seq_len=self.config['seq_len'],
                d_model=self.config['d_model'],
                N=self.config['n_layers'],
                h=self.config['n_heads'],
                dropout=0.0,  # No dropout for inference
                d_ff=self.config['d_ff']
            ).to(self.device)
            
            # Load weights
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            # Load tokenizer
            if tokenizer_path and os.path.exists(tokenizer_path):
                from tokenizers import Tokenizer
                self.tokenizer = Tokenizer.from_file(tokenizer_path)
            else:
                print("Warning: No tokenizer file found. Using default tokenizer.")
                # You'd need to recreate tokenizer from training data
                self.tokenizer = None
            
            print(f"✅ Model loaded successfully!")
            print(f"   Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
            print(f"   Vocabulary size: {self.config['vocab_size']}")
            print(f"   Max sequence length: {self.config['seq_len']}")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            self.model = None
    
    def generate_interactive(self, prompt="", max_length=100, temperature=0.8, 
                           top_k=50, top_p=0.9, show_tokens=False):
        """
        Interactive text generation with real-time display
        
        Args:
            prompt: Starting text
            max_length: Maximum tokens to generate
            temperature: Sampling temperature (0.1=focused, 2.0=creative)
            top_k: Only sample from top-k tokens (0=disabled)
            top_p: Nucleus sampling threshold (0.9=keep 90% probability mass)
            show_tokens: Show individual tokens as they're generated
        """
        if not self.model or not self.tokenizer:
            print("❌ No model loaded. Load a model first!")
            return
        
        print(f"\n🎯 Generating text...")
        print(f"Prompt: '{prompt}'")
        print(f"Settings: temp={temperature}, top_k={top_k}, top_p={top_p}")
        print("-" * 60)
        
        # Tokenize prompt
        if prompt:
            input_ids = torch.tensor([self.tokenizer.encode(prompt).ids]).to(self.device)
        else:
            # Start with a random token or special start token
            input_ids = torch.tensor([[1]]).to(self.device)  # Assuming 1 is a reasonable start token
        
        generated = input_ids.clone()
        
        print(prompt, end="", flush=True)
        
        with torch.no_grad():
            for step in range(max_length - input_ids.size(1)):
                # Get logits for next token
                logits = self.model(generated)[:, -1, :]  # (1, vocab_size)
                
                # Apply temperature
                if temperature != 1.0:
                    logits = logits / temperature
                
                # Apply top-k filtering
                if top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits = torch.full_like(logits, float('-inf'))
                    logits.scatter_(1, top_k_indices, top_k_logits)
                
                # Apply nucleus (top-p) sampling
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = float('-inf')
                
                # Sample next token
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                
                # Decode and display token
                token_text = self.tokenizer.decode([next_token.item()])
                
                if show_tokens:
                    print(f"[{token_text}]", end="", flush=True)
                else:
                    print(token_text, end="", flush=True)
                
                # Append to sequence
                generated = torch.cat([generated, next_token], dim=1)
                
                # Check for stop conditions (you could add special stop tokens)
                if next_token.item() == 0:  # Assuming 0 is end token
                    break
        
        print("\n" + "-" * 60)
        full_text = self.tokenizer.decode(generated[0].cpu().tolist())
        print(f"✅ Generated {generated.size(1) - input_ids.size(1)} tokens")
        return full_text
    
    def compare_sampling_strategies(self, prompt, strategies=None):
        """Compare different sampling strategies side by side"""
        if not strategies:
            strategies = [
                {"name": "Greedy", "temperature": 0.1, "top_k": 1, "top_p": 1.0},
                {"name": "Low Temp", "temperature": 0.5, "top_k": 50, "top_p": 0.9},
                {"name": "Balanced", "temperature": 0.8, "top_k": 50, "top_p": 0.9},
                {"name": "Creative", "temperature": 1.2, "top_k": 100, "top_p": 0.9},
                {"name": "Wild", "temperature": 2.0, "top_k": 0, "top_p": 1.0}
            ]
        
        print(f"\n🔬 Comparing Sampling Strategies")
        print(f"Prompt: '{prompt}'")
        print("=" * 80)
        
        for strategy in strategies:
            print(f"\n{strategy['name']} (T={strategy['temperature']}, K={strategy['top_k']}, P={strategy['top_p']}):")
            print("-" * 40)
            self.generate_interactive(
                prompt=prompt,
                max_length=50,
                temperature=strategy['temperature'],
                top_k=strategy['top_k'],
                top_p=strategy['top_p']
            )
    
    def interactive_mode(self):
        """Start interactive chat mode"""
        print("\n🎮 Interactive Mode Started!")
        print("Commands:")
        print("  /temp <value>    - Set temperature (0.1-2.0)")
        print("  /topk <value>    - Set top-k (0-100)")
        print("  /topp <value>    - Set top-p (0.1-1.0)")
        print("  /length <value>  - Set max length")
        print("  /compare <prompt> - Compare sampling strategies")
        print("  /tokens          - Toggle token display")
        print("  /quit            - Exit")
        print("-" * 50)
        
        # Default settings
        temperature = 0.8
        top_k = 50
        top_p = 0.9
        max_length = 100
        show_tokens = False
        
        while True:
            try:
                user_input = input("\n💭 Enter prompt (or command): ").strip()
                
                if user_input == "/quit":
                    break
                elif user_input.startswith("/temp "):
                    temperature = float(user_input.split()[1])
                    print(f"Temperature set to {temperature}")
                elif user_input.startswith("/topk "):
                    top_k = int(user_input.split()[1])
                    print(f"Top-k set to {top_k}")
                elif user_input.startswith("/topp "):
                    top_p = float(user_input.split()[1])
                    print(f"Top-p set to {top_p}")
                elif user_input.startswith("/length "):
                    max_length = int(user_input.split()[1])
                    print(f"Max length set to {max_length}")
                elif user_input.startswith("/compare "):
                    prompt = user_input[9:]  # Remove "/compare "
                    self.compare_sampling_strategies(prompt)
                elif user_input == "/tokens":
                    show_tokens = not show_tokens
                    print(f"Token display: {'ON' if show_tokens else 'OFF'}")
                elif user_input:
                    self.generate_interactive(
                        prompt=user_input,
                        max_length=max_length,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        show_tokens=show_tokens
                    )
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

def main():
    """Main playground interface"""
    print("🎮 Welcome to the LLM Playground!")
    print("=" * 50)
    
    playground = LLMPlayground()
    
    # Try to load a default model if it exists
    model_files = [
        'llm_final_model.pt',
        'llm_checkpoint_epoch_5.pt',
        'llm_checkpoint_epoch_4.pt',
        'llm_checkpoint_epoch_3.pt'
    ]
    
    for model_file in model_files:
        if os.path.exists(model_file):
            playground.load_model(model_file, 'llm_tokenizer.json')
            break
    
    if playground.model is None:
        print("\n📝 No trained model found. Train one first:")
        print("   python train_llm.py")
        return
    
    print("\n🎯 Quick Examples:")
    examples = [
        "The future of artificial intelligence",
        "Once upon a time in a distant galaxy",
        "The most important thing to remember is",
        "Scientists recently discovered"
    ]
    
    for example in examples:
        print(f"\nExample: '{example}'")
        playground.generate_interactive(example, max_length=30)
    
    print("\n" + "="*50)
    playground.interactive_mode()

if __name__ == "__main__":
    main() 