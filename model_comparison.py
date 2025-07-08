"""
Model Architecture Comparison Tool

This script provides side-by-side comparisons of different transformer architectures:
1. Translation Model (Encoder-Decoder) vs LLM (Decoder-Only)
2. Parameter count analysis
3. Architecture flow diagrams
4. Performance comparison on different tasks

Perfect for understanding architectural trade-offs and design choices!
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import numpy as np
from model import build_transformer
from llm_model import build_language_model
import seaborn as sns

class ModelComparator:
    """Compare different transformer architectures"""
    
    def __init__(self):
        print("🔄 Model Architecture Comparator")
        print("=" * 50)
    
    def count_parameters(self, model):
        """Count total and trainable parameters"""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return total_params, trainable_params
    
    def analyze_model_sizes(self):
        """Compare parameter counts across different model sizes"""
        print("\n📊 Parameter Count Analysis")
        print("-" * 40)
        
        # Standard configurations
        configs = [
            {"name": "Small", "d_model": 256, "n_layers": 4, "n_heads": 4, "d_ff": 1024},
            {"name": "Base", "d_model": 512, "n_layers": 6, "n_heads": 8, "d_ff": 2048},
            {"name": "Large", "d_model": 768, "n_layers": 12, "n_heads": 12, "d_ff": 3072},
            {"name": "XL", "d_model": 1024, "n_layers": 16, "n_heads": 16, "d_ff": 4096}
        ]
        
        vocab_size = 32000
        seq_len = 512
        
        results = []
        
        for config in configs:
            # Translation model (encoder-decoder)
            translation_model = build_transformer(
                src_vocab_size=vocab_size,
                tgt_vocab_size=vocab_size,
                src_seq_len=seq_len,
                tgt_seq_len=seq_len,
                d_model=config['d_model'],
                N=config['n_layers'],
                h=config['n_heads'],
                d_ff=config['d_ff']
            )
            
            # LLM (decoder-only)
            llm_model = build_language_model(
                vocab_size=vocab_size,
                seq_len=seq_len,
                d_model=config['d_model'],
                N=config['n_layers'],
                h=config['n_heads'],
                d_ff=config['d_ff']
            )
            
            trans_params, _ = self.count_parameters(translation_model)
            llm_params, _ = self.count_parameters(llm_model)
            
            results.append({
                'name': config['name'],
                'd_model': config['d_model'],
                'layers': config['n_layers'],
                'translation_params': trans_params,
                'llm_params': llm_params,
                'ratio': trans_params / llm_params
            })
            
            print(f"{config['name']:>6} | d_model: {config['d_model']:>4} | "
                  f"Translation: {trans_params/1e6:>6.1f}M | "
                  f"LLM: {llm_params/1e6:>6.1f}M | "
                  f"Ratio: {trans_params/llm_params:.2f}x")
        
        # Visualize parameter comparison
        self._plot_parameter_comparison(results)
        return results
    
    def _plot_parameter_comparison(self, results):
        """Plot parameter count comparison"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        names = [r['name'] for r in results]
        trans_params = [r['translation_params']/1e6 for r in results]
        llm_params = [r['llm_params']/1e6 for r in results]
        
        # Bar chart comparison
        x = np.arange(len(names))
        width = 0.35
        
        ax1.bar(x - width/2, trans_params, width, label='Translation Model', alpha=0.8)
        ax1.bar(x + width/2, llm_params, width, label='LLM Model', alpha=0.8)
        ax1.set_xlabel('Model Size')
        ax1.set_ylabel('Parameters (Millions)')
        ax1.set_title('Parameter Count Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels(names)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Ratio plot
        ratios = [r['ratio'] for r in results]
        ax2.plot(names, ratios, 'o-', linewidth=2, markersize=8)
        ax2.set_xlabel('Model Size')
        ax2.set_ylabel('Translation/LLM Parameter Ratio')
        ax2.set_title('Parameter Ratio: Translation vs LLM')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=2.0, color='r', linestyle='--', alpha=0.7, label='2x Ratio')
        ax2.legend()
        
        plt.tight_layout()
        plt.show()
    
    def draw_architecture_comparison(self):
        """Draw side-by-side architecture diagrams"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10))
        
        # Translation Model (Encoder-Decoder)
        self._draw_translation_architecture(ax1)
        
        # LLM (Decoder-Only)
        self._draw_llm_architecture(ax2)
        
        plt.tight_layout()
        plt.show()
    
    def _draw_translation_architecture(self, ax):
        """Draw encoder-decoder architecture"""
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 12)
        ax.set_title('Translation Model\n(Encoder-Decoder)', fontsize=14, fontweight='bold')
        
        # Colors
        encoder_color = '#E3F2FD'  # Light blue
        decoder_color = '#FFF3E0'  # Light orange
        attention_color = '#F3E5F5'  # Light purple
        
        # Input embeddings
        ax.add_patch(FancyBboxPatch((0.5, 0.5), 2, 0.8, boxstyle="round,pad=0.1", 
                                   facecolor='lightgray', edgecolor='black'))
        ax.text(1.5, 0.9, 'Source\nEmbeddings', ha='center', va='center', fontsize=9)
        
        ax.add_patch(FancyBboxPatch((6.5, 0.5), 2, 0.8, boxstyle="round,pad=0.1", 
                                   facecolor='lightgray', edgecolor='black'))
        ax.text(7.5, 0.9, 'Target\nEmbeddings', ha='center', va='center', fontsize=9)
        
        # Encoder stack
        for i in range(3):
            y_pos = 2 + i * 2.5
            # Self-attention
            ax.add_patch(FancyBboxPatch((0.5, y_pos), 2, 0.6, boxstyle="round,pad=0.05", 
                                       facecolor=attention_color, edgecolor='blue'))
            ax.text(1.5, y_pos + 0.3, 'Self-Attention', ha='center', va='center', fontsize=8)
            
            # Feed forward
            ax.add_patch(FancyBboxPatch((0.5, y_pos + 0.8), 2, 0.6, boxstyle="round,pad=0.05", 
                                       facecolor=encoder_color, edgecolor='blue'))
            ax.text(1.5, y_pos + 1.1, 'Feed Forward', ha='center', va='center', fontsize=8)
        
        # Decoder stack
        for i in range(3):
            y_pos = 2 + i * 2.5
            # Masked self-attention
            ax.add_patch(FancyBboxPatch((6.5, y_pos), 2, 0.4, boxstyle="round,pad=0.05", 
                                       facecolor=attention_color, edgecolor='orange'))
            ax.text(7.5, y_pos + 0.2, 'Masked\nSelf-Attn', ha='center', va='center', fontsize=7)
            
            # Cross-attention
            ax.add_patch(FancyBboxPatch((6.5, y_pos + 0.5), 2, 0.4, boxstyle="round,pad=0.05", 
                                       facecolor='#FFCDD2', edgecolor='red'))
            ax.text(7.5, y_pos + 0.7, 'Cross\nAttention', ha='center', va='center', fontsize=7)
            
            # Feed forward
            ax.add_patch(FancyBboxPatch((6.5, y_pos + 1.0), 2, 0.4, boxstyle="round,pad=0.05", 
                                       facecolor=decoder_color, edgecolor='orange'))
            ax.text(7.5, y_pos + 1.2, 'Feed Forward', ha='center', va='center', fontsize=7)
        
        # Output
        ax.add_patch(FancyBboxPatch((6.5, 10.5), 2, 0.8, boxstyle="round,pad=0.1", 
                                   facecolor='lightgreen', edgecolor='black'))
        ax.text(7.5, 10.9, 'Linear +\nSoftmax', ha='center', va='center', fontsize=9)
        
        # Arrows
        # Encoder flow
        ax.arrow(1.5, 1.3, 0, 0.5, head_width=0.1, head_length=0.1, fc='blue', ec='blue')
        ax.arrow(1.5, 4.3, 0, 0.5, head_width=0.1, head_length=0.1, fc='blue', ec='blue')
        ax.arrow(1.5, 6.8, 0, 0.5, head_width=0.1, head_length=0.1, fc='blue', ec='blue')
        
        # Decoder flow
        ax.arrow(7.5, 1.3, 0, 0.5, head_width=0.1, head_length=0.1, fc='orange', ec='orange')
        ax.arrow(7.5, 4.3, 0, 0.5, head_width=0.1, head_length=0.1, fc='orange', ec='orange')
        ax.arrow(7.5, 6.8, 0, 0.5, head_width=0.1, head_length=0.1, fc='orange', ec='orange')
        ax.arrow(7.5, 9.3, 0, 1.0, head_width=0.1, head_length=0.1, fc='orange', ec='orange')
        
        # Cross attention arrows
        ax.arrow(2.7, 5.5, 3.6, 0, head_width=0.1, head_length=0.1, fc='red', ec='red', linestyle='--')
        ax.arrow(2.7, 7.7, 3.6, 0, head_width=0.1, head_length=0.1, fc='red', ec='red', linestyle='--')
        ax.arrow(2.7, 3.3, 3.6, 0, head_width=0.1, head_length=0.1, fc='red', ec='red', linestyle='--')
        
        # Labels
        ax.text(1.5, 11.3, 'ENCODER', ha='center', va='center', fontsize=12, fontweight='bold')
        ax.text(7.5, 11.8, 'DECODER', ha='center', va='center', fontsize=12, fontweight='bold')
        
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
    
    def _draw_llm_architecture(self, ax):
        """Draw decoder-only LLM architecture"""
        ax.set_xlim(0, 6)
        ax.set_ylim(0, 12)
        ax.set_title('Language Model\n(Decoder-Only)', fontsize=14, fontweight='bold')
        
        # Colors
        decoder_color = '#FFF3E0'  # Light orange
        attention_color = '#F3E5F5'  # Light purple
        
        # Input embeddings
        ax.add_patch(FancyBboxPatch((2, 0.5), 2, 0.8, boxstyle="round,pad=0.1", 
                                   facecolor='lightgray', edgecolor='black'))
        ax.text(3, 0.9, 'Token\nEmbeddings', ha='center', va='center', fontsize=9)
        
        # Decoder blocks (more layers for LLM)
        for i in range(5):
            y_pos = 2 + i * 1.8
            # Causal self-attention
            ax.add_patch(FancyBboxPatch((2, y_pos), 2, 0.6, boxstyle="round,pad=0.05", 
                                       facecolor=attention_color, edgecolor='purple'))
            ax.text(3, y_pos + 0.3, 'Causal\nSelf-Attention', ha='center', va='center', fontsize=8)
            
            # Feed forward
            ax.add_patch(FancyBboxPatch((2, y_pos + 0.8), 2, 0.6, boxstyle="round,pad=0.05", 
                                       facecolor=decoder_color, edgecolor='orange'))
            ax.text(3, y_pos + 1.1, 'Feed Forward', ha='center', va='center', fontsize=8)
        
        # Language modeling head
        ax.add_patch(FancyBboxPatch((2, 11), 2, 0.8, boxstyle="round,pad=0.1", 
                                   facecolor='lightcoral', edgecolor='black'))
        ax.text(3, 11.4, 'Language\nHead', ha='center', va='center', fontsize=9)
        
        # Arrows (single flow)
        for i in range(6):
            y_start = 1.3 + i * 1.8
            y_end = y_start + 0.5
            ax.arrow(3, y_start, 0, 0.4, head_width=0.1, head_length=0.1, fc='purple', ec='purple')
        
        # Final arrow to output
        ax.arrow(3, 10.1, 0, 0.7, head_width=0.1, head_length=0.1, fc='purple', ec='purple')
        
        # Labels
        ax.text(3, 0.1, 'INPUT TOKENS', ha='center', va='center', fontsize=10, fontweight='bold')
        ax.text(3, 11.9, 'NEXT TOKEN\nPROBABILITIES', ha='center', va='center', fontsize=10, fontweight='bold')
        
        # Causal mask visualization
        ax.text(5.2, 6, 'Causal\nMasking\n(No Future\nAccess)', ha='center', va='center', 
               fontsize=8, bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7))
        
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
    
    def task_performance_comparison(self):
        """Compare theoretical performance on different tasks"""
        print("\n🎯 Task Suitability Comparison")
        print("-" * 50)
        
        tasks = {
            'Machine Translation': {
                'Translation Model': '⭐⭐⭐⭐⭐ (Designed for this)',
                'LLM': '⭐⭐⭐ (Can work with prompting)'
            },
            'Text Generation': {
                'Translation Model': '⭐⭐ (Not designed for this)',
                'LLM': '⭐⭐⭐⭐⭐ (Designed for this)'
            },
            'Question Answering': {
                'Translation Model': '⭐⭐ (Needs task adaptation)',
                'LLM': '⭐⭐⭐⭐ (Works well with prompting)'
            },
            'Text Summarization': {
                'Translation Model': '⭐⭐⭐ (Can be adapted)',
                'LLM': '⭐⭐⭐⭐ (Good with prompting)'
            },
            'Code Generation': {
                'Translation Model': '⭐⭐ (Limited applicability)',
                'LLM': '⭐⭐⭐⭐⭐ (Excellent with code training)'
            },
            'Few-shot Learning': {
                'Translation Model': '⭐ (Not designed for this)',
                'LLM': '⭐⭐⭐⭐⭐ (Emergent capability)'
            }
        }
        
        for task, ratings in tasks.items():
            print(f"\n{task}:")
            for model, rating in ratings.items():
                print(f"  {model:>18}: {rating}")
    
    def computational_complexity_analysis(self):
        """Analyze computational complexity differences"""
        print("\n⚡ Computational Complexity Analysis")
        print("-" * 50)
        
        print("Translation Model (Encoder-Decoder):")
        print("  • Training: O(n²) for both encoder and decoder")
        print("  • Inference: O(n²) for encoder + O(m²) for decoder")
        print("  • Memory: Stores both source and target representations")
        print("  • Parallelization: Encoder fully parallel, decoder sequential")
        
        print("\nLanguage Model (Decoder-Only):")
        print("  • Training: O(n²) with causal masking")
        print("  • Inference: O(n²) but can use KV-cache for efficiency")
        print("  • Memory: Single sequence representation")
        print("  • Parallelization: Training parallel, inference sequential")
        
        print("\nKey Differences:")
        print("  • LLM uses ~50% fewer parameters for similar capacity")
        print("  • Translation model better for fixed input-output tasks")
        print("  • LLM more flexible for varied generation tasks")
        print("  • LLM benefits more from scale (emergent capabilities)")
    
    def run_full_comparison(self):
        """Run comprehensive model comparison"""
        print("🔍 Comprehensive Model Architecture Comparison")
        print("=" * 60)
        
        # 1. Parameter analysis
        self.analyze_model_sizes()
        
        # 2. Architecture visualization
        print("\n🏗️ Drawing Architecture Diagrams...")
        self.draw_architecture_comparison()
        
        # 3. Task comparison
        self.task_performance_comparison()
        
        # 4. Complexity analysis
        self.computational_complexity_analysis()
        
        print("\n✅ Comparison Complete!")
        print("\nKey Takeaways:")
        print("- Translation models excel at sequence-to-sequence tasks")
        print("- LLMs are more parameter-efficient and flexible")
        print("- Choice depends on your specific use case")
        print("- LLMs show better scaling properties for general tasks")

def main():
    """Run the model comparison tool"""
    comparator = ModelComparator()
    comparator.run_full_comparison()

if __name__ == "__main__":
    main() 