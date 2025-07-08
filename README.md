# Transformer & LLM Educational Implementation

**Created by Roberto H Luna** - *Open source education for transformer technology*

This repository contains **complete from-scratch implementations** of both **Translation Transformers** and **Large Language Models (LLMs)**, designed specifically for **educational purposes**. Learn how modern AI works by building it yourself!

🔥 **What's Inside:**
- **Translation Model**: Encoder-decoder transformer (English ↔ Italian)
- **Language Model**: GPT-style decoder-only transformer for text generation
- **Attention Visualization**: See what the models actually learn
- **Educational Examples**: Step-by-step concept explanations

This implementation faithfully follows the architecture from ["Attention is All You Need"](https://arxiv.org/abs/1706.03762) by Vaswani et al., with educational extensions for modern LLM understanding.

## 🎯 Educational Goals

This implementation helps you understand:
- **Multi-head attention mechanism** and why it's revolutionary
- **Positional encoding** and how transformers handle sequence order
- **Encoder-decoder architecture** for sequence-to-sequence tasks (translation)
- **Decoder-only architecture** for autoregressive generation (LLMs)
- **Training dynamics** of both translation and language models
- **Attention visualization** to see what the models learn
- **Text generation** and sampling strategies

## 🏗️ Architecture Overview

### Translation Model (Encoder-Decoder)
```
Input Embeddings + Positional Encoding
         ↓
    Encoder Stack (6 layers)
    ┌─────────────────────┐
    │ Multi-Head Attention│
    │ + Residual & Norm   │
    ├─────────────────────┤
    │ Feed Forward        │
    │ + Residual & Norm   │
    └─────────────────────┘
         ↓
    Decoder Stack (6 layers)
    ┌─────────────────────┐
    │ Masked Self-Attn    │
    │ + Residual & Norm   │
    ├─────────────────────┤
    │ Cross Attention     │
    │ + Residual & Norm   │
    ├─────────────────────┤
    │ Feed Forward        │
    │ + Residual & Norm   │
    └─────────────────────┘
         ↓
    Linear + Softmax
```

### Language Model (Decoder-Only, GPT-style)
```
Input Embeddings + Positional Encoding
         ↓
    Decoder Stack (12+ layers)
    ┌─────────────────────┐
    │ Causal Self-Attn    │
    │ + Residual & Norm   │
    ├─────────────────────┤
    │ Feed Forward        │
    │ + Residual & Norm   │
    └─────────────────────┘
         ↓
    Language Head → Next Token
```

## 📁 File Structure

### Core Model Files
- **`model.py`** - Complete transformer architecture implementation
  - `LayerNormalization` - Layer normalization for stable training
  - `MultiHeadAttentionBlock` - The heart of the transformer
  - `FeedForwardBlock` - Position-wise feed-forward networks
  - `PositionalEncoding` - Sinusoidal positional embeddings
  - `EncoderBlock` & `DecoderBlock` - Complete encoder/decoder layers
  - `Transformer` - Full translation model combining all components

- **`llm_model.py`** - GPT-style language model implementation
  - `LLMDecoderBlock` - Simplified decoder for language modeling
  - `LanguageModel` - Complete LLM with text generation
  - `causal_mask` - Autoregressive attention masking
  - `build_language_model` - LLM factory function

### Training & Data
- **`train.py`** - Translation model training script with validation
- **`train_llm.py`** - Language model training script with text generation
- **`dataset.py`** - Bilingual dataset processing and tokenization
- **`config.py`** - Model and training configurations

### Inference & Analysis
- **`translate.py`** - Interactive translation interface
- **`inference.py`** - Model evaluation and testing
- **`attention.py`** - 🎨 **Attention visualization** (must-see!)
- **`beam_search.py`** - Advanced decoding strategies

### Educational Tools & Analysis 🎓
- **`educational_examples.py`** - 📚 **Step-by-step concept explanations**
- **`llm_playground.py`** - 🎮 **Interactive LLM experimentation**
- **`llm_attention.py`** - 🔍 **LLM-specific attention visualization**
- **`model_comparison.py`** - ⚖️ **Architecture comparison tool**

### Dataset Preparation & Analysis 📊
- **`dataset_preparation.py`** - 🛠️ **Complete dataset preprocessing pipeline**
- **`data_examples.py`** - 📝 **Dataset structure examples for different domains**

### Additional Training Options
- **`train_wb.py`** - Training with Weights & Biases integration
- **`local_train.py`** - Local training setup

## 🚀 Quick Start

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Train Models
```bash
# Translation model training
python train.py

# Language model training  
python train_llm.py

# With W&B logging
python train_wb.py
```

### 3. Use the Models
```bash
# Translate text (EN ↔ IT)
python translate.py

# Generate text with LLM
python -c "from train_llm import generate_text, torch; 
# Load your trained LLM and generate text"
```

### 4. Prepare Your Dataset (📊 Essential!)
```bash
# See dataset structure examples
python data_examples.py

# Prepare custom dataset
python dataset_preparation.py
```

### 5. Explore Educational Tools (🎓 Learn!)
```bash
# Step-by-step concept explanations
python educational_examples.py

# Interactive LLM playground
python llm_playground.py

# LLM attention patterns
python llm_attention.py

# Compare architectures
python model_comparison.py
```

### 6. Visualize Attention (🔥 Cool!)
```bash
python attention.py
```

## 🧠 Key Educational Features

### 1. **Complete Dataset Pipeline**
- **`dataset_preparation.py`** - Full preprocessing pipeline with quality filtering
- **`data_examples.py`** - Real dataset structure examples for 5+ domains
- Text cleaning, tokenization, and quality analysis tools

### 2. **Interactive Learning Tools**
- **`educational_examples.py`** - Step-by-step concept walkthroughs
- **`llm_playground.py`** - Real-time text generation experimentation
- **`model_comparison.py`** - Side-by-side architecture analysis

### 3. **Advanced Attention Visualization**
- **`attention.py`** - Translation model attention patterns
- **`llm_attention.py`** - LLM causal attention analysis
- Interactive exploration of different layers and heads
- Prompt vs generation attention comparison

### 4. **Clear Mathematical Implementation**
Every operation includes:
- Tensor shape comments (e.g., `# (batch, seq_len, d_model)`)
- Mathematical formulas from the paper
- Step-by-step transformations

### 5. **Educational Comments**
```python
# Apply the attention formula: Attention(Q,K,V) = softmax(QK^T/√d_k)V
attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
```

### 6. **Multiple Training Configurations**
- Standard training with TensorBoard
- Weights & Biases integration for experiment tracking
- Local training for development

### 7. **Architecture Comparison**
Visual side-by-side comparison of:
- Translation models vs LLMs
- Parameter count analysis across model sizes
- Task suitability comparisons
- Computational complexity analysis

## 📊 Dataset Preparation (Critical for Success!)

**Dataset quality is THE most important factor for LLM performance.** This repository includes comprehensive tools for preparing high-quality training data:

### **Dataset Structure Examples**
```bash
python data_examples.py
```
Shows 5 different dataset formats:
- **General Text** - Books, articles, web content
- **Code Generation** - Programming instruction-response pairs  
- **Conversational** - Multi-turn chat data
- **Instruction Following** - Task-oriented examples
- **Domain-Specific** - Medical, legal, technical content

### **Complete Preprocessing Pipeline**
```bash
python dataset_preparation.py
```
Full pipeline including:
- **Text Analysis** - Character/word distributions, quality metrics
- **Cleaning** - Remove URLs, normalize whitespace, filter low-quality
- **Quality Filtering** - Repetition detection, language identification
- **Tokenization** - BPE or WordLevel with custom vocabularies
- **Dataset Creation** - Proper train/val/test splits with HuggingFace format

### **Key Dataset Insights** 🎯
1. **Quality > Quantity** - 10K high-quality examples beat 100K poor ones
2. **Diversity Matters** - Mix sources, topics, and writing styles
3. **Domain-Specific** - Adapt preprocessing for your target use case
4. **Tokenization Strategy** - Choose BPE for efficiency, WordLevel for interpretability
5. **Proper Splits** - Prevent overfitting with clean validation sets

### **Example Usage**
```python
from dataset_preparation import DatasetPreparator

# Initialize preparator
prep = DatasetPreparator("my_dataset")

# Analyze raw data
analysis = prep.analyze_raw_text(my_texts)

# Clean and filter
cleaned = prep.quality_filter(my_texts)

# Create tokenizer and dataset
tokenizer = prep.create_tokenizer(cleaned, vocab_size=10000)
datasets = prep.create_training_dataset(cleaned, tokenizer)
```

## 🔧 Configuration

Edit `config.py` to experiment with different settings:

```python
{
    "d_model": 512,        # Model dimension
    "num_heads": 8,        # Number of attention heads  
    "num_layers": 6,       # Number of encoder/decoder layers
    "seq_len": 350,        # Maximum sequence length
    "dropout": 0.1,        # Dropout rate
    "lang_src": "en",      # Source language
    "lang_tgt": "it",      # Target language
}
```

## 📊 Understanding the Results

### Training Metrics
- **Loss curves** - Monitor convergence
- **BLEU scores** - Translation quality
- **Character/Word Error Rates** - Accuracy metrics

### Attention Patterns
Look for these interesting patterns in attention visualizations:
- **Diagonal patterns** in self-attention (attending to nearby words)
- **Alignment patterns** in cross-attention (source-target word relationships)
- **Specialized heads** that focus on different linguistic phenomena

## 🎓 Learning Path

### Beginner
1. Read the transformer paper
2. Understand the `MultiHeadAttentionBlock` class
3. Run `attention.py` to visualize attention
4. Train on a small dataset

### Intermediate  
1. Experiment with different model sizes
2. Implement custom decoding strategies
3. Add new positional encoding schemes
4. Try different datasets/language pairs

### Advanced
1. Implement optimization techniques (gradient clipping, warm-up)
2. Add regularization methods
3. Experiment with model architectures
4. Scale to larger datasets

## 🛠️ Technical Details

### Model Specifications
- **Default Model Size**: 512 dimensions, 8 heads, 6 layers
- **Vocabulary**: Built using HuggingFace tokenizers
- **Dataset**: OPUS Books (English ↔ Italian)
- **Training**: Adam optimizer with label smoothing

### Key Implementation Details
- **Xavier initialization** for stable training
- **Residual connections** prevent vanishing gradients
- **Layer normalization** for training stability
- **Causal masking** in decoder for autoregressive generation

## 🐛 Common Issues & Solutions

### Training Issues
- **OOM Errors**: Reduce batch size or sequence length
- **Slow Convergence**: Check learning rate and warm-up schedule
- **Poor Translation**: Ensure sufficient training data and epochs

### Attention Visualization
- **Blank Plots**: Check model weights are loaded correctly
- **Unclear Patterns**: Try different layers/heads or longer training

## 📚 Further Reading

- [Attention is All You Need](https://arxiv.org/abs/1706.03762) - Original paper
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/) - Excellent visual guide
- [Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html) - Harvard's implementation guide

## 🤝 Contributing

This is an educational project by **Roberto H Luna**, created to help more people understand transformer technology through open source learning! Contributions that improve education are welcome:
- Better documentation and comments
- Additional visualization tools
- More educational examples  
- Bug fixes and improvements
- New model architectures for learning

## 📄 License

MIT License - Feel free to use for educational purposes!

## 🙏 Acknowledgments

Created by **Roberto H Luna** with the mission of making transformer technology accessible to everyone through open source education. Special thanks to the authors of "Attention is All You Need" and the broader AI research community.

---

**Happy Learning! 🎉**

*Remember: The goal isn't just to run the code, but to understand how transformers and LLMs work under the hood.*

**Learn → Build → Teach → Repeat** 🚀
