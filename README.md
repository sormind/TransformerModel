# Transformer Model for Machine Translation - Educational Implementation

This repository contains a **from-scratch implementation** of the Transformer model for machine translation, designed specifically for **educational purposes**. It faithfully implements the architecture described in the seminal paper ["Attention is All You Need"](https://arxiv.org/abs/1706.03762) by Vaswani et al.

## 🎯 Educational Goals

This implementation helps you understand:
- **Multi-head attention mechanism** and why it's revolutionary
- **Positional encoding** and how transformers handle sequence order
- **Encoder-decoder architecture** for sequence-to-sequence tasks
- **Training dynamics** of large language models
- **Attention visualization** to see what the model learns

## 🏗️ Architecture Overview

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

## 📁 File Structure

### Core Model Files
- **`model.py`** - Complete transformer architecture implementation
  - `LayerNormalization` - Layer normalization for stable training
  - `MultiHeadAttentionBlock` - The heart of the transformer
  - `FeedForwardBlock` - Position-wise feed-forward networks
  - `PositionalEncoding` - Sinusoidal positional embeddings
  - `EncoderBlock` & `DecoderBlock` - Complete encoder/decoder layers
  - `Transformer` - Full model combining all components

### Training & Data
- **`train.py`** - Main training script with validation
- **`dataset.py`** - Bilingual dataset processing and tokenization
- **`config.py`** - Model and training configurations

### Inference & Analysis
- **`translate.py`** - Interactive translation interface
- **`inference.py`** - Model evaluation and testing
- **`attention.py`** - 🎨 **Attention visualization** (must-see!)
- **`beam_search.py`** - Advanced decoding strategies

### Additional Training Options
- **`train_wb.py`** - Training with Weights & Biases integration
- **`local_train.py`** - Local training setup

## 🚀 Quick Start

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Train the Model
```bash
# Basic training
python train.py

# With W&B logging
python train_wb.py
```

### 3. Translate Text
```bash
python translate.py
```

### 4. Visualize Attention (🔥 Cool!)
```bash
python attention.py
```

## 🧠 Key Educational Features

### 1. **Multi-Head Attention Visualization**
The `attention.py` script creates beautiful heatmaps showing:
- **Self-attention** patterns in encoder and decoder
- **Cross-attention** between source and target languages
- How different attention heads learn different relationships

### 2. **Clear Mathematical Implementation**
Every operation includes:
- Tensor shape comments (e.g., `# (batch, seq_len, d_model)`)
- Mathematical formulas from the paper
- Step-by-step transformations

### 3. **Educational Comments**
```python
# Apply the attention formula: Attention(Q,K,V) = softmax(QK^T/√d_k)V
attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
```

### 4. **Multiple Training Configurations**
- Standard training with TensorBoard
- Weights & Biases integration for experiment tracking
- Local training for development

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

This is an educational project! Contributions that improve learning are welcome:
- Better documentation and comments
- Additional visualization tools
- More educational examples
- Bug fixes and improvements

## 📄 License

MIT License - Feel free to use for educational purposes!

---

**Happy Learning! 🎉**

*Remember: The goal isn't just to run the code, but to understand how transformers work under the hood.*
