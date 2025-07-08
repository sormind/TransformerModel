"""
Dataset Preparation for LLM Training

This script provides comprehensive tools for preparing text datasets for language model training.
It covers the entire pipeline from raw text to training-ready data:

1. Data Collection & Sources
2. Text Cleaning & Preprocessing  
3. Quality Filtering
4. Tokenization Strategy
5. Dataset Structuring
6. Vocabulary Analysis
7. Train/Validation/Test Splits
8. Data Format Conversion

Essential for understanding how to prepare high-quality training data!
"""

import os
import json
import re
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple
from collections import Counter, defaultdict
import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict, load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, normalizers, processors
from tokenizers.models import BPE, WordLevel
from tokenizers.trainers import BpeTrainer, WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace, ByteLevel
from tokenizers.normalizers import NFD, Lowercase, StripAccents
from tokenizers.processors import TemplateProcessing
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

class DatasetPreparator:
    """Comprehensive dataset preparation for LLM training"""
    
    def __init__(self, output_dir: str = "prepared_data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        print(f"📁 Dataset Preparator initialized - Output: {self.output_dir}")
    
    def analyze_raw_text(self, texts: List[str]) -> Dict[str, Any]:
        """Analyze raw text data to understand its characteristics"""
        print("🔍 Analyzing raw text data...")
        
        # Basic statistics
        total_texts = len(texts)
        total_chars = sum(len(text) for text in texts)
        total_words = sum(len(text.split()) for text in texts)
        
        # Length distributions
        text_lengths = [len(text) for text in texts]
        word_counts = [len(text.split()) for text in texts]
        
        # Character frequency
        char_counter = Counter()
        for text in texts:
            char_counter.update(text.lower())
        
        # Language detection patterns
        english_chars = sum(char_counter[c] for c in 'abcdefghijklmnopqrstuvwxyz')
        total_alpha_chars = sum(char_counter[c] for c in char_counter if c.isalpha())
        english_ratio = english_chars / max(total_alpha_chars, 1)
        
        analysis = {
            'total_texts': total_texts,
            'total_characters': total_chars,
            'total_words': total_words,
            'avg_chars_per_text': total_chars / max(total_texts, 1),
            'avg_words_per_text': total_words / max(total_texts, 1),
            'text_length_stats': {
                'min': min(text_lengths) if text_lengths else 0,
                'max': max(text_lengths) if text_lengths else 0,
                'mean': np.mean(text_lengths) if text_lengths else 0,
                'median': np.median(text_lengths) if text_lengths else 0,
                'std': np.std(text_lengths) if text_lengths else 0
            },
            'word_count_stats': {
                'min': min(word_counts) if word_counts else 0,
                'max': max(word_counts) if word_counts else 0,
                'mean': np.mean(word_counts) if word_counts else 0,
                'median': np.median(word_counts) if word_counts else 0
            },
            'most_common_chars': char_counter.most_common(20),
            'english_ratio': english_ratio,
            'unique_chars': len(char_counter)
        }
        
        self._print_analysis(analysis)
        return analysis
    
    def _print_analysis(self, analysis: Dict[str, Any]):
        """Print formatted analysis results"""
        print("\n📊 Dataset Analysis Results:")
        print("=" * 50)
        print(f"Total texts: {analysis['total_texts']:,}")
        print(f"Total characters: {analysis['total_characters']:,}")
        print(f"Total words: {analysis['total_words']:,}")
        print(f"Average characters per text: {analysis['avg_chars_per_text']:.1f}")
        print(f"Average words per text: {analysis['avg_words_per_text']:.1f}")
        print(f"English character ratio: {analysis['english_ratio']:.2%}")
        print(f"Unique characters: {analysis['unique_chars']}")
        
        print(f"\nText Length Distribution:")
        stats = analysis['text_length_stats']
        print(f"  Min: {stats['min']:,} | Max: {stats['max']:,}")
        print(f"  Mean: {stats['mean']:.1f} | Median: {stats['median']:.1f}")
        print(f"  Std Dev: {stats['std']:.1f}")
        
        print(f"\nMost Common Characters:")
        for char, count in analysis['most_common_chars'][:10]:
            if char.isprintable():
                print(f"  '{char}': {count:,}")
    
    def clean_text(self, text: str, 
                   remove_urls: bool = True,
                   remove_emails: bool = True,
                   remove_phone_numbers: bool = True,
                   normalize_whitespace: bool = True,
                   remove_excessive_punctuation: bool = True,
                   min_length: int = 10,
                   max_length: int = 100000) -> str:
        """
        Clean and normalize text data
        
        Args:
            text: Raw text to clean
            remove_urls: Remove HTTP/HTTPS URLs
            remove_emails: Remove email addresses
            remove_phone_numbers: Remove phone numbers
            normalize_whitespace: Normalize whitespace
            remove_excessive_punctuation: Remove excessive punctuation
            min_length: Minimum text length to keep
            max_length: Maximum text length to keep
        """
        if not text or len(text) < min_length:
            return ""
        
        # Remove URLs
        if remove_urls:
            text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Remove emails
        if remove_emails:
            text = re.sub(r'\S+@\S+', '', text)
        
        # Remove phone numbers (basic pattern)
        if remove_phone_numbers:
            text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '', text)
        
        # Remove excessive punctuation
        if remove_excessive_punctuation:
            text = re.sub(r'[!.?]{3,}', '...', text)
            text = re.sub(r'[-]{3,}', '---', text)
        
        # Normalize whitespace
        if normalize_whitespace:
            text = re.sub(r'\s+', ' ', text)
            text = text.strip()
        
        # Length filtering
        if len(text) > max_length:
            text = text[:max_length]
        
        return text
    
    def quality_filter(self, texts: List[str], 
                      min_words: int = 5,
                      max_words: int = 5000,
                      min_unique_words: int = 3,
                      max_repetition_ratio: float = 0.7,
                      min_alpha_ratio: float = 0.6) -> List[str]:
        """
        Filter texts based on quality metrics
        
        Args:
            texts: List of texts to filter
            min_words: Minimum word count
            max_words: Maximum word count
            min_unique_words: Minimum unique words
            max_repetition_ratio: Maximum ratio of repeated content
            min_alpha_ratio: Minimum ratio of alphabetic characters
        """
        print(f"🔧 Quality filtering {len(texts)} texts...")
        
        filtered_texts = []
        stats = {
            'too_short': 0,
            'too_long': 0,
            'too_repetitive': 0,
            'insufficient_alpha': 0,
            'insufficient_unique': 0,
            'passed': 0
        }
        
        for text in tqdm(texts, desc="Filtering"):
            words = text.split()
            word_count = len(words)
            
            # Word count filter
            if word_count < min_words:
                stats['too_short'] += 1
                continue
            if word_count > max_words:
                stats['too_long'] += 1
                continue
            
            # Unique words filter
            unique_words = len(set(words))
            if unique_words < min_unique_words:
                stats['insufficient_unique'] += 1
                continue
            
            # Repetition filter
            if word_count > 0:
                repetition_ratio = 1 - (unique_words / word_count)
                if repetition_ratio > max_repetition_ratio:
                    stats['too_repetitive'] += 1
                    continue
            
            # Alphabetic character ratio
            alpha_chars = sum(1 for c in text if c.isalpha())
            total_chars = len(text)
            if total_chars > 0:
                alpha_ratio = alpha_chars / total_chars
                if alpha_ratio < min_alpha_ratio:
                    stats['insufficient_alpha'] += 1
                    continue
            
            filtered_texts.append(text)
            stats['passed'] += 1
        
        print(f"\n📈 Quality Filtering Results:")
        print(f"  Original texts: {len(texts):,}")
        print(f"  Passed filters: {stats['passed']:,} ({stats['passed']/len(texts):.1%})")
        print(f"  Too short: {stats['too_short']:,}")
        print(f"  Too long: {stats['too_long']:,}")
        print(f"  Too repetitive: {stats['too_repetitive']:,}")
        print(f"  Insufficient alphabetic: {stats['insufficient_alpha']:,}")
        print(f"  Insufficient unique words: {stats['insufficient_unique']:,}")
        
        return filtered_texts
    
    def create_tokenizer(self, texts: List[str], 
                        vocab_size: int = 10000,
                        tokenizer_type: str = "bpe",
                        special_tokens: List[str] = None) -> Tokenizer:
        """
        Create and train a tokenizer on the dataset
        """
        print(f"🔤 Training {tokenizer_type} tokenizer (vocab_size={vocab_size:,})...")
        
        if special_tokens is None:
            special_tokens = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
        
        if tokenizer_type == "bpe":
            tokenizer = Tokenizer(BPE(unk_token="<UNK>"))
            tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=True)
            trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=special_tokens)
        else:  # wordlevel
            tokenizer = Tokenizer(WordLevel(unk_token="<UNK>"))
            tokenizer.pre_tokenizer = Whitespace()
            trainer = WordLevelTrainer(vocab_size=vocab_size, special_tokens=special_tokens)
        
        # Train tokenizer
        tokenizer.train_from_iterator(texts, trainer)
        
        # Add post-processor
        tokenizer.post_processor = TemplateProcessing(
            single="<BOS> $A <EOS>",
            special_tokens=[("<BOS>", 2), ("<EOS>", 3)]
        )
        
        print(f"✅ Tokenizer trained! Vocabulary size: {tokenizer.get_vocab_size()}")
        return tokenizer
    
    def create_training_dataset(self, texts: List[str], tokenizer: Tokenizer, 
                               seq_length: int = 512, stride: int = 256,
                               train_ratio: float = 0.8, val_ratio: float = 0.1) -> Dict[str, Dataset]:
        """
        Create train/validation/test datasets with proper tokenization
        """
        print(f"📝 Creating training dataset (seq_len={seq_length}, stride={stride})...")
        
        # Tokenize all texts and create sequences
        all_sequences = []
        for text in tqdm(texts, desc="Tokenizing"):
            tokens = tokenizer.encode(text).ids
            
            # Create overlapping sequences
            for i in range(0, len(tokens) - seq_length, stride):
                sequence = tokens[i:i + seq_length + 1]
                if len(sequence) == seq_length + 1:
                    all_sequences.append({
                        'input_ids': sequence[:-1],
                        'labels': sequence[1:]
                    })
        
        print(f"Created {len(all_sequences):,} training sequences")
        
        # Split dataset
        random.shuffle(all_sequences)
        n_total = len(all_sequences)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        train_data = all_sequences[:n_train]
        val_data = all_sequences[n_train:n_train + n_val]
        test_data = all_sequences[n_train + n_val:]
        
        # Create HuggingFace datasets
        datasets = {
            'train': Dataset.from_list(train_data),
            'validation': Dataset.from_list(val_data),
            'test': Dataset.from_list(test_data)
        }
        
        print(f"Dataset splits:")
        print(f"  Train: {len(train_data):,} sequences")
        print(f"  Validation: {len(val_data):,} sequences")
        print(f"  Test: {len(test_data):,} sequences")
        
        return datasets
    
    def visualize_dataset_stats(self, texts: List[str], analysis: Dict[str, Any]):
        """Create visualizations of dataset statistics"""
        print("📊 Creating dataset visualizations...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Text length distribution
        text_lengths = [len(text) for text in texts[:10000]]  # Sample for speed
        axes[0, 0].hist(text_lengths, bins=50, alpha=0.7, color='blue')
        axes[0, 0].set_title('Text Length Distribution (Characters)')
        axes[0, 0].set_xlabel('Characters')
        axes[0, 0].set_ylabel('Frequency')
        
        # Word count distribution
        word_counts = [len(text.split()) for text in texts[:10000]]
        axes[0, 1].hist(word_counts, bins=50, alpha=0.7, color='green')
        axes[0, 1].set_title('Word Count Distribution')
        axes[0, 1].set_xlabel('Words')
        axes[0, 1].set_ylabel('Frequency')
        
        # Character frequency
        char_freq = dict(analysis['most_common_chars'][:20])
        chars = list(char_freq.keys())
        freqs = list(char_freq.values())
        axes[1, 0].bar(range(len(chars)), freqs, alpha=0.7, color='red')
        axes[1, 0].set_title('Most Common Characters')
        axes[1, 0].set_xlabel('Character')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_xticks(range(len(chars)))
        axes[1, 0].set_xticklabels(chars, rotation=45)
        
        # Dataset quality metrics
        quality_metrics = ['English Ratio', 'Avg Words/Text', 'Unique Chars']
        values = [analysis['english_ratio'], 
                 analysis['avg_words_per_text']/1000,  # Scale down
                 analysis['unique_chars']/100]  # Scale down
        axes[1, 1].bar(quality_metrics, values, alpha=0.7, color='purple')
        axes[1, 1].set_title('Dataset Quality Metrics (Scaled)')
        axes[1, 1].set_ylabel('Scaled Value')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'dataset_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def prepare_custom_dataset(self, data_source: str, 
                             domain: str = "general",
                             vocab_size: int = 10000,
                             seq_length: int = 512) -> Dict[str, Any]:
        """
        Complete pipeline to prepare a custom dataset
        """
        print(f"🚀 Preparing custom {domain} dataset from {data_source}")
        print("=" * 60)
        
        # Step 1: Load raw data
        texts = self.load_data_source(data_source, domain)
        
        # Step 2: Analyze raw data
        analysis = self.analyze_raw_text(texts)
        
        # Step 3: Clean texts
        print("\n🧹 Cleaning texts...")
        cleaned_texts = []
        for text in tqdm(texts, desc="Cleaning"):
            cleaned = self.clean_text(text)
            if cleaned:
                cleaned_texts.append(cleaned)
        
        print(f"Cleaned texts: {len(cleaned_texts):,} (from {len(texts):,})")
        
        # Step 4: Quality filtering
        filtered_texts = self.quality_filter(cleaned_texts)
        
        # Step 5: Create tokenizer
        tokenizer = self.create_tokenizer(filtered_texts, vocab_size=vocab_size)
        
        # Step 6: Create training dataset
        datasets = self.create_training_dataset(filtered_texts, tokenizer, seq_length)
        
        # Step 7: Visualizations
        self.visualize_dataset_stats(filtered_texts, analysis)
        
        # Step 8: Save everything
        output_paths = self.save_prepared_dataset(datasets, tokenizer, analysis, domain)
        
        print(f"\n✅ Dataset preparation complete!")
        print(f"📁 Output saved to: {self.output_dir}")
        
        return {
            'datasets': datasets,
            'tokenizer': tokenizer,
            'analysis': analysis,
            'output_paths': output_paths
        }
    
    def load_data_source(self, source: str, domain: str) -> List[str]:
        """Load text data from various sources"""
        print(f"📥 Loading data from {source} (domain: {domain})")
        
        texts = []
        
        if source.startswith("huggingface:"):
            # Load from HuggingFace datasets
            dataset_name = source.split(":", 1)[1]
            try:
                dataset = load_dataset(dataset_name)
                if 'train' in dataset:
                    texts = [item['text'] for item in dataset['train'] if 'text' in item]
                elif 'text' in dataset:
                    texts = [item['text'] for item in dataset['text']]
                print(f"Loaded {len(texts):,} texts from HuggingFace dataset")
            except Exception as e:
                print(f"Error loading HuggingFace dataset: {e}")
        
        elif os.path.isfile(source):
            # Load from file
            file_path = Path(source)
            if file_path.suffix == '.txt':
                with open(file_path, 'r', encoding='utf-8') as f:
                    texts = [line.strip() for line in f if line.strip()]
            elif file_path.suffix == '.json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        texts = [str(item) for item in data]
                    elif isinstance(data, dict) and 'texts' in data:
                        texts = data['texts']
            print(f"Loaded {len(texts):,} texts from file")
        
        elif os.path.isdir(source):
            # Load from directory of text files
            source_path = Path(source)
            for file_path in source_path.glob("*.txt"):
                with open(file_path, 'r', encoding='utf-8') as f:
                    file_texts = [line.strip() for line in f if line.strip()]
                    texts.extend(file_texts)
            print(f"Loaded {len(texts):,} texts from directory")
        
        else:
            # Demo data for testing
            texts = self.create_demo_data(domain)
            print(f"Created {len(texts):,} demo texts for domain: {domain}")
        
        return texts
    
    def create_demo_data(self, domain: str) -> List[str]:
        """Create sample data for different domains"""
        if domain == "code":
            return [
                "def hello_world():\n    print('Hello, World!')\n    return True",
                "class DataProcessor:\n    def __init__(self, data):\n        self.data = data",
                "import numpy as np\nimport pandas as pd\n\ndf = pd.read_csv('data.csv')",
                "for i in range(10):\n    if i % 2 == 0:\n        print(f'Even: {i}')",
                "try:\n    result = process_data()\nexcept Exception as e:\n    logging.error(e)"
            ] * 100
        
        elif domain == "literature":
            return [
                "It was the best of times, it was the worst of times, it was the age of wisdom.",
                "In a hole in the ground there lived a hobbit. Not a nasty, dirty, wet hole.",
                "Call me Ishmael. Some years ago—never mind how long precisely—having little money.",
                "All happy families are alike; each unhappy family is unhappy in its own way.",
                "It is a truth universally acknowledged, that a single man in possession of a good fortune."
            ] * 200
        
        else:  # general
            return [
                "The quick brown fox jumps over the lazy dog. This is a sample sentence.",
                "Machine learning is a subset of artificial intelligence that focuses on algorithms.",
                "Climate change is one of the most pressing issues of our time requiring global action.",
                "Technology has revolutionized the way we communicate and access information.",
                "Education plays a crucial role in developing critical thinking and problem-solving skills."
            ] * 150
    
    def save_prepared_dataset(self, datasets: Dict[str, Dataset], 
                            tokenizer: Tokenizer, analysis: Dict[str, Any], 
                            domain: str) -> Dict[str, str]:
        """Save all prepared dataset components"""
        print("💾 Saving prepared dataset...")
        
        # Create domain-specific directory
        domain_dir = self.output_dir / domain
        domain_dir.mkdir(exist_ok=True)
        
        output_paths = {}
        
        # Save datasets
        dataset_dict = DatasetDict(datasets)
        dataset_path = domain_dir / "dataset"
        dataset_dict.save_to_disk(str(dataset_path))
        output_paths['dataset'] = str(dataset_path)
        
        # Save tokenizer
        tokenizer_path = domain_dir / "tokenizer.json"
        tokenizer.save(str(tokenizer_path))
        output_paths['tokenizer'] = str(tokenizer_path)
        
        # Save analysis
        analysis_path = domain_dir / "analysis.json"
        with open(analysis_path, 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            serializable_analysis = self._make_json_serializable(analysis)
            json.dump(serializable_analysis, f, indent=2)
        output_paths['analysis'] = str(analysis_path)
        
        # Save preparation config
        config = {
            'domain': domain,
            'vocab_size': tokenizer.get_vocab_size(),
            'dataset_sizes': {split: len(dataset) for split, dataset in datasets.items()},
            'total_texts': analysis['total_texts'],
            'preparation_timestamp': pd.Timestamp.now().isoformat()
        }
        config_path = domain_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        output_paths['config'] = str(config_path)
        
        print(f"✅ Saved to {domain_dir}")
        return output_paths
    
    def _make_json_serializable(self, obj):
        """Convert numpy types to Python types for JSON serialization"""
        if isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

def create_dataset_tutorial():
    """Interactive tutorial for dataset preparation"""
    print("🎓 LLM Dataset Preparation Tutorial")
    print("=" * 50)
    
    preparator = DatasetPreparator("tutorial_output")
    
    print("\n1. Creating sample datasets for different domains...")
    
    # Prepare different domain examples
    domains = ["general", "code", "literature"]
    
    for domain in domains:
        print(f"\n📚 Preparing {domain} domain dataset:")
        result = preparator.prepare_custom_dataset(
            data_source="demo",
            domain=domain,
            vocab_size=5000,
            seq_length=256
        )
        
        # Show some examples
        print(f"\n🔍 Sample from {domain} dataset:")
        dataset = result['datasets']['train']
        tokenizer = result['tokenizer']
        
        for i in range(min(3, len(dataset))):
            tokens = dataset[i]['input_ids']
            text = tokenizer.decode(tokens)
            print(f"  Example {i+1}: {text[:100]}...")
    
    print("\n" + "="*50)
    print("🎉 Tutorial Complete!")
    print("\nKey Takeaways:")
    print("1. Dataset quality is crucial for LLM performance")
    print("2. Different domains need different preprocessing")
    print("3. Tokenization strategy affects vocabulary efficiency")
    print("4. Proper train/val/test splits prevent overfitting")
    print("5. Dataset analysis helps understand your data")

def main():
    """Run dataset preparation examples"""
    print("📊 Dataset Preparation for LLM Training")
    print("=" * 50)
    
    # Run the tutorial
    create_dataset_tutorial()
    
    print("\n🔧 Advanced Usage:")
    print("preparator = DatasetPreparator()")
    print("result = preparator.prepare_custom_dataset('path/to/data', 'domain_name')")

if __name__ == "__main__":
    main() 