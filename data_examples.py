"""
Dataset Examples for LLM Training

This script shows practical examples of how to structure datasets for different types of LLM training:

1. General Text (Books, Articles, Web Content)
2. Code Generation (Programming Languages)
3. Conversational AI (Chat, Q&A)
4. Domain-Specific (Technical, Medical, Legal)
5. Instruction Following (Task-Oriented)

Each example shows the data structure, preprocessing steps, and training considerations.
"""

import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
from datasets import Dataset, DatasetDict

class DatasetExamples:
    """Examples of different dataset structures for LLM training"""
    
    def __init__(self):
        print("📚 Dataset Structure Examples for LLM Training")
    
    def general_text_example(self):
        """Example: General text dataset (books, articles, web content)"""
        print("\n1. 📖 General Text Dataset")
        print("-" * 40)
        
        # Example structure for general text
        general_texts = [
            {
                "text": "The rise of artificial intelligence has transformed numerous industries. From healthcare to finance, AI applications are becoming increasingly sophisticated. Machine learning algorithms can now process vast amounts of data to identify patterns and make predictions that were previously impossible.",
                "source": "tech_article",
                "length": 267,
                "quality_score": 0.85
            },
            {
                "text": "Climate change represents one of the most pressing challenges of our time. Rising global temperatures have led to more frequent extreme weather events, melting ice caps, and rising sea levels. Scientists worldwide are working on solutions to mitigate these effects.",
                "source": "news_article", 
                "length": 248,
                "quality_score": 0.92
            },
            {
                "text": "In the quiet town of Millbrook, Sarah discovered an old diary hidden beneath the floorboards of her grandmother's attic. The yellowed pages contained stories of love, loss, and adventure that spanned decades. Each entry revealed another piece of her family's hidden history.",
                "source": "fiction",
                "length": 285,
                "quality_score": 0.88
            }
        ]
        
        print("Structure:")
        print("- text: The main content for training")
        print("- source: Where the text came from (for filtering)")
        print("- length: Character count (for sequence planning)")
        print("- quality_score: Content quality metric")
        
        print(f"\nExample entry:")
        print(json.dumps(general_texts[0], indent=2))
        
        print("\n🔧 Preprocessing Steps:")
        print("1. Remove duplicate content")
        print("2. Filter by quality score (> 0.7)")
        print("3. Normalize whitespace and encoding")
        print("4. Split long texts into chunks (< 2048 tokens)")
        print("5. Shuffle to mix different sources")
        
        return general_texts
    
    def code_generation_example(self):
        """Example: Code generation dataset"""
        print("\n2. 💻 Code Generation Dataset")
        print("-" * 40)
        
        code_examples = [
            {
                "instruction": "Write a Python function to calculate the factorial of a number",
                "input": "def factorial(n):",
                "output": "def factorial(n):\n    if n == 0 or n == 1:\n        return 1\n    else:\n        return n * factorial(n - 1)",
                "language": "python",
                "difficulty": "beginner",
                "tags": ["recursion", "math"]
            },
            {
                "instruction": "Create a JavaScript function to validate email addresses",
                "input": "function validateEmail(email) {",
                "output": "function validateEmail(email) {\n    const regex = /^[^\\s@]+@[^\\s@]+\\.[^\\s@]+$/;\n    return regex.test(email);\n}",
                "language": "javascript",
                "difficulty": "intermediate",
                "tags": ["regex", "validation"]
            },
            {
                "instruction": "Implement a binary search algorithm in Python",
                "input": "def binary_search(arr, target):",
                "output": "def binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    return -1",
                "language": "python", 
                "difficulty": "intermediate",
                "tags": ["algorithms", "search"]
            }
        ]
        
        print("Structure:")
        print("- instruction: What the code should do")
        print("- input: Starting point or function signature")
        print("- output: Complete, correct code")
        print("- language: Programming language")
        print("- difficulty: Complexity level")
        print("- tags: Relevant concepts")
        
        print(f"\nExample entry:")
        print(json.dumps(code_examples[0], indent=2))
        
        print("\n🔧 Training Format Options:")
        print("1. Instruction → Code (for instruction following)")
        print("2. Comment + Partial Code → Complete Code")
        print("3. Plain text description → Executable code")
        print("4. Code completion (given prefix, generate suffix)")
        
        return code_examples
    
    def conversational_example(self):
        """Example: Conversational AI dataset"""
        print("\n3. 💬 Conversational AI Dataset")
        print("-" * 40)
        
        conversations = [
            {
                "conversation_id": "conv_001",
                "messages": [
                    {
                        "role": "user",
                        "content": "Can you explain what machine learning is?",
                        "timestamp": "2024-01-15T10:30:00Z"
                    },
                    {
                        "role": "assistant", 
                        "content": "Machine learning is a subset of artificial intelligence where computers learn to make predictions or decisions by finding patterns in data, rather than being explicitly programmed for each task. For example, it can learn to recognize images, translate languages, or recommend products based on examples.",
                        "timestamp": "2024-01-15T10:30:15Z"
                    },
                    {
                        "role": "user",
                        "content": "What are some common types of machine learning?",
                        "timestamp": "2024-01-15T10:31:00Z"
                    },
                    {
                        "role": "assistant",
                        "content": "There are three main types: 1) Supervised learning - learns from labeled examples, 2) Unsupervised learning - finds patterns in unlabeled data, and 3) Reinforcement learning - learns through trial and error with rewards. Each type is suited for different problems.",
                        "timestamp": "2024-01-15T10:31:30Z"
                    }
                ],
                "topic": "machine_learning",
                "quality_rating": 4.5,
                "language": "en"
            }
        ]
        
        print("Structure:")
        print("- conversation_id: Unique identifier")
        print("- messages: List of alternating user/assistant messages")
        print("- role: 'user' or 'assistant'")
        print("- content: The actual message text")
        print("- topic/quality_rating/language: Metadata for filtering")
        
        print(f"\nExample conversation:")
        conv = conversations[0]
        for msg in conv['messages'][:2]:
            print(f"{msg['role']}: {msg['content']}")
        
        print("\n🔧 Training Formats:")
        print("1. Full conversation history → Next response")
        print("2. System prompt + User message → Assistant response")
        print("3. Multi-turn with conversation context")
        print("4. Role-specific fine-tuning")
        
        return conversations
    
    def instruction_following_example(self):
        """Example: Instruction following dataset"""
        print("\n4. 📋 Instruction Following Dataset")
        print("-" * 40)
        
        instructions = [
            {
                "instruction": "Summarize the following text in 2-3 sentences",
                "input": "Artificial intelligence has made remarkable progress in recent years, with breakthroughs in natural language processing, computer vision, and robotics. These advances have led to practical applications in healthcare, autonomous vehicles, and personal assistants. However, concerns about job displacement, privacy, and algorithmic bias have also emerged as important considerations for society.",
                "output": "AI has achieved significant breakthroughs in areas like language processing, vision, and robotics, leading to applications in healthcare, autonomous vehicles, and personal assistants. However, this progress has raised concerns about job displacement, privacy, and algorithmic bias that society must address.",
                "task_type": "summarization",
                "difficulty": "medium"
            },
            {
                "instruction": "Translate the following English text to French",
                "input": "Good morning! How are you today?",
                "output": "Bonjour ! Comment allez-vous aujourd'hui ?",
                "task_type": "translation",
                "difficulty": "easy"
            },
            {
                "instruction": "Extract the key information from this email and format it as bullet points",
                "input": "Subject: Project Meeting Tomorrow\n\nHi team, we have our weekly project meeting tomorrow at 2 PM in Conference Room B. Please bring your progress reports and any blockers you're facing. We'll also discuss the new requirements from the client and plan for next week. Thanks, Sarah",
                "output": "• Meeting: Tomorrow at 2 PM\n• Location: Conference Room B\n• Bring: Progress reports and blockers\n• Agenda: Discuss new client requirements and next week planning\n• Organizer: Sarah",
                "task_type": "information_extraction",
                "difficulty": "medium"
            }
        ]
        
        print("Structure:")
        print("- instruction: Clear task description")
        print("- input: Source material or context") 
        print("- output: Expected result")
        print("- task_type: Category of task")
        print("- difficulty: Complexity level")
        
        print(f"\nExample task:")
        print(json.dumps(instructions[0], indent=2))
        
        print("\n🔧 Training Format:")
        print("Template: '{instruction}\\n\\nInput: {input}\\n\\nOutput: {output}'")
        print("This teaches the model to follow instructions precisely")
        
        return instructions
    
    def domain_specific_example(self):
        """Example: Domain-specific dataset (medical)"""
        print("\n5. 🏥 Domain-Specific Dataset (Medical)")
        print("-" * 40)
        
        medical_data = [
            {
                "patient_query": "I've been experiencing chest pain and shortness of breath for the past few days. Should I be concerned?",
                "medical_response": "Chest pain and shortness of breath can be symptoms of serious conditions and should be evaluated promptly by a healthcare professional. Please seek medical attention immediately, especially if the pain is severe, radiating to your arm or jaw, or if you're experiencing difficulty breathing at rest.",
                "specialty": "cardiology",
                "urgency": "high",
                "disclaimer": "This is general information only. Always consult with qualified healthcare professionals for medical advice.",
                "verified": True
            },
            {
                "symptom_description": "Patient presents with persistent cough, fever (101.2°F), and fatigue lasting 5 days",
                "differential_diagnosis": [
                    "Viral upper respiratory infection",
                    "Bacterial pneumonia", 
                    "COVID-19",
                    "Influenza"
                ],
                "recommended_tests": [
                    "Chest X-ray",
                    "Complete blood count",
                    "COVID-19 PCR test"
                ],
                "specialty": "internal_medicine",
                "confidence": 0.85
            }
        ]
        
        print("Structure:")
        print("- patient_query/symptom_description: Input")
        print("- medical_response/diagnosis: Output")
        print("- specialty: Medical field")
        print("- urgency/confidence: Metadata")
        print("- disclaimer: Important safety notice")
        print("- verified: Quality assurance flag")
        
        print(f"\nExample entry:")
        print(json.dumps(medical_data[0], indent=2))
        
        print("\n⚠️  Special Considerations:")
        print("1. Include medical disclaimers")
        print("2. Verify with medical professionals")
        print("3. Filter for accuracy and safety")
        print("4. Consider liability implications")
        print("5. Regular updates with latest research")
        
        return medical_data
    
    def create_training_format_examples(self):
        """Show how to format data for different training objectives"""
        print("\n6. 🎯 Training Format Examples")
        print("-" * 40)
        
        formats = {
            "completion": {
                "description": "Simple text completion",
                "example": {
                    "text": "The capital of France is Paris. The capital of Germany is Berlin. The capital of Italy is"
                },
                "use_case": "General language modeling"
            },
            
            "instruction": {
                "description": "Instruction following format",
                "example": {
                    "text": "### Instruction:\nWrite a haiku about programming\n\n### Response:\nCode flows like water\nBugs hide in logic's shadows\nCoffee fuels the fix"
                },
                "use_case": "Task-specific fine-tuning"
            },
            
            "chat": {
                "description": "Conversational format",
                "example": {
                    "text": "<|user|>What is Python?<|assistant|>Python is a high-level, interpreted programming language known for its simplicity and readability. It's widely used for web development, data science, and automation.<|user|>Can you show me a simple example?"
                },
                "use_case": "Conversational AI"
            },
            
            "few_shot": {
                "description": "Few-shot learning examples",
                "example": {
                    "text": "Classify the sentiment:\n\nText: I love this product!\nSentiment: Positive\n\nText: This is terrible.\nSentiment: Negative\n\nText: It's okay, not great.\nSentiment: Neutral\n\nText: Amazing quality and fast shipping!\nSentiment:"
                },
                "use_case": "Learning from examples"
            }
        }
        
        for format_name, info in formats.items():
            print(f"\n{format_name.upper()} Format:")
            print(f"Description: {info['description']}")
            print(f"Use case: {info['use_case']}")
            print(f"Example: {info['example']['text'][:100]}...")
    
    def dataset_quality_checklist(self):
        """Provide a checklist for dataset quality"""
        print("\n7. ✅ Dataset Quality Checklist")
        print("-" * 40)
        
        checklist = {
            "Data Quality": [
                "Remove duplicates and near-duplicates",
                "Filter low-quality content (spam, gibberish)",
                "Verify factual accuracy where possible",
                "Check for consistent formatting",
                "Remove personally identifiable information (PII)"
            ],
            
            "Content Diversity": [
                "Balance different topics and domains",
                "Include various writing styles and formats",
                "Ensure demographic representation",
                "Mix difficulty levels",
                "Cover edge cases and exceptions"
            ],
            
            "Technical Considerations": [
                "Consistent tokenization strategy",
                "Appropriate sequence lengths",
                "Proper train/validation/test splits",
                "Balanced batch sizes",
                "Version control for datasets"
            ],
            
            "Legal & Ethical": [
                "Check data licensing and usage rights",
                "Remove copyrighted material",
                "Address potential biases",
                "Include content warnings where needed",
                "Document data sources and collection methods"
            ]
        }
        
        for category, items in checklist.items():
            print(f"\n{category}:")
            for item in items:
                print(f"  • {item}")
    
    def save_examples_to_files(self, output_dir: str = "dataset_examples"):
        """Save all examples to files for reference"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print(f"\n💾 Saving examples to {output_dir}/")
        
        # Save each example type
        examples = {
            "general_text": self.general_text_example(),
            "code_generation": self.code_generation_example(), 
            "conversational": self.conversational_example(),
            "instruction_following": self.instruction_following_example(),
            "domain_specific": self.domain_specific_example()
        }
        
        for name, data in examples.items():
            file_path = output_path / f"{name}.json"
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"  ✅ {file_path}")
        
        # Save a README with instructions
        readme_content = """# Dataset Examples for LLM Training

This directory contains examples of different dataset structures for training language models.

## Files:
- `general_text.json` - General text dataset structure
- `code_generation.json` - Code generation examples  
- `conversational.json` - Chat/conversation format
- `instruction_following.json` - Task instruction examples
- `domain_specific.json` - Specialized domain data

## Usage:
1. Choose the format that matches your use case
2. Adapt the structure to your specific data
3. Follow the preprocessing guidelines
4. Implement quality checks
5. Create proper train/validation splits

## Important Notes:
- Always verify data quality and licensing
- Consider bias and ethical implications
- Test with small datasets first
- Monitor training metrics closely
"""
        
        readme_path = output_path / "README.md"
        with open(readme_path, 'w') as f:
            f.write(readme_content)
        
        print(f"  📝 {readme_path}")
        print(f"\n✅ Examples saved! Check {output_dir}/ for reference files.")

def main():
    """Run all dataset examples"""
    examples = DatasetExamples()
    
    # Show all examples
    examples.general_text_example()
    examples.code_generation_example()
    examples.conversational_example()
    examples.instruction_following_example()
    examples.domain_specific_example()
    examples.create_training_format_examples()
    examples.dataset_quality_checklist()
    
    # Save examples to files
    examples.save_examples_to_files()
    
    print("\n" + "="*60)
    print("🎉 Dataset Examples Complete!")
    print("\nNext Steps:")
    print("1. Choose the format that fits your use case")
    print("2. Prepare your raw data using dataset_preparation.py")
    print("3. Structure it according to these examples")
    print("4. Train your LLM with train_llm.py")

if __name__ == "__main__":
    main() 