---
layer: "5"
category: "training"
subcategory: "preparation"
tags: ["dataset", "data-preparation", "preprocessing", "tokenization"]
rocm_version: "7.0+"
therock_included: true
last_updated: 2025-11-01
difficulty: "intermediate"
estimated_time: "40min"
---

# Dataset Preparation for LLM Training

Complete guide to preparing datasets for fine-tuning LLMs on AMD GPUs.

## Dataset Formats

### Instruction Format

```json
{
    "instruction": "What is machine learning?",
    "input": "",
    "output": "Machine learning is a subset of artificial intelligence..."
}
```

### Conversation Format

```json
{
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi! How can I help you today?"}
    ]
}
```

### Text Completion Format

```json
{
    "text": "The capital of France is Paris. It is known for..."
}
```

## Loading Datasets

### From Hugging Face Hub

```python
from datasets import load_dataset

# Load public dataset
dataset = load_dataset("timdettmers/openassistant-guanaco")
print(dataset)

# Load specific split
train_dataset = load_dataset("tatsu-lab/alpaca", split="train")

# Load with streaming (for large datasets)
dataset = load_dataset("c4", "en", streaming=True)
```

### From Local Files

```python
from datasets import load_dataset

# From JSON
dataset = load_dataset("json", data_files="data.json")

# From CSV
dataset = load_dataset("csv", data_files="data.csv")

# From multiple files
dataset = load_dataset("json", data_files={
    "train": ["train1.json", "train2.json"],
    "validation": "val.json"
})

# From directory
dataset = load_dataset("json", data_dir="./data")
```

### Creating Custom Datasets

```python
from datasets import Dataset, DatasetDict
import pandas as pd

# From pandas DataFrame
df = pd.DataFrame({
    "instruction": ["Question 1", "Question 2"],
    "output": ["Answer 1", "Answer 2"]
})
dataset = Dataset.from_pandas(df)

# From Python dict
data = {
    "instruction": ["Question 1", "Question 2"],
    "output": ["Answer 1", "Answer 2"]
}
dataset = Dataset.from_dict(data)

# Create train/val split
dataset_dict = DatasetDict({
    "train": dataset,
    "validation": dataset.select(range(100))
})
```

## Data Preprocessing

### Formatting for Chat Models

```python
def format_chat_template(example, tokenizer):
    """Format data for chat models (Llama-2-chat, etc.)"""
    messages = [
        {"role": "user", "content": example["instruction"]},
        {"role": "assistant", "content": example["output"]}
    ]
    
    # Apply chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )
    
    return {"text": text}

# Apply to dataset
dataset = dataset.map(
    lambda x: format_chat_template(x, tokenizer),
    remove_columns=dataset.column_names
)
```

### Instruction Formatting

```python
def format_instruction(example):
    """Format instruction-input-output"""
    if example.get("input", "").strip():
        prompt = f"""### Instruction:
{example['instruction']}

### Input:
{example['input']}

### Response:
{example['output']}"""
    else:
        prompt = f"""### Instruction:
{example['instruction']}

### Response:
{example['output']}"""
    
    return {"text": prompt}

dataset = dataset.map(format_instruction)
```

### Alpaca Format

```python
ALPACA_PROMPT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}"""

def format_alpaca(example):
    return {
        "text": ALPACA_PROMPT.format(
            instruction=example["instruction"],
            input=example.get("input", ""),
            output=example["output"]
        )
    }

dataset = dataset.map(format_alpaca)
```

## Tokenization

### Basic Tokenization

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

def tokenize_function(examples):
    """Tokenize text"""
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=512,
        padding="max_length",
        return_tensors="pt"
    )

tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=dataset.column_names
)
```

### Advanced Tokenization

```python
def tokenize_with_labels(examples):
    """Tokenize with proper labels for causal LM"""
    # Tokenize
    model_inputs = tokenizer(
        examples["text"],
        max_length=512,
        truncation=True,
        padding=False  # Don't pad yet
    )
    
    # Labels are the same as input_ids for causal LM
    model_inputs["labels"] = model_inputs["input_ids"].copy()
    
    return model_inputs

tokenized_dataset = dataset.map(
    tokenize_with_labels,
    batched=True,
    remove_columns=dataset.column_names
)
```

### Packing Sequences

```python
def pack_sequences(examples, max_length=2048):
    """Pack multiple examples into single sequence"""
    concatenated = {k: sum(examples[k], []) for k in examples.keys()}
    
    total_length = len(concatenated["input_ids"])
    
    # Drop remainder
    total_length = (total_length // max_length) * max_length
    
    # Split into chunks
    result = {
        k: [t[i:i + max_length] for i in range(0, total_length, max_length)]
        for k, t in concatenated.items()
    }
    
    result["labels"] = result["input_ids"].copy()
    return result

packed_dataset = tokenized_dataset.map(
    pack_sequences,
    batched=True
)
```

## Data Quality

### Filtering

```python
# Filter by length
def filter_length(example):
    return len(example["text"].split()) >= 10

dataset = dataset.filter(filter_length)

# Filter by content
def filter_quality(example):
    text = example["text"].lower()
    # Remove if contains certain keywords
    bad_keywords = ["error", "test", "debug"]
    return not any(kw in text for kw in bad_keywords)

dataset = dataset.filter(filter_quality)

# Filter by language (using langdetect)
from langdetect import detect

def is_english(example):
    try:
        return detect(example["text"]) == "en"
    except:
        return False

dataset = dataset.filter(is_english)
```

### Deduplication

```python
from datasets import Dataset

def deduplicate(dataset):
    """Remove duplicate examples"""
    seen = set()
    unique_indices = []
    
    for i, example in enumerate(dataset):
        text = example["text"]
        if text not in seen:
            seen.add(text)
            unique_indices.append(i)
    
    return dataset.select(unique_indices)

dataset = deduplicate(dataset)
```

### Data Validation

```python
def validate_dataset(dataset):
    """Validate dataset quality"""
    issues = []
    
    for i, example in enumerate(dataset):
        # Check for empty fields
        if not example.get("text", "").strip():
            issues.append(f"Empty text at index {i}")
        
        # Check length
        if len(example["text"]) < 10:
            issues.append(f"Too short at index {i}")
        
        # Check encoding
        try:
            example["text"].encode("utf-8")
        except UnicodeEncodeError:
            issues.append(f"Encoding error at index {i}")
    
    if issues:
        print(f"Found {len(issues)} issues:")
        for issue in issues[:10]:  # Show first 10
            print(f"  - {issue}")
    else:
        print("✓ No issues found")
    
    return len(issues) == 0

validate_dataset(dataset)
```

## Splits and Sampling

### Train/Val Split

```python
# Random split
dataset = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = dataset["train"]
val_dataset = dataset["test"]

# Stratified split (by category)
from sklearn.model_selection import train_test_split

indices = list(range(len(dataset)))
categories = [ex["category"] for ex in dataset]

train_idx, val_idx = train_test_split(
    indices,
    test_size=0.1,
    stratify=categories,
    random_state=42
)

train_dataset = dataset.select(train_idx)
val_dataset = dataset.select(val_idx)
```

### Sampling Strategies

```python
# Random sampling
small_dataset = dataset.shuffle(seed=42).select(range(1000))

# Stratified sampling
from collections import Counter

def stratified_sample(dataset, n_samples, category_key="category"):
    """Sample proportionally from each category"""
    # Count categories
    categories = [ex[category_key] for ex in dataset]
    category_counts = Counter(categories)
    
    # Calculate samples per category
    samples_per_cat = {
        cat: int(count / len(dataset) * n_samples)
        for cat, count in category_counts.items()
    }
    
    # Sample from each category
    sampled_indices = []
    for cat, n in samples_per_cat.items():
        cat_indices = [i for i, c in enumerate(categories) if c == cat]
        sampled_indices.extend(random.sample(cat_indices, min(n, len(cat_indices))))
    
    return dataset.select(sampled_indices)

sampled = stratified_sample(dataset, 1000)
```

## Data Augmentation

### Back Translation

```python
from transformers import MarianMTModel, MarianTokenizer

def back_translate(text, src_lang="en", tgt_lang="de"):
    """Augment by translating to another language and back"""
    # Load models
    forward_model_name = f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"
    backward_model_name = f"Helsinki-NLP/opus-mt-{tgt_lang}-{src_lang}"
    
    forward_tokenizer = MarianTokenizer.from_pretrained(forward_model_name)
    forward_model = MarianMTModel.from_pretrained(forward_model_name)
    
    backward_tokenizer = MarianTokenizer.from_pretrained(backward_model_name)
    backward_model = MarianMTModel.from_pretrained(backward_model_name)
    
    # Forward translation
    inputs = forward_tokenizer(text, return_tensors="pt", padding=True)
    translated = forward_model.generate(**inputs)
    intermediate = forward_tokenizer.decode(translated[0], skip_special_tokens=True)
    
    # Back translation
    inputs = backward_tokenizer(intermediate, return_tensors="pt", padding=True)
    translated = backward_model.generate(**inputs)
    result = backward_tokenizer.decode(translated[0], skip_special_tokens=True)
    
    return result
```

### Synonym Replacement

```python
import random
import nltk
from nltk.corpus import wordnet

def get_synonyms(word):
    """Get synonyms for a word"""
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name().replace("_", " "))
    return list(synonyms)

def synonym_replacement(text, n=3):
    """Replace n random words with synonyms"""
    words = text.split()
    random_word_list = list(set([word for word in words if word.isalnum()]))
    random.shuffle(random_word_list)
    
    num_replaced = 0
    for random_word in random_word_list:
        synonyms = get_synonyms(random_word)
        if len(synonyms) >= 1:
            synonym = random.choice(synonyms)
            words = [synonym if word == random_word else word for word in words]
            num_replaced += 1
        if num_replaced >= n:
            break
    
    return " ".join(words)
```

## Saving and Loading

### Save to Disk

```python
# Save dataset
dataset.save_to_disk("./processed_dataset")

# Save specific format
dataset.to_json("dataset.json")
dataset.to_csv("dataset.csv")
dataset.to_parquet("dataset.parquet")
```

### Load from Disk

```python
from datasets import load_from_disk

# Load saved dataset
dataset = load_from_disk("./processed_dataset")

# Load with specific format
from datasets import load_dataset
dataset = load_dataset("json", data_files="dataset.json")
```

### Push to Hub

```python
from huggingface_hub import login

# Login
login(token="your_token")

# Push dataset
dataset.push_to_hub("username/dataset-name")

# Load from hub
from datasets import load_dataset
dataset = load_dataset("username/dataset-name")
```

## Performance Optimization

### Parallel Processing

```python
# Use multiple cores
dataset = dataset.map(
    tokenize_function,
    batched=True,
    num_proc=8,  # Use 8 cores
    remove_columns=dataset.column_names
)
```

### Caching

```python
# Enable caching (default)
dataset = dataset.map(
    tokenize_function,
    batched=True,
    load_from_cache_file=True  # Use cached results if available
)

# Disable caching
dataset = dataset.map(
    tokenize_function,
    load_from_cache_file=False
)
```

## Complete Example

```python
from datasets import load_dataset
from transformers import AutoTokenizer

# Load dataset
dataset = load_dataset("tatsu-lab/alpaca")

# Format
def format_example(example):
    return {
        "text": f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"
    }

dataset = dataset.map(format_example)

# Filter quality
dataset = dataset.filter(lambda x: len(x["text"].split()) >= 10)

# Split
dataset = dataset["train"].train_test_split(test_size=0.1, seed=42)

# Tokenize
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer.pad_token = tokenizer.eos_token

def tokenize(examples):
    result = tokenizer(
        examples["text"],
        truncation=True,
        max_length=512,
        padding="max_length"
    )
    result["labels"] = result["input_ids"].copy()
    return result

tokenized = dataset.map(
    tokenize,
    batched=True,
    num_proc=8,
    remove_columns=dataset["train"].column_names
)

# Save
tokenized.save_to_disk("./processed_alpaca")

print(f"Train examples: {len(tokenized['train'])}")
print(f"Val examples: {len(tokenized['test'])}")
```

## References

- [Datasets Documentation](https://huggingface.co/docs/datasets/)
- [Tokenizers Documentation](https://huggingface.co/docs/tokenizers/)
- [Data Preparation Best Practices](https://huggingface.co/docs/transformers/tasks/language_modeling)

