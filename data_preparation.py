"""
Data preparation script for Swahili GPT-2 distillation.
This script downloads and processes Swahili text data for model training.
"""

import os
import argparse
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
import nltk
from tqdm import tqdm
import random

# Download NLTK resources
nltk.download('punkt')

def parse_args():
    parser = argparse.ArgumentParser(description="Prepare Swahili dataset for GPT-2 distillation")
    parser.add_argument("--output_dir", type=str, default="./data", help="Directory to save the processed data")
    parser.add_argument("--use_custom_data", action="store_true", help="Use custom data instead of HF datasets")
    parser.add_argument("--custom_data_path", type=str, default=None, help="Path to custom data file")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to use")
    parser.add_argument("--tokenizer", type=str, default="gpt2", help="Tokenizer to use for preprocessing")
    parser.add_argument("--val_split", type=float, default=0.1, help="Validation split ratio")
    return parser.parse_args()

def load_swahili_datasets():
    """Load Swahili datasets from Hugging Face."""
    print("Loading Swahili datasets from Hugging Face...")
    
    # Load multiple Swahili datasets
    datasets = []
    
    # KINNEWS and KIRNEWS dataset (Kinyarwanda and Kirundi, but contains some Swahili)
    try:
        news_dataset = load_dataset("castorini/kinnews-kirnews")
        swahili_texts = [item for item in news_dataset["train"]["text"] if detect_swahili(item)]
        if swahili_texts:
            datasets.extend(swahili_texts)
            print(f"Added {len(swahili_texts)} texts from KINNEWS-KIRNEWS")
    except Exception as e:
        print(f"Error loading KINNEWS-KIRNEWS: {e}")
    
    # MasakhaNER dataset (contains Swahili)
    try:
        ner_dataset = load_dataset("masakhane/masakhaner")
        swahili_subset = ner_dataset.filter(lambda example: example["lang"] == "swa")
        swahili_texts = [" ".join(item["tokens"]) for item in swahili_subset["train"]]
        datasets.extend(swahili_texts)
        print(f"Added {len(swahili_texts)} texts from MasakhaNER")
    except Exception as e:
        print(f"Error loading MasakhaNER: {e}")
    
    # XLSUM dataset (contains Swahili news)
    try:
        xlsum_dataset = load_dataset("csebuetnlp/xlsum")
        swahili_subset = xlsum_dataset.filter(lambda example: example["language"] == "swahili")
        swahili_texts = []
        for split in ["train", "validation", "test"]:
            if split in swahili_subset:
                for item in swahili_subset[split]:
                    swahili_texts.append(item["text"] + " " + item["summary"])
        datasets.extend(swahili_texts)
        print(f"Added {len(swahili_texts)} texts from XLSUM")
    except Exception as e:
        print(f"Error loading XLSUM: {e}")
    
    # AfriSpeech dataset (contains Swahili transcripts)
    try:
        speech_dataset = load_dataset("google/afrispeech-swahili")
        swahili_texts = [item["text"] for item in speech_dataset["train"]]
        datasets.extend(swahili_texts)
        print(f"Added {len(swahili_texts)} texts from AfriSpeech")
    except Exception as e:
        print(f"Error loading AfriSpeech: {e}")
    
    return datasets

def detect_swahili(text):
    """Simple heuristic to detect if text is likely Swahili."""
    # Common Swahili words
    swahili_words = ["na", "kwa", "ya", "wa", "ni", "katika", "za", "la", "kuwa", "kama"]
    words = text.lower().split()
    count = sum(1 for word in words if word in swahili_words)
    return count / max(len(words), 1) > 0.1  # If more than 10% are common Swahili words

def load_custom_data(file_path):
    """Load custom Swahili data from a file."""
    print(f"Loading custom data from {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    return [line.strip() for line in lines if line.strip()]

def preprocess_and_save(texts, output_dir, tokenizer_name, val_split, max_samples=None):
    """Preprocess texts and save them to disk."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Limit the number of samples if specified
    if max_samples and len(texts) > max_samples:
        print(f"Limiting to {max_samples} samples")
        random.shuffle(texts)
        texts = texts[:max_samples]
    
    print(f"Processing {len(texts)} text samples...")
    
    # Split into train and validation
    val_size = int(len(texts) * val_split)
    train_texts = texts[val_size:]
    val_texts = texts[:val_size]
    
    print(f"Train set: {len(train_texts)} samples")
    print(f"Validation set: {len(val_texts)} samples")
    
    # Save raw text files
    with open(os.path.join(output_dir, "train.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(train_texts))
    
    with open(os.path.join(output_dir, "valid.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(val_texts))
    
    print(f"Data saved to {output_dir}")
    
    # Calculate and print statistics
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    train_tokens = sum(len(tokenizer.encode(text)) for text in tqdm(train_texts, desc="Counting train tokens"))
    val_tokens = sum(len(tokenizer.encode(text)) for text in tqdm(val_texts, desc="Counting validation tokens"))
    
    print(f"Total train tokens: {train_tokens}")
    print(f"Total validation tokens: {val_tokens}")
    
    # Create dataset info file
    with open(os.path.join(output_dir, "dataset_info.txt"), "w", encoding="utf-8") as f:
        f.write(f"Total samples: {len(texts)}\n")
        f.write(f"Train samples: {len(train_texts)}\n")
        f.write(f"Validation samples: {len(val_texts)}\n")
        f.write(f"Train tokens: {train_tokens}\n")
        f.write(f"Validation tokens: {val_tokens}\n")
        f.write(f"Tokenizer: {tokenizer_name}\n")

def main():
    args = parse_args()
    
    if args.use_custom_data and args.custom_data_path:
        texts = load_custom_data(args.custom_data_path)
    else:
        texts = load_swahili_datasets()
    
    if not texts:
        print("No texts were loaded. Please check your data sources.")
        return
    
    print(f"Loaded {len(texts)} text samples")
    
    preprocess_and_save(
        texts=texts,
        output_dir=args.output_dir,
        tokenizer_name=args.tokenizer,
        val_split=args.val_split,
        max_samples=args.max_samples
    )

if __name__ == "__main__":
    main()
