"""
Script to download and prepare a large Swahili dataset with over 1 million words.
This script downloads data from various sources including:
1. OPUS corpus (CCAligned, CCMatrix, etc.)
2. Leipzig Corpora Collection
3. Swahili news websites
4. Masakhane datasets
"""

import os
import requests
import zipfile
import gzip
import shutil
import re
import argparse
import random
from tqdm import tqdm
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import word_tokenize

# Download NLTK resources
nltk.download('punkt')

def parse_args():
    parser = argparse.ArgumentParser(description="Download and prepare Swahili dataset")
    parser.add_argument("--output_dir", type=str, default="./data", help="Directory to save the downloaded data")
    parser.add_argument("--min_words", type=int, default=1000000, help="Minimum number of words to collect")
    parser.add_argument("--clean", action="store_true", help="Clean the output directory before downloading")
    return parser.parse_args()

def download_file(url, output_path):
    """Download a file from URL to the specified path."""
    print(f"Downloading {url} to {output_path}")
    
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024
    
    with open(output_path, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=os.path.basename(output_path)) as pbar:
            for data in response.iter_content(block_size):
                f.write(data)
                pbar.update(len(data))

def extract_zip(zip_path, extract_to):
    """Extract a zip file to the specified directory."""
    print(f"Extracting {zip_path} to {extract_to}")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

def extract_gz(gz_path, extract_to):
    """Extract a gzip file to the specified directory."""
    print(f"Extracting {gz_path} to {extract_to}")
    with gzip.open(gz_path, 'rb') as f_in:
        with open(extract_to, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

def download_leipzig_corpus(output_dir):
    """Download Swahili corpus from Leipzig Corpora Collection."""
    print("Downloading Leipzig Swahili corpus...")
    
    # Create directory
    leipzig_dir = os.path.join(output_dir, "leipzig")
    os.makedirs(leipzig_dir, exist_ok=True)
    
    # URLs for Swahili corpus files
    urls = [
        "https://downloads.wortschatz-leipzig.de/corpora/swa_news_2011_300K.tar.gz",
        "https://downloads.wortschatz-leipzig.de/corpora/swa_newscrawl_2011_300K.tar.gz",
        "https://downloads.wortschatz-leipzig.de/corpora/swa_wikipedia_2016_300K.tar.gz"
    ]
    
    collected_texts = []
    
    for url in urls:
        filename = os.path.basename(url)
        output_path = os.path.join(leipzig_dir, filename)
        
        # Download file
        download_file(url, output_path)
        
        # Extract tar.gz file
        extract_dir = os.path.join(leipzig_dir, filename.replace(".tar.gz", ""))
        os.makedirs(extract_dir, exist_ok=True)
        
        # Use shutil to extract tar.gz
        shutil.unpack_archive(output_path, extract_dir)
        
        # Find and process sentence files
        for root, _, files in os.walk(extract_dir):
            for file in files:
                if file.endswith("_sentences.txt"):
                    file_path = os.path.join(root, file)
                    with open(file_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            parts = line.strip().split('\t')
                            if len(parts) >= 2:
                                collected_texts.append(parts[1])
    
    print(f"Collected {len(collected_texts)} sentences from Leipzig corpus")
    return collected_texts

def download_opus_corpus(output_dir):
    """Download Swahili data from OPUS corpus."""
    print("Downloading OPUS Swahili corpus...")
    
    # Create directory
    opus_dir = os.path.join(output_dir, "opus")
    os.makedirs(opus_dir, exist_ok=True)
    
    # URLs for Swahili corpus files from OPUS
    urls = [
        "https://object.pouta.csc.fi/OPUS-CCAligned/v1/moses/en-sw.txt.zip",
        "https://object.pouta.csc.fi/OPUS-CCMatrix/v1/moses/en-sw.txt.zip",
        "https://object.pouta.csc.fi/OPUS-GlobalVoices/v2018q4/moses/en-sw.txt.zip",
        "https://object.pouta.csc.fi/OPUS-Tatoeba/v2022-03-03/moses/en-sw.txt.zip",
        "https://object.pouta.csc.fi/OPUS-bible-uedin/v1/moses/en-sw.txt.zip"
    ]
    
    collected_texts = []
    
    for url in urls:
        filename = os.path.basename(url)
        output_path = os.path.join(opus_dir, filename)
        
        # Download file
        download_file(url, output_path)
        
        # Extract zip file
        extract_dir = os.path.join(opus_dir, filename.replace(".zip", ""))
        os.makedirs(extract_dir, exist_ok=True)
        extract_zip(output_path, extract_dir)
        
        # Find and process Swahili text files
        for root, _, files in os.walk(extract_dir):
            for file in files:
                if file.endswith(".sw") or "sw." in file:
                    file_path = os.path.join(root, file)
                    with open(file_path, 'r', encoding='utf-8') as f:
                        collected_texts.extend([line.strip() for line in f if line.strip()])
    
    print(f"Collected {len(collected_texts)} sentences from OPUS corpus")
    return collected_texts

def download_masakhane_data(output_dir):
    """Download Swahili data from Masakhane project."""
    print("Downloading Masakhane Swahili data...")
    
    # Create directory
    masakhane_dir = os.path.join(output_dir, "masakhane")
    os.makedirs(masakhane_dir, exist_ok=True)
    
    # URLs for Masakhane datasets with Swahili
    urls = [
        "https://raw.githubusercontent.com/masakhane-io/masakhane-ner/main/data/swahili/train.txt",
        "https://raw.githubusercontent.com/masakhane-io/masakhane-ner/main/data/swahili/dev.txt",
        "https://raw.githubusercontent.com/masakhane-io/masakhane-ner/main/data/swahili/test.txt",
        "https://raw.githubusercontent.com/masakhane-io/masakhane-mt/main/benchmarks/sw-en/data/train.sw",
        "https://raw.githubusercontent.com/masakhane-io/masakhane-mt/main/benchmarks/sw-en/data/dev.sw",
        "https://raw.githubusercontent.com/masakhane-io/masakhane-mt/main/benchmarks/sw-en/data/test.sw"
    ]
    
    collected_texts = []
    
    for url in urls:
        filename = os.path.basename(url)
        output_path = os.path.join(masakhane_dir, filename)
        
        # Download file
        download_file(url, output_path)
        
        # Process file based on format
        with open(output_path, 'r', encoding='utf-8') as f:
            if filename.endswith(".txt"):  # NER data
                current_sentence = []
                for line in f:
                    line = line.strip()
                    if not line and current_sentence:
                        collected_texts.append(" ".join(current_sentence))
                        current_sentence = []
                    elif line and not line.startswith("-DOCSTART-"):
                        parts = line.split()
                        if parts:
                            current_sentence.append(parts[0])
                if current_sentence:
                    collected_texts.append(" ".join(current_sentence))
            else:  # MT data
                collected_texts.extend([line.strip() for line in f if line.strip()])
    
    print(f"Collected {len(collected_texts)} sentences from Masakhane data")
    return collected_texts

def clean_text(text):
    """Clean and normalize text."""
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove lines that are too short or likely not Swahili
    if len(text) < 20 or not re.search(r'[aeiou]', text.lower()):
        return ""
    
    return text

def process_and_save_data(texts, output_dir, min_words):
    """Process, clean, and save the collected texts."""
    print("Processing and cleaning collected texts...")
    
    # Clean texts
    cleaned_texts = []
    for text in tqdm(texts, desc="Cleaning texts"):
        cleaned = clean_text(text)
        if cleaned:
            cleaned_texts.append(cleaned)
    
    # Shuffle texts
    random.shuffle(cleaned_texts)
    
    # Count words
    total_words = 0
    for text in cleaned_texts:
        total_words += len(word_tokenize(text))
    
    print(f"Total cleaned texts: {len(cleaned_texts)}")
    print(f"Total words: {total_words}")
    
    if total_words < min_words:
        print(f"Warning: Collected only {total_words} words, which is less than the requested {min_words} words")
    
    # Split into train and validation sets (90% train, 10% validation)
    val_size = max(1, int(len(cleaned_texts) * 0.1))
    train_texts = cleaned_texts[val_size:]
    val_texts = cleaned_texts[:val_size]
    
    # Save to files
    train_file = os.path.join(output_dir, "train.txt")
    val_file = os.path.join(output_dir, "valid.txt")
    
    with open(train_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(train_texts))
    
    with open(val_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(val_texts))
    
    # Create dataset info file
    info_file = os.path.join(output_dir, "dataset_info.txt")
    train_words = sum(len(word_tokenize(text)) for text in train_texts)
    val_words = sum(len(word_tokenize(text)) for text in val_texts)
    
    with open(info_file, 'w', encoding='utf-8') as f:
        f.write(f"Total texts: {len(cleaned_texts)}\n")
        f.write(f"Total words: {total_words}\n")
        f.write(f"Train texts: {len(train_texts)}\n")
        f.write(f"Train words: {train_words}\n")
        f.write(f"Validation texts: {len(val_texts)}\n")
        f.write(f"Validation words: {val_words}\n")
    
    print(f"Data saved to {output_dir}")
    print(f"Train file: {train_file} ({len(train_texts)} texts, {train_words} words)")
    print(f"Validation file: {val_file} ({len(val_texts)} texts, {val_words} words)")

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Clean output directory if requested
    if args.clean:
        print(f"Cleaning output directory: {args.output_dir}")
        for item in os.listdir(args.output_dir):
            item_path = os.path.join(args.output_dir, item)
            if os.path.isfile(item_path):
                os.remove(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
    
    # Download data from different sources
    all_texts = []
    
    # Leipzig Corpora Collection
    leipzig_texts = download_leipzig_corpus(args.output_dir)
    all_texts.extend(leipzig_texts)
    
    # OPUS corpus
    opus_texts = download_opus_corpus(args.output_dir)
    all_texts.extend(opus_texts)
    
    # Masakhane data
    masakhane_texts = download_masakhane_data(args.output_dir)
    all_texts.extend(masakhane_texts)
    
    # Process and save data
    process_and_save_data(all_texts, args.output_dir, args.min_words)

if __name__ == "__main__":
    main()
