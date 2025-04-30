"""
Evaluation script for the distilled Swahili GPT-2 model.
This script evaluates the model's perplexity and generates sample texts.
"""

import argparse
import torch
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from tqdm import tqdm
import math
import random

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate distilled GPT-2 Swahili model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the distilled model")
    parser.add_argument("--test_file", type=str, default="./data/valid.txt", help="Path to test data file")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of text samples to generate")
    parser.add_argument("--max_length", type=int, default=100, help="Maximum length of generated text")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling parameter")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p (nucleus) sampling parameter")
    return parser.parse_args()

def calculate_perplexity(model, tokenizer, test_file, device):
    """Calculate perplexity on test data."""
    print("Calculating perplexity...")
    
    # Load test data
    with open(test_file, 'r', encoding='utf-8') as f:
        test_texts = [line.strip() for line in f if line.strip()]
    
    if not test_texts:
        print("No test data found.")
        return None
    
    # Sample a subset if there are too many texts
    if len(test_texts) > 100:
        random.shuffle(test_texts)
        test_texts = test_texts[:100]
    
    # Calculate perplexity
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for text in tqdm(test_texts, desc="Evaluating"):
            inputs = tokenizer(text, return_tensors="pt").to(device)
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss.item()
            
            # Count tokens
            num_tokens = inputs["input_ids"].numel()
            
            total_loss += loss * num_tokens
            total_tokens += num_tokens
    
    # Calculate perplexity
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    
    return perplexity

def generate_samples(model, tokenizer, num_samples, max_length, temperature, top_k, top_p):
    """Generate sample texts from the model."""
    print(f"Generating {num_samples} sample texts...")
    
    # Sample prompts
    prompts = [
        "Leo ni siku nzuri",  # Today is a good day
        "Habari ya asubuhi",  # Good morning
        "Ninataka kusafiri",  # I want to travel
        "Jambo la muhimu",    # An important matter
        "Katika nchi ya",     # In the country of
    ]
    
    # Use available prompts or generate from the beginning
    if num_samples <= len(prompts):
        selected_prompts = prompts[:num_samples]
    else:
        selected_prompts = prompts + [""] * (num_samples - len(prompts))
    
    samples = []
    
    for prompt in selected_prompts:
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        
        # Generate text
        output = model.generate(
            input_ids,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            num_return_sequences=1,
        )
        
        # Decode and add to samples
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        samples.append((prompt, generated_text))
    
    return samples

def main():
    args = parse_args()
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model and tokenizer
    print(f"Loading model from {args.model_path}")
    model = GPT2LMHeadModel.from_pretrained(args.model_path)
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_path)
    
    model.to(device)
    
    # Calculate perplexity
    perplexity = calculate_perplexity(model, tokenizer, args.test_file, device)
    if perplexity:
        print(f"Perplexity: {perplexity:.2f}")
    
    # Generate samples
    samples = generate_samples(
        model, 
        tokenizer, 
        args.num_samples, 
        args.max_length, 
        args.temperature, 
        args.top_k, 
        args.top_p
    )
    
    # Print samples
    print("\nGenerated Samples:")
    print("=" * 50)
    for i, (prompt, text) in enumerate(samples, 1):
        print(f"Sample {i}:")
        if prompt:
            print(f"Prompt: {prompt}")
        print(f"Generated: {text}")
        print("-" * 50)

if __name__ == "__main__":
    main()
