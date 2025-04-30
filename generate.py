"""
Text generation script for the distilled Swahili GPT-2 model.
This script generates text from the model given a prompt.
"""

import argparse
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def parse_args():
    parser = argparse.ArgumentParser(description="Generate text with distilled GPT-2 Swahili model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the distilled model")
    parser.add_argument("--prompt", type=str, default="", help="Text prompt to start generation")
    parser.add_argument("--max_length", type=int, default=100, help="Maximum length of generated text")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling parameter")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p (nucleus) sampling parameter")
    parser.add_argument("--num_return_sequences", type=int, default=1, help="Number of sequences to generate")
    return parser.parse_args()

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
    model.eval()
    
    # Encode the prompt
    input_ids = tokenizer.encode(args.prompt, return_tensors="pt").to(device)
    
    # Generate text
    print(f"Generating text with prompt: '{args.prompt}'")
    output_sequences = model.generate(
        input_ids,
        max_length=args.max_length,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        num_return_sequences=args.num_return_sequences,
    )
    
    # Print generated text
    print("\nGenerated Text:")
    print("=" * 50)
    for i, sequence in enumerate(output_sequences):
        text = tokenizer.decode(sequence, skip_special_tokens=True)
        if args.num_return_sequences > 1:
            print(f"Sequence {i+1}:")
        print(text)
        print("-" * 50)

if __name__ == "__main__":
    main()
