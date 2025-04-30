"""
GPT-2 distillation script for Swahili language modeling.
This script distills a GPT-2 model on Swahili text data.
"""

import os
import argparse
import torch
import numpy as np
from transformers import (
    GPT2LMHeadModel, 
    GPT2Config, 
    GPT2Tokenizer,
    TextDataset,
    DataCollatorForLanguageModeling,
    Trainer, 
    TrainingArguments,
    EarlyStoppingCallback
)
from datasets import load_dataset
from torch.nn import functional as F
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Distill GPT-2 for Swahili language modeling")
    
    # Data arguments
    parser.add_argument("--train_file", type=str, default="./data/train.txt", help="Path to training data file")
    parser.add_argument("--val_file", type=str, default="./data/valid.txt", help="Path to validation data file")
    
    # Model arguments
    parser.add_argument("--teacher_model", type=str, default="gpt2", help="Teacher model name or path")
    parser.add_argument("--output_dir", type=str, default="./models/distilled-gpt2-swahili", help="Output directory for the distilled model")
    parser.add_argument("--cache_dir", type=str, default="./cache", help="Cache directory for models and datasets")
    
    # Distillation arguments
    parser.add_argument("--temperature", type=float, default=2.0, help="Temperature for distillation")
    parser.add_argument("--alpha_ce", type=float, default=0.5, help="Weight for cross-entropy loss")
    parser.add_argument("--alpha_kl", type=float, default=0.5, help="Weight for KL divergence loss")
    
    # Training arguments
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=2, help="Batch size for training")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=2, help="Batch size for evaluation")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--max_seq_length", type=int, default=128, help="Maximum sequence length")
    parser.add_argument("--logging_steps", type=int, default=500, help="Logging steps")
    parser.add_argument("--eval_steps", type=int, default=500, help="Evaluation steps")
    parser.add_argument("--save_steps", type=int, default=1000, help="Save steps")
    parser.add_argument("--warmup_steps", type=int, default=1000, help="Warmup steps")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps")
    
    # Distilled model config
    parser.add_argument("--n_layers", type=int, default=6, help="Number of layers in the distilled model")
    parser.add_argument("--n_heads", type=int, default=8, help="Number of attention heads in the distilled model")
    parser.add_argument("--d_model", type=int, default=512, help="Hidden dimension in the distilled model")
    
    return parser.parse_args()

def create_distilled_config(teacher_model, args):
    """Create a configuration for the distilled model."""
    teacher_config = teacher_model.config
    
    # Create a smaller configuration for the student model
    student_config = GPT2Config(
        vocab_size=teacher_config.vocab_size,
        n_positions=teacher_config.n_positions,
        n_ctx=teacher_config.n_ctx,
        n_embd=args.d_model,
        n_layer=args.n_layers,
        n_head=args.n_heads,
        activation_function=teacher_config.activation_function,
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
        layer_norm_epsilon=teacher_config.layer_norm_epsilon,
        initializer_range=teacher_config.initializer_range,
        bos_token_id=teacher_config.bos_token_id,
        eos_token_id=teacher_config.eos_token_id,
    )
    
    return student_config

class DistillationTrainer(Trainer):
    def __init__(self, teacher_model=None, alpha_ce=0.5, alpha_kl=0.5, temperature=2.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.alpha_ce = alpha_ce
        self.alpha_kl = alpha_kl
        self.temperature = temperature
        
        # Put teacher model in eval mode
        if self.teacher_model is not None:
            self.teacher_model.eval()
    
    def compute_loss(self, model, inputs, return_outputs=False):
        if self.teacher_model is None:
            # If no teacher, use standard language modeling loss
            return super().compute_loss(model, inputs, return_outputs)
        
        # Get student outputs
        outputs_student = model(**inputs)
        student_logits = outputs_student.logits
        
        # Standard cross-entropy loss
        loss_ce = outputs_student.loss
        
        # Get teacher outputs (no gradients needed)
        with torch.no_grad():
            outputs_teacher = self.teacher_model(**inputs)
            teacher_logits = outputs_teacher.logits
        
        # KL divergence loss
        loss_kl = F.kl_div(
            F.log_softmax(student_logits / self.temperature, dim=-1),
            F.softmax(teacher_logits / self.temperature, dim=-1),
            reduction="batchmean",
        ) * (self.temperature ** 2)
        
        # Combined loss
        loss = self.alpha_ce * loss_ce + self.alpha_kl * loss_kl
        
        return (loss, outputs_student) if return_outputs else loss

def load_text_dataset(file_path, tokenizer, block_size):
    """Load a text dataset from file."""
    dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=block_size,
    )
    return dataset

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(args.teacher_model, cache_dir=args.cache_dir)
    
    # Load teacher model
    print(f"Loading teacher model: {args.teacher_model}")
    teacher_model = GPT2LMHeadModel.from_pretrained(args.teacher_model, cache_dir=args.cache_dir)
    teacher_model.to(device)
    
    # Create student model config
    student_config = create_distilled_config(teacher_model, args)
    
    # Initialize student model
    print("Initializing student model")
    student_model = GPT2LMHeadModel(config=student_config)
    
    # Load datasets
    print("Loading datasets")
    train_dataset = load_text_dataset(args.train_file, tokenizer, args.max_seq_length)
    eval_dataset = load_text_dataset(args.val_file, tokenizer, args.max_seq_length)
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_steps=args.logging_steps,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        warmup_steps=args.warmup_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=4,
        report_to="tensorboard",
    )
    
    # Initialize trainer
    trainer = DistillationTrainer(
        teacher_model=teacher_model,
        alpha_ce=args.alpha_ce,
        alpha_kl=args.alpha_kl,
        temperature=args.temperature,
        model=student_model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )
    
    # Train the model
    print("Starting training")
    trainer.train()
    
    # Save the final model
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    print(f"Model saved to {args.output_dir}")

if __name__ == "__main__":
    main()
