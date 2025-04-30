"""
Complete pipeline script for Swahili GPT-2 distillation.
This script runs the entire process:
1. Download Swahili data
2. Prepare the dataset
3. Distill the GPT-2 model
4. Evaluate the distilled model
"""

import os
import argparse
import subprocess
import time
import sys
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(description="Run the complete Swahili GPT-2 distillation pipeline")
    
    # Data arguments
    parser.add_argument("--data_dir", type=str, default="./data", help="Directory for data")
    parser.add_argument("--min_words", type=int, default=1000000, help="Minimum number of words to collect")
    parser.add_argument("--clean_data", action="store_true", help="Clean data directory before downloading")
    
    # Model arguments
    parser.add_argument("--teacher_model", type=str, default="gpt2", help="Teacher model name or path")
    parser.add_argument("--output_dir", type=str, default="./models/distilled-gpt2-swahili", help="Output directory for the distilled model")
    
    # Distillation arguments
    parser.add_argument("--temperature", type=float, default=2.0, help="Temperature for distillation")
    parser.add_argument("--alpha_ce", type=float, default=0.5, help="Weight for cross-entropy loss")
    parser.add_argument("--alpha_kl", type=float, default=0.5, help="Weight for KL divergence loss")
    
    # Training arguments
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--max_seq_length", type=int, default=128, help="Maximum sequence length")
    
    # Distilled model config
    parser.add_argument("--n_layers", type=int, default=6, help="Number of layers in the distilled model")
    parser.add_argument("--n_heads", type=int, default=8, help="Number of attention heads in the distilled model")
    parser.add_argument("--d_model", type=int, default=512, help="Hidden dimension in the distilled model")
    
    # Skip steps
    parser.add_argument("--skip_data_download", action="store_true", help="Skip data download step")
    parser.add_argument("--skip_training", action="store_true", help="Skip model training step")
    parser.add_argument("--skip_evaluation", action="store_true", help="Skip model evaluation step")
    
    return parser.parse_args()

def run_command(command, description):
    """Run a shell command and print its output."""
    print(f"\n{'=' * 80}")
    print(f"STEP: {description}")
    print(f"COMMAND: {' '.join(command)}")
    print(f"{'=' * 80}\n")
    
    start_time = time.time()
    
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True
    )
    
    # Print output in real-time
    for line in process.stdout:
        print(line, end='')
    
    process.wait()
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"\n{'=' * 80}")
    print(f"COMPLETED: {description}")
    print(f"DURATION: {duration:.2f} seconds ({duration/60:.2f} minutes)")
    print(f"EXIT CODE: {process.returncode}")
    print(f"{'=' * 80}\n")
    
    if process.returncode != 0:
        print(f"ERROR: Command failed with exit code {process.returncode}")
        return False
    
    return True

def create_directories(args):
    """Create necessary directories."""
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "logs"), exist_ok=True)

def download_data(args):
    """Download and prepare Swahili data."""
    command = [
        sys.executable, "download_swahili_data.py",
        "--output_dir", args.data_dir,
        "--min_words", str(args.min_words)
    ]
    
    if args.clean_data:
        command.append("--clean")
    
    return run_command(command, "Downloading Swahili data")

def train_model(args):
    """Train and distill the GPT-2 model."""
    command = [
        sys.executable, "distill.py",
        "--train_file", os.path.join(args.data_dir, "train.txt"),
        "--val_file", os.path.join(args.data_dir, "valid.txt"),
        "--teacher_model", args.teacher_model,
        "--output_dir", args.output_dir,
        "--temperature", str(args.temperature),
        "--alpha_ce", str(args.alpha_ce),
        "--alpha_kl", str(args.alpha_kl),
        "--num_train_epochs", str(args.num_train_epochs),
        "--per_device_train_batch_size", str(args.batch_size),
        "--per_device_eval_batch_size", str(args.batch_size),
        "--learning_rate", str(args.learning_rate),
        "--max_seq_length", str(args.max_seq_length),
        "--n_layers", str(args.n_layers),
        "--n_heads", str(args.n_heads),
        "--d_model", str(args.d_model)
    ]
    
    return run_command(command, "Training and distilling GPT-2 model")

def evaluate_model(args):
    """Evaluate the distilled model."""
    command = [
        sys.executable, "evaluate.py",
        "--model_path", args.output_dir,
        "--test_file", os.path.join(args.data_dir, "valid.txt"),
        "--num_samples", "5"
    ]
    
    return run_command(command, "Evaluating distilled model")

def generate_sample_text(args):
    """Generate sample text with the distilled model."""
    prompts = [
        "Habari ya leo",
        "Ninafurahi kukutana nawe",
        "Kiswahili ni lugha nzuri"
    ]
    
    success = True
    for prompt in prompts:
        command = [
            sys.executable, "generate.py",
            "--model_path", args.output_dir,
            "--prompt", prompt,
            "--max_length", "100",
            "--temperature", "0.7"
        ]
        
        description = f"Generating text with prompt: '{prompt}'"
        if not run_command(command, description):
            success = False
    
    return success

def save_pipeline_info(args):
    """Save information about the pipeline run."""
    info_file = os.path.join(args.output_dir, "pipeline_info.txt")
    
    with open(info_file, 'w', encoding='utf-8') as f:
        f.write(f"Pipeline run at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("Pipeline arguments:\n")
        for arg, value in vars(args).items():
            f.write(f"  {arg}: {value}\n")

def main():
    args = parse_args()
    
    # Create directories
    create_directories(args)
    
    # Save pipeline info
    save_pipeline_info(args)
    
    # Print pipeline configuration
    print("\nSwahili GPT-2 Distillation Pipeline")
    print("=" * 50)
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Teacher model: {args.teacher_model}")
    print(f"Distilled model config: {args.n_layers} layers, {args.n_heads} heads, {args.d_model} hidden dim")
    print(f"Training: {args.num_train_epochs} epochs, batch size {args.batch_size}, lr {args.learning_rate}")
    print("=" * 50)
    
    # Run pipeline steps
    pipeline_start_time = time.time()
    
    if not args.skip_data_download:
        if not download_data(args):
            print("Data download failed. Exiting pipeline.")
            return
    else:
        print("Skipping data download step.")
    
    if not args.skip_training:
        if not train_model(args):
            print("Model training failed. Exiting pipeline.")
            return
    else:
        print("Skipping model training step.")
    
    if not args.skip_evaluation:
        if not evaluate_model(args):
            print("Model evaluation failed, but continuing pipeline.")
        
        if not generate_sample_text(args):
            print("Text generation failed, but continuing pipeline.")
    else:
        print("Skipping model evaluation step.")
    
    pipeline_end_time = time.time()
    pipeline_duration = pipeline_end_time - pipeline_start_time
    
    print("\nPipeline Completed!")
    print("=" * 50)
    print(f"Total duration: {pipeline_duration:.2f} seconds ({pipeline_duration/60:.2f} minutes)")
    print(f"Distilled model saved to: {args.output_dir}")
    print("=" * 50)

if __name__ == "__main__":
    main()
