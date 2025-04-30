# Swahili GPT-2 Distillation on Google Colab

This guide will help you run the GPT-2 distillation process on Google Colab with T4 GPU acceleration, using your existing Swahili dataset.

## Step 1: Create a new Colab notebook

1. Go to [Google Colab](https://colab.research.google.com/)
2. Create a new notebook
3. Rename it to "Swahili_GPT2_Distillation"

## Step 2: Set up the GPU

1. Click on "Runtime" > "Change runtime type"
2. Select "GPU" from the hardware accelerator dropdown
3. Click "Save"

## Step 3: Mount Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

## Step 4: Install dependencies

```python
!pip install transformers==4.30.2 datasets==2.13.1 torch==2.0.1 accelerate==0.20.3 tensorboard==2.13.0 scikit-learn==1.3.0 tqdm==4.65.0 nltk==3.8.1 sentencepiece==0.1.99 evaluate==0.4.0
```

## Step 5: Upload the distillation script

1. Upload the `colab_distillation.py` script to your Google Drive
2. Or create it directly in Colab with this cell:

```python
%%writefile colab_distillation.py
# Paste the entire content of the colab_distillation.py file here
```

## Step 6: Run the distillation process

```python
!python colab_distillation.py \
  --train_file "/content/drive/MyDrive/msingi1/data/train.txt" \
  --val_file "/content/drive/MyDrive/msingi1/data/valid.txt" \
  --output_dir "/content/drive/MyDrive/msingi1/models/distilled-gpt2-swahili" \
  --num_train_epochs 3 \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --n_layers 6 \
  --n_heads 8 \
  --d_model 512 \
  --generate_samples
```

## Step 7: Monitor training with TensorBoard

```python
%load_ext tensorboard
%tensorboard --logdir "/content/drive/MyDrive/msingi1/models/distilled-gpt2-swahili/logs"
```

## Step 8: Evaluate the trained model

```python
!python colab_distillation.py \
  --val_file "/content/drive/MyDrive/msingi1/data/valid.txt" \
  --output_dir "/content/drive/MyDrive/msingi1/models/distilled-gpt2-swahili" \
  --evaluate_only \
  --generate_samples
```

## Step 9: Test the model with custom prompts

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load the trained model
model_path = "/content/drive/MyDrive/msingi1/models/distilled-gpt2-swahili"
model = GPT2LMHeadModel.from_pretrained(model_path)
tokenizer = GPT2Tokenizer.from_pretrained(model_path)

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Function to generate text
def generate_text(prompt, max_length=100):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    # Generate text
    output = model.generate(
        input_ids,
        max_length=max_length,
        temperature=0.7,
        top_k=50,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        num_return_sequences=1,
    )
    
    # Decode and return
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Test with some prompts
prompts = [
    "Leo ni siku nzuri",  # Today is a good day
    "Habari ya asubuhi",  # Good morning
    "Ninataka kusafiri",  # I want to travel
    "Jambo la muhimu",    # An important matter
    "Katika nchi ya",     # In the country of
]

for prompt in prompts:
    generated = generate_text(prompt)
    print(f"Prompt: {prompt}")
    print(f"Generated: {generated}")
    print("-" * 50)
```

## Step 10: Push the model to Hugging Face (optional)

```python
from huggingface_hub import notebook_login
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Login to Hugging Face
notebook_login()

# Load your model
model_path = "/content/drive/MyDrive/msingi1/models/distilled-gpt2-swahili"
model = GPT2LMHeadModel.from_pretrained(model_path)
tokenizer = GPT2Tokenizer.from_pretrained(model_path)

# Push to Hub
model_name = "swahili-gpt2-distilled"
model.push_to_hub(model_name)
tokenizer.push_to_hub(model_name)

print(f"Model pushed to: https://huggingface.co/YOUR_USERNAME/{model_name}")
```

## Customization Options

You can customize various aspects of the distillation process by adjusting the parameters:

### Model Size
- `--n_layers`: Number of transformer layers (default: 6)
- `--n_heads`: Number of attention heads (default: 8)
- `--d_model`: Hidden dimension size (default: 512)

### Training Parameters
- `--num_train_epochs`: Number of training epochs (default: 3)
- `--per_device_train_batch_size`: Batch size for training (default: 4)
- `--learning_rate`: Learning rate (default: 5e-5)
- `--max_seq_length`: Maximum sequence length (default: 128)

### Distillation Parameters
- `--temperature`: Temperature for distillation (default: 2.0)
- `--alpha_ce`: Weight for cross-entropy loss (default: 0.5)
- `--alpha_kl`: Weight for KL divergence loss (default: 0.5)

For example, to create an even smaller model:
```python
!python colab_distillation.py \
  --train_file "/content/drive/MyDrive/msingi1/data/train.txt" \
  --val_file "/content/drive/MyDrive/msingi1/data/valid.txt" \
  --output_dir "/content/drive/MyDrive/msingi1/models/distilled-gpt2-swahili-tiny" \
  --n_layers 4 \
  --n_heads 6 \
  --d_model 384
```
