# Swahili GPT-2 Distilled 
 
This repository hosts a distilled version of the GPT-2 model fine-tuned on a custom Swahili language corpus. The project aims to provide an efficient, smaller language model capable of generating Swahili text and performing other natural language processing (NLP) tasks. This model can be used in various applications including chatbots, content generation, and language modeling, especially in resource-constrained environments.

## Project Motivation
 
Swahili is spoken by over 100 million people, predominantly in East Africa. Despite its wide usage, Swahili has been underrepresented in state-of-the-art AI models, particularly in generative language modeling. The aim of this project is to bridge this gap by training a compact, efficient model that maintains the richness of the Swahili language while being computationally accessible.

By distilling a GPT-2 model, we seek to:
- Provide a lighter alternative to large-scale models for Swahili.
- Enable Swahili NLP applications in resource-constrained environments (e.g., mobile, embedded systems).
- Create a foundational model that can be expanded to other African languages.

## Key Features

- **Distilled GPT-2 Model**: A compact version of GPT-2 optimized for efficiency while retaining the core language generation capabilities.
- **Swahili-Centric Training**: Fine-tuned specifically on Swahili text from various domains (news, blogs, educational material).
- **Resource-Efficient**: The model is optimized for deployment in environments with limited computational resources (e.g., mobile devices).
- **Open Access**: The model and code are available for free, encouraging collaboration and expansion to other low-resource African languages.

## Table of Contents

1. [Installation](#installation)
2. [Data Preparation](#data-preparation)
3. [Training](#training)
4. [Evaluation](#evaluation)
5. [Usage](#usage)
6. [Contributing](#contributing)
7. [License](#license)
8. [Acknowledgements](#acknowledgements)
9. [References](#references)

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/kiplangatkorir/swahili-gpt2-distilled.git
cd swahili-gpt2-distilled
```

### 2. Install Dependencies

You can install the required dependencies using `pip`:

```bash
pip install -r requirements.txt
```

## Data Preparation

### 1. Dataset

The model is fine-tuned on a custom Swahili dataset, which includes:

- **train.txt**: Contains the training data (approx. 200K sentences)
- **valid.txt**: Contains the validation data (approx. 20K sentences)

The dataset covers a variety of domains, including news, social media, and educational content. The text is preprocessed to remove irrelevant content and ensure high-quality data.

### 2. Data Structure

The data is organized in plain text format (`.txt` files), with each line representing one sentence. The following is the structure of the dataset:

```
swahili-gpt2-distilled/
├── data/
│   ├── train.txt
│   └── valid.txt
```

Make sure to store your dataset in the `data/` directory of the repository for training.

## Training

### 1. Training Process

![W B Chart 4_25_2025, 6_12_13 PM](https://github.com/user-attachments/assets/48a5a40e-d3af-4803-ab70-55779fd1ee26)

The training is done using the **Hugging Face Transformers** library. The key steps include:

- **Model Architecture**: The original GPT-2 small model is distilled to reduce its size, which includes modifications such as reducing the number of layers and attention heads.
- **Training Loop**: We use the `Trainer` API from Hugging Face to handle the training and validation process.
- **Environment**: The training is conducted on **Google Colab Pro** with GPU acceleration for efficiency.

The training is done using a **batch size of 2**, **3 epochs**, and **learning rate scheduler** to ensure stable convergence.

To start training, run the following:

```bash
python train.py
```

Alternatively, you can access the Jupyter notebook for training and experimentation in the `notebooks/` directory:

```
notebooks/distillation-colab.ipynb
```

This notebook can be opened directly in Google Colab for seamless GPU access.

## Evaluation

### 1. Evaluation Metrics

We evaluate the model using the following metrics:

- **Perplexity**: Measures how well the model predicts the next word in a sentence. A lower perplexity indicates better performance.
- **Generated Samples**: We evaluate the quality of generated Swahili text based on coherence, grammar, and diversity.

To evaluate the trained model, you can run:

```bash
python evaluate.py --model_path ./models/distilled-gpt2-swahili
```

This will output the perplexity and show some sample generated text.

## Usage

Once trained, the distilled GPT-2 Swahili model can be used for various NLP tasks like text generation, completion, and more. To use the model for text generation, you can use the following code:

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load the trained model and tokenizer
model = GPT2LMHeadModel.from_pretrained('./models/distilled-gpt2-swahili')
tokenizer = GPT2Tokenizer.from_pretrained('./models/distilled-gpt2-swahili')

# Encode input prompt
input_text = "Habari ya leo"
input_ids = tokenizer.encode(input_text, return_tensors='pt')

# Generate text
output = model.generate(input_ids, max_length=50, num_return_sequences=1)

# Decode and print the output
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)
```

This will generate text in Swahili starting from the given prompt, `"Habari ya leo"` (Good morning).

## Contributing

We welcome contributions to improve this project! If you have suggestions, bug fixes, or want to contribute code, feel free to open an issue or submit a pull request.

Steps to contribute:
1. Fork the repository.
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/swahili-gpt2-distilled.git`
3. Create a new branch: `git checkout -b feature-branch`
4. Commit your changes: `git commit -am 'Add new feature'`
5. Push to the branch: `git push origin feature-branch`
6. Open a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## References

- Radford, A., Narasimhan, K., Salimans, T., & Sutskever, I. (2019). **Language Models are Unsupervised Multitask Learners**. OpenAI.
- Hinton, G., Vinyals, O., & Dean, J. (2015). **Distilling the Knowledge in a Neural Network**.
- Masakhane, S. et al. (2020). **The Masakhane African Language NLP Research**.
