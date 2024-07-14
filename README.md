# Model Quantization with llama-cpp in Google Colab

## Table of Contents

1. [Introduction](#introduction)
2. [Features](#features)
3. [What is Quantization?](#what-is-quantization)
4. [Why Use Quantization?](#why-use-quantization)
5. [Installation](#installation)
6. [Model Calling](#model-calling)
7. [Quantization](#quantization)
8. [Configuration](#configuration)
9. [Technologies Used](#technologies-used)
10. [Contributing](#contributing)
11. [License](#license)

## Introduction

This project demonstrates how to quantize a large language model using the llama-cpp library in Google Colab. The goal is to reduce the model size while maintaining its performance, facilitating easier deployment and faster inference.

## Features

- **Model Download**: Fetches a pre-trained model from Hugging Face.
- **Model Conversion**: Converts the model to a quantized format using GGUF.
- **Embedding Generation**: Uses the quantized model to generate embeddings.

## What is Quantization?

Quantization is the process of reducing the number of bits that represent a number in a model's parameters and activations. This often involves converting 32-bit floating point numbers (FP32) to 16-bit (FP16) or even lower bit representations like 8-bit integers (INT8).

## Why Use Quantization?

Quantization is used to:

- **Reduce Model Size**: Smaller models require less storage space, making them easier to deploy, especially on edge devices.
- **Speed Up Inference**: Quantized models can process data faster due to reduced computational complexity.
- **Lower Power Consumption**: Reduced bit-width operations consume less power, which is critical for battery-operated devices.

## Installation

Follow these steps to set up the project in Google Colab.

### Prerequisites

- Google Colab account

### Step-by-Step Guide

1. **Open Google Colab**: Go to [Google Colab](https://colab.research.google.com/).

2. **Create a new notebook**: Click on `File` > `New notebook`.

3. **Run the following commands in Colab**:

   ```python
   # Update and install dependencies
   !apt-get update
   !apt-get install -y build-essential cmake

   # Install llama-cpp-python
   !CMAKE_ARGS="-DGGML_CUDA=ON" pip install llama-cpp-python --no-cache-dir

   # Verify CUDA installation
   !nvcc --version
   !nvidia-smi

   # Clone the llama.cpp repository
   !git clone https://github.com/ggerganov/llama.cpp

   # Install the required Python packages
   !pip install -r llama.cpp/requirements.txt
   ```

## Model Calling

Download the pre-trained model and call it in your notebook:

1. **Download the model from Hugging Face**:

   ```python
   import os
   from huggingface_hub import snapshot_download

   model_name = "BAAI/bge-large-en-v1.5"
   base_model = "./original_model/"
   quantized_model = "./quantized_model/"

   snapshot_download(repo_id=model_name, local_dir=base_model, local_dir_use_symlinks=False)
   ```

2. **Use the model**:

   ```python
   from llama_cpp import Llama

   texts = "This is an example"
   model = Llama("./quantized_model/bge-large-en-1.5.gguf", embedding=True)
   embed = model.embed(texts)
   print(embed)
   ```

## Quantization

Convert the downloaded model to a quantized format:

1. **Create a directory for the quantized model**:

   ```python
   !mkdir ./quantized_model/
   ```

2. **Convert the model to GGUF format**:
   ```python
   !python llama.cpp/convert_hf_to_gguf.py ./original_model/ --outtype bf16 --outfile ./quantized_model/bge-large-en-1.5.gguf
   ```

## Configuration

Check the file sizes to ensure the model has been correctly downloaded and converted:

```python
import os

# Quantized model file size
file_path = './quantized_model/bge-large-en-1.5.gguf'
if os.path.exists(file_path):
    file_size = os.stat(file_path).st_size
    file_size_gb = file_size / (1024 * 1024 * 1024)
    print(f"Size of '{file_path}': {file_size} bytes ({file_size_gb:.2f} GB)")
else:
    print(f"File '{file_path}' not found.")

# Original model file size
file_path = './original_model/onnx/model.onnx'
if os.path.exists(file_path):
    file_size = os.stat(file_path).st_size
    file_size_gb = file_size / (1024 * 1024 * 1024)
    print(f"Size of '{file_path}': {file_size} bytes ({file_size_gb:.2f} GB)")
else:
    print(f"File '{file_path}' not found.")
```

## Technologies Used

- **Python**: Programming language used for scripting and model handling.
- **CUDA**: For GPU acceleration.
- **Hugging Face Hub**: Model repository.
- **llama-cpp**: Library for model conversion and embedding generation.

## Contributing

Contributions are welcome! Please follow these steps to contribute:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes and commit them (`git commit -m 'Add some feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Authors

- [Rajesh](https://github.com/RajeshK1006)
