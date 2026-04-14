# Cognix Documentation

Cognix is designed to be the simplest way to interact with AI models. It wraps powerful libraries like `transformers`, `peft`, and `bitsandbytes` into a clean, intuitive API.

## Table of Contents
- [Installation](#installation)
- [Authentication](#authentication)
- [Model Management](#model-management)
  - [Chat Models](#chat-models)
  - [Vision Models](#vision-models)
  - [Text Classification](#text-classification)
- [Dataset Loading](#dataset-loading)
- [Fine-Tuning (Trainer)](#fine-tuning-trainer)
- [Saving and Loading](#saving-and-loading)

---

## Installation

To install Cognix locally for development:
```bash
pip install -e .
```

Once published, you can install it via pip:
```bash
pip install cognix
```

---

## Authentication

### `cognix.login(token: str)`
Authenticates your session with Hugging Face Hub. This is required to download gated models (like Llama-3).
- **token**: Your Hugging Face API token (get it at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)).

```python
import cognix
cognix.login("hf_your_token_here")
```

---

## Model Management

### `cognix.load_model(model_id, task="chat", quantization="none", device=None)`
The central factory function to load any supported model.

- **model_id**: The Hugging Face model repository ID (e.g., `"gpt2"`, `"meta-llama/Llama-3-8B"`) or a local path.
- **task**: The type of model to load:
  - `"chat"`: For text generation and conversation.
  - `"vision"`: For image-to-text or image description.
  - `"classification"`: For sentiment analysis or category prediction.
- **quantization**: Memory optimization level:
  - `"none"`: Full precision (default).
  - `"4bit"`: Uses QLoRA (requires `bitsandbytes`, best for large models on consumer GPUs).
  - `"8bit"`: Uses 8-bit quantization.
- **device**: `"cuda"` or `"cpu"`. Automatically detected if not provided.

### Chat Models
Used for generating text or answering questions.
```python
model = cognix.load_model("gpt2", task="chat")
response = model.generate("What is the meaning of life?", max_new_tokens=100)
```
- **`.generate(prompt, max_new_tokens=256, temperature=0.7, top_p=0.9)`**: Generates text.

### Vision Models
Used for describing images.
```python
vision_model = cognix.load_model("Salesforce/blip-image-captioning-base", task="vision")
description = vision_model.predict("https://example.com/photo.jpg")
```
- **`.predict(image_input, prompt="Describe this image:")`**: Takes a URL, local path, or PIL Image.

### Text Classification
Used for labeling text.
```python
classifier = cognix.load_model("distilbert-base-uncased-finetuned-sst-2-english", task="classification")
result = classifier.predict("I love this library!")
# Returns: {"label": "POSITIVE", "score": 0.99}
```
- **`.predict(text)`**: Returns the predicted label and confidence score.

---

## Dataset Loading

### `cognix.load_dataset(path_or_id, format=None, split="train")`
Loads data for training or evaluation.
- **path_or_id**: Local path (`.json`, `.csv`, `.txt`) or HF Dataset ID (e.g., `"imdb"`).
- **format**: Optional. Automatically detected from file extension if not provided.

---

## Fine-Tuning (Trainer)

### `cognix.Trainer(model_wrapper, dataset, output_dir="./output")`
A simplified class to fine-tune models using **LoRA (Low-Rank Adaptation)**.

#### `trainer.train(epochs=3, lr=5e-5, batch_size=4, lora_r=16, lora_alpha=32)`
Runs the training loop.
- **epochs**: Number of training passes.
- **lr**: Learning rate.
- **batch_size**: Number of samples per step.
- **lora_r / lora_alpha**: LoRA hyperparameters (higher = more parameters updated).

```python
model = cognix.load_model("gpt2", task="chat", quantization="4bit")
dataset = cognix.load_dataset("data.json")

trainer = cognix.Trainer(model, dataset)
trainer.train(epochs=1)
trainer.save("my_custom_model")
```

---

## Saving and Loading

Any model (including fine-tuned ones) can be saved:
```python
model.save("path/to/directory")
```
To load a saved model later:
```python
model = cognix.load_model("path/to/directory", task="chat")
```
