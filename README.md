# Cognix AI Library

Cognix is a simplified Python library for model management, inference, and fine-tuning. It provides a high-level API to interact with Hugging Face models effortlessly.

## Installation

```bash
pip install -e .
```

## Features

- **Simplified Model Loading**: Load models for Chat, Vision, or Classification with one line.
- **Easy Fine-Tuning**: Built-in support for LoRA/QLoRA for efficient training.
- **Hugging Face Integration**: Simple login and model downloading.
- **Unified Interface**: `generate()` for LLMs, `predict()` for Vision/Classification.

## Quick Start

### 1. Login to Hugging Face
```python
import cognix
cognix.login("your_hf_token")
```

### 2. Load and Use an LLM (Chat)
```python
import cognix

# Load a model for chat (supports 4bit/8bit quantization)
model = cognix.load_model("gpt2", task="chat")

# Generate text
response = model.generate("The future of AI is")
print(response)
```

### 3. Image Classification / Vision
```python
import cognix

# Load a vision model
vision_model = cognix.load_model("google/vit-base-patch16-224", task="vision")

# Predict from URL or local path
result = vision_model.predict("https://example.com/image.jpg")
print(result)
```

### 4. Fine-Tuning with LoRA
```python
import cognix

# Load model and dataset
model = cognix.load_model("gpt2", task="chat", quantization="4bit")
dataset = cognix.load_dataset("my_data.json")

# Initialize trainer
trainer = cognix.Trainer(model, dataset, output_dir="./my_fine_tuned_model")

# Train!
trainer.train(epochs=1, lr=5e-5, batch_size=4)

# Save the adapter
trainer.save()
```

## License
MIT
