# Cognix AI Library 🚀

Cognix is a powerful yet simplified Python library for model management, inference, and fine-tuning. It provides a high-level API to interact with Hugging Face models effortlessly, built for developers who want to integrate AI without the boilerplate.

## Installation

```bash
pip install cognix-ai
```

## Features

- **Simplified Model Loading**: Load models for Chat, Vision, or Classification with one line.
- **Robust Error Handling**: Improved fallback mechanisms for `ImportError` and model compatibility.
- **Easy Fine-Tuning**: Built-in support for LoRA/QLoRA for efficient training.
- **Hugging Face Integration**: Secure login and seamless model downloading.
- **Unified Interface**: `generate()` for LLMs, `predict()` for Vision/Classification.

---

## Quick Start

### 1. Login to Hugging Face
```python
import cognix

# Method 1: Explicit token
cognix.login("your_hf_token")

# Method 2: Uses HUGGING_FACE_HUB_TOKEN environment variable
cognix.login()
```

### 2. Load and Use an LLM (Chat)
```python
import cognix

try:
    # Load a model for chat (supports 4bit/8bit quantization)
    model = cognix.load_model("gpt2", task="chat")
    
    # Generate text
    response = model.generate("The future of AI is")
    print(f"AI: {response}")
except Exception as e:
    print(f"Error loading model: {e}")
```

### 3. Image Classification / Vision (Improved)
Cognix now handles older versions of `transformers` gracefully.
```python
import cognix

# Load a vision model (e.g., BLIP or ViT)
vision_model = cognix.load_model("Salesforce/blip-image-captioning-base", task="vision")

# Predict from URL, local path, or PIL Image
result = vision_model.predict("https://example.com/image.jpg")
print(f"Description: {result}")
```

---

## Error Handling & Edge Cases

Cognix is designed to fail gracefully. Here are common scenarios handled:

| Scenario | Behavior |
|----------|----------|
| **Missing `bitsandbytes`** | Automatically disables 4-bit/8-bit quantization and tries to load in full precision. |
| **Invalid Image URL** | Returns a clear error message instead of crashing. |
| **Model Incompatibility** | Falls back to generic `AutoModel` if specialized classes like `AutoModelForVision2Seq` fail. |
| **Login Failure** | Provides detailed feedback if the token is invalid or there's no connection. |
| **Missing PEFT** | Informs the user that `peft` is required only when trying to use the `Trainer`. |

---

## Fine-Tuning with LoRA
```python
import cognix

# Load model and dataset
model = cognix.load_model("gpt2", task="chat", quantization="4bit")
dataset = cognix.load_dataset("my_data.json")

# Initialize trainer
trainer = cognix.Trainer(model, dataset, output_dir="./my_fine_tuned_model")

# Train with built-in validation!
if trainer.train(epochs=1, lr=5e-5, batch_size=4):
    trainer.save()
else:
    print("Training failed. Check logs for details.")
```

## License
MIT

## Contributing
Visit our GitHub: [github.com/golemgamer/cognix](https://github.com/golemgamer/cognix)


[![Buy Me a Coffee](https://img.buymeacoffee.com/button-api/?text=Buy%20me%20a%20coffee&emoji=&slug=bueorm&button_colour=5F7FFF&font_colour=ffffff&font_family=Cookie&outline_colour=000000&coffee_colour=FFDD00)](https://www.buymeacoffee.com/bueorm)
