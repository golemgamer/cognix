from .chat import ChatModel
from .vision import VisionModel
from .classifier import TextClassifier

def load(model_id, task="chat", quantization="none", device=None):
    """
    Factory function to load models based on task.
    
    Args:
        model_id (str): Hugging Face model ID or path to local model.
        task (str): The task for which to load the model ("chat", "vision", "classification").
        quantization (str): Quantization level ("none", "4bit", "8bit"). Default: "none".
        device (str): Device to use ("cuda", "cpu"). Default: auto-detect.
    """
    if task == "chat":
        return ChatModel(model_id, quantization, device)
    elif task == "vision":
        return VisionModel(model_id, device)
    elif task == "classification":
        return TextClassifier(model_id, device)
    else:
        raise ValueError(f"Unknown task: {task}. Choose from 'chat', 'vision', 'classification'.")
