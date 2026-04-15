import torch
import os
from transformers import AutoTokenizer, AutoConfig

class BaseModel:
    def __init__(self, model_id, device=None):
        self.model_id = model_id
        try:
            self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        except Exception:
            self.device = "cpu"
        self.tokenizer = None
        self.model = None

    def save(self, path):
        """
        Save the model and tokenizer to a directory with error handling.
        """
        try:
            os.makedirs(path, exist_ok=True)
            if self.model:
                self.model.save_pretrained(path)
            if self.tokenizer:
                self.tokenizer.save_pretrained(path)
            print(f"Model and tokenizer saved successfully to {path}")
            return True
        except Exception as e:
            print(f"Error saving model to {path}: {e}")
            return False

    def load_tokenizer(self):
        """
        Load the tokenizer for the model with error handling.
        """
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            return self.tokenizer
        except Exception as e:
            raise RuntimeError(f"Failed to load tokenizer for '{self.model_id}': {str(e)}")
