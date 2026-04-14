import torch
from transformers import AutoTokenizer, AutoConfig

class BaseModel:
    def __init__(self, model_id, device=None):
        self.model_id = model_id
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None

    def save(self, path):
        """
        Save the model and tokenizer to a directory.
        """
        if self.model:
            self.model.save_pretrained(path)
        if self.tokenizer:
            self.tokenizer.save_pretrained(path)
        print(f"Model saved to {path}")

    def load_tokenizer(self):
        """
        Load the tokenizer for the model.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        return self.tokenizer
