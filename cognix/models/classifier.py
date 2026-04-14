import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from .base import BaseModel

class TextClassifier(BaseModel):
    def __init__(self, model_id, device=None):
        super().__init__(model_id, device)
        self.load_model()

    def load_model(self):
        """
        Load the classification model.
        """
        print(f"Loading classification model {self.model_id}...")
        self.tokenizer = self.load_tokenizer()
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_id, device_map="auto")
        print(f"Model loaded on {self.device}.")

    def predict(self, text):
        """
        Classify text input.
        """
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        predicted_class_id = torch.argmax(probabilities, dim=-1).item()
        
        # Get label from config if available
        label = self.model.config.id2label.get(predicted_class_id, f"Class {predicted_class_id}")
        
        return {
            "label": label,
            "score": probabilities[0][predicted_class_id].item()
        }
