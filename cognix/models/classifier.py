import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from .base import BaseModel

class TextClassifier(BaseModel):
    def __init__(self, model_id, device=None):
        super().__init__(model_id, device)
        self.load_model()

    def load_model(self):
        """
        Load the classification model with robust error handling.
        """
        try:
            print(f"Loading classification model {self.model_id}...")
            self.tokenizer = self.load_tokenizer()
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_id, 
                device_map="auto",
                trust_remote_code=True
            )
            print(f"Model loaded successfully on {self.model.device}.")
        except Exception as e:
            raise RuntimeError(f"Failed to load classification model '{self.model_id}': {str(e)}")

    def predict(self, text):
        """
        Classify text input with robust validation and error handling.
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model or tokenizer not loaded properly.")

        if not text or not isinstance(text, str):
            raise ValueError("Input text must be a non-empty string.")

        try:
            inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            predicted_class_id = torch.argmax(probabilities, dim=-1).item()
            
            # Get label from config safely
            try:
                label = self.model.config.id2label.get(predicted_class_id, f"Class {predicted_class_id}")
            except (AttributeError, KeyError):
                label = f"Class {predicted_class_id}"
            
            return {
                "label": label,
                "score": float(probabilities[0][predicted_class_id].item())
            }
        except Exception as e:
            return {"error": f"Error during classification: {str(e)}"}
