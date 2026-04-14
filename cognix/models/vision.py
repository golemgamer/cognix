import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image
import requests
from .base import BaseModel

class VisionModel(BaseModel):
    def __init__(self, model_id, device=None):
        super().__init__(model_id, device)
        self.processor = None
        self.load_model()

    def load_model(self):
        """
        Load the vision model.
        """
        print(f"Loading vision model {self.model_id}...")
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForVision2Seq.from_pretrained(self.model_id, device_map="auto")
        print(f"Model loaded on {self.device}.")

    def predict(self, image_input, prompt="Describe this image:"):
        """
        Predict/describe based on image input.
        """
        if isinstance(image_input, str):
            if image_input.startswith("http"):
                image = Image.open(requests.get(image_input, stream=True).raw)
            else:
                image = Image.open(image_input)
        else:
            image = image_input

        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=256)
        
        return self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
