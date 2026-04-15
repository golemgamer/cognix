import torch
from PIL import Image
import requests
import os
from .base import BaseModel

try:
    from transformers import AutoProcessor, AutoModelForVision2Seq
except ImportError:
    from transformers import AutoProcessor, AutoModel as AutoModelForVision2Seq

class VisionModel(BaseModel):
    def __init__(self, model_id, device=None):
        super().__init__(model_id, device)
        self.processor = None
        self.load_model()

    def load_model(self):
        """
        Load the vision model with error handling.
        """
        try:
            print(f"Loading vision model {self.model_id}...")
            self.processor = AutoProcessor.from_pretrained(self.model_id)
            
            # Attempt to load with AutoModelForVision2Seq, fallback to AutoModel
            try:
                self.model = AutoModelForVision2Seq.from_pretrained(
                    self.model_id, 
                    device_map="auto",
                    trust_remote_code=True
                )
            except Exception as e:
                print(f"Warning: Could not load with specialized class. Trying generic AutoModel. Error: {e}")
                from transformers import AutoModel
                self.model = AutoModel.from_pretrained(self.model_id, device_map="auto")
            
            print(f"Model loaded successfully on {self.model.device}.")
        except Exception as e:
            raise RuntimeError(f"Failed to load vision model '{self.model_id}': {str(e)}")

    def predict(self, image_input, prompt="Describe this image:"):
        """
        Predict/describe based on image input with robust validation.
        """
        if self.model is None or self.processor is None:
            raise ValueError("Model or processor not loaded. Call load_model() first.")

        try:
            # Handle image input
            if isinstance(image_input, str):
                if image_input.startswith("http"):
                    try:
                        response = requests.get(image_input, stream=True, timeout=10)
                        response.raise_for_status()
                        image = Image.open(response.raw).convert("RGB")
                    except Exception as e:
                        raise ValueError(f"Failed to download image from URL: {e}")
                elif os.path.exists(image_input):
                    image = Image.open(image_input).convert("RGB")
                else:
                    raise FileNotFoundError(f"Image file not found: {image_input}")
            elif isinstance(image_input, Image.Image):
                image = image_input.convert("RGB")
            else:
                raise TypeError("image_input must be a URL, a local path, or a PIL Image object.")

            # Prepare inputs
            inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.model.device)
            
            # Inference
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_new_tokens=256)
            
            # Decode
            result = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
            return result.strip()

        except Exception as e:
            return f"Error during vision prediction: {str(e)}"
