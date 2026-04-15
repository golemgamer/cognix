import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from .base import BaseModel

class ChatModel(BaseModel):
    def __init__(self, model_id, quantization="none", device=None):
        super().__init__(model_id, device)
        self.quantization = quantization
        self.load_model()

    def load_model(self):
        """
        Load the chat model with specified quantization and robust error handling.
        """
        try:
            print(f"Loading chat model {self.model_id}...")
            kwargs = {"device_map": "auto", "trust_remote_code": True}
            
            if self.quantization == "4bit":
                try:
                    from transformers import BitsAndBytesConfig
                    kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4",
                    )
                except ImportError:
                    print("Warning: bitsandbytes not found. 4-bit quantization disabled.")
            elif self.quantization == "8bit":
                try:
                    kwargs["load_in_8bit"] = True
                except Exception as e:
                    print(f"Warning: Failed to enable 8-bit quantization: {e}")
            
            self.tokenizer = self.load_tokenizer()
            self.model = AutoModelForCausalLM.from_pretrained(self.model_id, **kwargs)
            print(f"Model loaded successfully on {self.model.device}.")
        except Exception as e:
            raise RuntimeError(f"Failed to load chat model '{self.model_id}': {str(e)}")

    def generate(self, prompt, max_new_tokens=256, temperature=0.7, top_p=0.9):
        """
        Generate text with prompt validation and error handling.
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model or tokenizer not loaded properly.")

        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True if temperature > 0 else False,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            
            # Decode only the generated tokens
            result = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            return result.strip()
        except Exception as e:
            return f"Error during text generation: {str(e)}"
