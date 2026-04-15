import torch
import os
from transformers import TrainingArguments, Trainer as HFTrainer, DataCollatorForLanguageModeling

try:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

class Trainer:
    def __init__(self, model_wrapper, dataset, output_dir="./output"):
        """
        Initialize the simplified trainer with robust error handling.
        """
        self.model_wrapper = model_wrapper
        self.dataset = dataset
        self.output_dir = output_dir
        self.tokenizer = model_wrapper.tokenizer
        
        if self.tokenizer is None:
            raise ValueError("Model wrapper must have a valid tokenizer.")

        # Pre-process dataset
        try:
            self.processed_dataset = self.prepare_dataset()
        except Exception as e:
            raise RuntimeError(f"Failed to prepare dataset for training: {str(e)}")

    def prepare_dataset(self):
        """
        Tokenize the dataset with column validation.
        """
        def tokenize_function(examples):
            if 'text' in examples:
                return self.tokenizer(examples['text'], truncation=True, padding="max_length", max_length=512)
            elif 'prompt' in examples and 'completion' in examples:
                texts = [str(p) + str(c) for p, c in zip(examples['prompt'], examples['completion'])]
                return self.tokenizer(texts, truncation=True, padding="max_length", max_length=512)
            else:
                raise ValueError("Dataset must contain a 'text' column or both 'prompt' and 'completion' columns.")

        print("Tokenizing dataset...")
        return self.dataset.map(tokenize_function, batched=True, remove_columns=self.dataset.column_names)

    def train(self, epochs=3, lr=5e-5, batch_size=4, lora_r=16, lora_alpha=32):
        """
        Run the training process with LoRA and comprehensive error handling.
        """
        if not PEFT_AVAILABLE:
            raise ImportError("PEFT library is required for training with LoRA. Please install it with 'pip install peft'.")

        try:
            print("Preparing model for training with LoRA...")
            model = self.model_wrapper.model
            
            if model is None:
                raise ValueError("Model wrapper does not contain a loaded model.")

            # Prepare for training (LoRA/QLoRA)
            if hasattr(model, "is_loaded_in_4bit") and model.is_loaded_in_4bit:
                model = prepare_model_for_kbit_training(model)
                
            peft_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=["q_proj", "v_proj"], # Common for Llama/most LLMs
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM"
            )
            
            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()
            
            training_args = TrainingArguments(
                output_dir=self.output_dir,
                num_train_epochs=epochs,
                learning_rate=lr,
                per_device_train_batch_size=batch_size,
                gradient_accumulation_steps=4,
                optim="adamw_torch",
                logging_steps=10,
                save_strategy="epoch",
                fp16=True if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7 else False,
                remove_unused_columns=False,
            )
            
            hf_trainer = HFTrainer(
                model=model,
                args=training_args,
                train_dataset=self.processed_dataset,
                data_collator=DataCollatorForLanguageModeling(self.tokenizer, mlm=False),
            )
            
            print("Starting training...")
            hf_trainer.train()
            
            # Save the final adapter
            os.makedirs(self.output_dir, exist_ok=True)
            model.save_pretrained(self.output_dir)
            print(f"Training complete. Adapter saved to {self.output_dir}")
            return True
        except Exception as e:
            print(f"Error during training process: {str(e)}")
            return False

    def save(self, path=None):
        """
        Save the model wrapper with the trained adapter.
        """
        save_path = path if path else self.output_dir
        return self.model_wrapper.save(save_path)
