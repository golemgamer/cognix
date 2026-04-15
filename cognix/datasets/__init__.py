from datasets import load_dataset as hf_load_dataset
import os

def load(path_or_id, format=None, split="train", **kwargs):
    """
    Load a dataset from local path or Hugging Face Hub with error handling.
    """
    try:
        if os.path.exists(path_or_id):
            # Local file
            if format is None:
                if path_or_id.endswith(".json") or path_or_id.endswith(".jsonl"):
                    format = "json"
                elif path_or_id.endswith(".csv"):
                    format = "csv"
                elif path_or_id.endswith(".txt"):
                    format = "text"
                else:
                    raise ValueError(f"Could not determine format for '{path_or_id}'. Please specify 'format' (json, csv, text).")
            
            print(f"Loading local dataset from {path_or_id} in {format} format...")
            return hf_load_dataset(format, data_files=path_or_id, split=split, **kwargs)
        else:
            # Hugging Face dataset
            print(f"Loading Hugging Face dataset '{path_or_id}'...")
            return hf_load_dataset(path_or_id, split=split, **kwargs)
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset '{path_or_id}': {str(e)}")
