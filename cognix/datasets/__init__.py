from datasets import load_dataset as hf_load_dataset
import os

def load(path_or_id, format=None, split="train", **kwargs):
    """
    Load a dataset from local path or Hugging Face Hub.
    
    Args:
        path_or_id (str): Path to local file (JSON, CSV, TXT) or Hugging Face dataset ID.
        format (str, optional): Format of local file ("json", "csv", "text").
        split (str): Dataset split to load. Default: "train".
    """
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
                raise ValueError("Could not determine format. Please specify 'format'.")
        
        return hf_load_dataset(format, data_files=path_or_id, split=split, **kwargs)
    else:
        # Hugging Face dataset
        return hf_load_dataset(path_or_id, split=split, **kwargs)
