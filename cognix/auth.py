from huggingface_hub import login as hf_login

def login(token: str):
    """
    Login to Hugging Face Hub.
    """
    hf_login(token=token)
    print("Successfully logged in to Hugging Face Hub.")
