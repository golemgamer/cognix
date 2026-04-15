from huggingface_hub import login as hf_login
import os

def login(token: str = None):
    """
    Login to Hugging Face Hub with robust error handling.
    """
    if token is None:
        token = os.getenv("HUGGING_FACE_HUB_TOKEN")
        if not token:
            print("Error: No token provided. Provide a token as an argument or set HUGGING_FACE_HUB_TOKEN env var.")
            return False

    try:
        # Perform the login
        hf_login(token=token)
        print("Successfully logged in to Hugging Face Hub.")
        return True
    except Exception as e:
        print(f"Error: Failed to login to Hugging Face Hub. Details: {e}")
        return False
