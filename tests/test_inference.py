import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cognix

def test_chat_model():
    print("Testing ChatModel with GPT-2...")
    try:
        model = cognix.load_model("gpt2", task="chat")
        response = model.generate("Hello, Cognix is")
        print(f"Response: {response}")
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    test_chat_model()
