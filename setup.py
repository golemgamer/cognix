from setuptools import setup, find_packages

setup(
    name="cognix-ai",
    version="0.1.0.1",
    packages=find_packages(),
    install_requires=[
        "transformers",
        "datasets",
        "peft",
        "accelerate",
        "bitsandbytes",
        "torch",
        "torchvision",
        "huggingface_hub",
        "pillow",
    ],
    author="BUEORM",
    description="A simplified AI library for model management and fine-tuning.",
    python_requires=">=3.8",
)
