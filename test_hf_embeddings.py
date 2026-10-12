import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()

token = os.getenv("HF_TOKEN")
if not token:
    raise ValueError("HF_TOKEN was not found in .env")

client = InferenceClient(
    provider="hf-inference",
    api_key=token
)
try:
    result = client.feature_extraction(
        "DevVibe provides software development and AI services.",
        model="BAAI/bge-base-en-v1.5"
    )
    print("Embedding request succeeded!")
    print("Result type:", type(result))
    print("Embedding shape:", getattr(result, "shape", "Shape unavailable"))
except Exception as e:
    print("Embedding request failed:")
    print(type(e).__name__, str(e))