
from pathlib import Path
import json
import ollama
import os
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = PROJECT_DIR / "data" / "chunks_preview.json"
OUTPUT_FILE = PROJECT_DIR / "data" / "embedded_chunks.json"

# Load the prepared website chunks
chunks = json.loads(INPUT_FILE.read_text(encoding="utf-8"))

print(f"Loaded {len(chunks)} chunks.")
print("Generating embeddings with nomic-embed-text...")

# Generate and attach an embedding to each chunk
for index, chunk in enumerate(chunks, start=1):
    result = ollama.embed(
        model="nomic-embed-text",
        input=chunk["text"]
    )

    chunk["embedding"] = result["embeddings"][0]

    if index % 10 == 0 or index == len(chunks):
        print(f"Processed {index}/{len(chunks)} chunks")

print("\nUploading chunks to Supabase...")

for start in range(0, len(chunks), 25):
    batch = chunks[start:start + 25]

    records = [
        {
            "page_id": chunk["page_id"],
            "title": chunk["title"],
            "content": chunk["text"],
            "embedding": chunk["embedding"]
        }
        for chunk in batch
    ]

    supabase.table("devvibe_documents").insert(records).execute()
    print(f"Uploaded {min(start + 25, len(chunks))}/{len(chunks)} chunks")

print("Supabase upload complete.")

# Save the embedded chunks
OUTPUT_FILE.write_text(
    json.dumps(chunks, ensure_ascii=False),
    encoding="utf-8"
)

print("\nEmbedding generation complete.")
print(f"Total embedded chunks: {len(chunks)}")
print(f"Embedding dimensions: {len(chunks[0]['embedding'])}")
print(f"Saved to: {OUTPUT_FILE}")