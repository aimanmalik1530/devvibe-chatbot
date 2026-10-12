
import os
import time

from dotenv import load_dotenv
from fastembed import TextEmbedding
from supabase import create_client

load_dotenv()

supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_KEY"),
)

print("Loading FastEmbed model...")
model = TextEmbedding("BAAI/bge-small-en-v1.5")
print("Model loaded.")

BATCH_SIZE = 10
updated = 0

while True:
    result = (
        supabase.table("devvibe_documents")
        .select("id, content")
        .is_("embedding_fastembed", "null")
        .order("id")
        .limit(BATCH_SIZE)
        .execute()
    )

    rows = result.data

    if not rows:
        break

    texts = [row["content"] or "" for row in rows]
    vectors = list(model.embed(texts))

    for row, vector in zip(rows, vectors):
        supabase.table("devvibe_documents").update(
            {"embedding_fastembed": vector.tolist()}
        ).eq("id", row["id"]).execute()

        updated += 1

    print(f"Embedded {updated} chunks so far.")

print(f"Finished. Total chunks embedded in this run: {updated}")
