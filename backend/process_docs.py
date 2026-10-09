
from pathlib import Path
import re
import json
import ollama

# Project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_DIR / "data" / "DevVibe-Website-Content.md"
OUTPUT_FILE = PROJECT_DIR / "data" / "chunks_preview.json"

# Load the original Markdown document
content = DATA_FILE.read_text(encoding="utf-8")

# Locate page sections
pattern = r"(?m)^## (P\d{3})\s*[—-]\s*(.+)$"
matches = list(re.finditer(pattern, content))

pages = []
skipped_pages = []

for i, match in enumerate(matches):
    end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
    section = content[match.start():end].strip()

    # Keep published page content rather than the separate heading outline
    published_marker = "### Published page content" 

    if published_marker in section:
        section = section.split(published_marker, 1)[1].strip()

    # Fix the single malformed service heading in the source
    section = section.replace(
        "***A**I**D**e**v**e**l**o**p**m**e**n**t******S**o**f**t**w**a**r**e******S**a**a**s**&**M**V**P******R**o**b**o**t**i**c**s**/**I**o**T**/**I**I**o**T******C**l**o**u**d**&**E**R**P*** Services",
        "AI Development Software SaaS & MVP Robotics/IoT/IIoT Cloud & ERP Services"
    )
    
     # Your existing reference-only page check continues below

    # Skip pages that only refer to content recorded on another page
    if re.search(
        r"(?:The retrieved page body is identical to|"
        r"full published text is recorded there)",
        section,
        re.IGNORECASE
    ):
        skipped_pages.append(match.group(1))
        continue

    pages.append({
        "page_id": match.group(1),
        "title": match.group(2).strip(),
        "text": section
    })

# Chunk settings
CHUNK_SIZE = 1800
OVERLAP = 250
chunks = []

for page in pages:
    paragraphs = re.split(r"\n\s*\n", page["text"])
    page_chunks = []
    current = ""

    for paragraph in paragraphs:
        paragraph = paragraph.strip()

        if not paragraph:
            continue

        candidate = (
            f"{current}\n\n{paragraph}".strip()
            if current else paragraph
        )

        if len(candidate) <= CHUNK_SIZE:
            current = candidate
            continue

        if current:
            page_chunks.append(current)
            current = ""

        # Split unusually long paragraphs without dropping their content
        while len(paragraph) > CHUNK_SIZE:
            split_at = paragraph.rfind(" ", 0, CHUNK_SIZE)

            if split_at <= 0:
                split_at = CHUNK_SIZE

            piece = paragraph[:split_at].strip()
            page_chunks.append(piece)

            # Retain overlap for continuity
            paragraph = paragraph[max(0, split_at - OVERLAP):].strip()

        current = paragraph

    if current:
        page_chunks.append(current)

    # Add overlap between adjacent chunks from the same page
    for i, chunk_text in enumerate(page_chunks):
        if i > 0:
            previous_tail = page_chunks[i - 1][-OVERLAP:]
            chunk_text = previous_tail + "\n\n" + chunk_text

        chunks.append({
            "page_id": page["page_id"],
            "title": page["title"],
            "text": chunk_text
        })

# Save the preview; the original source file remains unchanged
OUTPUT_FILE.write_text(
    json.dumps(chunks, ensure_ascii=False, indent=2),
    encoding="utf-8"
)

# Print processing summary
print("=" * 55)
print("DevVibe Markdown Processing Summary")
print("=" * 55)
print(f"Original page markers: {len(matches)}")
print(f"Pages retained: {len(pages)}")
print(f"Reference-only pages skipped: {len(skipped_pages)}")
print(f"Chunks created: {len(chunks)}")
print(f"Preview saved to: {OUTPUT_FILE}")

if chunks:
    print("\nFirst chunk:")
    print(json.dumps(chunks[0], ensure_ascii=False, indent=2)[:1500])

    if len(chunks) > 1:
        print("\nSecond chunk beginning:")
        print(chunks[1]["text"][:500])