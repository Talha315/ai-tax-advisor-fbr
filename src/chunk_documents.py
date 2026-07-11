from pathlib import Path
import json
import re


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TEXT_DIR = PROJECT_ROOT / "data" / "extracted_text"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "document_chunks.jsonl"

CHUNK_SIZE = 1800
CHUNK_OVERLAP = 250


def clean_text(text):
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def detect_language(text):
    if re.search(r"[\u0600-\u06FF]", text):
        return "urdu"
    return "english"


def get_category(text_file):
    relative_path = text_file.relative_to(TEXT_DIR)
    return relative_path.parts[0]


def parse_pages(text):
    """
    The text extraction file contains markers like:
    --- Page 12 ---

    This function separates text into pages.
    """
    parts = re.split(r"\n*--- Page\s+(\d+)\s+---\n*", text)

    pages = []

    if len(parts) < 3:
        return [{"page_number": None, "text": text}]

    for i in range(1, len(parts), 2):
        page_number = int(parts[i])
        page_text = clean_text(parts[i + 1])

        if page_text:
            pages.append({
                "page_number": page_number,
                "text": page_text
            })

    return pages


def split_long_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """
    Fallback splitter for very long paragraphs.
    """
    pieces = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        piece = text[start:end].strip()

        if piece:
            pieces.append(piece)

        if end >= len(text):
            break

        start = end - overlap

    return pieces


def make_chunks_from_pages(pages):
    """
    Recursive paragraph chunking:

    1. Split each page into paragraphs.
    2. Add paragraphs into a chunk until chunk size is reached.
    3. If a paragraph is too long, split it by characters.
    4. Keep page_start and page_end for citation support.
    """
    chunks = []

    current_text = ""
    current_start_page = None
    current_end_page = None

    for page in pages:
        page_number = page["page_number"]
        paragraphs = re.split(r"\n\s*\n", page["text"])

        for paragraph in paragraphs:
            paragraph = clean_text(paragraph)

            if not paragraph:
                continue

            if len(paragraph) > CHUNK_SIZE:
                if current_text:
                    chunks.append({
                        "text": clean_text(current_text),
                        "page_start": current_start_page,
                        "page_end": current_end_page
                    })

                    current_text = ""
                    current_start_page = None
                    current_end_page = None

                small_parts = split_long_text(paragraph)

                for part in small_parts:
                    chunks.append({
                        "text": part,
                        "page_start": page_number,
                        "page_end": page_number
                    })

                continue

            new_text = current_text + "\n\n" + paragraph if current_text else paragraph

            if len(new_text) <= CHUNK_SIZE:
                current_text = new_text

                if current_start_page is None:
                    current_start_page = page_number

                current_end_page = page_number

            else:
                chunks.append({
                    "text": clean_text(current_text),
                    "page_start": current_start_page,
                    "page_end": current_end_page
                })

                overlap_text = current_text[-CHUNK_OVERLAP:] if current_text else ""

                current_text = clean_text(overlap_text + "\n\n" + paragraph)
                current_start_page = current_end_page
                current_end_page = page_number

    if current_text:
        chunks.append({
            "text": clean_text(current_text),
            "page_start": current_start_page,
            "page_end": current_end_page
        })

    return chunks


def make_chunk_id(text_file, chunk_index):
    file_name = text_file.stem.lower()
    file_name = re.sub(r"[^a-z0-9]+", "_", file_name)
    file_name = file_name.strip("_")
    file_name = file_name[:80]

    return f"{file_name}_chunk_{chunk_index}"


def process_file(text_file):
    with open(text_file, "r", encoding="utf-8") as file:
        text = file.read()

    text = clean_text(text)
    pages = parse_pages(text)
    chunks = make_chunks_from_pages(pages)

    category = get_category(text_file)
    language = detect_language(text)

    records = []

    for chunk_index, chunk in enumerate(chunks):
        record = {
            "id": make_chunk_id(text_file, chunk_index),
            "text": chunk["text"],
            "metadata": {
                "source_file": text_file.name,
                "category": category,
                "language": language,
                "chunk_index": chunk_index,
                "page_start": chunk["page_start"],
                "page_end": chunk["page_end"],
                "chunking_method": "recursive_paragraph_chunking",
                "character_count": len(chunk["text"])
            }
        }

        records.append(record)

    return records


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    text_files = sorted(TEXT_DIR.rglob("*.txt"))

    print(f"Found {len(text_files)} extracted text files.")

    total_chunks = 0

    with open(OUTPUT_PATH, "w", encoding="utf-8") as output_file:
        for file_number, text_file in enumerate(text_files, start=1):
            print(f"[{file_number}/{len(text_files)}] Chunking: {text_file.name}")

            try:
                records = process_file(text_file)

                for record in records:
                    output_file.write(json.dumps(record, ensure_ascii=False) + "\n")

                total_chunks += len(records)
                print(f"Chunks created: {len(records)}")

            except Exception as error:
                print(f"Failed to chunk: {text_file.name}")
                print(f"Reason: {error}")

    print("\nChunking completed.")
    print(f"Total chunks created: {total_chunks}")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()