from pathlib import Path
import json

import chromadb
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CHUNKS_PATH = PROJECT_ROOT / "data" / "processed" / "document_chunks.jsonl"
VECTORSTORE_DIR = PROJECT_ROOT / "vectorstore"

COLLECTION_NAME = "fbr_tax_documents"
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

BATCH_SIZE = 64


def load_chunks():
    chunks = []

    with open(CHUNKS_PATH, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            record = json.loads(line)
            chunks.append(record)

    return chunks


def clean_metadata(metadata):
    """
    ChromaDB metadata should contain simple values only.
    This function keeps metadata safe before saving.
    """
    clean_data = {}

    for key, value in metadata.items():
        if value is None:
            clean_data[key] = ""
        elif isinstance(value, (str, int, float, bool)):
            clean_data[key] = value
        else:
            clean_data[key] = str(value)

    return clean_data


def create_chroma_collection():
    client = chromadb.PersistentClient(path=str(VECTORSTORE_DIR))

    existing_collections = client.list_collections()
    existing_collection_names = [collection.name for collection in existing_collections]

    if COLLECTION_NAME in existing_collection_names:
        print(f"Deleting old collection: {COLLECTION_NAME}")
        client.delete_collection(name=COLLECTION_NAME)

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={
            "description": "FBR tax documents for RAG-based tax advisor",
            "hnsw:space": "cosine"
        }
    )

    return collection


def add_chunks_to_chroma(collection, chunks, model):
    total_chunks = len(chunks)

    for start_index in range(0, total_chunks, BATCH_SIZE):
        batch = chunks[start_index:start_index + BATCH_SIZE]

        ids = []
        texts = []
        metadatas = []

        for local_index, record in enumerate(batch):
            global_index = start_index + local_index

            # ChromaDB requires every ID to be unique.
            # Some files can create duplicate chunk IDs, so we add global_index.
            unique_id = f"{record['id']}_{global_index}"

            ids.append(unique_id)
            texts.append(record["text"])
            metadatas.append(clean_metadata(record["metadata"]))

        embeddings = model.encode(
            texts,
            show_progress_bar=False,
            normalize_embeddings=True
        )

        collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
            embeddings=embeddings.tolist()
        )

        print(f"Added {min(start_index + BATCH_SIZE, total_chunks)} / {total_chunks} chunks")


def main():
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(
            f"Chunks file not found: {CHUNKS_PATH}\n"
            "Please run chunk_documents.py first."
        )

    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading chunks...")
    chunks = load_chunks()
    print(f"Total chunks loaded: {len(chunks)}")

    print("\nLoading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    print("\nCreating ChromaDB collection...")
    collection = create_chroma_collection()

    print("\nAdding chunks to ChromaDB...")
    add_chunks_to_chroma(collection, chunks, model)

    print("\nVector store created successfully.")
    print(f"Saved inside: {VECTORSTORE_DIR}")


if __name__ == "__main__":
    main()