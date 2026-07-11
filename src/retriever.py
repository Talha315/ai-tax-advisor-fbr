from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[1]

VECTORSTORE_DIR = PROJECT_ROOT / "vectorstore"

COLLECTION_NAME = "fbr_tax_documents"
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

TOP_K = 6


def load_embedding_model():
    print("Loading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return model


def load_collection():
    client = chromadb.PersistentClient(path=str(VECTORSTORE_DIR))
    collection = client.get_collection(name=COLLECTION_NAME)
    return collection


def search_documents(question, collection, model, top_k=TOP_K):
    question_embedding = model.encode(
        question,
        normalize_embeddings=True
    )

    results = collection.query(
        query_embeddings=[question_embedding.tolist()],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    chunks = []

    for index in range(len(results["ids"][0])):
        chunk = {
            "id": results["ids"][0][index],
            "text": results["documents"][0][index],
            "metadata": results["metadatas"][0][index],
            "distance": results["distances"][0][index]
        }

        chunks.append(chunk)

    return chunks


def print_results(chunks):
    for index, chunk in enumerate(chunks, start=1):
        metadata = chunk["metadata"]

        print("\n" + "=" * 90)
        print(f"Result {index}")
        print("=" * 90)

        print(f"Source file : {metadata.get('source_file')}")
        print(f"Category    : {metadata.get('category')}")
        print(f"Language    : {metadata.get('language')}")
        print(f"Pages       : {metadata.get('page_start')} - {metadata.get('page_end')}")
        print(f"Chunk index : {metadata.get('chunk_index')}")
        print(f"Distance    : {chunk['distance']}")

        print("\nText preview:")
        print("-" * 90)
        print(chunk["text"][:1500])


def main():
    collection = load_collection()
    model = load_embedding_model()

    print("\nRetriever is ready.")
    print("Type your question. Type 'exit' to stop.\n")

    while True:
        question = input("Ask a tax question: ").strip()

        if question.lower() in ["exit", "quit"]:
            print("Goodbye.")
            break

        if not question:
            continue

        chunks = search_documents(
            question=question,
            collection=collection,
            model=model,
            top_k=TOP_K
        )

        print_results(chunks)


if __name__ == "__main__":
    main()