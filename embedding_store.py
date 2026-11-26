
import ollama
import numpy as np
import faiss

from config import OLLAMA_HOST

# Connect to local Ollama
client = ollama.Client(host=OLLAMA_HOST)

EMBEDDING_MODEL = "nomic-embed-text"


def generate_embeddings(texts):

    response = client.embed(
        model=EMBEDDING_MODEL,
        input=texts
    )

    embeddings = np.array(
        response["embeddings"],
        dtype="float32"
    )

    return embeddings


def create_vector_database(chunks):

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = generate_embeddings(texts)

    # Normalize vectors for cosine similarity
    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]

    # Create FAISS index
    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    return index, chunks


def search_database(question, index, chunks, top_k=3):

    query_embedding = generate_embeddings([question])

    faiss.normalize_L2(query_embedding)

    scores, indices = index.search(
        query_embedding,
        top_k
    )

    results = []

    for score, idx in zip(scores[0], indices[0]):

        if idx == -1:
            continue

        results.append({
            "text": chunks[idx]["text"],
            "page": chunks[idx]["page"],
            "similarity": float(score)
        })

    return results
