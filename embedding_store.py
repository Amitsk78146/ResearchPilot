
import ollama
import numpy as np
import faiss
import re

from rank_bm25 import BM25Okapi
from config import OLLAMA_HOST

# Import scientific terminology expansion
from scientific_terms import expand_scientific_terms


# ==========================================
# INITIALIZE OLLAMA
# ==========================================

client = ollama.Client(host=OLLAMA_HOST)

EMBEDDING_MODEL = "nomic-embed-text"


# ==========================================
# STEP 1: TOKENIZATION FOR BM25
# ==========================================

def tokenize(text):

    # Convert Unicode chemical subscripts
    translation = str.maketrans(
        "₀₁₂₃₄₅₆₇₈₉",
        "0123456789"
    )

    text = text.translate(translation)

    # Preserve chemical formulas and numerical values
    tokens = re.findall(
        r"[a-z0-9]+(?:[./+-][a-z0-9]+)*",
        text.lower()
    )

    return tokens


# ==========================================
# STEP 2: GENERATE EMBEDDINGS
# ==========================================

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


# ==========================================
# STEP 3: CREATE HYBRID DATABASE
# ==========================================

def create_vector_database(chunks):

    if not chunks:
        raise ValueError("No chunks provided.")

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    # --------------------------------------
    # A. Generate original semantic embeddings
    # --------------------------------------

    embeddings = generate_embeddings(texts)

    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]

    faiss_index = faiss.IndexFlatIP(dimension)

    faiss_index.add(embeddings)

    # --------------------------------------
    # B. Scientific terminology expansion
    # --------------------------------------

    expanded_documents = []

    for text in texts:

        expanded_text = expand_scientific_terms(text)

        expanded_documents.append(expanded_text)

    # --------------------------------------
    # C. Build BM25 using expanded documents
    # --------------------------------------

    tokenized_corpus = [
        tokenize(text)
        for text in expanded_documents
    ]

    bm25_index = BM25Okapi(tokenized_corpus)

    print("\nHybrid database created successfully!")

    print("Total FAISS vectors:", faiss_index.ntotal)

    print("Total BM25 documents:", len(tokenized_corpus))

    print("Embedding dimension:", dimension)

    print("Scientific terminology expansion: Enabled")

    return faiss_index, chunks, bm25_index


# ==========================================
# STEP 4: HYBRID SEARCH
# FAISS + BM25 + RRF
# ==========================================

def search_database(
    question,
    index,
    chunks,
    bm25_index,
    top_k=3
):

    if index.ntotal == 0:
        return []

    candidate_k = min(top_k, len(chunks))

    # --------------------------------------
    # A. FAISS SEMANTIC RETRIEVAL
    # --------------------------------------

    # Keep original query for semantic embeddings
    query_embedding = generate_embeddings([question])

    faiss.normalize_L2(query_embedding)

    faiss_scores, faiss_indices = index.search(
        query_embedding,
        candidate_k
    )

    # --------------------------------------
    # B. SCIENTIFIC TERMINOLOGY EXPANSION
    # --------------------------------------

    expanded_question = expand_scientific_terms(
        question
    )

    # Tokenize expanded query for BM25
    query_tokens = tokenize(expanded_question)

    # --------------------------------------
    # C. BM25 KEYWORD RETRIEVAL
    # --------------------------------------

    bm25_scores = bm25_index.get_scores(
        query_tokens
    )

    bm25_indices = np.argsort(
        bm25_scores
    )[::-1][:candidate_k]

    # --------------------------------------
    # D. RECIPROCAL RANK FUSION
    # --------------------------------------

    rrf_scores = {}

    RRF_K = 60

    # Add FAISS rankings
    for rank, idx in enumerate(
        faiss_indices[0],
        start=1
    ):

        if idx == -1:
            continue

        rrf_scores[idx] = (
            rrf_scores.get(idx, 0)
            + 1 / (RRF_K + rank)
        )

    # Add BM25 rankings
    for rank, idx in enumerate(
        bm25_indices,
        start=1
    ):

        rrf_scores[idx] = (
            rrf_scores.get(idx, 0)
            + 1 / (RRF_K + rank)
        )

    # Sort combined results
    ranked_indices = sorted(
        rrf_scores,
        key=rrf_scores.get,
        reverse=True
    )[:top_k]

    # --------------------------------------
    # E. PREPARE FINAL RESULTS
    # --------------------------------------

    results = []

    faiss_score_map = {

        int(idx): float(score)

        for idx, score in zip(
            faiss_indices[0],
            faiss_scores[0]
        )

        if idx != -1
    }

    for idx in ranked_indices:

        chunk = chunks[idx]

        metadata = chunk.get(
            "metadata",
            {}
        )

        results.append({

            "text": chunk["text"],

            "source": metadata.get(
                "source",
                "unknown"
            ),

            "page": metadata.get(
                "page",
                chunk.get("page")
            ),

            "chunk_id": metadata.get(
                "chunk_id",
                idx
            ),

            "similarity": faiss_score_map.get(idx),

            "bm25_score": float(
                bm25_scores[idx]
            ),

            "hybrid_score": float(
                rrf_scores[idx]
            )

        })

    return results
