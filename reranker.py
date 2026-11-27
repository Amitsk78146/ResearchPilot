
from sentence_transformers import CrossEncoder


# Load reranking model
RERANKER_MODEL = "BAAI/bge-reranker-base"

print("Loading reranking model...")

reranker = CrossEncoder(
    RERANKER_MODEL,
    max_length=512,
    device="cpu"
)

print("Reranking model loaded successfully!")


# Rerank retrieved research chunks
def rerank_results(question, results, top_k=5):

    if not results:
        return []

    # Create question-passage pairs
    pairs = [
        [question, result["text"]]
        for result in results
    ]

    # Calculate relevance scores
    scores = reranker.predict(
        pairs,
        show_progress_bar=False
    )

    # Attach scores to results
    for result, score in zip(results, scores):

        result["rerank_score"] = float(score)

    # Sort by reranking score
    reranked_results = sorted(
        results,
        key=lambda x: x["rerank_score"],
        reverse=True
    )

    # Return only the most relevant chunks
    return reranked_results[:top_k]
