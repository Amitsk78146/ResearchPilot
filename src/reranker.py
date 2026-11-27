
from sentence_transformers import CrossEncoder


class BGEreranker:

    def __init__(self):

        print("Loading BGE Cross-Encoder...")

        self.model = CrossEncoder(
            "BAAI/bge-reranker-base"
        )

        print("BGE Reranker loaded successfully!")

    def rerank(self, query, documents, top_k=5):

        if not documents:
            return []

        # Support both strings and retrieval dictionaries
        texts = [
            doc["text"] if isinstance(doc, dict) else doc
            for doc in documents
        ]

        # Create query-document pairs
        pairs = [
            [query, text]
            for text in texts
        ]

        # Calculate relevance scores
        scores = self.model.predict(pairs)

        # Preserve original documents and metadata
        ranked_results = sorted(
            zip(documents, scores),
            key=lambda x: float(x[1]),
            reverse=True
        )

        return [
            {
                "document": doc,
                "score": float(score)
            }
            for doc, score in ranked_results[:top_k]
        ]
