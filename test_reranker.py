
from src.reranker import BGEreranker

reranker = BGEreranker()

query = "Which catalysts improve stability in dry reforming of methane?"

documents = [
    "Nickel catalysts supported on ceria show improved resistance to carbon deposition.",
    "Solar photovoltaic cells convert sunlight into electrical energy.",
    "Dry reforming of methane produces synthesis gas using carbon dioxide and methane.",
    "Catalyst deactivation can occur due to sintering and carbon formation."
]

results = reranker.rerank(
    query,
    documents,
    top_k=3
)

for i, result in enumerate(results, 1):
    print(f"\nRank {i}")
    print("Score:", result["score"])
    print("Document:", result["document"])
