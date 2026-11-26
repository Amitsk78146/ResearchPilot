
from document_processor import extract_text_from_pdf
from text_chunker import create_chunks

from embedding_store import (
    create_vector_database,
    search_database
)

# Load research paper
pdf_path = "Data/Paper/EJ1172284.pdf"

pages = extract_text_from_pdf(pdf_path)

# Create chunks
chunks = create_chunks(pages)

print("Total chunks:", len(chunks))

# Generate embeddings and create FAISS index
index, stored_chunks = create_vector_database(chunks)

print("Vector database created successfully!")

# Ask a research question
question = input("\nEnter your research question: ")

results = search_database(
    question,
    index,
    stored_chunks,
    top_k=3
)

print("\nMOST RELEVANT RESEARCH SECTIONS:\n")

for result in results:

    print("Page:", result["page"])
    print("Similarity:", round(result["similarity"], 4))
    print(result["text"][:500])
    print("-" * 60)
