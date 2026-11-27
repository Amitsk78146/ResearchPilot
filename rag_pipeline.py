
from document_processor import extract_text_from_pdf
from text_chunker import create_chunks

from embedding_store import (
    create_vector_database,
    search_database
)

from config import (
    OLLAMA_MODEL,
    OLLAMA_HOST,
    TEMPERATURE
)

import ollama


# Initialize Ollama
client = ollama.Client(host=OLLAMA_HOST)


# STEP 1: Process research paper
def load_research_paper(pdf_path):

    print("\nExtracting PDF text...")

    pages = extract_text_from_pdf(pdf_path)

    print("Creating text chunks...")

    chunks = create_chunks(
        pages,
        source=pdf_path
    )

    print("Generating embeddings and building hybrid database...")

    index, stored_chunks, bm25_index = create_vector_database(chunks)

    return index, stored_chunks, bm25_index


# STEP 2: Hybrid retrieval using FAISS + BM25 + RRF
def retrieve_information(
    question,
    index,
    chunks,
    bm25_index
):

    print("\nSearching research paper...")

    results = search_database(
        question,
        index,
        chunks,
        bm25_index,
        top_k=10
    )

    return results


# STEP 3: Generate evidence-based answer
def generate_answer(question, results):

    if not results:
        return "No relevant information was retrieved."

    context = ""

    for i, result in enumerate(results):

        context += (
            f"\n[Source: {result['source']}, "
            f"Page: {result['page']}, "
            f"Chunk: {result['chunk_id']}]\n"
            f"{result['text']}\n"
        )

    response = client.chat(

        model=OLLAMA_MODEL,

        messages=[
            {
                "role": "system",
                "content": """
                You are ResearchPilot, a scientific
                research assistant.

                Answer using ONLY the supplied
                research excerpts.

                Instructions:

                1. Do not invent experimental results.
                2. Do not introduce unsupported facts.
                3. Cite sources using their PDF filename
                   and page number.
                4. Use citations in this format:
                   [Source: filename, Page N].
                5. If evidence is insufficient,
                   explicitly say so.
                6. Distinguish reported findings
                   from your interpretation.
                7. Never fabricate a citation.
                """
            },

            {
                "role": "user",
                "content": f"""
                RESEARCH QUESTION:

                {question}

                RETRIEVED RESEARCH EXCERPTS:

                {context}

                Answer the question using these excerpts.
                """
            }
        ],

        options={
            "temperature": TEMPERATURE
        }
    )

    return response["message"]["content"]


# STEP 4: Display retrieved sources
def display_sources(results):

    print("\n" + "=" * 60)
    print("HYBRID RETRIEVAL RESULTS")
    print("=" * 60)

    for i, result in enumerate(results):

        print(f"\n--- RESULT {i+1} ---")

        print("Source:", result["source"])
        print("Page:", result["page"])
        print("Chunk ID:", result["chunk_id"])

        print("FAISS similarity:", result["similarity"])

        print("BM25 score:", f"{result['bm25_score']:.3f}")

        print("Hybrid score:", f"{result['hybrid_score']:.5f}")

        print("\nRetrieved text:")
        print(result["text"][:500])


# STEP 5: Run ResearchPilot
if __name__ == "__main__":

    pdf_path = input(
        "Enter research paper path: "
    ).strip().strip('"')

    print("\nLoading research paper...")

    index, chunks, bm25_index = load_research_paper(pdf_path)

    print("\nResearch paper processed successfully.")

    print("Total chunks:", len(chunks))
    print("Total FAISS vectors:", index.ntotal)

    print("\nResearchPilot is ready!")
    print("Type 'exit' to stop.")

    while True:

        question = input("\nYou: ").strip()

        if question.lower() == "exit":
            print("\nResearchPilot session ended.")
            break

        if not question:
            continue

        # Hybrid retrieval
        results = retrieve_information(
            question,
            index,
            chunks,
            bm25_index
        )

        # Generate answer directly from hybrid results
        print("\nResearchPilot is analysing...\n")

        answer = generate_answer(
            question,
            results
        )

        print("\n" + "=" * 60)
        print("RESEARCHPILOT ANSWER")
        print("=" * 60)

        print(answer)

        # Display retrieved sources
        display_sources(results)
