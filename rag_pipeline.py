
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

from src.reranker import BGEreranker

import ollama


# ==========================================
# INITIALIZE OLLAMA
# ==========================================

client = ollama.Client(host=OLLAMA_HOST)



# ==========================================
# STEP 1: PROCESS MULTIPLE RESEARCH PAPERS
# ==========================================

def load_research_papers(pdf_paths):

    all_chunks = []

    for paper_number, pdf_path in enumerate(pdf_paths, 1):

        print("\n" + "=" * 60)
        print(f"PROCESSING PAPER {paper_number}")
        print("=" * 60)

        print("PDF:", pdf_path)

        print("\nExtracting PDF text...")

        pages = extract_text_from_pdf(pdf_path)

        print("Creating text chunks...")

        chunks = create_chunks(
            pages,
            source=pdf_path
        )

        print("Chunks created:", len(chunks))

        all_chunks.extend(chunks)

    if not all_chunks:
        raise ValueError("No research paper chunks were created.")

    print("\n" + "=" * 60)
    print("BUILDING MULTI-PAPER DATABASE")
    print("=" * 60)

    index, stored_chunks, bm25_index = create_vector_database(
        all_chunks
    )

    print("\nTotal research papers:", len(pdf_paths))
    print("Total chunks:", len(stored_chunks))
    print("Total FAISS vectors:", index.ntotal)

    return index, stored_chunks, bm25_index



# ==========================================
# STEP 2: HYBRID RETRIEVAL + BGE RERANKING
# ==========================================

def retrieve_information(
    question,
    index,
    chunks,
    bm25_index,
    reranker
):

    print("\nSearching using FAISS + BM25 + RRF...")

    # Retrieve top 20 hybrid candidates

    candidates = search_database(
        question,
        index,
        chunks,
        bm25_index,
        top_k=20
    )

    print("Hybrid candidates retrieved:", len(candidates))

    if not candidates:
        return []

    # Apply BGE Cross-Encoder

    print("\nApplying BGE Cross-Encoder reranking...")

    reranked_results = reranker.rerank(
        question,
        candidates,
        top_k=5
    )

    # Preserve original metadata and retrieval scores

    final_results = []

    for item in reranked_results:

        result = item["document"].copy()

        result["reranker_score"] = item["score"]

        final_results.append(result)

    print("\nBGE reranking completed!")

    print("Final relevant chunks:", len(final_results))

    return final_results


# ==========================================
# STEP 3: GENERATE EVIDENCE-BASED ANSWER
# ==========================================

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


# ==========================================
# STEP 4: DISPLAY RETRIEVED SOURCES
# ==========================================

def display_sources(results):

    print("\n" + "=" * 65)

    print("HYBRID RETRIEVAL + BGE RERANKING RESULTS")

    print("=" * 65)

    for i, result in enumerate(results, start=1):

        print(f"\n--- RESULT {i} ---")

        print("Source:", result["source"])

        print("Page:", result["page"])

        print("Chunk ID:", result["chunk_id"])

        print(
            "FAISS similarity:",
            result["similarity"]
        )

        print(
            "BM25 score:",
            f"{result['bm25_score']:.5f}"
        )

        print(
            "Hybrid score:",
            f"{result['hybrid_score']:.5f}"
        )

        print(
            "BGE relevance score:",
            f"{result['reranker_score']:.5f}"
        )

        print("\nRetrieved text:")

        print(result["text"][:500])



# ==========================================
# STEP 5: RUN MULTI-PAPER RESEARCHPILOT
# ==========================================

if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("RESEARCHPILOT - MULTI-PAPER MODE")
    print("=" * 60)

    print("\nEnter PDF paths separated by semicolons (;).")

    pdf_input = input("\nPDF paths: ").strip()

    pdf_paths = [
        path.strip().strip('"')
        for path in pdf_input.split(";")
        if path.strip()
    ]

    if not pdf_paths:
        raise ValueError("No PDF paths provided.")

    print("\nLoading research papers...")

    index, chunks, bm25_index = load_research_papers(
        pdf_paths
    )

    print("\nInitializing BGE Reranker...")

    reranker = BGEreranker()

    print("\nResearchPilot is ready!")

    print("Total papers:", len(pdf_paths))
    print("Total chunks:", len(chunks))

    print("\nType 'exit' to stop.")

    while True:

        question = input("\nYou: ").strip()

        if question.lower() == "exit":

            print("\nResearchPilot session ended.")
            break

        if not question:
            continue

        results = retrieve_information(
            question,
            index,
            chunks,
            bm25_index,
            reranker
        )

        print("\nResearchPilot is analysing...\n")

        answer = generate_answer(
            question,
            results
        )

        print("\n" + "=" * 60)
        print("RESEARCHPILOT ANSWER")
        print("=" * 60)

        print(answer)

        display_sources(results)

