
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


# Step 1: Process research paper
def load_research_paper(pdf_path):

    pages = extract_text_from_pdf(pdf_path)

    chunks = create_chunks(pages)

    index, stored_chunks = create_vector_database(chunks)

    return index, stored_chunks


# Step 2: Retrieve relevant information
def retrieve_information(question, index, chunks):

    results = search_database(
        question,
        index,
        chunks,
        top_k=10
    )

    return results


# Step 3: Generate an evidence-based answer
def generate_answer(question, results):

    if not results:
        return "No relevant information was retrieved."

    context = ""

    for i, result in enumerate(results):

        context += (
            f"\n[Source {i+1}, Page {result['page']}]\n"
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
                3. Cite source pages using [Page N].
                4. If the evidence is insufficient,
                   explicitly say so.
                5. Distinguish reported findings
                   from your interpretation.
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


# Step 4: Run ResearchPilot
if __name__ == "__main__":

    pdf_path = input(
        "Enter research paper path: "
    )

    print("\nLoading research paper...")

    index, chunks = load_research_paper(pdf_path)

    print("Research paper processed.")
    print("Total chunks:", len(chunks))

    while True:

        question = input("\nYou: ")

        if question.lower() == "exit":
            break

        results = retrieve_information(
            question,
            index,
            chunks
        )

        print("\nResearchPilot is analysing...\n")

        answer = generate_answer(
            question,
            results
        )

        print("RESEARCHPILOT:\n")
        print(answer)

        print("\nRETRIEVED SOURCES:")

        for result in results:

            print(
                f"Page {result['page']} | "
                f"Similarity: {result['similarity']:.3f}"
            )
