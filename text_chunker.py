
def create_chunks(
    extracted_pages,
    source="unknown",
    chunk_size=1000,
    overlap=150
):

    chunks = []

    chunk_id = 0

    for page in extracted_pages:

        text = page["text"]
        page_number = page["page"]

        start = 0

        while start < len(text):

            end = start + chunk_size

            chunk_text = text[start:end]

            if chunk_text.strip():

                chunks.append({

                    "text": chunk_text,

                    "metadata": {
                        "source": source,
                        "page": page_number,
                        "chunk_id": chunk_id
                    }

                })

                chunk_id += 1

            start += chunk_size - overlap

    return chunks


if __name__ == "__main__":

    from document_processor import extract_text_from_pdf

    pdf_path = input("Enter PDF path: ")

    pages = extract_text_from_pdf(pdf_path)

    chunks = create_chunks(
        pages,
        source=pdf_path
    )

    print("\nTotal pages:", len(pages))
    print("Total chunks:", len(chunks))

    for i, chunk in enumerate(chunks[:5]):

        print(f"\n--- CHUNK {i+1} ---")

        print("Metadata:", chunk["metadata"])

        print(chunk["text"][:300])
