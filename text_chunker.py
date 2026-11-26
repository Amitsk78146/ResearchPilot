
def create_chunks(
    extracted_pages,
    chunk_size=1000,
    overlap=150
):

    chunks = []

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
                    "page": page_number
                })

            start += chunk_size - overlap

    return chunks


if __name__ == "__main__":

    from document_processor import extract_text_from_pdf

    pdf_path = input("Enter PDF path: ")

    pages = extract_text_from_pdf(pdf_path)

    chunks = create_chunks(pages)

    print("\nTotal pages:", len(pages))
    print("Total chunks:", len(chunks))

    for i, chunk in enumerate(chunks[:5]):

        print(f"\n--- CHUNK {i+1} ---")
        print("Page:", chunk["page"])
        print(chunk["text"][:300])
