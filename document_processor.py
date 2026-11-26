
import fitz
import os


def extract_text_from_pdf(pdf_path):

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(
            f"PDF not found: {pdf_path}"
        )

    document = fitz.open(pdf_path)

    extracted_pages = []

    for page_number, page in enumerate(document):

        text = page.get_text("text")

        extracted_pages.append({
            "page": page_number + 1,
            "text": text
        })

    document.close()

    return extracted_pages


def display_pdf_text(pdf_path):

    pages = extract_text_from_pdf(pdf_path)

    for page in pages:

        print(f"\n--- PAGE {page['page']} ---\n")

        print(page["text"])


if __name__ == "__main__":

    pdf_path = input(
        "Enter the PDF file path: "
    )

    display_pdf_text(pdf_path)
