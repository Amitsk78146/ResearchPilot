
import os
import re
import json
import hashlib
import ollama

from document_processor import extract_text_from_pdf
from config import (
    OLLAMA_MODEL,
    OLLAMA_HOST,
    TEMPERATURE
)

client = ollama.Client(host=OLLAMA_HOST)

# ==============================
# CONFIGURATION
# ==============================

CHUNK_SIZE = 8000
CACHE_DIR = "Summaries/cache"
OUTPUT_DIR = "Summaries"

NUM_CTX = 8192

ANALYSIS_TOKENS = 800
FINAL_TOKENS = 2500


# ==============================
# SYSTEM PROMPT
# ==============================

SYSTEM_PROMPT = """
You are ResearchPilot, an AI scientific research assistant
specialising in chemical engineering and catalysis.

Analyse scientific research papers accurately.

Rules:
1. Never invent experimental results or numerical values.
2. Preserve units and chemical formulas.
3. Distinguish experimental observations from interpretations.
4. Retain original PDF page references.
5. State when information is not reported.
6. Distinguish catalyst activity from stability.
7. Focus on catalyst deactivation and carbon deposition.
"""


ANALYSIS_PROMPT = """
Analyse this scientific research paper extract.

Extract the following information:

1. Research objective
2. Problem statement
3. Catalyst composition and materials
4. Experimental methodology
5. Reaction conditions
6. Major findings
7. Catalyst activity and stability
8. Carbon deposition and deactivation mechanisms
9. Limitations and future research scope

Instructions:
- Use only supplied evidence.
- Preserve numerical values and units.
- Include PDF page references.
- Do not assume missing information.
- Be concise but scientifically accurate.

Paper: {paper_name}

Extract:
{paper_text}
"""


FINAL_PROMPT = """
Prepare a structured scientific summary of this research paper.

Paper: {paper_name}

Use the following structure:

# 1. Paper Identification
- Title
- Authors
- Publication year
- Research area

# 2. Research Objective

# 3. Problem Statement

# 4. Catalyst Composition and Materials
- Active metals
- Support
- Promoters
- Preparation method

# 5. Experimental Methodology

# 6. Reaction Conditions
- Temperature
- Pressure
- CH4/CO2 ratio
- Feed composition
- Gas flow rate or space velocity
- Reaction duration

# 7. Major Findings

# 8. Catalyst Activity and Stability
- CH4 conversion
- CO2 conversion
- H2/CO ratio
- Time-on-stream performance
- Stability

# 9. Carbon Deposition and Deactivation
- Carbon formation
- Carbon characterisation
- Sintering
- Deactivation mechanisms
- Regeneration

# 10. Limitations and Future Research Scope

# 11. Key Research Takeaways
Focus on improving DRM catalyst stability.

# 12. Source References
List original PDF page numbers.

Rules:
- Never invent information.
- Preserve numerical values and units.
- Use "Not reported in the paper" when information is absent.
- Retain page references beside findings.
- Distinguish reported facts from interpretations.

Evidence:
{evidence}
"""


# ==============================
# LLM FUNCTION
# ==============================

def ask_llm(prompt, max_tokens=ANALYSIS_TOKENS):

    response = client.chat(
        model=OLLAMA_MODEL,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        options={
            "temperature": TEMPERATURE,
            "num_predict": max_tokens,
            "num_ctx": NUM_CTX
        }
    )

    return response["message"]["content"].strip()


# ==============================
# CACHE MANAGEMENT
# ==============================

def get_cache_path(paper_name, chunk_index, text):

    content_hash = hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()[:16]

    safe_name = re.sub(
        r"[^a-zA-Z0-9_-]",
        "_",
        os.path.splitext(paper_name)[0]
    )

    filename = (
        f"{safe_name}_chunk_{chunk_index}_"
        f"{content_hash}.json"
    )

    return os.path.join(CACHE_DIR, filename)


def load_cache(cache_path):

    if not os.path.exists(cache_path):
        return None

    try:
        with open(
            cache_path,
            "r",
            encoding="utf-8"
        ) as file:
            return json.load(file)

    except (json.JSONDecodeError, OSError):
        return None


def save_cache(cache_path, evidence):

    os.makedirs(
        CACHE_DIR,
        exist_ok=True
    )

    with open(
        cache_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            {"evidence": evidence},
            file,
            ensure_ascii=False,
            indent=2
        )


# ==============================
# PDF CHUNKING
# ==============================

def split_pages_into_chunks(
    pages,
    chunk_size=CHUNK_SIZE
):

    chunks = []

    current_text = ""
    current_pages = []

    for page in pages:

        page_text = page["text"].strip()
        page_number = page["page"]

        if not page_text:
            continue

        # Split long pages while preserving page number
        for start in range(
            0,
            len(page_text),
            chunk_size
        ):

            section = page_text[
                start:start + chunk_size
            ]

            labelled_section = (
                f"\n[PDF PAGE {page_number}]\n"
                f"{section}"
            )

            if (
                current_text
                and len(current_text) + len(labelled_section)
                > chunk_size
            ):

                chunks.append({
                    "text": current_text,
                    "pages": sorted(set(current_pages))
                })

                current_text = ""
                current_pages = []

            current_text += labelled_section
            current_pages.append(page_number)

    if current_text:

        chunks.append({
            "text": current_text,
            "pages": sorted(set(current_pages))
        })

    return chunks


# ==============================
# CHUNK ANALYSIS
# ==============================

def analyse_chunk(
    chunk,
    paper_name,
    chunk_index
):

    cache_path = get_cache_path(
        paper_name,
        chunk_index,
        chunk["text"]
    )

    # Reuse previously analysed chunks
    cached = load_cache(cache_path)

    if cached is not None:

        print(
            f"Using cached chunk {chunk_index + 1}"
        )

        return cached["evidence"]

    prompt = ANALYSIS_PROMPT.format(
        paper_name=paper_name,
        paper_text=chunk["text"]
    )

    evidence = ask_llm(
        prompt,
        max_tokens=ANALYSIS_TOKENS
    )

    save_cache(
        cache_path,
        evidence
    )

    return evidence


# ==============================
# EVIDENCE CONSOLIDATION
# ==============================

def consolidate_evidence(
    evidence_list,
    paper_name
):

    # One call is sufficient for a manageable evidence set.
    evidence = "\n\n".join(evidence_list)

    if not evidence.strip():
        raise ValueError("No research evidence extracted.")

    prompt = f"""
Consolidate the following scientific evidence
from the same research paper.

Paper: {paper_name}

Requirements:
- Remove repetitive information.
- Preserve experimental values and units.
- Retain original PDF page references.
- Preserve conflicting findings with their sources.
- Do not invent missing information.
- Keep all nine research parameters represented.

Evidence:

{evidence}
"""

    return ask_llm(
        prompt,
        max_tokens=1800
    )


# ==============================
# PAPER SUMMARISATION
# ==============================

def summarize_research_paper(
    pdf_path,
    output_dir=OUTPUT_DIR
):

    if not os.path.exists(pdf_path):

        raise FileNotFoundError(
            f"Paper not found: {pdf_path}"
        )

    paper_name = os.path.basename(pdf_path)

    print("\n" + "=" * 60)
    print("ANALYSING:", paper_name)
    print("=" * 60)

    # Step 1: Extract PDF text
    pages = extract_text_from_pdf(pdf_path)

    print("Total pages:", len(pages))

    # Step 2: Create chunks
    chunks = split_pages_into_chunks(pages)

    print("Total text chunks:", len(chunks))

    if not chunks:

        raise ValueError(
            "No extractable text found."
        )

    # Step 3: Analyse chunks
    evidence_list = []

    for index, chunk in enumerate(chunks):

        print(
            f"\nAnalysing chunk "
            f"{index + 1}/{len(chunks)} "
            f"(Pages: {chunk['pages']})..."
        )

        evidence = analyse_chunk(
            chunk,
            paper_name,
            index
        )

        evidence_list.append(evidence)

    # Step 4: Consolidate evidence
    print("\nConsolidating research evidence...")

    consolidated_evidence = consolidate_evidence(
        evidence_list,
        paper_name
    )

    # Step 5: Final summary
    print("\nGenerating final scientific summary...")

    final_prompt = FINAL_PROMPT.format(
        paper_name=paper_name,
        evidence=consolidated_evidence
    )

    final_summary = ask_llm(
        final_prompt,
        max_tokens=FINAL_TOKENS
    )

    # Step 6: Save summary
    os.makedirs(
        output_dir,
        exist_ok=True
    )

    output_name = os.path.splitext(
        paper_name
    )[0]

    output_path = os.path.join(
        output_dir,
        output_name + "_summary.md"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            f"# ResearchPilot Summary\n\n"
            f"**Source PDF:** {paper_name}\n\n"
            f"**Model:** {OLLAMA_MODEL}\n\n"
            f"---\n\n"
        )

        file.write(final_summary)

    print("\nSummary completed!")
    print("Saved to:", output_path)

    return final_summary


# ==============================
# SUMMARISE ALL PAPERS
# ==============================

def summarize_all_papers(
    paper_directory="Data/Paper"
):

    if not os.path.isdir(paper_directory):

        raise FileNotFoundError(
            f"Paper directory not found: {paper_directory}"
        )

    pdf_files = sorted(
        file
        for file in os.listdir(paper_directory)
        if file.lower().endswith(".pdf")
    )

    if not pdf_files:

        print("No PDF files found.")
        return

    print("\nResearchPilot Automatic Summarisation")
    print("Total papers:", len(pdf_files))

    successful = 0
    failed = []

    for pdf_file in pdf_files:

        pdf_path = os.path.join(
            paper_directory,
            pdf_file
        )

        try:

            summarize_research_paper(pdf_path)
            successful += 1

        except Exception as error:

            print(
                f"\nERROR processing "
                f"{pdf_file}: {error}"
            )

            failed.append(pdf_file)

    print("\n" + "=" * 60)
    print("SUMMARISATION REPORT")
    print("=" * 60)

    print("Total papers:", len(pdf_files))
    print("Successful:", successful)
    print("Failed:", len(failed))

    if failed:
        print("Failed papers:", failed)

    print("Output directory: Summaries/")


# ==============================
# MAIN
# ==============================

if __name__ == "__main__":

    summarize_all_papers()
