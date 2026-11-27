
import re
import unicodedata


# ==========================================
# 1. SCIENTIFIC TERMINOLOGY DICTIONARY
# ==========================================

SCIENTIFIC_TERMS = {

    "dry reforming of methane": [
        "drm",
        "dry reforming",
        "methane dry reforming"
    ],

    "carbon dioxide": [
        "co2",
        "carbon dioxide"
    ],

    "methane": [
        "ch4",
        "methane gas"
    ],

    "nickel": [
        "ni",
        "nickel metal"
    ],

    "cerium oxide": [
        "ceria",
        "ceo2",
        "cerium dioxide"
    ],

    "magnesium oxide": [
        "mgo",
        "magnesia"
    ],

    "aluminium oxide": [
        "alumina",
        "al2o3"
    ],

    "carbon deposition": [
        "coking",
        "carbon formation",
        "carbon accumulation",
        "carbon deposits"
    ],

    "catalyst deactivation": [
        "catalyst deterioration",
        "catalyst degradation",
        "loss of catalytic activity"
    ],

    "sintering": [
        "particle growth",
        "metal particle agglomeration"
    ],

    "synthesis gas": [
        "syngas",
        "hydrogen and carbon monoxide"
    ],

    "hydrogen": [
        "h2",
        "molecular hydrogen"
    ],

    "carbon monoxide": [
        "co",
        "molecular carbon monoxide"
    ],

    "catalyst stability": [
        "catalytic stability",
        "long-term stability",
        "catalyst durability"
    ]
}


# ==========================================
# 2. NORMALIZE SCIENTIFIC TEXT
# ==========================================

def normalize_scientific_text(text):

    if not isinstance(text, str):
        raise TypeError("Input must be a string.")

    # Normalize Unicode characters
    text = unicodedata.normalize("NFKC", text)

    # Convert Greek symbols to readable names
    greek_symbols = {
        "α": "alpha",
        "β": "beta",
        "γ": "gamma",
        "Δ": "delta",
        "δ": "delta",
        "μ": "micro",
        "λ": "lambda"
    }

    for symbol, replacement in greek_symbols.items():
        text = text.replace(symbol, replacement)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ==========================================
# 3. EXPAND SCIENTIFIC TERMINOLOGY
# ==========================================

def expand_scientific_terms(text):

    normalized_text = normalize_scientific_text(text)

    expanded_terms = [normalized_text]

    text_lower = normalized_text.lower()

    for canonical, synonyms in SCIENTIFIC_TERMS.items():

        all_terms = [canonical] + synonyms

        # Check whether any term is present
        for term in all_terms:

            pattern = (
                r"(?<![a-z0-9])"
                + re.escape(term.lower())
                + r"(?![a-z0-9])"
            )

            if re.search(pattern, text_lower):

                expanded_terms.extend(all_terms)

                break

    # Remove duplicate terms
    expanded_terms = list(dict.fromkeys(expanded_terms))

    return " ".join(expanded_terms)
