
import ollama

from config import (
    OLLAMA_MODEL,
    OLLAMA_HOST,
    TEMPERATURE
)

# Initialise the local Ollama client
client = ollama.Client(host=OLLAMA_HOST)


def generate_text(
    prompt,
    system_prompt=None,
    max_tokens=800,
    num_ctx=8192
):
    """
    Generate text using the configured local LLM.

    This interface will later support switching
    between Ollama and the Colab Transformers backend.
    """

    messages = []

    if system_prompt:
        messages.append({
            "role": "system",
            "content": system_prompt
        })

    messages.append({
        "role": "user",
        "content": prompt
    })

    response = client.chat(
        model=OLLAMA_MODEL,
        messages=messages,
        options={
            "temperature": TEMPERATURE,
            "num_predict": max_tokens,
            "num_ctx": num_ctx
        }
    )

    return response["message"]["content"].strip()
