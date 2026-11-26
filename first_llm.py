
import ollama

from config import (
    OLLAMA_MODEL,
    OLLAMA_HOST,
    TEMPERATURE
)

client = ollama.Client(host=OLLAMA_HOST)

SYSTEM_PROMPT = """
You are ResearchPilot, an AI scientific research assistant.

Explain scientific concepts clearly.
Use structured scientific explanations.
Never invent research papers or experimental results.
Maintain context from previous questions.
"""

# Store conversation history
messages = [
    {
        "role": "system",
        "content": SYSTEM_PROMPT
    }
]


def ask_researchpilot(question):

    messages.append({
        "role": "user",
        "content": question
    })

    response = client.chat(
        model=OLLAMA_MODEL,
        messages=messages,
        options={
            "temperature": TEMPERATURE
        }
    )

    answer = response["message"]["content"]

    messages.append({
        "role": "assistant",
        "content": answer
    })

    return answer


if __name__ == "__main__":

    print("=== RESEARCHPILOT ===")
    print("Type 'exit' to quit.\n")

    while True:

        question = input("You: ")

        if question.lower() == "exit":
            break

        answer = ask_researchpilot(question)

        print("\nResearchPilot:", answer, "\n")
