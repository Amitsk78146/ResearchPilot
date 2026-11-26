
import os
from dotenv import load_dotenv

load_dotenv()

# Project configuration
PROJECT_NAME = "ResearchPilot"
APP_ENV = os.getenv("APP_ENV", "development")

# Local AI configuration
OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "qwen2.5:3b"
)

OLLAMA_HOST = os.getenv(
    "OLLAMA_HOST",
    "http://localhost:11434"
)

TEMPERATURE = 0.3

print("Project:", PROJECT_NAME)
print("Environment:", APP_ENV)
print("AI Model:", OLLAMA_MODEL)



