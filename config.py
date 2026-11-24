import os
from dotenv import load_dotenv

load_dotenv()

PROJECT_NAME = os.getenv("PROJECT_NAME")
APP_ENV = os.getenv("APP_ENV")

print("Project:", PROJECT_NAME)
print("Environment:", APP_ENV)