import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_BASE_URL: str = os.getenv(
        "OPENAI_BASE_URL",
        "https://generativelanguage.googleapis.com/v1beta/openai/",
    )
    MODEL_NAME: str = os.getenv("MODEL_NAME", "gemini-3.5-flash-lite")

settings = Settings()

print("API key loaded:", bool(settings.OPENAI_API_KEY))
print("Base URL:", settings.OPENAI_BASE_URL)
print("Model:", settings.MODEL_NAME)