import os
from pathlib import Path
from dotenv import load_dotenv

# Search for .env relative to config.py location (backend/.env or root/.env)
_current_dir = Path(__file__).resolve().parent
_backend_env = _current_dir / ".env"
_root_env = _current_dir.parent / ".env"

if _backend_env.exists():
    load_dotenv(dotenv_path=_backend_env)
elif _root_env.exists():
    load_dotenv(dotenv_path=_root_env)
else:
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
