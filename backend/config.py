import os

from dotenv import load_dotenv


load_dotenv()


class Settings:
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_BASE_URL: str = os.getenv(
        "OPENAI_BASE_URL",
        "https://ws-ptbx8vsgapbf8qeg.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1",
    )
    MODEL_NAME: str = os.getenv(
        "MODEL_NAME",
        "qwen3.8-flash",
    )


settings = Settings()


print("API key loaded:", bool(settings.OPENAI_API_KEY))
print("Base URL:", settings.OPENAI_BASE_URL)
print("Model:", settings.MODEL_NAME)