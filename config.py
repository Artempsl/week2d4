"""Configuration management for news summarizer."""
import os
from dotenv import load_dotenv

# Load variables from .env (does NOT override system env variables by default)
load_dotenv()

class Config:
    """Application configuration."""

    # ==============================
    # API Keys
    # ==============================

    # OpenAI key comes from Windows global environment variable
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

    # Cohere and NewsAPI keys come from .env
    COHERE_API_KEY = os.getenv("COHERE_API_KEY")
    NEWS_API_KEY = os.getenv("NEWS_API_KEY")

    # ==============================
    # Environment
    # ==============================

    ENVIRONMENT = os.getenv("ENVIRONMENT", "development")

    # ==============================
    # API Configuration
    # ==============================

    MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
    REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "30"))

    # ==============================
    # Models
    # ==============================

    OPENAI_MODEL = "gpt-4o-mini"
    COHERE_MODEL = "command-a-03-2025"

    # ==============================
    # Cost Control
    # ==============================

    DAILY_BUDGET = float(os.getenv("DAILY_BUDGET", "5.00"))

    # ==============================
    # Rate Limits (requests per minute)
    # ==============================

    OPENAI_RPM = 500
    COHERE_RPM = 100
    NEWS_API_RPM = 100

    @classmethod
    def validate(cls):
        """Validate that required configuration is present."""
        required = [
            ("OPENAI_API_KEY (Windows env)", cls.OPENAI_API_KEY),
            ("COHERE_API_KEY (.env)", cls.COHERE_API_KEY),
            ("NEWS_API_KEY (.env)", cls.NEWS_API_KEY),
        ]

        missing = [name for name, value in required if not value]

        if missing:
            raise ValueError(f"Missing required configuration: {', '.join(missing)}")

        print(f"✓ Configuration validated for {cls.ENVIRONMENT} environment")


# Validate on import
Config.validate()
