import os

from pydantic_settings import BaseSettings


def to_boolean(value: str) -> bool:
    """Convert a string value to a boolean.

    Args:
        value (str): The string to convert.

    Returns:
        bool: True if the value is one of ["yes", "true", "y", "1"],
            False otherwise.
    """

    if value.lower() in ["yes", "true", "y", "1"]:
        return True
    return False


class LangchainSettings(BaseSettings):
    """Configurations related to Langchain"""
    LANGCHAIN_PROJECT: str = os.getenv("LANGCHAIN_PROJECT")
    LANGCHAIN_API_KEY: str = os.getenv("LANGCHAIN_API_KEY")
    LANGCHAIN_TRACING: bool = to_boolean(
        os.getenv("LANGCHAIN_TRACING", "false")
    )
    LANGCHAIN_ENDPOINT: str = os.getenv(
        "LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com"
    )


class Settings(LangchainSettings):
    """Main configurations inheriting from other settings."""
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_USE_BASIC_FORMAT: bool = to_boolean(
        os.getenv("LOG_USE_BASIC_FORMAT", "false")
    )


config = Settings()
