from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, validator


class Settings(BaseSettings):
    # --- API Keys ---
    gemini_api_key: str
    huggingface_api_key: str

    # --- Vector Store Configuration ---
    vector_store_path: str = "./chroma_db"
    collection_name: str = "ad_documents"

    # --- File Upload Configuration ---
    upload_dir: str = "./uploaded_pdfs"
    max_file_size_mb: int = 50
    allowed_extensions: List[str] = [".pdf"]

    # --- Embedding Model Configuration ---
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # --- Text Splitting Configuration ---
    chunk_size: int = 500
    chunk_overlap: int = 50

    # --- Image Generation Configuration ---
    default_image_width: int = 1024
    default_image_height: int = 1024
    flux_model: str = "black-forest-labs/FLUX.1-schnell"
    flux_steps: int = 4
    flux_guidance_scale: float = 7.5  # ✅ default float

    # --- Gemini Configuration ---
    gemini_model: str = "gemini-1.5-flash"

    # --- API Configuration ---
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_origins: List[str] = ["*"]

    # Pydantic config
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # ✅ Ensure flux_guidance_scale is always float
    @validator("flux_guidance_scale", pre=True)
    def cast_guidance_scale(cls, v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return 0.0


def get_settings() -> Settings:
    """Create and return settings instance."""
    return Settings()


def validate_environment() -> tuple[bool, list[str]]:
    """
    Validate that required environment variables and paths are set.
    Returns:
        (is_valid, errors)
    """
    errors = []
    settings = get_settings()

    # API Keys
    if not settings.gemini_api_key:
        errors.append("GEMINI_API_KEY not set")
    if not settings.huggingface_api_key:
        errors.append("HUGGINGFACE_API_KEY not set")

    # Directories
    import os
    if not os.path.exists(settings.upload_dir):
        try:
            os.makedirs(settings.upload_dir, exist_ok=True)
        except Exception as e:
            errors.append(f"Failed to create upload_dir '{settings.upload_dir}': {e}")

    if not os.path.exists(settings.vector_store_path):
        try:
            os.makedirs(settings.vector_store_path, exist_ok=True)
        except Exception as e:
            errors.append(f"Failed to create vector_store_path '{settings.vector_store_path}': {e}")

    return (len(errors) == 0, errors)
