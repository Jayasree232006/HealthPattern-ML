import os
from functools import lru_cache
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Locate and load backend/.env if present, as well as root .env
_current_file = Path(__file__).resolve()
_backend_dir = _current_file.parent.parent
_backend_env = _backend_dir / ".env"
_repo_root_env = _backend_dir.parent / ".env"

if _backend_env.exists():
    # Allow .env to override empty string or unset environment variables
    # (e.g. if uvicorn was started with GROQ_API_KEY="" in the environment)
    load_dotenv(dotenv_path=_backend_env, override=True)
elif _repo_root_env.exists():
    load_dotenv(dotenv_path=_repo_root_env, override=True)
else:
    load_dotenv(override=False)


class Settings(BaseModel):
    allowed_origins: list[str] = Field(
        default_factory=lambda: [
            origin.strip()
            for origin in os.getenv(
                "ALLOWED_ORIGINS",
                "http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000,http://127.0.0.1:3000",
            ).split(",")
            if origin.strip()
        ]
    )
    api_v1_prefix: str = "/api/v1"
    project_name: str = "Rare & Common Disease Detector API"
    version: str = "0.1.0"
    model_artifacts_dir: Optional[str] = Field(
        default_factory=lambda: os.getenv("MODEL_ARTIFACTS_DIR", None)
    )

    # Groq AI Service Configuration
    groq_api_key: Optional[str] = Field(
        default_factory=lambda: os.getenv("GROQ_API_KEY", "").strip() or None
    )
    groq_model: str = Field(
        default_factory=lambda: os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile").strip()
    )
    groq_timeout_seconds: float = Field(
        default_factory=lambda: float(os.getenv("GROQ_TIMEOUT_SECONDS", "10.0"))
    )
    groq_max_retries: int = Field(
        default_factory=lambda: int(os.getenv("GROQ_MAX_RETRIES", "2"))
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
