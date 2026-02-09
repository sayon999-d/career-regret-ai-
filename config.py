import os
from typing import Optional, List
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    HOST: str = "127.0.0.1"
    PORT: int = 8000
    DEBUG: bool = os.getenv("DEBUG", "true").lower() == "true"

    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3.2"
    OLLAMA_TIMEOUT: int = 120

    DATABASE_URL: str = "sqlite+aiosqlite:///./career_regret.db"

    MODEL_PATH: str = "./models"
    ML_LEARNING_RATE: float = 0.001
    ENSEMBLE_DL_WEIGHT: float = 0.7
    ENSEMBLE_ML_WEIGHT: float = 0.3
    DECAY_FACTOR: float = 0.95
    TEMPORAL_DECAY: float = 0.99
    MONTE_CARLO_SIMULATIONS: int = 1000
    FEEDBACK_BATCH_SIZE: int = 10
    FEEDBACK_WEIGHT: float = 0.3
    ENABLE_AB_TESTING: bool = True

    CHROMA_PERSIST_DIR: str = "./chroma_db"
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

    JWT_SECRET_KEY: Optional[str] = None
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    RATE_LIMIT_RPM: int = 30
    RATE_LIMIT_RPH: int = 500
    RATE_LIMIT_BURST: int = 5

    CORS_ORIGINS: Optional[str] = "http://localhost:8000,http://127.0.0.1:8000,http://localhost:3000"

    IP_BLACKLIST: Optional[str] = None
    IP_WHITELIST: str = "127.0.0.1,::1"
    TRUSTED_PROXIES: str = "127.0.0.1"

    MAX_LOGIN_ATTEMPTS: int = 5
    LOCKOUT_DURATION_MINUTES: int = 15

    MAX_REQUEST_SIZE_MB: int = 1

    ENABLE_API_KEYS: bool = False

    ADMIN_PASSWORD: Optional[str] = None

    CACHE_MAX_SIZE: int = 500
    CACHE_TTL: int = 300

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

settings = Settings()

def validate_production_settings():
    warnings = []
    errors = []

    if not settings.DEBUG:
        if not settings.JWT_SECRET_KEY:
            errors.append("JWT_SECRET_KEY must be set in production")
        elif len(settings.JWT_SECRET_KEY) < 32:
            errors.append("JWT_SECRET_KEY must be at least 32 characters")

        if not settings.CORS_ORIGINS:
            warnings.append("CORS_ORIGINS not set - API may be inaccessible from browsers")

        if not settings.ADMIN_PASSWORD:
            warnings.append("ADMIN_PASSWORD not set - using auto-generated password")

    return errors, warnings

_errors, _warnings = validate_production_settings()
for w in _warnings:
    print(f"WARNING: {w}")
for e in _errors:
    print(f"ERROR: {e}")
