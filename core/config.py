# ╔════════════════════════════════════════════════════════════╗
# ║ 🚚 IMPORTS
# ╚════════════════════════════════════════════════════════════╝

from typing import ClassVar, List
from pydantic_settings import BaseSettings, SettingsConfigDict

from pathlib import Path


# ╔════════════════════════════════════════════════════════════╗
# ║ ⚙️ CONFIG
# ╚════════════════════════════════════════════════════════════╝
class Settings(BaseSettings):

    # ═════════════════════ APP ═════════════════════
    APP_NAME: str = "Review Sentiment Analysis"
    DESCRIPTION: str = ""
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:8000",
        "http://localhost:127.0.0.1",
    ]

    # ═════════════════════  PATH ═════════════════════

    BASE_DIR: ClassVar[Path] = Path(__file__).resolve().parents[1]
    BASELINE_PATH: ClassVar[Path] = BASE_DIR / "models" / "cnn_tokenization.keras"
    REVIEWS_DATA_URL: str = ""

    # ═════════════════════ CONFIG ═════════════════════
    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        extra="ignore",
        case_sensitive=False,
    )


settings = Settings()
