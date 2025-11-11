from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


# Resolve repository root:
# settings.py -> config -> cv_search -> src -> <repo root>
REPO_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    """Centralized configuration shared across CLI, Streamlit, and services."""

    model_config = SettingsConfigDict(
        env_file=REPO_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    use_azure_openai: bool = Field(
        default=False,
        description="Set to True to use Azure OpenAI, False for standard OpenAI.",
    )

    openai_api_key: Optional[SecretStr] = None

    azure_endpoint: Optional[str] = Field(default=None)
    azure_api_version: Optional[str] = Field(default=None)

    openai_model: str = Field(default="gpt-4.1-mini")
    openai_embed_model: str = Field(default="text-embedding-3-large")

    search_mode: str = "hybrid"
    search_vs_topk: int = 8
    search_w_lex: float = 1.0
    search_w_sem: float = 0.8

    db_path: Path = Field(default_factory=lambda: REPO_ROOT / "data" / "db" / "cvsearch.db")
    data_dir: Path = Field(default_factory=lambda: REPO_ROOT / "data")
    lexicon_dir: Path = Field(default_factory=lambda: REPO_ROOT / "data" / "lexicons")
    schema_file: Path = Field(default_factory=lambda: REPO_ROOT / "src" / "cv_search" / "db" / "schema.sql")

    faiss_index_path: Path = Field(default_factory=lambda: REPO_ROOT / "data" / "cv_search.faiss")
    faiss_doc_map_path: Path = Field(default_factory=lambda: REPO_ROOT / "data" / "cv_search_docs.json")

    gdrive_rclone_config_path: Optional[Path] = Field(default=None)
    gdrive_remote_name: str = Field(default="gdrive")
    gdrive_source_dir: str = Field(default="CV_Inbox")
    gdrive_local_dest_dir: Path = Field(default_factory=lambda: REPO_ROOT / "data" / "gdrive_inbox")

    @property
    def openai_api_key_str(self) -> str | None:
        return self.openai_api_key.get_secret_value() if self.openai_api_key else None
