from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class ChunkerConfig(BaseSettings):
    DOCS_REPO_DIR: Path = Path("data/fastapi")
    DOCS_SUBDIR: str = "docs/en/docs"
    DOCS_EXCLUDE: list[str] = ["release-notes.md", "reference/", "_llm-test.md", "translation-banner.md", "agent-banner.md"]

    EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"
    EMBEDDING_MAX_TOKENS: int = Field(default=512, gt=0)
    CHUNK_MAX_TOKENS: int = Field(default=400, gt=0)

    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', extra='ignore')

    @property
    def docs_dir(self) -> Path:
        return self.DOCS_REPO_DIR / self.DOCS_SUBDIR

@lru_cache
def get_settings():
    return ChunkerConfig()
