from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict

class LLMConfig(BaseSettings):
    LLM_BASE_URL: str
    LLM_MODEL_API: str
    LLM_TIMEOUT: int

    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', extra='ignore')

@lru_cache
def get_settings():
    return LLMConfig()
