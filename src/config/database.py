from functools import lru_cache

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

class DbConfig(BaseSettings):
    POSTGRES_USER: str
    POSTGRES_PASSWORD: SecretStr
    POSTGRES_DB: str
    POSTGRES_HOST: str
    POSTGRES_HOST_PORT: int
    POSTGRES_CONNECT_TIMEOUT: int

    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', extra='ignore')

@lru_cache
def get_settings():
    return DbConfig()

connection_config = {
    "dbname": get_settings().POSTGRES_DB,
    "user": get_settings().POSTGRES_USER,
    "password": get_settings().POSTGRES_PASSWORD.get_secret_value(),
    "host": get_settings().POSTGRES_HOST,
    "port": get_settings().POSTGRES_HOST_PORT,
    "connect_timeout": get_settings().POSTGRES_CONNECT_TIMEOUT
}
