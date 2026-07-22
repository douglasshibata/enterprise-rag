from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    opensearch_host: str = "localhost"
    opensearch_port: int = 9200
    opensearch_index: str = "enterprise_docs"
    opensearch_user: str = "admin"
    opensearch_pass: str = "admin"
    api_key: str = "default-secret-key"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()
