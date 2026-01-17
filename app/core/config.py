from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    PROJECT_NAME: str = "FastAPI Workshop"
    ENV: str = "development"
    LOG_LEVEL: str = "INFO"
    MODEL_NAME: str = "sentence-transformers/all-mpnet-base-v2"
    vectorDBPath: str = "app/utils/vectorDB"
    product_data_path: str = "data/product_catalog_real.csv"
    OPENAI_MODEL_NAME: str = "gpt-4o-mini"
    GROQ_MODEL_NAME: str = "llama-3.3-70b-versatile"
    TOOL_CHOICE: str = "auto"
    MAX_TOKENS: int = 1024
    TEMPERATURE: float = 0.7
    GROQ_API_KEY: str = Field(..., env="GROQ_API_KEY")
    OPENAI_API_KEY: str = Field(..., env="OPENAI_API_KEY")

    class Config:
        env_file = ".env"

settings = Settings()
