from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    PROJECT_NAME: str = "Shifaa IA"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    
    # AI APIs
    GROQ_API_KEY: str = ""
    OPENAI_API_KEY: str = ""
    
    # Model configs
    GROQ_MODEL: str = "llama-3.3-70b-versatile" # Fast & capable Llama 3 70B
    OPENAI_MODEL: str = "gpt-4-turbo-preview"
    
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

settings = Settings()
