import os

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ルーターモード: 'logit_router' (Stage 1) または 'ruri_cpu' (Stage 2)
    ROUTER_TYPE: str = os.getenv("ROUTER_TYPE", "logit_router")

    # Stage 1: Logit-Router 設定
    ROUTER_MODEL_ID: str = "Qwen/Qwen2.5-1.5B-Instruct"
    ROUTER_DEVICE: str | None = None  # None の場合は cuda があれば cuda、無ければ cpu
    ROUTER_TORCH_DTYPE: str = "bfloat16"
    LOAD_IN_4BIT: bool = False
    LOAD_IN_8BIT: bool = False

    # 不確実性（Uncertainty）判定閾値
    ENTROPY_THRESHOLD: float = 0.35
    MARGIN_THRESHOLD: float = 0.80

    # バックエンドエンドポイント
    LOCAL_LLM_URL: str = "http://localhost:8000/v1"
    LOCAL_LLM_MODEL: str = "qwen2.5-14b-instruct"
    LOCAL_LLM_API_KEY: str = "EMPTY"

    COMMERCIAL_FAST_URL: str = "https://generativelanguage.googleapis.com/v1beta/openai"
    COMMERCIAL_FAST_MODEL: str = "gemini-1.5-flash"
    GEMINI_API_KEY: str | None = None

    COMMERCIAL_EXPERT_URL: str = "https://api.openai.com/v1"
    COMMERCIAL_EXPERT_MODEL: str = "gpt-4o"
    OPENAI_API_KEY: str | None = None

    # HTTP コネクション設定
    HTTP_TIMEOUT_SECONDS: float = 120.0
    HTTP_CONNECT_TIMEOUT_SECONDS: float = 5.0
    HTTP_MAX_KEEP_ALIVE_CONNECTIONS: int = 100
    HTTP_MAX_CONNECTIONS: int = 200

    # 永続ストレージ
    DB_PATH: str = "data/transactions.db"


settings = Settings()
