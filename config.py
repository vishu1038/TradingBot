"""Central configuration. Loads settings from environment / .env file.

Usage:
    from config import CONFIG
    client = BinanceFuturesClient(CONFIG.binance_public_key, CONFIG.binance_secret_key,
                                  CONFIG.use_testnet)

Credentials are intentionally NOT hard-coded (see main.py history). Copy .env.example
to .env and fill in your testnet keys.
"""

import os
from dataclasses import dataclass

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # python-dotenv not installed; rely on real environment variables.
    pass


def _as_bool(value: str, default: bool = True) -> bool:
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class Config:
    binance_public_key: str = os.getenv("BINANCE_PUBLIC_KEY", "")
    binance_secret_key: str = os.getenv("BINANCE_SECRET_KEY", "")
    bitmex_public_key: str = os.getenv("BITMEX_PUBLIC_KEY", "")
    bitmex_secret_key: str = os.getenv("BITMEX_SECRET_KEY", "")

    use_testnet: bool = _as_bool(os.getenv("USE_TESTNET"), default=True)

    database_path: str = os.getenv("DATABASE_PATH", "data/tradingbot.db")
    model_dir: str = os.getenv("MODEL_DIR", "models")

    telegram_bot_token: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    telegram_chat_id: str = os.getenv("TELEGRAM_CHAT_ID", "")


CONFIG = Config()
