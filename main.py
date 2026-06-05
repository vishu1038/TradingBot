import logging

from config import CONFIG

from connectors.binance_futures import BinanceFuturesClient
from connectors.bitmex import BitmexClient

from data.database import Database
from strategies.ema_cross import EmaCrossStrategy
from risk.risk_manager import RiskManager, RiskLimits
from engine.trading_engine import TradingEngine
from alerts.notifier import build_default_notifier
from promotion.evaluator import PromotionEvaluator

from interface.root_component import Root

logger = logging.getLogger()

logger.setLevel(logging.INFO)

stream_handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s %(levelname)s :: %(message)s")

stream_handler.setFormatter(formatter)
stream_handler.setLevel(logging.INFO)

file_handler = logging.FileHandler('info.log')
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.DEBUG)

logger.addHandler(stream_handler)
logger.addHandler(file_handler)


if __name__ == "__main__":
    # Connectors — credentials come from CONFIG (env / .env), never hard-coded.
    binance = None
    bitmex = None

    try:
        binance = BinanceFuturesClient(CONFIG.binance_public_key,
                                       CONFIG.binance_secret_key,
                                       CONFIG.use_testnet)
    except Exception as e:
        logger.error("Failed to initialize Binance client: %s", e)

    try:
        bitmex = BitmexClient(CONFIG.bitmex_public_key,
                              CONFIG.bitmex_secret_key,
                              CONFIG.use_testnet)
    except Exception as e:
        logger.error("Failed to initialize Bitmex client: %s", e)

    # Build a paper-mode trading engine on the first available connector.
    engine = None
    try:
        database = Database(CONFIG.database_path)
        strategy = EmaCrossStrategy()
        risk_manager = RiskManager(RiskLimits(), starting_equity=10_000)
        notifier = build_default_notifier(CONFIG)        # Telegram if configured, else console
        evaluator = PromotionEvaluator()                 # "ready for real money?" gate
        active_client = binance or bitmex
        if active_client is not None:
            engine = TradingEngine(
                client=active_client,
                strategy=strategy,
                risk_manager=risk_manager,
                database=database,
                symbol="BTCUSDT",
                timeframe="1h",
                mode="paper",
                initial_capital=10_000,
                notifier=notifier,
                evaluator=evaluator,
            )
        else:
            logger.error("No exchange connector available; engine not constructed.")
    except Exception as e:
        logger.error("Failed to construct TradingEngine: %s", e)

    root = Root(binance, bitmex, engine=engine)
    root.mainloop()
