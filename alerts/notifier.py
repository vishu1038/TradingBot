"""Notifications — how the bot tells a human what it just did.

Defines a small `Notifier` interface with a console-only implementation (always
works, zero config), a Telegram-backed implementation, and a fan-out wrapper.
Sending an alert must never crash the trading loop, so every implementation
swallows its own errors and returns a success bool instead of raising.

Use `build_default_notifier(CONFIG)` to get the right one for the current config:
Telegram if a token + chat id are present, otherwise console. The result is
always wrapped so callers can fire-and-forget.

The convenience helpers (`notify_fill`, `notify_daily_summary`, `notify_risk_halt`,
`notify_promotion`) format the handful of events the bot actually cares about.
"""

import logging

from alerts.telegram_bot import TelegramBot

logger = logging.getLogger()

# Map a notification level to a log level + a small emoji prefix for chat clients.
_LEVELS = {
    "info": (logging.INFO, "ℹ️"),
    "success": (logging.INFO, "✅"),
    "warning": (logging.WARNING, "⚠️"),
    "error": (logging.ERROR, "🚨"),
    "critical": (logging.CRITICAL, "🚨"),
}


class Notifier:
    """Base notifier interface. Subclasses implement `send`."""

    def send(self, message: str, level: str = "info") -> bool:
        """Deliver `message`. Returns True if delivered. Never raises."""
        raise NotImplementedError


class ConsoleNotifier(Notifier):
    """Logs the message. Always available, needs no configuration."""

    def send(self, message: str, level: str = "info") -> bool:
        log_level, prefix = _LEVELS.get(level, (logging.INFO, "ℹ️"))
        logger.log(log_level, "%s %s", prefix, message)
        return True


class TelegramNotifier(Notifier):
    """Sends messages to a Telegram chat. Degrades to a no-op (logged) when the
    token/chat id are missing rather than raising."""

    def __init__(self, token: str, chat_id: str) -> None:
        self.token = token or ""
        self.chat_id = chat_id or ""
        self._bot = TelegramBot(self.token, self.chat_id)

    def send(self, message: str, level: str = "info") -> bool:
        _, prefix = _LEVELS.get(level, (logging.INFO, "ℹ️"))
        if not self.token or not self.chat_id:
            logger.warning(
                "TelegramNotifier not configured (missing token/chat_id); "
                "dropping message: %s",
                message,
            )
            return False
        try:
            return self._bot.send_message(f"{prefix} {message}")
        except Exception as exc:  # defensive: alerts must never crash the caller
            logger.error("TelegramNotifier.send failed: %s", exc)
            return False


class MultiNotifier(Notifier):
    """Fans a message out to several notifiers. Returns True if *any* succeeds;
    one failing channel never blocks the others and never raises."""

    def __init__(self, notifiers: list[Notifier]) -> None:
        self.notifiers = list(notifiers)

    def send(self, message: str, level: str = "info") -> bool:
        delivered = False
        for notifier in self.notifiers:
            try:
                delivered = notifier.send(message, level) or delivered
            except Exception as exc:  # defensive: keep fanning out
                logger.error("MultiNotifier: %r failed: %s", notifier, exc)
        return delivered


def build_default_notifier(config) -> Notifier:
    """Pick a notifier based on config.

    Returns a Telegram notifier (with console as a fallback channel) when a
    token + chat id are configured, otherwise a console-only notifier. The result
    is always a `MultiNotifier`, so the caller can fire-and-forget without ever
    crashing on a missing/broken channel.
    """
    token = getattr(config, "telegram_bot_token", "") or ""
    chat_id = getattr(config, "telegram_chat_id", "") or ""
    console = ConsoleNotifier()
    if token and chat_id:
        return MultiNotifier([TelegramNotifier(token, chat_id), console])
    logger.info("Telegram not configured; using console notifier only.")
    return MultiNotifier([console])


# --- convenience helpers for the events the bot cares about -------------------


def notify_fill(notifier: Notifier, trade: dict) -> bool:
    """Announce an order fill. `trade` duck-types on side/symbol/qty/price."""
    side = str(trade.get("side", "?")).upper()
    symbol = trade.get("symbol", "?")
    qty = trade.get("qty", trade.get("quantity", "?"))
    price = trade.get("price", trade.get("fill_price", "?"))
    pnl = trade.get("pnl")
    msg = f"FILL {side} {qty} {symbol} @ {price}"
    if pnl is not None:
        msg += f" | PnL {pnl}"
    return notifier.send(msg, level="info")


def notify_daily_summary(notifier: Notifier, metrics: dict) -> bool:
    """Daily roll-up from a `summarize()`-style metrics dict."""
    lines = [
        "Daily summary",
        f"  Return:    {metrics.get('total_return_pct', '?')}%",
        f"  Equity:    {metrics.get('final_equity', '?')}",
        f"  Sharpe:    {metrics.get('sharpe', '?')}",
        f"  Max DD:    {metrics.get('max_drawdown_pct', '?')}%",
        f"  Trades:    {metrics.get('trades', '?')}",
        f"  Win rate:  {metrics.get('win_rate_pct', '?')}%",
    ]
    return notifier.send("\n".join(lines), level="info")


def notify_risk_halt(notifier: Notifier, reason: str) -> bool:
    """Loud alert that the risk manager flattened/halted trading."""
    return notifier.send(f"RISK HALT - trading stopped: {reason}", level="critical")


def notify_promotion(notifier: Notifier, verdict: dict) -> bool:
    """The headline event: the bot judged itself ready to risk real money.

    `verdict` duck-types on the PromotionVerdict shape (ready/score/reasons/
    metrics/recommendation). When ready, sends a prominent "READY FOR LIVE"
    banner with the supporting metrics and reasons; when not ready, sends a
    lower-key status update so progress is still visible.
    """
    ready = bool(verdict.get("ready", False))
    score = verdict.get("score")
    metrics = verdict.get("metrics", {}) or {}
    reasons = verdict.get("reasons", []) or []
    recommendation = verdict.get("recommendation", "")

    score_txt = f"{score:.0%}" if isinstance(score, (int, float)) else str(score)

    if ready:
        header = "🎓 READY FOR LIVE - all graduation criteria met"
    else:
        header = f"📈 Promotion check: not yet ready (score {score_txt})"

    lines = [header, ""]
    if metrics:
        lines.append("Metrics:")
        for key in (
            "total_return_pct",
            "sharpe",
            "max_drawdown_pct",
            "win_rate_pct",
            "profit_factor",
            "trades",
            "days_running",
        ):
            if key in metrics:
                lines.append(f"  {key}: {metrics[key]}")
        lines.append("")
    if reasons:
        lines.append("Reasons:")
        lines.extend(f"  - {r}" for r in reasons)
        lines.append("")
    if recommendation:
        lines.append(recommendation)
    if ready:
        lines.append("")
        lines.append(
            "NOTE: paper success != live success. Start with a reduced-size "
            "live trial before scaling up."
        )

    level = "success" if ready else "info"
    return notifier.send("\n".join(lines).rstrip(), level=level)
