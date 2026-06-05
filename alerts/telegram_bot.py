"""Thin Telegram Bot API client.

Just enough to push a text message to a chat via the Bot API. The bot does not
need to *receive* messages, so there is no polling/webhook machinery here — only
`send_message`. Keep this module dependency-light (stdlib + `requests`).

Get a token from @BotFather and your chat id from @userinfobot, then either pass
them in or export TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID and run this file directly
for a smoke test.
"""

import logging

import requests

logger = logging.getLogger()

# Telegram caps message text at 4096 chars; leave a little headroom.
_MAX_LEN = 4000
_TIMEOUT = 10  # seconds


class TelegramBot:
    """Minimal Telegram Bot API client for outbound notifications."""

    def __init__(self, token: str, chat_id: str) -> None:
        self.token = token or ""
        self.chat_id = chat_id or ""

    @property
    def _url(self) -> str:
        return f"https://api.telegram.org/bot{self.token}/sendMessage"

    def send_message(self, text: str) -> bool:
        """Post `text` to the configured chat. Returns True on success.

        Never raises: network/API errors are logged and reported as False so a
        failed alert can never take down the trading loop.
        """
        if not self.token or not self.chat_id:
            logger.warning("TelegramBot: missing token/chat_id; cannot send message")
            return False

        if len(text) > _MAX_LEN:
            text = text[: _MAX_LEN - 3] + "..."

        payload = {"chat_id": self.chat_id, "text": text}
        try:
            resp = requests.post(self._url, json=payload, timeout=_TIMEOUT)
        except requests.RequestException as exc:
            logger.error("TelegramBot: request failed: %s", exc)
            return False

        if resp.status_code != 200:
            logger.error(
                "TelegramBot: API returned %s: %s", resp.status_code, resp.text[:200]
            )
            return False

        try:
            ok = bool(resp.json().get("ok", False))
        except ValueError:
            logger.error("TelegramBot: non-JSON response: %s", resp.text[:200])
            return False

        if not ok:
            logger.error("TelegramBot: API reported not-ok: %s", resp.text[:200])
        return ok


if __name__ == "__main__":
    # Smoke test: only fires if both env vars are present. Safe to run blind.
    import os

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s :: %(message)s"
    )
    _token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    _chat = os.getenv("TELEGRAM_CHAT_ID", "")
    if not _token or not _chat:
        logger.warning(
            "Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID to run the smoke test."
        )
    else:
        bot = TelegramBot(_token, _chat)
        if bot.send_message("TradingBot Telegram smoke test - ignore."):
            logger.info("Test message sent successfully.")
        else:
            logger.error("Failed to send test message.")
