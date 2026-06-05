"""Claude advisor via the **`claude` CLI** — no API key required.

Why the CLI (not the API)
-------------------------
You don't have an Anthropic API key, but you DO have the `claude` Code CLI authenticated with
your subscription. This module shells out to it headlessly:

    claude -p "<prompt>" --output-format json [--model <model>]

and reads the model's answer out of the JSON envelope's ``result`` field. That means the
trading script can "ask Claude" on demand using your subscription, with zero extra credentials.

WHAT THIS IS — AND IS NOT
-------------------------
* ADVISORY ONLY. The advisor returns a small structured opinion per symbol
  ``{bias in [-1,1], confidence in [0,1], veto: bool, rationale}``. It NEVER places orders and
  has no access to keys. The deterministic engine (EMA / RL + RiskManager + the live-trading
  safety gate) stays fully in charge; the advice is at most a bias nudge or a veto-to-flat.
* SLOW. Each call spawns the CLI and takes ~3-10s and counts against your subscription usage
  limits. This is a minutes/hours cadence "macro" advisor, NOT a per-tick signal source.
* NOT an edge. LLMs do not predict prices. Treat any advice as a soft, optional overlay and
  validate out-of-sample before trusting it with real money.

Every failure mode (CLI missing, timeout, non-zero exit, unparseable output) degrades to
NEUTRAL advice so it can never crash or block the trading loop.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
import threading
import time
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# On Windows the npm shim is `claude.cmd`; the extensionless `claude` is a shell script that
# CreateProcess can't run, so prefer the .cmd. shutil.which honours PATHEXT elsewhere.
CLAUDE_BIN: Optional[str] = (shutil.which("claude.cmd") or shutil.which("claude"))

# Opus per user preference — the strongest read (slower + more subscription usage per call,
# so keep the advisory cadence slow). "opus" is the CLI alias for the latest Opus.
DEFAULT_MODEL = "opus"

# `claude -p` normally runs the full Claude Code *agent* (a coding assistant), so without this
# it replies "how can I help with your project?" instead of trading advice. --system-prompt
# REPLACES that framing with a pure JSON-only advisor persona.
SYSTEM_PROMPT = (
    "You are a terse quantitative market risk advisor embedded in an automated trading system. "
    "You are NOT a chat assistant: never greet, never ask questions, never offer help, never "
    "explain. Read the user's market summary and reply with EXACTLY ONE LINE of strict JSON "
    "matching the requested schema and nothing else."
)

NEUTRAL = {"bias": 0.0, "confidence": 0.0, "veto": False, "rationale": "neutral (no advice)"}


def available() -> bool:
    """True if the `claude` CLI is on PATH (so the advisor can actually run)."""
    return CLAUDE_BIN is not None


# --------------------------------------------------------------------------- #
# Prompt construction — a compact, cheap market summary
# --------------------------------------------------------------------------- #
def _summarize(df: pd.DataFrame, lookback: int = 60) -> dict:
    """Reduce recent OHLCV to a few scalars so the prompt stays small and cheap."""
    tail = df.tail(lookback)
    close = tail["close"].to_numpy(dtype=float)
    rets = np.diff(close) / close[:-1] if len(close) > 1 else np.array([0.0])
    last = float(close[-1])
    return {
        "bars": int(len(tail)),
        "last_price": round(last, 6),
        "pct_change_window": round(float(close[-1] / close[0] - 1.0) * 100, 3) if len(close) else 0.0,
        "pct_change_recent10": round(float(close[-1] / close[-min(10, len(close))] - 1.0) * 100, 3),
        "ann_vol_pct": round(float(np.std(rets) * 100), 4),
        "max": round(float(tail["high"].max()), 6),
        "min": round(float(tail["low"].min()), 6),
        "pos_in_range": round(float((last - tail["low"].min()) /
                                    max(1e-9, tail["high"].max() - tail["low"].min())), 3),
    }


def _build_prompt(symbol: str, df: pd.DataFrame, timeframe: str) -> str:
    s = _summarize(df)
    return (
        "You are a conservative crypto trading risk advisor. Based ONLY on the price summary "
        "below, give a SHORT directional read for the next few bars. You are advisory only — a "
        "deterministic engine makes the actual trades.\n\n"
        f"Symbol: {symbol}  Timeframe: {timeframe}\n"
        f"Recent summary (last {s['bars']} bars): {json.dumps(s)}\n\n"
        "Respond with ONLY a single line of strict JSON, no prose, no code fences:\n"
        '{"bias": <float -1..1, negative=short positive=long>, '
        '"confidence": <float 0..1>, '
        '"veto": <true if conditions look too risky to trade, else false>, '
        '"rationale": "<=15 words"}'
    )


# --------------------------------------------------------------------------- #
# CLI invocation + parsing
# --------------------------------------------------------------------------- #
def _extract_json_obj(text: str) -> Optional[dict]:
    """Pull the first {...} JSON object out of `text` (tolerates stray prose/code fences)."""
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def ask_claude(prompt: str, *, timeout: float = 60.0,
               model: str = DEFAULT_MODEL) -> Optional[dict]:
    """Run one headless `claude -p` call and return the parsed JSON answer (or None)."""
    if not available():
        return None
    cmd = [CLAUDE_BIN, "-p", prompt, "--output-format", "json",
           "--system-prompt", SYSTEM_PROMPT, "--exclude-dynamic-system-prompt-sections"]
    if model:
        cmd += ["--model", model]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        logger.warning("claude CLI timed out after %.0fs", timeout)
        return None
    except Exception as e:  # noqa: BLE001 - never let the advisor crash the caller
        logger.warning("claude CLI invocation failed: %s", e)
        return None
    if proc.returncode != 0:
        logger.warning("claude CLI rc=%s: %s", proc.returncode, (proc.stderr or "")[:200])
        return None
    try:
        envelope = json.loads(proc.stdout)
        result_text = envelope.get("result", "") if isinstance(envelope, dict) else ""
    except Exception:
        result_text = proc.stdout  # some versions print bare text
    return _extract_json_obj(result_text)


def get_advice(symbol: str, df: pd.DataFrame, *, timeframe: str = "1m",
               model: str = DEFAULT_MODEL, timeout: float = 60.0) -> dict:
    """Ask Claude for an advisory read on `symbol`. Always returns a valid, clamped dict;
    degrades to NEUTRAL on any failure so the trading loop is never blocked."""
    if not available() or df is None or len(df) < 20:
        return dict(NEUTRAL)
    raw = ask_claude(_build_prompt(symbol, df, timeframe), timeout=timeout, model=model)
    if not isinstance(raw, dict):
        return dict(NEUTRAL)
    try:
        return {
            "bias": float(np.clip(float(raw.get("bias", 0.0)), -1.0, 1.0)),
            "confidence": float(np.clip(float(raw.get("confidence", 0.0)), 0.0, 1.0)),
            "veto": bool(raw.get("veto", False)),
            "rationale": str(raw.get("rationale", ""))[:300],
        }
    except (TypeError, ValueError):
        return dict(NEUTRAL)


# --------------------------------------------------------------------------- #
# Background advisory loop + a Strategy wrapper that lets the advice gate trades
# --------------------------------------------------------------------------- #
class ClaudeAdvisor:
    """SLOW background loop that periodically refreshes a per-symbol advisory read and stores
    it. The CLI call is subprocess I/O (it blocks on a child process, NOT on the GIL), so this
    runs happily on a thread without starving the dashboard. Default cadence is intentionally
    slow (minutes) because each call costs latency + subscription usage. Default OFF."""

    def __init__(self, symbols, fetch_df, event_bus=None, *, interval: float = 300.0,
                 timeframe: str = "1m", model: str = DEFAULT_MODEL, timeout: float = 60.0):
        self.symbols = list(symbols)
        self.fetch_df = fetch_df              # callable(symbol) -> DataFrame | None
        self.event_bus = event_bus
        self.interval = float(interval)
        self.timeframe = timeframe
        self.model = model
        self.timeout = timeout
        self.advice = {s: dict(NEUTRAL) for s in self.symbols}
        self._stop = threading.Event()
        self._thread = None

    def bias(self, symbol: str) -> float:
        return float(self.advice.get(symbol, NEUTRAL)["bias"])

    def veto(self, symbol: str) -> bool:
        return bool(self.advice.get(symbol, NEUTRAL)["veto"])

    def start(self):
        if not available():
            logger.warning("ClaudeAdvisor: `claude` CLI not found on PATH; advisor disabled.")
            return
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="claude-advisor")
        self._thread.start()
        logger.info("ClaudeAdvisor started: %d symbols, model=%s, every %.0fs (advisory only).",
                    len(self.symbols), self.model, self.interval)

    def stop(self):
        self._stop.set()

    def _loop(self):
        while not self._stop.is_set():
            for sym in self.symbols:
                if self._stop.is_set():
                    break
                try:
                    df = self.fetch_df(sym)
                    adv = get_advice(sym, df, timeframe=self.timeframe, model=self.model,
                                     timeout=self.timeout)
                    self.advice[sym] = adv
                    logger.info("[advisor] %s: bias=%+.2f conf=%.2f veto=%s — %s", sym,
                                adv["bias"], adv["confidence"], adv["veto"], adv["rationale"])
                    if self.event_bus is not None:
                        try:
                            self.event_bus.publish("advice", {"symbol": sym, **adv})
                        except Exception:
                            pass
                except Exception as e:  # noqa: BLE001 - advisory must never crash the app
                    logger.warning("[advisor] %s failed: %s", sym, e)
            self._stop.wait(self.interval)


class AdvisedStrategy:
    """Wraps any Strategy and lets the Claude advice GATE its signals (advisory overlay):
      * veto -> force flat (0) everywhere,
      * otherwise drop entries that fight a confident directional bias (long-only when bias is
        firmly positive, short-only when firmly negative).
    It never invents new trades — it can only suppress the base strategy's, so the deterministic
    engine + RiskManager remain authoritative."""

    name = "advised"

    def __init__(self, base, advisor: ClaudeAdvisor, symbol: str, *, conf_gate: float = 0.5):
        self.base = base
        self.advisor = advisor
        self.symbol = symbol
        self.conf_gate = conf_gate

    def generate_signals(self, df):
        sig = self.base.generate_signals(df)
        adv = self.advisor.advice.get(self.symbol, NEUTRAL)
        if adv.get("veto"):
            return sig * 0
        bias, conf = adv.get("bias", 0.0), adv.get("confidence", 0.0)
        if conf >= self.conf_gate:
            if bias > 0.33:      # confident long view -> suppress shorts
                sig = sig.clip(lower=0)
            elif bias < -0.33:   # confident short view -> suppress longs
                sig = sig.clip(upper=0)
        return sig
