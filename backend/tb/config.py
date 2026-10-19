"""Settings, paths, .env loading and mode selection.

Keys are read only from the environment / ``.env`` and are never logged: ``Secrets.__repr__``
masks them and the dataclass fields are excluded from repr.
"""
from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]

MODES = ("paper_local", "binance_demo", "live")

PROD_BASE = "https://fapi.binance.com"
DEMO_BASE = "https://demo-fapi.binance.com"
TESTNET_BASE = "https://testnet.binancefuture.com"


# ----------------------------------------------------------------------------- paths
def var_dir() -> Path:
    p = Path(os.environ.get("TB_VAR") or ROOT / "var")
    p.mkdir(parents=True, exist_ok=True)
    return p


def db_path() -> Path:
    env = os.environ.get("TB_DB")
    if env:
        p = Path(env)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p
    return var_dir() / "tb.sqlite"


def models_dir() -> Path:
    return Path(os.environ.get("TB_MODELS") or ROOT / "models")


def ui_dir() -> Path:
    return Path(os.environ.get("TB_UI") or ROOT / "ui")


def archive_dir() -> Path:
    return Path(os.environ.get("TB_ARCHIVE") or ROOT / "data" / "archive")


# ----------------------------------------------------------------------------- env
_ENV_LOADED = False


def load_env(force: bool = False) -> None:
    """Load ``.env`` (repo root, then backend/) and ``config.json`` without overriding real env vars."""
    global _ENV_LOADED
    if _ENV_LOADED and not force:
        return
    try:
        from dotenv import load_dotenv

        for p in (ROOT / ".env", ROOT / "backend" / ".env"):
            if p.exists():
                load_dotenv(p, override=False)
    except Exception:  # dotenv missing is not fatal; env vars may be set directly
        pass
    _load_config_json(ROOT / "config.json")
    _ENV_LOADED = True


def _load_config_json(path: Path) -> None:
    """``{"binance": {"api_key", "api_secret", "env"}}`` -> BINANCE_* env vars (gitignored, mode 600)."""
    if not path.exists():
        return
    try:
        import json

        b = json.loads(path.read_text()).get("binance", {})
    except Exception:  # a broken file must not leak its content into a traceback
        print(f"config: {path.name} is not valid JSON; ignored")
        return
    for var, key in (("BINANCE_API_KEY", "api_key"), ("BINANCE_API_SECRET", "api_secret"),
                     ("BINANCE_ENV", "env")):
        val = str(b.get(key, "")).strip()
        if val and not os.environ.get(var):
            os.environ[var] = val


def _mask(s: str) -> str:
    if not s:
        return "<unset>"
    return s[:3] + "***" + f"({len(s)})"


@dataclass(frozen=True)
class Secrets:
    api_key: str = field(default="", repr=False)
    api_secret: str = field(default="", repr=False)
    chat_webhook: str = field(default="", repr=False)   # Google Chat incoming webhook (contains key+token)

    def __repr__(self) -> str:  # never print the real values
        return (f"Secrets(api_key={_mask(self.api_key)}, api_secret={_mask(self.api_secret)}, "
                f"chat_webhook={_mask(self.chat_webhook)})")

    __str__ = __repr__

    @property
    def has_binance(self) -> bool:
        return bool(self.api_key and self.api_secret)


def load_secrets(env: Mapping[str, str] | None = None) -> Secrets:
    load_env()
    e = env if env is not None else os.environ
    return Secrets(
        api_key=e.get("BINANCE_API_KEY", "").strip(),
        api_secret=e.get("BINANCE_API_SECRET", "").strip(),
        chat_webhook=e.get("GOOGLE_CHAT_WEBHOOK_URL", "").strip(),
    )


# ----------------------------------------------------------------------------- modes
class LiveNotAllowed(RuntimeError):
    pass


def resolve_mode(requested: str | None = None, *, live_flag: bool = False,
                 env: Mapping[str, str] | None = None) -> str:
    """Pick the trading mode. ``live`` needs ALLOW_LIVE_TRADING=true AND an explicit flag."""
    load_env()
    e = env if env is not None else os.environ
    mode = (requested or e.get("TB_MODE") or "paper_local").strip().lower()
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")
    if mode == "live":
        allow = e.get("ALLOW_LIVE_TRADING", "").strip().lower() == "true"
        flag = live_flag or e.get("TB_CONFIRM_LIVE", "").strip().lower() == "true"
        if not (allow and flag):
            raise LiveNotAllowed(
                "live mode needs env ALLOW_LIVE_TRADING=true AND the explicit confirm flag "
                "(--confirm-live or TB_CONFIRM_LIVE=true)")
    return mode


def trading_base_url(mode: str, env: Mapping[str, str] | None = None) -> str:
    e = env if env is not None else os.environ
    if mode == "live":
        return PROD_BASE
    if mode == "binance_demo":
        if e.get("BINANCE_ENV", "").strip().lower() == "testnet":
            return TESTNET_BASE
        return e.get("BINANCE_DEMO_BASE", "").strip() or DEMO_BASE
    return PROD_BASE


# ----------------------------------------------------------------------------- settings
@dataclass(frozen=True)
class TrailCfg:
    enabled: bool = True                # profit protection (2026-10-19): trail on by default, after break-even
    activate_r: float = 1.5
    distance_atr: float = 1.0


@dataclass(frozen=True)
class CorrCap:
    max_corr: float = 0.8
    max_correlated: int = 2


# keys exposed through the API contract (GET/PUT /api/settings)
CONTRACT_KEYS = (
    "slice_margin_usd", "leverage", "max_positions", "max_leverage", "max_total_leverage", "max_margin_ratio",
    "max_open_risk_pct", "exit_mode", "tp_usd", "sl_usd", "tp_atr",
    "sl_atr", "trail", "short_enabled", "prob_threshold_long", "prob_threshold_short", "corr_cap",
    "universe_size", "universe_refresh_s", "universe_whitelist", "kill_dd_pct", "dd_basis", "fee_taker", "sound", "capital_usd",
    "decision_every_bars", "cooldown_bars", "max_trades_per_day", "entry_mode", "maker_timeout_s",
    "maker_reprices", "maker_on_timeout",
    "breakeven_r", "sl_cooldown_bars", "max_side_risk_pct", "partial_tp_r", "partial_tp_frac",
)


@dataclass(frozen=True)
class Settings:
    slice_margin_usd: float = 20.0
    leverage: int = 8
    max_positions: int | None = None    # optional hard cap on concurrent trades (null = governed by the rules below)
    max_leverage: int = 20              # per-trade leverage is chosen automatically, never above this (1..125)
    max_total_leverage: float = 20.0    # sum(open notional) / bot equity, including a new entry
    max_margin_ratio: float = 0.50      # sum(maintenance margin) / bot equity, including a new entry
    max_open_risk_pct: float = 20.0     # sum(loss if every SL fills, fees+slippage incl.) / bot equity, in %
    exit_mode: str = "fixed"            # fixed | atr
    tp_usd: float = 5.0
    sl_usd: float = 10.0
    tp_atr: float = 1.5
    sl_atr: float = 1.0
    trail: TrailCfg = TrailCfg()
    short_enabled: bool = False
    prob_threshold_long: float = 0.70
    prob_threshold_short: float = 0.80
    corr_cap: CorrCap = CorrCap()
    universe_size: int = 8
    universe_refresh_s: int = 3600
    universe_whitelist: list[str] | None = None   # null = hot-list ranking; else exactly these coins (∩ symbols TRADING on the order venue)
    kill_dd_pct: float = 50.0           # paper default; live is capped at LIVE_KILL_DD_PCT
    dd_basis: str = "start"             # drawdown for kill/derating vs the starting stake ("start") or the high-water mark ("peak")
    fee_taker: float = 0.0005
    sound: bool = True
    capital_usd: float | None = 100.0   # bot stake in binance_demo/live (null = whole wallet); see tb.capital
    # ---- entry controls (V3). Defaults for every model; a champion's / challenger's own ``entry_cfg`` overrides per key.
    decision_every_bars: int = 1        # decide only on every N-th closed 5m bar, aligned to the UTC grid (12 = hourly)
    cooldown_bars: int = 0              # per coin: no new entry for N 5m bars after its last entry / exit
    max_trades_per_day: int | None = None   # entries per UTC day, all coins (null = unlimited)
    entry_mode: str = "taker"           # taker (market) | maker (post-only GTX limit at best bid/ask)
    maker_timeout_s: int = 60           # how long one resting post-only order may wait before it is re-priced
    maker_reprices: int = 2             # re-price (cancel + repost at the new best bid/ask) at most this many times
    maker_on_timeout: str = "skip"      # skip | taker: what to do when no maker order filled at all
    # ---- profit protection (docs/ARCHITECTURE.md "Profit protection"; same maths in engine, shadow and tb.sim)
    breakeven_r: float = 1.0            # favourable move (x initial risk) after which the stop moves to entry +- round-trip fees; 0 = off
    sl_cooldown_bars: int = 24          # per coin: no new entry for N 5m bars after a LOSING sl / trail / kill / timeout / liquidation exit
    max_side_risk_pct: float = 60.0     # open stop-out risk of ONE side (longs or shorts), incl. a new entry, <= this % of the open-risk budget
    partial_tp_r: float = 1.5           # at this favourable move (x initial risk) close partial_tp_frac of the position; 0 = off
    partial_tp_frac: float = 0.5        # share of the quantity closed by the partial take-profit (once per position)
    # ---- engine-internal (not part of the API contract; stored but not exposed)
    paper_start_equity: float = 100.0
    slippage_pct: float = 0.0002        # market-order slippage (fraction)
    stop_slippage_pct: float = 0.0005   # extra slippage on stop-market fills (fraction)
    maint_margin_rate: float = 0.01     # maintenance margin rate when a symbol's brackets are unknown (conservative)
    min_universe_volume_usd: float = 50e6
    min_listing_days: int = 7
    max_abs_funding: float = 0.001
    # ---- safety & robustness layer (engine-internal)
    fault_mature_n: int = 3             # consecutive bad cycles before a fault matures
    fault_recover_m: int = 5            # consecutive good cycles before it clears (hysteresis)
    max_bar_return_pct: float = 15.0    # plausibility: single-bar |return| limit (or 8x ATR, if larger)
    max_bar_return_atr: float = 8.0
    max_mark_last_diff_pct: float = 1.0
    drift_n: int = 30                   # closed trades in the drift window
    slippage_drift_mult: float = 2.0    # measured entry slippage > this x assumption -> alert + half size
    guard_interval_s: float = 5.0
    heartbeat_stale_s: float = 30.0
    live_stage_trades: int = 20         # first N live trades run at live_stage_mult x size
    live_stage_mult: float = 0.5
    shadow_min_trades: int = 50         # promotion needs this many shadow trades
    fee_maker: float = 0.0002           # maker fee per side (fraction); used by the paper venue and the shadow simulator

    # -- helpers
    @property
    def notional_usd(self) -> float:
        return self.slice_margin_usd * self.leverage

    def kill_limit(self, mode: str) -> float:
        """Effective drawdown limit: paper uses kill_dd_pct, real-money modes are capped at 25."""
        if mode == "paper_local":
            return float(self.kill_dd_pct)
        return min(float(self.kill_dd_pct), LIVE_KILL_DD_PCT)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_api(self) -> dict[str, Any]:
        d = asdict(self)
        return {k: d[k] for k in CONTRACT_KEYS}

    def merge(self, patch: Mapping[str, Any], *, allow_internal: bool = True) -> "Settings":
        """Return a new Settings with ``patch`` merged in (nested dicts merge too); validates."""
        known = {f.name for f in fields(self)}
        allowed = known if allow_internal else set(CONTRACT_KEYS)
        cur = self.to_dict()
        for k, v in patch.items():
            if k not in allowed:
                raise ValueError(f"unknown setting {k!r}")
            if k in ("trail", "corr_cap"):
                if not isinstance(v, Mapping):
                    raise ValueError(f"{k} must be an object")
                sub = dict(cur[k])
                for sk, sv in v.items():
                    if sk not in sub:
                        raise ValueError(f"unknown setting {k}.{sk}")
                    sub[sk] = sv
                cur[k] = sub
            else:
                cur[k] = v
        s = Settings(**{**cur, "trail": TrailCfg(**_coerce(TrailCfg, cur["trail"])),
                        "corr_cap": CorrCap(**_coerce(CorrCap, cur["corr_cap"]))})
        s = _coerce_settings(s)
        s.validate()
        return s

    def validate(self) -> None:
        def chk(cond: bool, msg: str) -> None:
            if not cond:
                raise ValueError(msg)

        chk(self.slice_margin_usd > 0, "slice_margin_usd must be > 0")
        chk(1 <= self.leverage <= 50, "leverage must be 1..50")
        chk(self.max_positions is None or 1 <= self.max_positions <= 50, "max_positions must be 1..50 or null")
        chk(1 <= self.max_leverage <= 125, "max_leverage must be 1..125")
        chk(1 <= self.max_total_leverage <= 125, "max_total_leverage must be 1..125")
        chk(0.05 <= self.max_margin_ratio <= 0.9, "max_margin_ratio must be 0.05..0.9")
        chk(1 <= self.max_open_risk_pct <= 100, "max_open_risk_pct must be 1..100")
        chk(self.exit_mode in ("fixed", "atr"), "exit_mode must be fixed|atr")
        chk(1 <= self.decision_every_bars <= 4032, "decision_every_bars must be 1..4032")
        chk(0 <= self.cooldown_bars <= 100_000, "cooldown_bars must be >= 0")
        chk(self.max_trades_per_day is None or 1 <= self.max_trades_per_day <= 1000, "max_trades_per_day must be 1..1000 or null")
        chk(self.entry_mode in ("taker", "maker"), "entry_mode must be taker|maker")
        chk(5 <= self.maker_timeout_s <= 900, "maker_timeout_s must be 5..900")
        chk(0 <= self.maker_reprices <= 10, "maker_reprices must be 0..10")
        chk(self.maker_on_timeout in ("skip", "taker"), "maker_on_timeout must be skip|taker")
        chk(0 <= self.fee_maker < 0.01, "fee_maker must be a fraction in [0, 0.01)")
        chk(self.tp_usd > 0 and self.sl_usd > 0, "tp_usd/sl_usd must be > 0")
        chk(self.tp_atr > 0 and self.sl_atr > 0, "tp_atr/sl_atr must be > 0")
        chk(0 <= self.breakeven_r <= 20, "breakeven_r must be 0..20 (0 = off)")
        chk(0 <= self.sl_cooldown_bars <= 100_000, "sl_cooldown_bars must be >= 0")
        chk(1 <= self.max_side_risk_pct <= 100, "max_side_risk_pct must be 1..100")
        chk(0 <= self.partial_tp_r <= 20, "partial_tp_r must be 0..20 (0 = off)")
        chk(0.05 <= self.partial_tp_frac <= 0.95, "partial_tp_frac must be 0.05..0.95")
        chk(self.trail.activate_r > 0 and self.trail.distance_atr > 0, "trail values must be > 0")
        chk(0 <= self.prob_threshold_long <= 1, "prob_threshold_long must be 0..1")
        chk(0 <= self.prob_threshold_short <= 1, "prob_threshold_short must be 0..1")
        chk(0 < self.corr_cap.max_corr <= 1, "corr_cap.max_corr must be (0,1]")
        chk(self.corr_cap.max_correlated >= 1, "corr_cap.max_correlated must be >= 1")
        chk(1 <= self.universe_size <= 50, "universe_size must be 1..50")
        chk(self.universe_refresh_s >= 30, "universe_refresh_s must be >= 30")
        wl = self.universe_whitelist
        chk(wl is None or (isinstance(wl, list) and 1 <= len(wl) <= 50 and len(set(wl)) == len(wl)
                           and all(isinstance(x, str) and _SYMBOL_RE.match(x) for x in wl)),
            "universe_whitelist must be null or 1..50 distinct symbols like BTCUSDT")
        chk(0 < self.kill_dd_pct <= 100, "kill_dd_pct must be (0,100]")
        chk(0 <= self.fee_taker < 0.01, "fee_taker must be a fraction in [0, 0.01)")
        chk(self.paper_start_equity > 0, "paper_start_equity must be > 0")
        chk(self.dd_basis in ("start", "peak"), "dd_basis must be 'start' or 'peak'")
        chk(self.capital_usd is None or self.capital_usd > 0, "capital_usd must be > 0 or null")


LIVE_KILL_DD_PCT = 25.0


def _coerce(cls: type, d: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for f in fields(cls):
        v = d[f.name]
        out[f.name] = _cast(f.type if not isinstance(f.type, str) else f.type, v, f.name)
    return out


def _cast(tp: Any, v: Any, name: str) -> Any:
    t = tp if isinstance(tp, str) else getattr(tp, "__name__", str(tp))
    if t.endswith("| None"):                      # optional field, e.g. "float | None"
        if v is None:
            return None
        t = t[:-len("| None")].strip()
    try:
        if t == "bool":
            if isinstance(v, str):
                return v.strip().lower() in ("1", "true", "yes", "on")
            if isinstance(v, bool):
                return v
            if isinstance(v, (int, float)):
                return bool(v)
            raise ValueError
        if t == "int":
            if isinstance(v, bool):
                raise ValueError
            f = float(v)
            if f != int(f):
                raise ValueError
            return int(f)
        if t == "float":
            if isinstance(v, bool):
                raise ValueError
            f = float(v)
            if f != f or f in (float("inf"), float("-inf")):
                raise ValueError
            return f
        if t == "str":
            return str(v)
    except (TypeError, ValueError):
        raise ValueError(f"setting {name!r} has invalid value {v!r}") from None
    return v


_SYMBOL_RE = re.compile(r"^[A-Z0-9]{2,20}(USDT|USDC|BUSD)$")


def _coerce_whitelist(v: Any) -> list[str] | None:
    """null stays null; a list (or comma-separated string) of symbols is upper-cased, stripped and de-duplicated (order kept)."""
    if v is None:
        return None
    if isinstance(v, str):
        v = [x for x in v.replace(";", ",").split(",")]
    if not isinstance(v, (list, tuple)) or not all(isinstance(x, str) for x in v):
        raise ValueError("setting 'universe_whitelist' must be null or a list of symbols")
    out: list[str] = []
    for x in v:
        x = x.strip().upper()
        if x and x not in out:
            out.append(x)
    return out


def _coerce_settings(s: Settings) -> Settings:
    kw: dict[str, Any] = {}
    for f in fields(s):
        v = getattr(s, f.name)
        if f.name in ("trail", "corr_cap"):
            kw[f.name] = v
        elif f.name == "universe_whitelist":
            kw[f.name] = _coerce_whitelist(v)
        else:
            kw[f.name] = _cast(f.type, v, f.name)
    return Settings(**kw)


def settings_from_rows(rows: Mapping[str, Any]) -> Settings:
    """Defaults overlaid with stored rows; bad/unknown stored values are ignored (defaults kept)."""
    s = Settings()
    for k, v in rows.items():
        try:
            s = s.merge({k: v})
        except ValueError:
            continue
    return s


__all__ = [
    "ROOT", "MODES", "Settings", "TrailCfg", "CorrCap", "Secrets", "CONTRACT_KEYS", "LiveNotAllowed",
    "LIVE_KILL_DD_PCT", "PROD_BASE", "DEMO_BASE", "TESTNET_BASE", "db_path", "var_dir", "models_dir",
    "ui_dir", "archive_dir", "load_env", "load_secrets", "resolve_mode", "trading_base_url",
    "settings_from_rows", "replace",
]
