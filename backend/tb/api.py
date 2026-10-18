"""FastAPI backend for the UI (``python -m tb.api``, default http://127.0.0.1:8787).

Reads SQLite only; every action is a row in the ``commands`` table that the engine polls. The API
never talks to the exchange, so closing the browser or killing this process changes nothing for
open trades. Timestamps are unix seconds.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from fastapi import Body, FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from . import __version__, capital
from .config import Settings, db_path, models_dir, resolve_mode, ui_dir
from .risk import drawdown_pct
from .safety import merged_safety
from .scorer import scan_models
from .shadow import Registry, shadow_stats, vs_champion
from .store import Store

log = logging.getLogger("tb.api")

ENGINE_ALIVE_S = 15.0
CONTROL_ACTIONS = ("start", "stop", "kill", "reset_kill", "close_position", "close_all", "promote_model", "rollback_model",
                   "pause_symbol", "unpause_symbol", "refresh_universe")
LOCAL_ORIGIN = r"^https?://(localhost|127\.0\.0\.1|\[::1\])(:\d+)?$"


def _num(x: Any, nd: int | None = None) -> Any:
    if x is None:
        return None
    return round(x, nd) if nd is not None and isinstance(x, float) else x


def create_app(db: str | Path | None = None, *, control_wait_s: float = 4.0, ui: Path | None = None) -> FastAPI:
    store = Store(db or db_path())
    app = FastAPI(title="TradingBot API", version=__version__)
    app.add_middleware(CORSMiddleware, allow_origin_regex=LOCAL_ORIGIN, allow_methods=["*"], allow_headers=["*"])
    app.state.store = store

    @app.middleware("http")
    async def no_stale_ui(request, call_next):
        # The UI is unbundled ES modules; without this Safari keeps old modules after a UI update
        # and mixes versions. no-cache = always revalidate (ETag makes it a cheap 304).
        resp = await call_next(request)
        if not request.url.path.startswith("/api"):
            resp.headers["Cache-Control"] = "no-cache"
        return resp

    # ------------------------------------------------------------------ builders
    def heartbeat_age() -> tuple[float | None, dict | None]:
        hb = store.heartbeat()
        if not hb:
            return None, None
        return max(0.0, time.time() - hb["t"]), hb

    def build_status() -> dict:
        cfg = store.get_settings()
        kv = store.kv_get_many(["status", "safety_guard", "safety_engine", "guard_heartbeat"])
        st = kv.get("status") or {}
        age, hb = heartbeat_age()
        mode = st.get("mode") or (hb or {}).get("mode") or os.environ.get("TB_MODE") or "paper_local"
        stats = store.trade_stats()
        now = datetime.now(timezone.utc)
        day0 = datetime(now.year, now.month, now.day, tzinfo=timezone.utc).timestamp()
        today = store.trade_stats(day0)
        equity = st.get("equity")
        if equity is None:
            equity = cfg.paper_start_equity
        start = st.get("start_equity") if st.get("start_equity") is not None else equity
        peak = st.get("peak_equity") if st.get("peak_equity") is not None else max(equity, start)
        n_open = store.one("SELECT COUNT(*) n FROM positions")["n"]
        beat = kv.get("guard_heartbeat")
        guard_age = (time.time() - beat["t"]) if beat else None
        safety = merged_safety(kv.get("safety_guard"), kv.get("safety_engine"), guard_age,
                               guard_required=(mode != "paper_local"))
        safety.pop("parts", None)
        return {
            "mode": mode, "running": bool(st.get("running", False)) and age is not None and age < ENGINE_ALIVE_S,
            "equity": equity, "wallet_equity": st.get("wallet_equity", equity), "start_equity": start, "peak_equity": peak, "cash": st.get("cash", equity),
            "realized_pnl": stats["net"], "unrealized_pnl": st.get("unrealized_pnl", 0.0),
            "drawdown_pct": drawdown_pct(equity, peak), "open_positions": n_open,
            "max_positions": st.get("max_positions", cfg.max_positions), "trades_today": today["n"],
            "win_rate": (stats["wins"] / stats["n"]) if stats["n"] else 0.0,
            "kill": st.get("kill") or {"tripped": False, "reason": "", "dd_limit_pct": cfg.kill_limit(mode)},
            "model": st.get("model") or {"name": "null", "version": "0", "trained_at": None,
                                         "threshold_long": cfg.prob_threshold_long,
                                         "threshold_short": cfg.prob_threshold_short},
            "universe_updated_at": st.get("universe_updated_at"),
            "safety": safety,
            "paused_symbols": sorted(store.kv_get("paused_symbols", []) or []),
        }

    def build_shadow() -> list[dict]:
        reg = Registry.from_dict(store.kv_get("model_registry"))
        entries = {e.name: e.status for e in scan_models(models_dir(), reg.overrides)}
        names = list(entries)
        for n in store.shadow_models():
            if n not in entries:
                names.append(n)
                entries[n] = "retired"
        champ = next((n for n, s in entries.items() if s == "champion"), None)
        out = []
        for n in names:
            trades = store.get_shadow_trades(n)
            s = shadow_stats(trades)
            vs = None
            if champ and n != champ and trades:
                since = min(t["entry_time"] for t in trades)
                vs = vs_champion(s, shadow_stats(store.get_shadow_trades(champ, since)))
            out.append({"model": n, "status": entries[n], "n_trades": s["n_trades"], "win_rate": s["win_rate"],
                        "expectancy": s["expectancy"], "pnl": s["pnl"], "vs_champion": vs})
        return out

    # ------------------------------------------------------------------ endpoints
    @app.get("/api/health")
    def health() -> dict:
        age, hb = heartbeat_age()
        alive = age is not None and age < ENGINE_ALIVE_S
        return {"ok": True, "engine_alive": alive, "engine_heartbeat_age_s": _num(age, 2),
                "mode": (hb or {}).get("mode") or os.environ.get("TB_MODE") or "paper_local", "version": __version__}

    @app.get("/api/status")
    def status() -> dict:
        return build_status()

    @app.get("/api/market")
    def market() -> dict:
        m = store.kv_get("market")
        return m or {"btc_price": None, "btc_change_24h_pct": None, "btc_trend": "flat", "market_volatility_pct": None,
                     "avg_funding_rate": None, "long_short_ratio": None, "updated_at": None}

    @app.get("/api/universe")
    def universe() -> list[dict]:
        keys = ("symbol", "score", "atr_pct", "quote_volume_24h", "change_24h_pct", "funding_rate",
                "open_interest_usd", "oi_change_1h_pct", "last_price", "in_universe_since")
        paused = set(store.kv_get("paused_symbols", []) or [])
        return [{**{k: r.get(k) for k in keys}, "paused": r["symbol"] in paused} for r in store.get_universe()]

    @app.get("/api/orderbook")
    def orderbook(symbol: str = Query(...)) -> dict:
        snap = store.get_orderbook(symbol.upper())
        if not snap:
            raise HTTPException(status_code=404, detail=f"no order book snapshot for {symbol.upper()} (not in the universe, or the engine has not fetched it yet)")
        keys = ("symbol", "t", "mid", "spread", "spread_pct", "cover_pct", "bands", "bids", "asks")
        return {k: snap.get(k) for k in keys}

    @app.get("/api/candles")
    def candles(symbol: str = Query(...), tf: str = "5m", limit: int = Query(500, ge=1, le=1500)) -> list[dict]:
        return store.get_candles(symbol.upper(), tf, limit)

    @app.get("/api/signals")
    def signals(symbol: str | None = None, limit: int = Query(200, ge=1, le=2000)) -> list[dict]:
        return store.get_signals(symbol.upper() if symbol else None, limit)

    @app.get("/api/positions")
    def positions() -> list[dict]:
        keys = ("id", "symbol", "side", "entry_price", "qty", "notional", "leverage", "margin", "sl_price", "tp_price",
                "trail_active", "trail_price", "mark_price", "upnl", "opened_at", "prob", "reasons")
        return [{k: p.get(k) for k in keys} for p in store.get_positions()]

    @app.get("/api/trades")
    def trades(limit: int = Query(200, ge=1, le=5000)) -> list[dict]:
        keys = ("id", "symbol", "side", "entry_price", "exit_price", "qty", "opened_at", "closed_at", "pnl", "fees",
                "funding", "exit_reason", "prob")
        return [{k: t.get(k) for k in keys} for t in store.get_trades(limit)]

    @app.get("/api/equity")
    def equity(limit: int = Query(2000, ge=1, le=20000)) -> list[dict]:
        # In demo/live the history before the $-stake started is wallet-sized (e.g. 5000 USDT on
        # testnet) and would dwarf the bot's own curve; show only points since the stake began.
        since = capital.since_t(store)
        rows = store.get_equity(limit)
        return [r for r in rows if r["t"] >= since] if since is not None else rows

    @app.get("/api/logs")
    def logs(limit: int = Query(200, ge=1, le=2000)) -> list[dict]:
        return store.get_logs(limit)

    @app.get("/api/settings")
    def get_settings() -> dict:
        return store.get_settings().to_api()

    @app.put("/api/settings")
    def put_settings(patch: dict = Body(...)) -> dict:
        try:
            new = store.patch_settings(patch, allow_internal=False)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))
        store.add_log("INFO", f"settings updated: {', '.join(patch)}")
        return new.to_api()

    @app.post("/api/control")
    def control(body: dict = Body(...)) -> dict:
        action = body.get("action")
        if action not in CONTROL_ACTIONS:
            return {"ok": False, "error": f"unknown action {action!r}; expected one of {list(CONTROL_ACTIONS)}"}
        payload: dict[str, Any] = {}
        if action == "close_position":
            if body.get("position_id") is None:
                return {"ok": False, "error": "position_id is required"}
            payload["position_id"] = body["position_id"]
        if action in ("pause_symbol", "unpause_symbol"):
            sym = str(body.get("symbol") or "").upper().strip()
            if not sym or not sym.isalnum():
                return {"ok": False, "error": "symbol is required"}
            payload["symbol"] = sym
        if action == "promote_model":
            if not body.get("model"):
                return {"ok": False, "error": "model is required"}
        if body.get("model"):
            payload["model"] = body["model"]
        age, _ = heartbeat_age()
        if age is None or age > ENGINE_ALIVE_S:
            return {"ok": False, "error": "engine is not running (no recent heartbeat); command not queued"}
        cid = store.add_command(action, payload)
        deadline = time.time() + control_wait_s
        while time.time() < deadline:
            res = store.command_result(cid)
            if res is not None:
                out = {"ok": bool(res.get("ok"))}
                if res.get("error"):
                    out["error"] = res["error"]
                return out
            time.sleep(0.1)
        return {"ok": True, "queued": True}

    @app.get("/api/lab/reports")
    def lab_reports() -> list[dict]:
        return store.get_lab_reports()

    @app.get("/api/shadow")
    def shadow() -> list[dict]:
        return build_shadow()

    # ------------------------------------------------------------------ websocket
    @app.websocket("/ws")
    async def ws(sock: WebSocket) -> None:
        await sock.accept()
        closed = asyncio.Event()

        async def reader() -> None:
            try:
                while True:
                    await sock.receive_text()
            except Exception:
                closed.set()

        rtask = asyncio.create_task(reader())
        last_id = store.max_event_id()
        last_status = 0.0
        try:
            while not closed.is_set():
                now = time.time()
                if now - last_status >= 2.0:
                    last_status = now
                    await sock.send_json({"type": "status", "data": build_status()})
                for ev in store.events_after(last_id, 200):
                    last_id = ev["id"]
                    data = build_status() if ev["type"] == "status" else ev["data"]
                    await sock.send_json({"type": ev["type"], "data": data})
                await asyncio.sleep(0.25)
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            rtask.cancel()

    # ------------------------------------------------------------------ static UI (last: routes win)
    ui_path = ui or ui_dir()
    if ui_path.is_dir():
        app.mount("/", StaticFiles(directory=str(ui_path), html=True), name="ui")
    return app


def main(argv: list[str] | None = None) -> int:
    import argparse

    import uvicorn

    ap = argparse.ArgumentParser(prog="tb.api")
    ap.add_argument("--port", type=int, default=int(os.environ.get("TB_API_PORT", "8787")))
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args(argv)
    if args.host not in ("127.0.0.1", "localhost", "::1"):
        print("refusing to bind a non-loopback host (the API has no authentication)", file=sys.stderr)
        return 2
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    uvicorn.run(create_app(), host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    sys.exit(main())
