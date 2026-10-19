"""SQLite (WAL) store shared by the engine and the API processes.

Short transactions, busy_timeout, one connection per thread. All timestamps are unix seconds.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .config import Settings, settings_from_rows

SCHEMA = """
CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT NOT NULL, updated_at REAL);
CREATE TABLE IF NOT EXISTS positions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL, side TEXT NOT NULL,
    entry_price REAL NOT NULL, qty REAL NOT NULL, notional REAL NOT NULL,
    leverage INTEGER NOT NULL, margin REAL NOT NULL,
    sl_price REAL NOT NULL, tp_price REAL, init_sl_price REAL NOT NULL,
    atr REAL, trail_active INTEGER NOT NULL DEFAULT 0, trail_price REAL,
    best_price REAL, mark_price REAL, upnl REAL DEFAULT 0,
    opened_at INTEGER NOT NULL, prob REAL, reasons TEXT DEFAULT '[]',
    sl_order_id TEXT, tp_order_id TEXT, entry_fee REAL DEFAULT 0, funding REAL DEFAULT 0,
    liq_price REAL, exit_mode TEXT
);
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    position_id INTEGER, symbol TEXT NOT NULL, side TEXT NOT NULL,
    entry_price REAL NOT NULL, exit_price REAL NOT NULL, qty REAL NOT NULL,
    opened_at INTEGER NOT NULL, closed_at INTEGER NOT NULL,
    pnl REAL NOT NULL, fees REAL NOT NULL DEFAULT 0, funding REAL NOT NULL DEFAULT 0,
    exit_reason TEXT NOT NULL, prob REAL
);
CREATE INDEX IF NOT EXISTS ix_trades_closed ON trades(closed_at);
CREATE TABLE IF NOT EXISTS equity (t INTEGER PRIMARY KEY, equity REAL NOT NULL);
CREATE TABLE IF NOT EXISTS signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    t INTEGER NOT NULL, symbol TEXT NOT NULL, prob_long REAL NOT NULL, prob_short REAL NOT NULL,
    action TEXT NOT NULL, reasons TEXT NOT NULL DEFAULT '[]'
);
CREATE INDEX IF NOT EXISTS ix_signals_sym ON signals(symbol, t);
CREATE INDEX IF NOT EXISTS ix_signals_t ON signals(t);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT, t REAL NOT NULL, type TEXT NOT NULL, data TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS commands (
    id INTEGER PRIMARY KEY AUTOINCREMENT, t REAL NOT NULL, action TEXT NOT NULL,
    payload TEXT NOT NULL DEFAULT '{}', status TEXT NOT NULL DEFAULT 'pending',
    result TEXT, done_at REAL
);
CREATE INDEX IF NOT EXISTS ix_commands_status ON commands(status, id);
CREATE TABLE IF NOT EXISTS logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT, t REAL NOT NULL, level TEXT NOT NULL, msg TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS universe (
    symbol TEXT PRIMARY KEY, rank INTEGER, score REAL, atr_pct REAL, quote_volume_24h REAL,
    change_24h_pct REAL, funding_rate REAL, open_interest_usd REAL, oi_change_1h_pct REAL,
    last_price REAL, in_universe_since INTEGER, updated_at INTEGER
);
CREATE TABLE IF NOT EXISTS heartbeat (
    id INTEGER PRIMARY KEY CHECK (id = 1), t REAL NOT NULL, pid INTEGER, mode TEXT,
    version TEXT, data TEXT
);
CREATE TABLE IF NOT EXISTS lab_reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT, created_at INTEGER NOT NULL, kind TEXT NOT NULL,
    title TEXT NOT NULL, summary TEXT NOT NULL DEFAULT '{}', equity TEXT NOT NULL DEFAULT '[]'
);
CREATE TABLE IF NOT EXISTS shadow_trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model TEXT NOT NULL, symbol TEXT NOT NULL, side TEXT NOT NULL,
    entry_time INTEGER NOT NULL, exit_time INTEGER NOT NULL,
    entry_price REAL NOT NULL, exit_price REAL NOT NULL, qty REAL NOT NULL,
    pnl REAL NOT NULL, fees REAL NOT NULL DEFAULT 0, exit_reason TEXT NOT NULL, prob REAL
);
CREATE INDEX IF NOT EXISTS ix_shadow_model ON shadow_trades(model, exit_time);
CREATE TABLE IF NOT EXISTS depth_5m (
    symbol TEXT NOT NULL, open_time INTEGER NOT NULL, t REAL NOT NULL, data TEXT NOT NULL,
    PRIMARY KEY (symbol, open_time)
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS candles (
    symbol TEXT NOT NULL, tf TEXT NOT NULL, t INTEGER NOT NULL,
    o REAL, h REAL, l REAL, c REAL, v REAL,
    PRIMARY KEY (symbol, tf, t)
) WITHOUT ROWID;
"""


def _j(x: Any) -> str:
    return json.dumps(x, separators=(",", ":"), default=_default)


def _default(o: Any) -> Any:
    try:
        import numpy as np

        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
    except Exception:
        pass
    if hasattr(o, "__dict__"):
        return vars(o)
    raise TypeError(f"not JSON serialisable: {type(o).__name__}")


def _load(s: str | None, default: Any = None) -> Any:
    if s is None:
        return default
    try:
        return json.loads(s)
    except Exception:
        return default


class Store:
    def __init__(self, path: str | Path, *, readonly: bool = False):
        self.path = str(path)
        self.readonly = readonly
        self._local = threading.local()
        self._init_lock = threading.Lock()
        if not readonly:
            with self._init_lock:
                c = self._conn()
                c.executescript(SCHEMA)

    # ------------------------------------------------------------------ plumbing
    def _conn(self) -> sqlite3.Connection:
        c = getattr(self._local, "conn", None)
        if c is None:
            c = sqlite3.connect(self.path, timeout=10.0, isolation_level=None, check_same_thread=False)
            c.row_factory = sqlite3.Row
            c.execute("PRAGMA busy_timeout=10000")
            c.execute("PRAGMA journal_mode=WAL")
            c.execute("PRAGMA synchronous=NORMAL")
            self._local.conn = c
        return c

    def close(self) -> None:
        c = getattr(self._local, "conn", None)
        if c is not None:
            c.close()
            self._local.conn = None

    def execute(self, sql: str, params: Sequence[Any] = ()) -> sqlite3.Cursor:
        return self._conn().execute(sql, params)

    def query(self, sql: str, params: Sequence[Any] = ()) -> list[sqlite3.Row]:
        return self._conn().execute(sql, params).fetchall()

    def one(self, sql: str, params: Sequence[Any] = ()) -> sqlite3.Row | None:
        return self._conn().execute(sql, params).fetchone()

    def tx(self) -> "_Tx":
        return _Tx(self._conn())

    # ------------------------------------------------------------------ settings
    def get_settings(self) -> Settings:
        rows = {r["key"]: _load(r["value"]) for r in self.query("SELECT key, value FROM settings")}
        return settings_from_rows(rows)

    def save_settings(self, s: Settings) -> None:
        d = s.to_dict()
        with self.tx() as c:
            for k, v in d.items():
                c.execute("INSERT INTO settings(key,value) VALUES(?,?) "
                          "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (k, _j(v)))

    def patch_settings(self, patch: Mapping[str, Any], *, allow_internal: bool = False) -> Settings:
        """Merge a partial update (validated); raises ValueError on bad input."""
        cur = self.get_settings()
        new = cur.merge(patch, allow_internal=allow_internal)
        self.save_settings(new)
        return new

    # ------------------------------------------------------------------ kv
    def kv_get(self, key: str, default: Any = None) -> Any:
        r = self.one("SELECT value FROM kv WHERE key=?", (key,))
        return _load(r["value"], default) if r else default

    def kv_set(self, key: str, value: Any) -> None:
        self.execute("INSERT INTO kv(key,value,updated_at) VALUES(?,?,?) "
                     "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
                     (key, _j(value), time.time()))

    def kv_get_many(self, keys: Iterable[str]) -> dict[str, Any]:
        keys = list(keys)
        if not keys:
            return {}
        q = ",".join("?" * len(keys))
        return {r["key"]: _load(r["value"]) for r in self.query(f"SELECT key,value FROM kv WHERE key IN ({q})", keys)}

    # ------------------------------------------------------------------ events / logs
    def add_event(self, type_: str, data: Any, t: float | None = None) -> int:
        cur = self.execute("INSERT INTO events(t,type,data) VALUES(?,?,?)", (t or time.time(), type_, _j(data)))
        return int(cur.lastrowid)

    def events_after(self, last_id: int, limit: int = 500) -> list[dict]:
        rows = self.query("SELECT id,t,type,data FROM events WHERE id>? ORDER BY id LIMIT ?", (last_id, limit))
        return [{"id": r["id"], "t": r["t"], "type": r["type"], "data": _load(r["data"], {})} for r in rows]

    def max_event_id(self) -> int:
        r = self.one("SELECT COALESCE((SELECT seq FROM sqlite_sequence WHERE name='events'),0) m")
        return int(r["m"])

    def prune_events(self, older_than_s: float = 600.0, keep_types: Sequence[str] = ()) -> None:
        cutoff = time.time() - older_than_s
        if keep_types:
            q = ",".join("?" * len(keep_types))
            self.execute(f"DELETE FROM events WHERE t<? AND type NOT IN ({q})", (cutoff, *keep_types))
        else:
            self.execute("DELETE FROM events WHERE t<?", (cutoff,))

    def add_log(self, level: str, msg: str, t: float | None = None) -> None:
        self.execute("INSERT INTO logs(t,level,msg) VALUES(?,?,?)", (t or time.time(), level.upper(), msg[:2000]))

    def get_logs(self, limit: int = 200) -> list[dict]:
        rows = self.query("SELECT t,level,msg FROM logs ORDER BY id DESC LIMIT ?", (limit,))
        return [{"t": int(r["t"]), "level": r["level"], "msg": r["msg"]} for r in reversed(rows)]

    def prune_logs(self, keep: int = 5000) -> None:
        self.execute("DELETE FROM logs WHERE id <= (SELECT COALESCE(MAX(id),0) FROM logs) - ?", (keep,))

    # ------------------------------------------------------------------ commands
    def add_command(self, action: str, payload: Mapping[str, Any] | None = None) -> int:
        cur = self.execute("INSERT INTO commands(t,action,payload) VALUES(?,?,?)",
                           (time.time(), action, _j(payload or {})))
        return int(cur.lastrowid)

    def pending_commands(self) -> list[dict]:
        rows = self.query("SELECT id,t,action,payload FROM commands WHERE status='pending' ORDER BY id")
        return [{"id": r["id"], "t": r["t"], "action": r["action"], "payload": _load(r["payload"], {})} for r in rows]

    def finish_command(self, cmd_id: int, ok: bool, error: str | None = None) -> None:
        self.execute("UPDATE commands SET status=?, result=?, done_at=? WHERE id=?",
                     ("done" if ok else "error", _j({"ok": ok, "error": error}), time.time(), cmd_id))

    def command_result(self, cmd_id: int) -> dict | None:
        r = self.one("SELECT status,result FROM commands WHERE id=?", (cmd_id,))
        if not r or r["status"] == "pending":
            return None
        return _load(r["result"], {"ok": r["status"] == "done"})

    # ------------------------------------------------------------------ heartbeat
    def beat(self, mode: str, version: str, pid: int, data: Mapping[str, Any] | None = None) -> None:
        self.execute("INSERT INTO heartbeat(id,t,pid,mode,version,data) VALUES(1,?,?,?,?,?) "
                     "ON CONFLICT(id) DO UPDATE SET t=excluded.t,pid=excluded.pid,mode=excluded.mode,"
                     "version=excluded.version,data=excluded.data",
                     (time.time(), pid, mode, version, _j(data or {})))

    def heartbeat(self) -> dict | None:
        r = self.one("SELECT t,pid,mode,version,data FROM heartbeat WHERE id=1")
        if not r:
            return None
        return {"t": r["t"], "pid": r["pid"], "mode": r["mode"], "version": r["version"],
                "data": _load(r["data"], {})}

    # ------------------------------------------------------------------ positions
    POS_COLS = ("symbol", "side", "entry_price", "qty", "notional", "leverage", "margin", "sl_price",
                "tp_price", "init_sl_price", "atr", "trail_active", "trail_price", "best_price",
                "mark_price", "upnl", "opened_at", "prob", "reasons", "sl_order_id", "tp_order_id",
                "entry_fee", "funding", "liq_price", "exit_mode")

    def insert_position(self, p: Mapping[str, Any]) -> int:
        d = dict(p)
        d["reasons"] = _j(d.get("reasons") or [])
        d["trail_active"] = int(bool(d.get("trail_active", False)))
        cols = [c for c in self.POS_COLS if c in d]
        cur = self.execute(f"INSERT INTO positions({','.join(cols)}) VALUES({','.join('?' * len(cols))})",
                           [d[c] for c in cols])
        return int(cur.lastrowid)

    def update_position(self, pos_id: int, **fields: Any) -> None:
        if not fields:
            return
        bad = set(fields) - set(self.POS_COLS)
        if bad:
            raise KeyError(f"unknown position columns {bad}")
        if "reasons" in fields:
            fields["reasons"] = _j(fields["reasons"])
        if "trail_active" in fields:
            fields["trail_active"] = int(bool(fields["trail_active"]))
        sets = ",".join(f"{k}=?" for k in fields)
        self.execute(f"UPDATE positions SET {sets} WHERE id=?", [*fields.values(), pos_id])

    def delete_position(self, pos_id: int) -> None:
        self.execute("DELETE FROM positions WHERE id=?", (pos_id,))

    def get_positions(self) -> list[dict]:
        out = []
        for r in self.query("SELECT * FROM positions ORDER BY id"):
            d = dict(r)
            d["reasons"] = _load(d.get("reasons"), [])
            d["trail_active"] = bool(d["trail_active"])
            out.append(d)
        return out

    def get_position(self, pos_id: int) -> dict | None:
        return next((p for p in self.get_positions() if p["id"] == pos_id), None)

    # ------------------------------------------------------------------ trades
    def insert_trade(self, t: Mapping[str, Any]) -> int:
        cur = self.execute(
            "INSERT INTO trades(position_id,symbol,side,entry_price,exit_price,qty,opened_at,closed_at,"
            "pnl,fees,funding,exit_reason,prob) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (t.get("position_id"), t["symbol"], t["side"], t["entry_price"], t["exit_price"], t["qty"],
             int(t["opened_at"]), int(t["closed_at"]), t["pnl"], t.get("fees", 0.0), t.get("funding", 0.0),
             t["exit_reason"], t.get("prob")))
        return int(cur.lastrowid)

    def get_trades(self, limit: int = 200) -> list[dict]:
        rows = self.query("SELECT * FROM trades ORDER BY id DESC LIMIT ?", (limit,))
        return [dict(r) for r in rows]

    def trade_stats(self, since: float | None = None) -> dict:
        w = "WHERE closed_at>=?" if since is not None else ""
        a = (since,) if since is not None else ()
        # trades.pnl is NET of fees and funding (fees/funding are COSTS, > 0 = paid); win = net pnl > 0
        r = self.one(f"SELECT COALESCE(SUM(pnl),0) pnl, COALESCE(SUM(fees),0) fees, "
                     f"COALESCE(SUM(funding),0) funding FROM trades {w}", a)
        # a position scaled out by a partial take-profit has several rows (exit_reason 'partial' + the final one):
        # trade COUNT and wins are per position (summed net pnl), money sums are per row
        g = self.one(f"SELECT COUNT(*) n, COALESCE(SUM(CASE WHEN p>0 THEN 1 ELSE 0 END),0) wins FROM "
                     f"(SELECT SUM(pnl) p FROM trades {w} GROUP BY CASE WHEN position_id IN "
                     f"(SELECT position_id FROM trades WHERE exit_reason='partial') THEN position_id ELSE -id END)", a)
        r = {"n": g["n"], "wins": g["wins"], "pnl": r["pnl"], "fees": r["fees"], "funding": r["funding"]}
        return {"n": r["n"], "net": r["pnl"], "fees": r["fees"], "funding": r["funding"],
                "gross": r["pnl"] + r["fees"] + r["funding"], "wins": r["wins"]}

    # ------------------------------------------------------------------ shadow trades
    def add_shadow_trade(self, t: Mapping[str, Any]) -> int:
        cur = self.execute(
            "INSERT INTO shadow_trades(model,symbol,side,entry_time,exit_time,entry_price,exit_price,qty,pnl,fees,"
            "exit_reason,prob) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
            (t["model"], t["symbol"], t["side"], int(t["entry_time"]), int(t["exit_time"]), t["entry_price"],
             t["exit_price"], t["qty"], t["pnl"], t.get("fees", 0.0), t["exit_reason"], t.get("prob")))
        return int(cur.lastrowid)

    def get_shadow_trades(self, model: str | None = None, since: int | None = None) -> list[dict]:
        q, a = "SELECT * FROM shadow_trades WHERE 1=1", []
        if model:
            q += " AND model=?"
            a.append(model)
        if since is not None:
            q += " AND entry_time>=?"
            a.append(int(since))
        return [dict(r) for r in self.query(q + " ORDER BY id", a)]

    def shadow_models(self) -> list[str]:
        return [r["model"] for r in self.query("SELECT DISTINCT model FROM shadow_trades")]

    # ------------------------------------------------------------------ equity
    def add_equity(self, t: int, equity: float) -> None:
        self.execute("INSERT INTO equity(t,equity) VALUES(?,?) ON CONFLICT(t) DO UPDATE SET equity=excluded.equity",
                     (int(t), float(equity)))

    def get_equity(self, limit: int = 2000) -> list[dict]:
        rows = self.query("SELECT t,equity FROM equity ORDER BY t DESC LIMIT ?", (limit,))
        return [{"t": r["t"], "equity": r["equity"]} for r in reversed(rows)]

    # ------------------------------------------------------------------ signals
    def add_signal(self, t: int, symbol: str, prob_long: float, prob_short: float, action: str,
                   reasons: Sequence[Any] | None = None) -> int:
        cur = self.execute("INSERT INTO signals(t,symbol,prob_long,prob_short,action,reasons) VALUES(?,?,?,?,?,?)",
                           (int(t), symbol, float(prob_long), float(prob_short), action, _j(list(reasons or []))))
        return int(cur.lastrowid)

    def get_signals(self, symbol: str | None = None, limit: int = 200) -> list[dict]:
        if symbol:
            rows = self.query("SELECT * FROM signals WHERE symbol=? ORDER BY id DESC LIMIT ?", (symbol, limit))
        else:
            rows = self.query("SELECT * FROM signals ORDER BY id DESC LIMIT ?", (limit,))
        out = []
        for r in reversed(rows):
            out.append({"t": r["t"], "symbol": r["symbol"], "prob_long": r["prob_long"],
                        "prob_short": r["prob_short"], "action": r["action"],
                        "reasons": _load(r["reasons"], [])})
        return out

    def prune_signals(self, keep: int = 20000) -> None:
        self.execute("DELETE FROM signals WHERE id <= (SELECT COALESCE(MAX(id),0) FROM signals) - ?", (keep,))

    # ------------------------------------------------------------------ candles
    def upsert_candles(self, symbol: str, tf: str, rows: Iterable[Sequence[float]]) -> None:
        """rows: (t_seconds, o, h, l, c, v)."""
        data = [(symbol, tf, int(r[0]), r[1], r[2], r[3], r[4], r[5]) for r in rows]
        if not data:
            return
        with self.tx() as c:
            c.executemany("INSERT INTO candles(symbol,tf,t,o,h,l,c,v) VALUES(?,?,?,?,?,?,?,?) "
                          "ON CONFLICT(symbol,tf,t) DO UPDATE SET o=excluded.o,h=excluded.h,l=excluded.l,"
                          "c=excluded.c,v=excluded.v", data)

    def get_candles(self, symbol: str, tf: str = "5m", limit: int = 500) -> list[dict]:
        rows = self.query("SELECT t,o,h,l,c,v FROM candles WHERE symbol=? AND tf=? ORDER BY t DESC LIMIT ?",
                          (symbol, tf, limit))
        return [{"t": r["t"], "o": r["o"], "h": r["h"], "l": r["l"], "c": r["c"], "v": r["v"]}
                for r in reversed(rows)]

    def prune_candles(self, symbol: str, tf: str = "5m", keep: int = 1500) -> None:
        self.execute("DELETE FROM candles WHERE symbol=? AND tf=? AND t < "
                     "(SELECT COALESCE(MAX(t),0) FROM candles WHERE symbol=? AND tf=?) - ?",
                     (symbol, tf, symbol, tf, keep * 300))

    # ------------------------------------------------------------------ universe
    UNI_COLS = ("symbol", "rank", "score", "atr_pct", "quote_volume_24h", "change_24h_pct", "funding_rate",
                "open_interest_usd", "oi_change_1h_pct", "last_price", "in_universe_since", "updated_at")

    def replace_universe(self, rows: Sequence[Mapping[str, Any]]) -> None:
        with self.tx() as c:
            c.execute("DELETE FROM universe")
            for r in rows:
                c.execute(f"INSERT INTO universe({','.join(self.UNI_COLS)}) VALUES({','.join('?' * len(self.UNI_COLS))})",
                          [r.get(k) for k in self.UNI_COLS])

    def update_universe_prices(self, prices: Mapping[str, float]) -> None:
        with self.tx() as c:
            c.executemany("UPDATE universe SET last_price=? WHERE symbol=?", [(p, s) for s, p in prices.items()])

    def get_universe(self) -> list[dict]:
        return [dict(r) for r in self.query("SELECT * FROM universe ORDER BY rank")]

    # ------------------------------------------------------------------ order book
    def set_orderbook(self, symbol: str, snap: Mapping[str, Any]) -> None:
        self.kv_set(f"orderbook:{symbol}", snap)

    def get_orderbook(self, symbol: str) -> dict | None:
        return self.kv_get(f"orderbook:{symbol}")

    def upsert_depth_5m(self, symbol: str, open_time: int, t: float, row: Mapping[str, Any]) -> None:
        self.execute("INSERT INTO depth_5m(symbol,open_time,t,data) VALUES(?,?,?,?) "
                     "ON CONFLICT(symbol,open_time) DO UPDATE SET t=excluded.t,data=excluded.data WHERE excluded.t>=t",
                     (symbol, int(open_time), float(t), _j(row)))

    def get_depth_5m(self, symbol: str, limit: int = 2016) -> list[dict]:
        rows = self.query("SELECT open_time,t,data FROM depth_5m WHERE symbol=? ORDER BY open_time DESC LIMIT ?",
                          (symbol, limit))
        return [{"open_time": r["open_time"], "t": r["t"], "data": _load(r["data"], {})} for r in reversed(rows)]

    def prune_depth_5m(self, symbol: str, keep: int = 2016) -> None:
        self.execute("DELETE FROM depth_5m WHERE symbol=? AND open_time < "
                     "(SELECT COALESCE(MAX(open_time),0) FROM depth_5m WHERE symbol=?) - ?",
                     (symbol, symbol, keep * 300_000))

    # ------------------------------------------------------------------ lab reports
    def add_lab_report(self, kind: str, title: str, summary: Mapping[str, Any],
                       equity: Sequence[Mapping[str, Any]] | None = None, created_at: int | None = None) -> int:
        cur = self.execute("INSERT INTO lab_reports(created_at,kind,title,summary,equity) VALUES(?,?,?,?,?)",
                           (int(created_at or time.time()), kind, title, _j(summary), _j(list(equity or []))))
        return int(cur.lastrowid)

    def get_lab_reports(self, limit: int = 100) -> list[dict]:
        rows = self.query("SELECT * FROM lab_reports ORDER BY id DESC LIMIT ?", (limit,))
        return [{"id": r["id"], "created_at": r["created_at"], "kind": r["kind"], "title": r["title"],
                 "summary": _load(r["summary"], {}), "equity": _load(r["equity"], [])} for r in rows]


class _Tx:
    def __init__(self, conn: sqlite3.Connection):
        self.c = conn

    def __enter__(self) -> sqlite3.Connection:
        self.c.execute("BEGIN IMMEDIATE")
        return self.c

    def __exit__(self, et, ev, tb) -> None:
        if et is None:
            self.c.execute("COMMIT")
        else:
            self.c.execute("ROLLBACK")
