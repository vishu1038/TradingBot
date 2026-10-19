#!/usr/bin/env python3
"""Periodic bot health/performance check (run every 3 h by launchd).

* Reads the running API (default http://127.0.0.1:8787): status, trades, equity, shadow models.
* Summarises the last WINDOW hours, posts a Google Chat card, writes var/reports/check-<ts>.json
  and var/reports/latest.json (read by the Claude-side check that decides on retraining).
* Model update: asks the engine to promote a shadow challenger that looks better. The engine's
  own promotion rules decide (>= 50 shadow trades, higher expectancy than the champion, no drift
  alarm; never a PAPER-TEST-ONLY model in live), so this script cannot force a bad model in.

Exit code 0 = checked (even if the bot needs attention), 2 = API unreachable.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

import requests  # noqa: E402

from tb.alerts import AMBER, GREEN, RED, AlertDispatcher, build_card  # noqa: E402
from tb.config import load_env  # noqa: E402


def get(api: str, path: str):
    r = requests.get(api + path, timeout=15)
    r.raise_for_status()
    return r.json()


def summarise(status: dict, trades: list[dict], equity: list[dict], shadow: list[dict], window_h: float) -> dict:
    now = time.time()
    since = now - window_h * 3600
    recent = [t for t in trades if (t.get("closed_at") or 0) >= since]
    net = sum(t["pnl"] for t in recent)
    wins = sum(1 for t in recent if t["pnl"] > 0)
    eq_then = next((e["equity"] for e in equity if e["t"] >= since), equity[0]["equity"] if equity else None)
    safety = status.get("safety") or {}
    drift = safety.get("drift") or {}
    attention = []
    if safety.get("state", "NORMAL") != "NORMAL":
        attention.append(f"safety {safety.get('state')}: {'; '.join(safety.get('reasons') or [])}")
    if (status.get("kill") or {}).get("tripped"):
        attention.append(f"kill switch tripped: {status['kill'].get('reason')}")
    z = drift.get("z")
    if z is not None and z < -2:
        attention.append(f"model drift z={z:.2f} (realised win rate below predicted)")
    if status.get("drawdown_pct", 0) >= 10:
        attention.append(f"drawdown {status['drawdown_pct']:.1f}%")
    return {
        "t": int(now), "window_h": window_h, "mode": status.get("mode"), "running": status.get("running"),
        "equity": status.get("equity"), "equity_window_start": eq_then,
        "equity_change": (status.get("equity") - eq_then) if eq_then is not None and status.get("equity") is not None else None,
        "drawdown_pct": status.get("drawdown_pct"), "open_positions": status.get("open_positions"),
        "unrealized_pnl": status.get("unrealized_pnl"), "realized_pnl_total": status.get("realized_pnl"),
        "closed_in_window": len(recent), "wins_in_window": wins, "net_pnl_in_window": net,
        "win_rate_total": status.get("win_rate"), "safety": safety.get("state"), "drift": drift,
        "model": (status.get("model") or {}).get("name"), "shadow": shadow, "attention": attention,
    }


def try_promotions(api: str, summary: dict) -> list[str]:
    """Ask the engine to promote challengers that look better; the engine enforces the real rules."""
    out = []
    if summary["mode"] == "live":
        return ["live mode: no automatic promotion (manual only)"]
    for m in summary["shadow"]:
        if m.get("status") != "challenger":
            continue
        vs = m.get("vs_champion")
        if m.get("n_trades", 0) >= 50 and vs is not None and vs > 0:
            r = requests.post(api + "/api/control", json={"action": "promote_model", "model": m["model"]}, timeout=15)
            res = r.json() if r.ok else {"ok": False, "error": f"HTTP {r.status_code}"}
            out.append(f"promote {m['model']}: {'OK' if res.get('ok') else res.get('error')}")
    return out


def card_for(s: dict, promos: list[str]) -> tuple[str, dict]:
    chg = s["equity_change"]
    rows = [("Bot equity", f"{s['equity']:.2f}" if s["equity"] is not None else "n/a", None),
            (f"Change ({s['window_h']:g} h)", f"{chg:+.2f}" if chg is not None else "n/a",
             None if chg is None else (GREEN if chg >= 0 else RED)),
            ("Closed trades", f"{s['closed_in_window']} ({s['wins_in_window']} wins) net {s['net_pnl_in_window']:+.2f}",
             None),
            ("Open", f"{s['open_positions']} · uPnL {s['unrealized_pnl']:+.2f}" if s["unrealized_pnl"] is not None
             else str(s["open_positions"]), None),
            ("Drawdown", f"{s['drawdown_pct']:.2f}%", RED if (s["drawdown_pct"] or 0) >= 10 else None),
            ("Safety", s["safety"] or "?", GREEN if s["safety"] == "NORMAL" else AMBER),
            ("Model", s["model"] or "none", None)]
    for a in s["attention"]:
        rows.append(("Attention", a, RED))
    for p in promos:
        rows.append(("Model update", p, AMBER))
    if not promos:
        rows.append(("Model update", "none needed (no challenger qualifies)", None))
    head = "⏱ 3-hourly report" + (" · ⚠️ needs attention" if s["attention"] else "")
    return head, build_card(head, s["mode"] or "", rows, "report")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://127.0.0.1:8787")
    ap.add_argument("--window-h", type=float, default=3.0)
    ap.add_argument("--no-chat", action="store_true")
    ap.add_argument("--no-promote", action="store_true")
    a = ap.parse_args()
    load_env()
    reports = ROOT / "var" / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    try:
        status = get(a.api, "/api/status")
        trades = get(a.api, "/api/trades?limit=500")
        equity = get(a.api, "/api/equity?limit=5000")
        shadow = get(a.api, "/api/shadow")
    except Exception as e:
        msg = f"bot check: API unreachable at {a.api}: {type(e).__name__}"
        print(msg)
        if not a.no_chat:
            d = AlertDispatcher()
            d.send(msg, headline="🔌 Bot check: API unreachable",
                   card=build_card("🔌 Bot check failed", "API unreachable", [("API", a.api, RED)], "report"))
            d.flush(15)
            d.close()
        return 2
    s = summarise(status, trades, equity, shadow, a.window_h)
    promos = [] if a.no_promote else try_promotions(a.api, s)
    s["promotions"] = promos
    out = json.dumps(s, indent=2, default=str)
    # One latest-file per bot: latest.json for the default (testnet, :8787), latest_<port>.json otherwise.
    port = a.api.rstrip("/").rsplit(":", 1)[-1]
    suffix = "" if port == "8787" else f"_{port}"
    (reports / f"check{suffix}-{s['t']}.json").write_text(out)
    (reports / f"latest{suffix}.json").write_text(out)
    print(out)
    if not a.no_chat:
        head, card = card_for(s, promos)
        d = AlertDispatcher()
        d.send(head + "\n" + "; ".join(s["attention"]), headline=head, card=card)
        d.flush(15)
        d.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
