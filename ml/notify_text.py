"""Post a training/ops notice to Google Chat as a readable card: notify_text.py "message" [file].

Google Chat does not render markdown tables, so tables in ``file`` (e.g. ml/compare.py output) are turned
into card sections: one row per table line ("tag: trades 618 · win 62.1 % · $/trade -0.53 · …"), plus the
VERDICT/holdout lines. Every notice ends with both bots' current stats (testnet :8787, paper :8788).
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

import requests  # noqa: E402

from tb.alerts import AlertDispatcher, build_card, stats_section, status_rows  # noqa: E402
from tb.config import load_env  # noqa: E402

BOTS = (("TESTNET", "http://127.0.0.1:8787"), ("PAPER", "http://127.0.0.1:8788"))
KEEP = ("trades", "win", "exp", "random", "atr", "z vs", "auc", "net", "sharpe", "dd", "verdict")
MAX_ROWS = 40  # Chat card size limits


def _cell(h: str) -> bool:
    h = h.lower()
    return any(k in h for k in KEEP)


def tables_to_sections(text: str) -> list[dict]:
    sections: list[dict] = []
    title, header, widgets, rows = "", None, [], 0

    def flush():
        nonlocal widgets
        if widgets:
            sections.append({"header": title[:120] or "Results", "widgets": widgets})
        widgets = []

    for line in text.splitlines():
        s = line.strip()
        if s.startswith("#"):
            flush()
            title, header = s.lstrip("#").strip(), None
        elif s.startswith("|") and s.endswith("|"):
            cells = [c.strip() for c in s.strip("|").split("|")]
            if all(re.fullmatch(r":?-{2,}:?", c) for c in cells):
                continue
            if header is None:
                header = cells
                continue
            if rows >= MAX_ROWS:
                continue
            parts = [f"{h} {v}" for h, v in zip(header[1:], cells[1:]) if _cell(h)]
            widgets.append({"decoratedText": {"topLabel": cells[0][:60], "text": " · ".join(parts)[:350],
                                              "wrapText": True}})
            rows += 1
        elif s and (s.upper().startswith(("VERDICT", "LOCKED", "EXPORTED", "HOLDOUT")) or "pooled OOS" in s
                    or s.lower().startswith(("long:", "short:", "paired"))):
            header = None if s.lower().startswith(("long:", "short:")) else header
            if rows < MAX_ROWS:
                widgets.append({"textParagraph": {"text": s[:500]}})
                rows += 1
    flush()
    return sections


def bot_rows() -> list[tuple]:
    rows: list[tuple] = []
    for name, api in BOTS:
        try:
            rows += status_rows(requests.get(api + "/api/status", timeout=5).json(), name)
        except Exception:
            rows.append((name, "API unreachable", None))
    return rows


def main() -> None:
    load_env()
    msg = sys.argv[1] if len(sys.argv) > 1 else "notice"
    body = ""
    if len(sys.argv) > 2 and Path(sys.argv[2]).exists():
        body = Path(sys.argv[2]).read_text(errors="replace")
    card = build_card("🧪 " + msg.splitlines()[0][:110], "TradingBot training", [], "train")
    sections = tables_to_sections(body) if body else []
    if body and not sections:  # a log tail, not a table report
        tail = "\n".join(body.splitlines()[-15:])
        sections = [{"header": "Log tail", "widgets": [{"textParagraph": {"text": tail[-2500:]}}]}]
    extra = msg.splitlines()[1:]
    if extra:
        sections.insert(0, {"widgets": [{"textParagraph": {"text": "\n".join(extra)[:1500]}}]})
    card["card"]["sections"] = [s for s in card["card"]["sections"] if s.get("widgets")] + sections
    d = AlertDispatcher(stats=bot_rows)
    d.send(msg[:300] + ("\n" + body[-1500:] if body else ""), headline="🧪 " + msg.splitlines()[0][:120], card=card)
    d.flush(30)
    d.close()


if __name__ == "__main__":
    main()
