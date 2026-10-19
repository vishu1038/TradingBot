import json
import threading
import time

from tb.alerts import AlertDispatcher, GoogleChatNotifier, NotifyError, mask_url, notifier_from_env

URL = "https://chat.googleapis.com/v1/spaces/AAAA123/messages?key=SECRETKEY&token=SECRETTOKEN"


class Resp:
    def __init__(self, code=200, headers=None):
        self.status_code = code
        self.headers = headers or {}


class Sess:
    def __init__(self, *codes):
        self.calls = []
        self.codes = list(codes)
        self.times = []

    def post(self, url, data=None, headers=None, timeout=None):
        self.calls.append({"url": url, "data": data, "headers": headers})
        self.times.append(time.monotonic())
        c = self.codes.pop(0) if self.codes else 200
        if isinstance(c, Exception):
            raise c
        if isinstance(c, tuple):
            return Resp(c[0], c[1])
        return Resp(c)


def test_payload_shape_and_headers():
    s = Sess()
    GoogleChatNotifier(URL, session=s).send("hello ✓")
    c = s.calls[0]
    assert c["url"] == URL
    assert c["headers"]["Content-Type"] == "application/json; charset=UTF-8"
    assert json.loads(c["data"].decode("utf-8")) == {"text": "hello ✓"}


def test_masking_in_repr_and_errors():
    n = GoogleChatNotifier(URL, session=Sess(ConnectionError(f"HTTPSConnectionPool: failed for url: {URL}")))
    for txt in (repr(n), str(n), mask_url(URL)):
        assert "SECRETKEY" not in txt and "SECRETTOKEN" not in txt
    try:
        n.send("x")
    except NotifyError as e:
        assert "SECRETKEY" not in str(e) and "SECRETTOKEN" not in str(e) and e.retryable
    else:
        raise AssertionError("expected NotifyError")
    n2 = GoogleChatNotifier(URL, session=Sess(403))
    try:
        n2.send("x")
    except NotifyError as e:
        assert not e.retryable and "SECRET" not in str(e)


def test_noop_when_unset():
    assert notifier_from_env({}) == []
    d = AlertDispatcher(notifier_from_env({}))
    d.send("x")
    d.trade_opened("A", "long", 1, 1, 1, 1, 1)
    assert not d.enabled and d.sent == 0
    assert isinstance(notifier_from_env({"GOOGLE_CHAT_WEBHOOK_URL": URL})[0], GoogleChatNotifier)


def test_dispatch_is_non_blocking_and_coalesces_bursts():
    s = Sess()
    d = AlertDispatcher([GoogleChatNotifier(URL, session=s)], min_interval=0.3)
    t0 = time.monotonic()
    for i in range(20):
        d.send(f"msg {i}")
    assert time.monotonic() - t0 < 0.05            # engine never blocks
    assert d.flush(5)
    texts = [json.loads(c["data"])["text"] for c in s.calls]
    assert len(s.calls) <= 3                       # 20 messages coalesced into a few posts
    assert sum(t.count("msg ") for t in texts) == 20
    assert "msg 0" in texts[0] and "msg 19" in texts[-1]
    d.close()


def test_rate_limit_spacing():
    s = Sess()
    d = AlertDispatcher([GoogleChatNotifier(URL, session=s)], min_interval=0.25)
    d.send("a")
    d.flush(3)
    time.sleep(0.01)
    d.send("b")
    d.flush(3)
    d.close()
    assert len(s.calls) == 2 and s.times[1] - s.times[0] >= 0.24


def test_retry_with_backoff_on_429_and_5xx():
    s = Sess((429, {"Retry-After": "0.05"}), 503, 200)
    d = AlertDispatcher([GoogleChatNotifier(URL, session=s)], min_interval=0, backoff_base=0.05)
    d.send("retry me")
    assert d.flush(5)
    d.close()
    assert len(s.calls) == 3 and d.sent == 1 and d.failed == 0


def test_gives_up_after_max_attempts_and_on_permanent_error():
    s = Sess(500, 500, 500)
    d = AlertDispatcher([GoogleChatNotifier(URL, session=s)], min_interval=0, max_attempts=3, backoff_base=0.01)
    d.send("x")
    d.flush(5)
    assert len(s.calls) == 3 and d.failed == 1
    s2 = Sess(404)
    d2 = AlertDispatcher([GoogleChatNotifier(URL, session=s2)], min_interval=0, max_attempts=5)
    d2.send("y")
    d2.flush(5)
    assert len(s2.calls) == 1 and d2.failed == 1


def test_error_alerts_deduped():
    s = Sess()
    d = AlertDispatcher([GoogleChatNotifier(URL, session=s)], min_interval=0)
    for _ in range(5):
        d.engine_error("bars", "boom")
    d.engine_error("prices", "other")
    d.flush(3)
    text = "".join(json.loads(c["data"])["text"] for c in s.calls)
    assert text.count("Engine error in bars") == 1 and text.count("Engine error in prices") == 1


def test_event_formats():
    got = []

    class N:
        name = "mem"

        def send(self, text):
            got.append(text)

    d = AlertDispatcher([N()], min_interval=0)
    d.trade_opened("ETHUSDT", "long", 2364.8, 0.067, 158.4, 2215.5, 2439.4, 0.81, "paper_local")
    d.flush(2)
    d.trade_closed("ETHUSDT", "long", 2364.8, 2439.4, 5.0, 0.16, "tp", "paper_local")
    d.flush(2)
    d.kill("drawdown", 49.0, 51.0, "paper_local")
    d.daily_summary(date="2026-10-18", trades=4, wins=3, pnl=2.5, equity=102.5, mode="paper_local")
    d.flush(2)
    joined = "\n".join(got)
    for needle in ("OPEN LONG ETHUSDT", "CLOSE LONG ETHUSDT", "reason=tp", "KILL SWITCH", "DAILY 2026-10-18", "win rate 75%"):
        assert needle in joined


def test_google_chat_gets_cards_with_headline_text():
    s = Sess()
    d = AlertDispatcher([GoogleChatNotifier(URL, session=s)], min_interval=0.2)
    d.trade_opened("ETHUSDT", "long", 2364.8, 0.067, 158.4, 2215.5, 2439.4, 0.81, "binance_demo")
    d.trade_closed("ETH<USDT>", "long", 2364.8, 2215.5, -10.2, 0.16, "sl", "binance_demo")
    d.safety("DEGRADED", ["guard heartbeat stale"], "binance_demo")
    assert d.flush(5)
    d.close()
    bodies = [json.loads(c["data"]) for c in s.calls]
    cards = [c for b in bodies for c in b.get("cardsV2", [])]
    assert len(cards) == 3 and len({c["cardId"] for c in cards}) == 3       # coalesced, unique ids
    titles = [c["card"]["header"]["title"] for c in cards]
    assert titles[0] == "▲ OPEN LONG · ETHUSDT" and "SL hit" in titles[1] and "DEGRADED" in titles[2]
    rows = {w["decoratedText"]["topLabel"]: w["decoratedText"]["text"]
            for w in cards[1]["card"]["sections"][0]["widgets"] if "decoratedText" in w}
    assert '<font color="#d93025">-10.20 USDT</font>' == rows["Net PnL"]   # loss in red
    assert "ETH&lt;USDT&gt;" in titles[1]                                  # HTML escaped
    text = "\n".join(b["text"] for b in bodies)
    assert "OPEN LONG ETHUSDT" in text and "net PnL" not in text            # short headlines, details in cards
    assert "SECRET" not in json.dumps(bodies)


def test_dashboard_button_only_when_configured(monkeypatch):
    from tb.alerts import build_card
    monkeypatch.delenv("TB_DASHBOARD_URL", raising=False)
    assert all("buttonList" not in w for w in build_card("t", "s", [("a", 1, None)])["card"]["sections"][0]["widgets"])
    monkeypatch.setenv("TB_DASHBOARD_URL", "http://mac-pro.example.ts.net:8787/")
    w = build_card("t", "s", [("a", 1, None)])["card"]["sections"][0]["widgets"][-1]
    assert w["buttonList"]["buttons"][0]["onClick"]["openLink"]["url"] == "http://mac-pro.example.ts.net:8787/"


def test_worker_survives_notifier_exception():
    class Bad:
        name = "bad"

        def send(self, text):
            raise RuntimeError("kaboom")

    ok = Sess()
    d = AlertDispatcher([Bad(), GoogleChatNotifier(URL, session=ok)], min_interval=0)
    d.send("x")
    d.flush(3)
    assert len(ok.calls) == 1


def test_every_message_carries_bot_stats_footer():
    from tb.alerts import status_rows
    s = Sess()
    st = {"mode": "paper_local", "equity": 152.1, "start_equity": 200.0, "peak_equity": 210.0, "drawdown_pct": 27.6,
          "open_positions": 9, "open_risk_pct": 31.0, "unrealized_pnl": -5.5,
          "kill": {"tripped": False, "dd_limit_pct": 50.0}}
    d = AlertDispatcher([GoogleChatNotifier(URL, session=s)], min_interval=0, stats=lambda: status_rows(st, "PAPER"))
    d.send("plain text notice")                                   # no card: one is made for it
    d.flush(3)
    d.trade_opened("ETHUSDT", "long", 2364.8, 0.067, 158.4, 2215.5, 2439.4, 0.81, "paper_local")
    d.flush(3)
    d.close()
    assert len(s.calls) == 2
    for c in s.calls:
        body = json.loads(c["data"])
        footer = body["cardsV2"][-1]["card"]["sections"][-1]
        labels = [w["decoratedText"]["topLabel"] for w in footer["widgets"]]
        assert footer["header"] == "🤖 Bot" and labels[:3] == ["PAPER · Equity", "PAPER · PnL", "PAPER · Drawdown"]
        assert "-47.90 (-24.0 %)" in footer["widgets"][1]["decoratedText"]["text"]
        assert "27.6 % of 50 % kill" in footer["widgets"][2]["decoratedText"]["text"]


def test_stats_failure_never_blocks_alert():
    s = Sess()
    d = AlertDispatcher([GoogleChatNotifier(URL, session=s)], min_interval=0, stats=lambda: 1 / 0)
    d.send("x")
    d.flush(3)
    d.close()
    assert len(s.calls) == 1 and json.loads(s.calls[0]["data"]) == {"text": "x"}
