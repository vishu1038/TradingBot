"""User rule (2026-10-19): the bot never SENDS a reduce-only order. Entries and closes are normal orders, TP/SL are
closePosition conditional orders. (Binance itself echoes reduceOnly=true on closePosition orders; that is not ours.)"""
import io
import tokenize
from pathlib import Path

from tb.exchange.binance import BinanceExchange
from tb.exchange.base import SymbolFilters

TB = Path(__file__).resolve().parents[1] / "tb"


def _ex():
    ex = BinanceExchange("https://example.invalid", "k", "s", mode="binance_demo")
    ex.filters = lambda symbol: SymbolFilters(0.01, 0.001, 0.001, 5.0)
    return ex


def test_order_param_builders_never_carry_reduce_only():
    ex = _ex()
    sent = [ex.market_order_params("AAAUSDT", "SELL", 0.5)]
    for kind in ("SL", "TP"):
        for algo in (False, True):
            sent.append(ex.conditional_params(kind, "AAAUSDT", "SELL", 99.0, 0.5, algo))
    for p in sent:
        assert not any(k.lower() in ("reduceonly", "reduce_only") for k in p), p
    for p in sent[1:]:
        assert p["closePosition"] is True and "quantity" not in p


def test_no_reduce_only_string_in_the_order_code():
    """Every string literal in the order path (comments and docstrings excluded) is free of reduceOnly."""
    for f in [TB / "exchange" / "binance.py", TB / "exchange" / "paper.py", TB / "emergency.py", TB / "mirror.py",
              TB / "engine.py", TB / "guard.py"]:
        toks = tokenize.generate_tokens(io.StringIO(f.read_text()).readline)
        prev = None
        for t in toks:
            if t.type == tokenize.STRING and prev not in (tokenize.INDENT, tokenize.NEWLINE, tokenize.NL, None):
                assert "reduceonly" not in t.string.lower().replace("_", ""), f"{f.name}:{t.start[0]} {t.string[:60]}"
            if t.type not in (tokenize.COMMENT,):
                prev = t.type
