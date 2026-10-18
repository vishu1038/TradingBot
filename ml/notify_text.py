"""Post a short message (plus the tail of a log file) to Google Chat: notify_text.py "message" [logfile]."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from tb.alerts import AlertDispatcher  # noqa: E402
from tb.config import load_env  # noqa: E402

load_env()
msg = sys.argv[1] if len(sys.argv) > 1 else "notice"
if len(sys.argv) > 2 and Path(sys.argv[2]).exists():
    tail = Path(sys.argv[2]).read_text(errors="replace").splitlines()[-25:]
    msg += "\n" + "\n".join(tail)
d = AlertDispatcher()
d.send(msg[:3500])
d.flush(30)
d.close()
