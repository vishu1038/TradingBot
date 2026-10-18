"""After the paired depth test: run ml/compare.py and post the result to Google Chat (no Claude needed).

Usage: .venv/bin/python ml/notify_depth_result.py
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from tb.alerts import AMBER, AlertDispatcher, build_card  # noqa: E402
from tb.config import load_env  # noqa: E402

load_env()
cmd = [str(ROOT / ".venv" / "bin" / "python"), str(ROOT / "ml" / "compare.py"),
       "--arch", "cnn", "--modes", "fixed,atr32", "--tags", "depth_off,depth_on"]
r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=600)
out = (r.stdout or "") + (("\n[stderr]\n" + r.stderr[-1500:]) if r.returncode else "")
stamp = time.strftime("%Y%m%d-%H%M")
report = ROOT / "var" / "ml" / f"depth_pair_result_{stamp}.txt"
report.parent.mkdir(parents=True, exist_ok=True)
report.write_text(out)
print(out)

d = AlertDispatcher()
card = build_card("🧪 Order-book depth test finished", "with vs without depth features",
                  [("Result file", str(report.relative_to(ROOT)), None),
                   ("Compare exit code", r.returncode, AMBER if r.returncode else None),
                   ("Read", "full table in the next message / result file", None)], "depth")
d.send("Depth test finished", headline="🧪 Order-book depth test finished", card=card)
d.send("Depth test (depth_off vs depth_on):\n" + out[-3500:])
d.flush(30)
d.close()
