# Deploying TradingBot on a BeagleBone (headless, web-accessible GUI)

The dashboard was built specifically so it can run on a BeagleBone Black / BeagleBone AI
and be reached from your laptop or phone over the network. This document is the plan to get
there. Nothing here has been run on real hardware yet — it is the deployment design plus the
exact commands to execute on the board.

---

## 1. Why this architecture fits the BeagleBone

| Constraint on a BBB | How the design handles it |
|---|---|
| **Headless** (no monitor; Tkinter is useless) | The live GUI is a **web dashboard** (`web/server.py`), reached from any browser on the LAN/phone. Tkinter (`main.py`) is for a desktop only and is *not* used on the BBB. |
| **No reliable `pip`** (limited flash, slow/absent network) | The dashboard + event bus + replay feed are **stdlib-only** (`http.server`, `queue`, `threading`, `json`). Only the *strategies* need numpy/pandas. |
| **Weak CPU / ~512 MB RAM** | The default RL agent is a tiny numpy **CEM linear policy** — no torch/SB3. EMA/ML strategies are light. Keep candle history modest (≤3000 bars). |
| **ARM architecture** | Pure-Python + numpy/pandas have ARM wheels. **Avoid LightGBM** (needs a compiler/OpenMP); the code already falls back to scikit-learn's `GradientBoostingClassifier`. |
| **Always-on, unattended** | Run under **systemd** so it auto-starts on boot and restarts on crash (unit below). |

---

## 2. One-time setup on the board

SSH in (default `debian@beaglebone.local`). Then:

```bash
# System Python 3 + venv + the few build-light deps
sudo apt-get update
sudo apt-get install -y python3 python3-venv python3-pip git

# Get the code (or rsync/scp it over)
git clone https://github.com/vishu1038/TradingBot.git
cd TradingBot
git checkout feature/automated-trading-scaffold

# Isolated environment
python3 -m venv .venv
source .venv/bin/activate

# Install ONLY what runs on the board. numpy/pandas/scikit-learn have ARM wheels.
# Do NOT install lightgbm (needs a compiler); the ML strategy falls back to sklearn.
pip install --upgrade pip
pip install numpy pandas scikit-learn requests websocket-client python-dateutil python-dotenv
```

If `pip install numpy` tries to compile from source (slow/може fail on a BBB), prefer the
distro packages instead: `sudo apt-get install -y python3-numpy python3-pandas python3-sklearn`
and create the venv with `--system-site-packages`.

### Credentials (only needed for real/testnet trading, not the replay demo)
```bash
cp .env.example .env
nano .env          # add Binance *testnet* keys; leave USE_TESTNET=true
```

---

## 3. Run it

### a) Offline demo (no exchange, no keys) — verify the board works
```bash
source .venv/bin/activate
python run_live_demo.py --host 0.0.0.0 --port 8765 --no-browser
```
`--host 0.0.0.0` binds all interfaces so it's reachable off-box. Then from your laptop/phone
on the same network:
```
http://beaglebone.local:8765/        # or  http://<board-ip>:8765/
```
Click **Start** — paper trades stream to your phone in real time. This proves the GUI,
event stream, and strategies all run on the hardware before any money/keys are involved.

### b) Live testnet paper trading (real market data, fake money)
Needs network access to Binance from the board and testnet keys in `.env`. Use a live
runner that swaps `ReplayClient` for `BinanceFuturesClient` (see §6). Same dashboard, same
URL — the only change is the data source.

---

## 4. Auto-start on boot (systemd)

Create `/etc/systemd/system/tradingbot.service`:

```ini
[Unit]
Description=TradingBot live dashboard
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=debian
WorkingDirectory=/home/debian/TradingBot
ExecStart=/home/debian/TradingBot/.venv/bin/python run_live_demo.py --host 0.0.0.0 --port 8765 --no-browser
Restart=on-failure
RestartSec=5
# Keep memory in check on a small board
MemoryMax=300M

[Install]
WantedBy=multi-user.target
```

Enable + start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable --now tradingbot
sudo systemctl status tradingbot          # check it's running
journalctl -u tradingbot -f               # follow logs
```

Now the dashboard is up at `http://<board-ip>:8765/` on every boot, restarting itself if it
crashes. For real trading, point `ExecStart` at the live runner from §6 instead.

---

## 5. Accessing it from your phone

- **Same Wi-Fi / LAN:** just open `http://<board-ip>:8765/`. The page is responsive (it
  reflows to one column on narrow screens) and uses Server-Sent Events, which work fine on
  mobile browsers.
- **Find the board IP:** `hostname -I` on the board, or use `beaglebone.local` (mDNS).
- **From anywhere (outside the LAN):** do **not** port-forward this to the open internet —
  there's no auth yet. Use one of:
  - **Tailscale / WireGuard** (recommended): install on the board and phone; reach it over
    the private mesh as if local. Zero exposure.
  - **SSH tunnel:** `ssh -L 8765:localhost:8765 debian@<board>` then browse `localhost:8765`.
- Phone push alerts already work independently of the GUI via the **Telegram notifier**
  (`alerts/`): set `TELEGRAM_BOT_TOKEN` + `TELEGRAM_CHAT_ID` in `.env` and you get fills,
  risk halts, and the "🎓 READY FOR LIVE" message on your phone even when the dashboard tab
  is closed.

---

## 6. Live runner (swap replay feed for the real exchange)

`run_live_demo.py` is paper-only by design (it uses `ReplayClient`). For live/testnet, use a
runner that builds a real connector and passes the **same** `event_bus`, `engine`, and
`start_training` into `AppContext` — the dashboard code does not change. Sketch:

```python
from connectors.binance_futures import BinanceFuturesClient
from config import CONFIG
client = BinanceFuturesClient(CONFIG.binance_public_key, CONFIG.binance_secret_key,
                              CONFIG.use_testnet)        # use_testnet=True for paper-on-real-data
engine = TradingEngine(client=client, ..., mode="paper", event_bus=GLOBAL_BUS)
# real money requires BOTH:  env ALLOW_LIVE_TRADING=true  AND  confirm_live=True
```
The promotion evaluator + Telegram alert tell you when the paper track record is good enough
to consider `mode="live"`. **Graduating is necessary but not sufficient** — start live with a
reduced position size.

---

## 7. Security hardening checklist (before any real money)

- [ ] Never expose port 8765 directly to the internet — use Tailscale/WireGuard or an SSH tunnel.
- [ ] `.env` is gitignored; keep keys off the board's git checkout and `chmod 600 .env`.
- [ ] Use **testnet keys** until the bot has graduated and you've done a reduced-size live trial.
- [ ] Keep `ALLOW_LIVE_TRADING` unset on the board until you explicitly want live orders.
- [ ] Set `MemoryMax` in the unit so a runaway can't take the board down.
- [ ] (Optional) Put the dashboard behind a reverse proxy with basic auth if more than you
      will ever touch it.

---

## 8. Hardware notes

- BeagleBone Black (1 GHz ARM Cortex-A8, 512 MB RAM, 4 GB eMMC) is enough for the replay
  demo and light live paper trading. RL **training** is CPU-heavy — train on your laptop and
  copy `models/rl_policy.npz` to the board, rather than training on the BBB.
- A CAN cape is irrelevant here (that's the other River projects) — this only needs network.
- Use a good 5 V supply; brownouts corrupt the eMMC. Consider running from an SD card so a
  corrupt filesystem is a cheap re-flash.
