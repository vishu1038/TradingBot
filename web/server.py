"""A dependency-free web dashboard for the trading bot — stdlib http.server + SSE.

Why stdlib (no Flask/FastAPI)
-----------------------------
* It must run on the locked-down corporate machine, where `pip install` is unreliable.
* It must run on a BeagleBone (ARM, often offline) with nothing but CPython.
* The surface is tiny — a handful of routes and one event stream — so a framework buys
  nothing here.

What it serves
--------------
    GET  /                -> the single-page dashboard (web/dashboard.html)
    GET  /events          -> Server-Sent Events stream of EventBus events (live updates)
    GET  /api/state       -> current engine state snapshot (JSON)
    POST /api/start       -> start the trading engine
    POST /api/stop        -> stop the trading engine
    POST /api/train       -> kick off RL training on a background thread
    GET  /api/readiness   -> run the promotion evaluator and return its verdict

The server is wired with an `AppContext` carrying the engine, event bus, and a callable to
launch training — it never imports those modules itself, so it stays decoupled and testable.

Server-Sent Events (SSE) are used rather than WebSockets because they are one-directional
(server->browser, which is all a live dashboard needs), trivial over plain HTTP, survive
proxies, and need zero client libraries — `new EventSource('/events')` in the browser.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import typing
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

logger = logging.getLogger(__name__)

_HTML_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard.html")


class AppContext:
    """Everything the HTTP handlers need, injected so the server stays decoupled.

    Parameters
    ----------
    event_bus : the EventBus to stream to browsers.
    engine : object exposing start()/stop()/get_state()/assess_readiness() (the TradingEngine).
        May be None (dashboard then runs in view-only mode).
    start_training : callable taking no args that launches training (returns a summary dict).
        May be None to disable the Train button.
    """

    def __init__(self, event_bus, engine=None, start_training=None):
        self.event_bus = event_bus
        self.engine = engine
        self.start_training = start_training
        self._training_thread: typing.Optional[threading.Thread] = None
        self._lock = threading.Lock()

    def launch_training(self) -> dict:
        """Start training on a background thread if not already running."""
        if self.start_training is None:
            return {"ok": False, "error": "training not configured"}
        with self._lock:
            if self._training_thread is not None and self._training_thread.is_alive():
                return {"ok": False, "error": "training already in progress"}
            self._training_thread = threading.Thread(
                target=self._run_training, name="rl-training", daemon=True)
            self._training_thread.start()
        return {"ok": True, "started": True}

    def _run_training(self) -> None:
        try:
            self.start_training()
        except Exception as e:
            logger.error("Training thread crashed: %s", e, exc_info=True)


def make_handler(ctx: AppContext):
    """Build a request-handler class closed over the application context."""

    class Handler(BaseHTTPRequestHandler):
        # Quieter logging: route normal request lines through our logger at DEBUG.
        def log_message(self, fmt, *args):  # noqa: N802
            logger.debug("http: " + fmt, *args)

        # ----------------------------------------------------------- helpers
        def _send_json(self, obj, status: int = 200) -> None:
            body = json.dumps(obj).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _send_text(self, text: str, content_type: str = "text/html",
                       status: int = 200) -> None:
            body = text.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type + "; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        # ----------------------------------------------------------------- GET
        def do_GET(self):  # noqa: N802
            if self.path == "/" or self.path.startswith("/index"):
                return self._serve_index()
            if self.path.startswith("/events"):
                return self._serve_events()
            if self.path.startswith("/api/state"):
                state = ctx.engine.get_state() if ctx.engine else {}
                return self._send_json(state)
            if self.path.startswith("/api/readiness"):
                verdict = ctx.engine.assess_readiness() if ctx.engine else {
                    "ready": False, "reasons": ["no engine"]}
                return self._send_json(verdict)
            return self._send_text("not found", "text/plain", status=404)

        # ---------------------------------------------------------------- POST
        def do_POST(self):  # noqa: N802
            if self.path.startswith("/api/start"):
                if ctx.engine:
                    ctx.engine.start()
                    return self._send_json({"ok": True, "state": ctx.engine.get_state()})
                return self._send_json({"ok": False, "error": "no engine"}, status=400)
            if self.path.startswith("/api/stop"):
                if ctx.engine:
                    ctx.engine.stop()
                    return self._send_json({"ok": True, "state": ctx.engine.get_state()})
                return self._send_json({"ok": False, "error": "no engine"}, status=400)
            if self.path.startswith("/api/train"):
                return self._send_json(ctx.launch_training())
            return self._send_text("not found", "text/plain", status=404)

        # ------------------------------------------------------------- routes
        def _serve_index(self):
            try:
                with open(_HTML_PATH, "r", encoding="utf-8") as f:
                    return self._send_text(f.read())
            except OSError as e:
                return self._send_text(f"dashboard.html missing: {e}", "text/plain",
                                       status=500)

        def _serve_events(self):
            """Stream EventBus events to the browser as Server-Sent Events."""
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.send_header("X-Accel-Buffering", "no")  # disable proxy buffering
            self.end_headers()

            sub = ctx.event_bus.subscribe(replay=True)
            try:
                # Initial hello so the client knows it's connected.
                self._write_sse("hello", {"ok": True})
                for evt in sub.events(timeout=10.0):
                    if evt is None:
                        continue
                    self._write_sse(evt.type, evt.data, event_id=evt.seq)
            except (BrokenPipeError, ConnectionResetError):
                pass  # browser tab closed — normal
            except Exception as e:
                logger.debug("SSE stream ended: %s", e)
            finally:
                sub.close()

        def _write_sse(self, event: str, data, event_id=None) -> None:
            chunk = ""
            if event_id is not None:
                chunk += f"id: {event_id}\n"
            chunk += f"event: {event}\n"
            chunk += f"data: {json.dumps(data)}\n\n"
            self.wfile.write(chunk.encode("utf-8"))
            self.wfile.flush()

    return Handler


class DashboardServer:
    """Owns a ThreadingHTTPServer on a background thread."""

    def __init__(self, ctx: AppContext, host: str = "127.0.0.1", port: int = 8765):
        self.ctx = ctx
        self.host = host
        self.port = port
        self._httpd: typing.Optional[ThreadingHTTPServer] = None
        self._thread: typing.Optional[threading.Thread] = None

    @property
    def url(self) -> str:
        shown = "localhost" if self.host in ("127.0.0.1", "0.0.0.0") else self.host
        return f"http://{shown}:{self.port}/"

    def start(self) -> None:
        handler = make_handler(self.ctx)
        self._httpd = ThreadingHTTPServer((self.host, self.port), handler)
        self._thread = threading.Thread(target=self._httpd.serve_forever,
                                        name="dashboard-http", daemon=True)
        self._thread.start()
        logger.info("Dashboard serving at %s (bound %s:%d)", self.url, self.host, self.port)

    def stop(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            logger.info("Dashboard server stopped.")
