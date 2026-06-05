"""A tiny thread-safe publish/subscribe event bus — stdlib only.

The trading engine and the RL trainer run on background threads and need to push
live updates (fills, equity points, log lines, training progress) to any number of
listeners — chiefly the web dashboard's Server-Sent-Events stream, but also a console
logger or a test harness. They must never block on a slow/absent listener, and a
listener that goes away (browser tab closed) must not break the publishers.

Design
------
* `EventBus.publish(type, data)` fan-outs an `Event` to every live subscriber queue.
  It never blocks: if a subscriber's bounded queue is full, the OLDEST event in that
  queue is dropped to make room (live dashboards care about recent state, not history).
* `EventBus.subscribe()` returns a `Subscription` whose `.events()` generator yields
  events until the caller stops iterating or calls `.close()`. Each subscriber gets its
  own queue, so a slow subscriber can't slow the publisher or other subscribers.
* A bounded ring of recent events is kept so a subscriber that connects late (e.g. a
  browser that opens the dashboard after trading started) immediately receives a replay
  of recent state instead of a blank screen.

No third-party dependencies — works on the locked-down corporate box and on a BeagleBone.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
import typing

logger = logging.getLogger(__name__)

# Per-subscriber queue depth. Small: a live UI only needs recent events.
_SUBSCRIBER_QUEUE_MAX = 1000
# How many recent events to retain for late subscribers.
_REPLAY_BUFFER_MAX = 200


class Event:
    """An immutable-ish event: a type string, a JSON-serializable payload, and a timestamp."""

    __slots__ = ("type", "data", "ts_ms", "seq")

    def __init__(self, type: str, data: typing.Any, ts_ms: int, seq: int):
        self.type = type
        self.data = data
        self.ts_ms = ts_ms
        self.seq = seq

    def as_dict(self) -> dict:
        return {"type": self.type, "data": self.data, "ts": self.ts_ms, "seq": self.seq}


class Subscription:
    """A single listener's view of the bus. Iterate `.events()` to receive events."""

    def __init__(self, bus: "EventBus", q: "queue.Queue[Event]"):
        self._bus = bus
        self._q = q
        self._closed = False

    def events(self, timeout: float = 1.0) -> typing.Iterator[Event]:
        """Yield events as they arrive.

        Blocks up to `timeout` seconds waiting for each event; if none arrives it yields
        nothing for that tick and loops, which lets a consumer (e.g. an SSE handler) send
        periodic keep-alives and notice client disconnects. Stops when `.close()` is called.
        """
        while not self._closed:
            try:
                yield self._q.get(timeout=timeout)
            except queue.Empty:
                continue

    def get_nowait(self) -> typing.Optional[Event]:
        try:
            return self._q.get_nowait()
        except queue.Empty:
            return None

    def close(self) -> None:
        self._closed = True
        self._bus._remove(self)


class EventBus:
    """Fan-out events to many subscribers without ever blocking a publisher."""

    def __init__(self):
        self._subscribers: typing.List[typing.Tuple[Subscription, "queue.Queue[Event]"]] = []
        self._lock = threading.Lock()
        self._seq = 0
        self._replay: typing.List[Event] = []

    # ----------------------------------------------------------------- publish
    def publish(self, type: str, data: typing.Any = None) -> Event:
        """Broadcast an event to all subscribers. Never blocks; never raises on a bad sink."""
        with self._lock:
            self._seq += 1
            evt = Event(type=type, data=data, ts_ms=int(time.time() * 1000), seq=self._seq)
            self._replay.append(evt)
            if len(self._replay) > _REPLAY_BUFFER_MAX:
                del self._replay[: len(self._replay) - _REPLAY_BUFFER_MAX]
            targets = list(self._subscribers)

        for _sub, q in targets:
            self._offer(q, evt)
        return evt

    @staticmethod
    def _offer(q: "queue.Queue[Event]", evt: Event) -> None:
        """Put `evt` on `q`, dropping the oldest item if the queue is full (no blocking)."""
        try:
            q.put_nowait(evt)
        except queue.Full:
            try:
                q.get_nowait()       # drop oldest
            except queue.Empty:
                pass
            try:
                q.put_nowait(evt)
            except queue.Full:
                pass                 # give up; a stuck subscriber must not stall publishers

    # --------------------------------------------------------------- subscribe
    def subscribe(self, replay: bool = True) -> Subscription:
        """Register a new subscriber. If `replay`, seed it with recent buffered events."""
        q: "queue.Queue[Event]" = queue.Queue(maxsize=_SUBSCRIBER_QUEUE_MAX)
        sub = Subscription(self, q)
        with self._lock:
            if replay:
                for evt in self._replay:
                    self._offer(q, evt)
            self._subscribers.append((sub, q))
        return sub

    def _remove(self, sub: "Subscription") -> None:
        with self._lock:
            self._subscribers = [(s, q) for (s, q) in self._subscribers if s is not sub]

    @property
    def subscriber_count(self) -> int:
        with self._lock:
            return len(self._subscribers)


class BusLogHandler(logging.Handler):
    """A logging.Handler that re-publishes log records onto an EventBus as `log` events.

    Attach this to the root logger and the dashboard's log pane mirrors everything the
    bot logs (fills, warnings, risk halts) without any extra plumbing at each call site.
    """

    def __init__(self, bus: "EventBus", level: int = logging.INFO):
        super().__init__(level=level)
        self._bus = bus

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._bus.publish("log", {
                "level": record.levelname,
                "logger": record.name,
                "message": self.format(record),
            })
        except Exception:
            # A logging handler must never raise into the logging machinery.
            pass


# A process-wide default bus is convenient: the engine, trainer, and web server can all
# import the same instance without threading it through every constructor. Components that
# want isolation (tests) can still create their own EventBus.
GLOBAL_BUS = EventBus()
