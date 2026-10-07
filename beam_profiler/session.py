"""Thread-safe handoff between the acquisition thread and HTTP clients.

Publish a whole snapshot at once. Readers keep that snapshot for their entire
request; they never access camera drivers or native UI objects. Control requests
are queued and applied by the acquisition thread before its next frame.
"""
from collections import deque
from dataclasses import dataclass, field
from threading import Lock

from .processing.pipeline import FrameResult


@dataclass(frozen=True)
class SessionSnapshot:
    result: FrameResult | None = None
    status: dict = field(default_factory=dict)
    angular: dict | None = None


class LiveSession:
    def __init__(self):
        self._lock = Lock()
        self._snapshot = SessionSnapshot()
        self._commands = deque()

    def publish(self, snapshot: SessionSnapshot):
        with self._lock:
            self._snapshot = snapshot

    def read(self) -> SessionSnapshot:
        with self._lock:
            return self._snapshot

    def set_region(self, name, coords):
        if name not in ('rect_sensor', 'fit_rect_sensor'):
            raise ValueError('Unknown region')
        with self._lock:
            self._commands.append((name, coords))

    def set_fit_config(self, changes):
        with self._lock:
            self._commands.append(("fit_config", dict(changes)))

    def take_commands(self):
        with self._lock:
            commands = list(self._commands)
            self._commands.clear()
        return commands
