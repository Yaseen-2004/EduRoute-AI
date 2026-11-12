"""
Access control and load limits for a deployed app.

- An access code gate (one shared code per deployment, e.g. per class), with
  a slowing delay after wrong attempts. The studio can have its own code.
- A per-session question rate limit.
- A server-wide cap on answers generated at the same time, so a burst of
  students queues instead of exhausting GPU memory.

The limiter and slot classes are plain Python (unit-tested); require_access
is the Streamlit part.
"""

import hmac
import threading
import time
from collections import defaultdict, deque


class SlidingWindowLimiter:
    """At most max_events per key in any window_s seconds (0 = unlimited)."""

    def __init__(self, max_events, window_s=3600):
        self.max_events, self.window_s = max_events, window_s
        self._events = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, key, now=None):
        if self.max_events <= 0:
            return True
        now = time.monotonic() if now is None else now
        with self._lock:
            if len(self._events) > 5000:     # forget sessions idle for a whole window
                for stale in [k for k, e in self._events.items() if not e or now - e[-1] >= self.window_s]:
                    del self._events[stale]
            events = self._events[key]
            while events and now - events[0] >= self.window_s:
                events.popleft()
            if len(events) >= self.max_events:
                return False
            events.append(now)
            return True

    def retry_after(self, key, now=None):
        """Seconds until the next event is allowed."""
        now = time.monotonic() if now is None else now
        events = self._events.get(key)
        if not events or len(events) < self.max_events:
            return 0
        return max(0, int(self.window_s - (now - events[0])) + 1)


class AnswerSlots:
    """Server-wide cap on concurrent answers (0 = unlimited). Use as
    `with slots.acquire(timeout) as ok:`; ok is False when no slot freed in time."""

    def __init__(self, limit):
        self.limit = limit
        self._sem = threading.BoundedSemaphore(limit) if limit > 0 else None
        self._busy = 0
        self._lock = threading.Lock()

    @property
    def busy(self):
        return self._busy

    def acquire(self, timeout):
        slots = self

        class _Slot:
            def __enter__(self):
                self.ok = slots._sem is None or slots._sem.acquire(timeout=timeout)
                if self.ok:
                    with slots._lock:
                        slots._busy += 1
                return self.ok

            def __exit__(self, *exc):
                if self.ok:
                    with slots._lock:
                        slots._busy -= 1
                    if slots._sem is not None:
                        slots._sem.release()
                return False
        return _Slot()


def code_matches(given, expected):
    """Constant-time comparison; an empty expected code means no gate."""
    if not expected:
        return True
    return hmac.compare_digest((given or "").strip().encode(), expected.encode())


_failures = {"count": 0}
_failures_lock = threading.Lock()


def require_access(scope, expected_code, title="EduRoute AI"):
    """Stop the page until this session has entered the code for `scope`."""
    import streamlit as st

    if not expected_code:
        return
    granted = st.session_state.setdefault("access_granted", set())
    if scope in granted:
        return
    st.title(title)
    st.caption("This deployment is restricted. Enter the access code from your instructor.")
    with st.form(f"access_{scope}"):
        code = st.text_input("Access code", type="password")
        submitted = st.form_submit_button("Continue", type="primary")
    if submitted:
        if code_matches(code, expected_code):
            granted.add(scope)
            with _failures_lock:
                _failures["count"] = 0
            st.rerun()
        with _failures_lock:   # wrong codes slow down every further attempt on this server
            _failures["count"] += 1
            delay = min(10, 0.5 * _failures["count"])
        time.sleep(delay)
        st.error("That code is not right.")
    st.stop()
