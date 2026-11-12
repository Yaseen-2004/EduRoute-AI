"""Access code, rate limit and concurrent-answer cap (production deployment)."""

import threading
import time

from src.utils.access import AnswerSlots, SlidingWindowLimiter, code_matches


def test_access_code():
    assert code_matches("anything", "")                 # no code configured = open
    assert code_matches(" class-2026 ", "class-2026")   # stray spaces forgiven
    assert not code_matches("class-2025", "class-2026")
    assert not code_matches(None, "class-2026")


def test_rate_limit_window():
    limiter = SlidingWindowLimiter(3, window_s=3600)
    assert all(limiter.allow("s1", now=t) for t in (0, 10, 20))
    assert not limiter.allow("s1", now=30)
    assert limiter.retry_after("s1", now=30) == 3571
    assert limiter.allow("s2", now=30)                  # sessions are independent
    assert limiter.allow("s1", now=3601)                # the first question left the window


def test_idle_sessions_are_forgotten():
    limiter = SlidingWindowLimiter(5, window_s=60)
    for i in range(5001):
        limiter.allow(f"s{i}", now=0)
    limiter.allow("new", now=120)            # everyone else has been idle for a full window
    assert len(limiter._events) == 1


def test_rate_limit_zero_means_unlimited():
    limiter = SlidingWindowLimiter(0)
    assert all(limiter.allow("s", now=t) for t in range(1000))


def test_answer_slots_queue_and_time_out():
    slots = AnswerSlots(1)
    holding, release = threading.Event(), threading.Event()

    def hold():
        with slots.acquire(1) as ok:
            assert ok
            holding.set()
            release.wait(5)

    worker = threading.Thread(target=hold)
    worker.start()
    holding.wait(5)
    assert slots.busy == 1
    start = time.monotonic()
    with slots.acquire(0.2) as ok:                      # the only slot is taken
        assert not ok
    assert time.monotonic() - start >= 0.2
    release.set()
    worker.join()
    assert slots.busy == 0
    with slots.acquire(0.2) as ok:                      # freed again
        assert ok


def test_unlimited_slots_never_block():
    slots = AnswerSlots(0)
    with slots.acquire(0) as a, slots.acquire(0) as b:
        assert a and b
