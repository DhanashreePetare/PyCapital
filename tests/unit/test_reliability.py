"""Offline tests for pyequity.reliability (fake clock, fake sleep, no network)."""
import json
import logging

import pandas as pd
import pytest
import requests

from pyequity.reliability.outcome import FetchOutcome, Status
from pyequity.reliability.rate_limiter import RateLimiter
from pyequity.reliability.retry import (
    PermanentError, RetryableError, describe_response, retry_call)
from pyequity.reliability.session import SessionManager


class FakeClock:
    """Clock whose time only moves when sleep() is called."""

    def __init__(self):
        self.t = 100.0
        self.sleeps = []

    def now(self):
        return self.t

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.t += seconds


def http_error(status, headers=None):
    resp = requests.Response()
    resp.status_code = status
    resp.headers.update(headers or {})
    return requests.HTTPError(f"HTTP {status}", response=resp)


def script(*steps):
    """Operation that plays steps in order: exceptions are raised, others returned."""
    calls = {"n": 0}
    items = list(steps)

    def op():
        step = items[min(calls["n"], len(items) - 1)]
        calls["n"] += 1
        if isinstance(step, BaseException):
            raise step
        return step

    op.calls = calls
    return op


def run(op, **kw):
    clock = FakeClock()
    kw.setdefault("rng", lambda: 0.5)  # zero jitter
    outcome = retry_call(op, sleep=clock.sleep, **kw)
    return outcome, clock


# ----------------------------- outcome ------------------------------------

def test_outcome_flags():
    assert FetchOutcome(Status.OK, 1).ok
    assert FetchOutcome(Status.EMPTY).empty
    assert FetchOutcome(Status.FAILED).failed
    o = FetchOutcome(Status.OK, 1)
    assert not o.empty and not o.failed


def test_outcome_is_immutable():
    with pytest.raises(Exception):
        FetchOutcome(Status.OK).status = Status.FAILED


# ----------------------------- retry --------------------------------------

def test_success_first_try_no_sleep():
    outcome, clock = run(script(pd.DataFrame({"a": [1]})))
    assert outcome.ok and outcome.attempts == 1 and clock.sleeps == []


def test_retries_transient_then_succeeds():
    op = script(requests.ConnectionError("down"), requests.Timeout("slow"), [1, 2])
    outcome, clock = run(op, base_delay=1.0)
    assert outcome.ok and outcome.data == [1, 2] and outcome.attempts == 3
    assert clock.sleeps == [1.0, 2.0]  # exponential backoff


def test_backoff_is_capped():
    op = script(*[requests.ConnectionError("x")] * 6)
    outcome, clock = run(op, max_attempts=6, base_delay=10, max_delay=15)
    assert outcome.failed and outcome.attempts == 6
    assert clock.sleeps == [10, 15, 15, 15, 15]


def test_jitter_stays_within_bounds():
    op = script(requests.ConnectionError("x"), requests.ConnectionError("x"), 1)
    low, _ = run(op, base_delay=10, jitter=0.1, rng=lambda: 0.0)
    assert low.ok
    clock = FakeClock()
    retry_call(script(requests.ConnectionError("x"), 1), base_delay=10, jitter=0.1,
               rng=lambda: 0.0, sleep=clock.sleep)
    assert clock.sleeps == [pytest.approx(9.0)]
    clock = FakeClock()
    retry_call(script(requests.ConnectionError("x"), 1), base_delay=10, jitter=0.1,
               rng=lambda: 0.999999, sleep=clock.sleep)
    assert clock.sleeps == [pytest.approx(11.0, abs=1e-3)]


def test_gives_up_after_max_attempts_and_reports_failed():
    op = script(requests.ConnectionError("down"))
    outcome, clock = run(op, max_attempts=3)
    assert outcome.failed and outcome.data is None and outcome.attempts == 3
    assert "ConnectionError" in outcome.reason and op.calls["n"] == 3


def test_empty_dataframe_is_not_retried():
    op = script(pd.DataFrame())
    outcome, clock = run(op)
    assert outcome.empty and outcome.attempts == 1 and op.calls["n"] == 1
    assert clock.sleeps == []


@pytest.mark.parametrize("value", [None, [], {}, ""])
def test_default_empty_values_are_empty_not_failed(value):
    outcome, _ = run(script(value))
    assert outcome.empty and not outcome.failed


def test_custom_is_empty():
    outcome, _ = run(script({"data": []}), is_empty=lambda v: not v["data"])
    assert outcome.empty


def test_non_sized_value_counts_as_data():
    outcome, _ = run(script(42))
    assert outcome.ok and outcome.data == 42


@pytest.mark.parametrize("status", [403, 429, 500, 502, 503, 504])
def test_retryable_http_statuses(status):
    outcome, _ = run(script(http_error(status), "ok-data"))
    assert outcome.ok and outcome.attempts == 2


@pytest.mark.parametrize("status", [400, 401, 404])
def test_other_http_statuses_fail_immediately(status):
    op = script(http_error(status), "never reached")
    outcome, clock = run(op)
    assert outcome.failed and outcome.attempts == 1 and op.calls["n"] == 1
    assert clock.sleeps == []


def test_retry_after_header_is_honoured():
    op = script(http_error(429, {"Retry-After": "7"}), "ok-data")
    outcome, clock = run(op, base_delay=1.0)
    assert outcome.ok and clock.sleeps == [7.0]


def test_retry_after_is_capped_by_max_delay():
    op = script(http_error(429, {"Retry-After": "999"}), "ok-data")
    outcome, clock = run(op, max_delay=20)
    assert clock.sleeps == [20.0]


def test_unparseable_body_is_retried():
    op = script(json.JSONDecodeError("Expecting value", "", 0), [{"a": 1}])
    outcome, _ = run(op)
    assert outcome.ok and outcome.attempts == 2


def test_explicit_error_classes():
    outcome, _ = run(script(RetryableError("blocked"), "ok-data"))
    assert outcome.ok
    op = script(PermanentError("bad symbol"), "never reached")
    outcome, _ = run(op)
    assert outcome.failed and op.calls["n"] == 1


def test_programming_error_is_not_retried_but_never_raised(caplog):
    op = script(KeyError("missing"), "never reached")
    with caplog.at_level(logging.ERROR):
        outcome, clock = run(op)
    assert outcome.failed and op.calls["n"] == 1 and clock.sleeps == []
    assert "KeyError" in outcome.reason


def test_keyboard_interrupt_is_not_swallowed():
    with pytest.raises(KeyboardInterrupt):
        retry_call(script(KeyboardInterrupt()), sleep=lambda s: None)


def test_max_attempts_floor_is_one():
    op = script(requests.ConnectionError("x"))
    outcome, _ = run(op, max_attempts=0)
    assert outcome.failed and outcome.attempts == 1


def test_failed_and_empty_stay_distinct():
    failed, _ = run(script(requests.ConnectionError("x")), max_attempts=2)
    empty, _ = run(script(pd.DataFrame()))
    assert failed.status is Status.FAILED and empty.status is Status.EMPTY


def test_retry_logs_a_warning_per_retry(caplog):
    with caplog.at_level(logging.WARNING, logger="pyequity.reliability.retry"):
        run(script(requests.ConnectionError("x"), 1))
    assert any("retrying" in r.message for r in caplog.records)


def test_describe_response():
    r = requests.Response()
    r.status_code = 200
    r.headers.update({"Content-Type": "application/json", "Content-Encoding": "br"})
    text = describe_response(r)
    assert "status=200" in text and "content-encoding=br" in text
    assert describe_response(None) == "no response"


def test_http_error_log_includes_response_details():
    err = http_error(404, {"Content-Type": "text/html"})
    outcome, _ = run(script(err))
    assert "status=404" in outcome.reason and "text/html" in outcome.reason


# --------------------------- rate limiter ---------------------------------

def make_limiter(interval):
    clock = FakeClock()
    return RateLimiter(interval, clock=clock.now, sleep=clock.sleep), clock


def test_first_call_does_not_wait():
    limiter, clock = make_limiter(1.0)
    assert limiter.wait("nse") == 0.0 and clock.sleeps == []


def test_second_call_waits_for_interval():
    limiter, clock = make_limiter(1.0)
    limiter.wait("nse")
    assert limiter.wait("nse") == pytest.approx(1.0)


def test_no_wait_if_enough_time_passed():
    limiter, clock = make_limiter(1.0)
    limiter.wait("nse")
    clock.t += 5
    assert limiter.wait("nse") == 0.0


def test_partial_wait_after_partial_elapsed_time():
    limiter, clock = make_limiter(2.0)
    limiter.wait("nse")
    clock.t += 0.5
    assert limiter.wait("nse") == pytest.approx(1.5)


def test_keys_are_independent():
    limiter, clock = make_limiter(1.0)
    limiter.wait("nse")
    assert limiter.wait("bse") == 0.0


def test_burst_of_calls_is_spaced_out():
    limiter, clock = make_limiter(1.0)
    for _ in range(4):
        limiter.wait("nse")
    assert clock.t == pytest.approx(103.0)  # clock starts at 100; 3 waits of 1s


def test_zero_interval_never_waits():
    limiter, clock = make_limiter(0)
    for _ in range(5):
        assert limiter.wait() == 0.0
    assert clock.sleeps == []


def test_negative_interval_treated_as_zero():
    limiter, clock = make_limiter(-3)
    limiter.wait()
    assert limiter.wait() == 0.0


# --------------------------- session manager ------------------------------

class FakeSession:
    def __init__(self, name):
        self.name = name
        self.closed = False

    def close(self):
        self.closed = True


def make_manager(factory=None, **kw):
    clock = FakeClock()
    made = []

    def default_factory(key):
        s = FakeSession(f"{key}-{len(made)}")
        made.append(s)
        return s

    mgr = SessionManager(factory or default_factory, clock=clock.now,
                         sleep=clock.sleep, **kw)
    return mgr, clock, made


def test_session_is_reused_within_max_age():
    mgr, clock, made = make_manager(max_age=600)
    a = mgr.get("RELIANCE")
    clock.t += 100
    b = mgr.get("RELIANCE")
    assert a.ok and b.ok and a.data is b.data and len(made) == 1


def test_session_refreshed_after_max_age_and_old_one_closed():
    mgr, clock, made = make_manager(max_age=600)
    first = mgr.get("X").data
    clock.t += 601
    second = mgr.get("X").data
    assert first is not second and first.closed and not second.closed


def test_sessions_are_per_key():
    mgr, _, made = make_manager()
    a = mgr.get("TCS").data
    b = mgr.get(None).data
    assert a is not b and len(made) == 2


def test_invalidate_forces_new_session():
    mgr, _, made = make_manager()
    first = mgr.get("X").data
    mgr.invalidate("X")
    second = mgr.get("X").data
    assert first is not second and first.closed


def test_invalidate_unknown_key_is_harmless():
    mgr, _, _ = make_manager()
    mgr.invalidate("nope")


def test_creation_failure_returns_failed_and_never_raises():
    def broken(key):
        raise requests.ConnectionError("network down")

    mgr, clock, _ = make_manager(broken, max_attempts=3, base_delay=2)
    outcome = mgr.get("X")
    assert outcome.failed and outcome.data is None and outcome.attempts == 3
    assert "ConnectionError" in outcome.reason
    # default jitter is +/-10% around 2s then 4s
    assert 1.8 <= clock.sleeps[0] <= 2.2 and 3.6 <= clock.sleeps[1] <= 4.4
    assert len(clock.sleeps) == 2


def test_creation_recovers_after_transient_failure():
    state = {"n": 0}

    def flaky(key):
        state["n"] += 1
        if state["n"] < 3:
            raise requests.Timeout("slow")
        return FakeSession("ok")

    mgr, _, _ = make_manager(flaky, max_attempts=3)
    outcome = mgr.get("X")
    assert outcome.ok and outcome.attempts == 3


def test_failed_creation_is_not_cached():
    state = {"fail": True}

    def factory(key):
        if state["fail"]:
            raise requests.ConnectionError("down")
        return FakeSession("ok")

    mgr, _, _ = make_manager(factory, max_attempts=1)
    assert mgr.get("X").failed
    state["fail"] = False
    assert mgr.get("X").ok


def test_close_closes_everything():
    mgr, _, made = make_manager()
    mgr.get("A")
    mgr.get("B")
    mgr.close()
    assert all(s.closed for s in made)
    assert mgr.get("A").data is not made[0]


def test_close_error_does_not_propagate():
    class BadClose:
        def close(self):
            raise RuntimeError("boom")

    mgr, _, _ = make_manager(lambda key: BadClose())
    mgr.get("X")
    mgr.close()


def test_session_without_close_method_is_fine():
    mgr, _, _ = make_manager(lambda key: object())
    mgr.get("X")
    mgr.invalidate("X")


def test_default_factory_wraps_create_nse_session(monkeypatch):
    calls = []
    monkeypatch.setattr("pyequity.connectors._session_utils.create_nse_session",
                        lambda symbol=None: calls.append(symbol) or FakeSession("nse"))
    mgr = SessionManager(max_attempts=1)
    outcome = mgr.get("RELIANCE")
    assert outcome.ok and calls == ["RELIANCE"]


def test_no_network_is_touched(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("network used in an offline test")

    monkeypatch.setattr(requests.Session, "request", boom)
    run(script(1))
    make_manager()[0].get("X")