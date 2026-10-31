"""Ограничение частоты и антиспам-предупреждения."""
import pytest

import main


@pytest.fixture
def clock(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(main.time, "monotonic", lambda: now[0])
    return now


def test_min_interval(clock):
    assert main.check_rate_limit(1) == 0
    assert main.check_rate_limit(1) > 0
    clock[0] += main.MIN_INTERVAL
    assert main.check_rate_limit(1) == 0


def test_per_minute_limit_and_window(clock):
    for _ in range(main.RATE_LIMIT_REQUESTS):
        assert main.check_rate_limit(1) == 0
        clock[0] += main.MIN_INTERVAL
    wait = main.check_rate_limit(1)
    assert wait > 0
    clock[0] += wait
    assert main.check_rate_limit(1) == 0


def test_rejected_requests_do_not_extend_block(clock):
    main.check_rate_limit(1)
    for _ in range(5):
        main.check_rate_limit(1)          # отклонены, но не записаны
    clock[0] += main.MIN_INTERVAL
    assert main.check_rate_limit(1) == 0


def test_users_are_independent(clock):
    assert main.check_rate_limit(1) == 0
    assert main.check_rate_limit(2) == 0


def test_should_warn_throttled(clock):
    assert main.should_warn(1) is True
    assert main.should_warn(1) is False
    clock[0] += main.MIN_INTERVAL
    assert main.should_warn(1) is True
