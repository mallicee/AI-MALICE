"""Защита от запуска второй копии бота."""
import socket

import pytest

import main


@pytest.fixture
def free_port(monkeypatch):
    # Отдельный свободный порт: тест не мешает запущенному боту
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    monkeypatch.setattr(main, "INSTANCE_LOCK_PORT", port)
    monkeypatch.setattr(main, "_instance_lock", None)
    yield port
    if main._instance_lock is not None:
        main._instance_lock.close()


def test_second_instance_blocked(free_port):
    assert main.acquire_instance_lock() is True
    first = main._instance_lock
    assert main.acquire_instance_lock() is False
    first.close()
    assert main.acquire_instance_lock() is True
