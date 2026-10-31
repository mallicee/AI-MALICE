"""Конфигурация: модель, провайдер, таймауты, обработчики."""
import ast

import pytest

import check_model
import main
from conftest import ROOT


@pytest.mark.parametrize("name", ["main.py", "check_model.py"])
def test_syntax(name):
    ast.parse((ROOT / name).read_text(encoding="utf-8"), filename=name)


def test_model_and_provider_unchanged():
    assert main.MODEL == "gemini-3.1-flash-lite"
    assert str(main.ai.base_url).rstrip("/") == "https://api.provod.ai/v1"
    assert main.MAX_TOKENS == 1500


def test_timeouts_and_retries():
    timeout = main.ai.timeout
    assert timeout.connect == 10.0
    assert timeout.read == 45.0
    assert timeout.write == 45.0
    assert main.ai.max_retries == 2


def test_handlers_registered():
    names = [h.callback.__name__ for h in main.dp.message.handlers]
    assert names == ["start", "reset", "chat", "photo", "image_document"]


def test_tests_do_not_use_real_secrets():
    assert main.BOT_TOKEN == "123456:TEST-TOKEN-NOT-REAL"
    assert main.PROVOD_API_KEY == "test-key-not-real"


def test_check_model_redacts_key(monkeypatch):
    monkeypatch.setattr(check_model, "key", "secret-value-123")
    text = check_model.safe("Invalid key secret-value-123 provided")
    assert "secret-value-123" not in text
    assert "***" in text
