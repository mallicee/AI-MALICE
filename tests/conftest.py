"""Общие фикстуры: имитация Telegram и Provod.ai, без сети и без секретов."""
import asyncio
import itertools
import os
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Фиктивные значения задаются ДО импорта main: load_dotenv не перезаписывает
# уже заданные переменные, поэтому настоящий .env в тестах не используется.
os.environ["BOT_TOKEN"] = "123456:TEST-TOKEN-NOT-REAL"
os.environ["OPENAI_API_KEY"] = "test-key-not-real"

import httpx2 as httpx  # noqa: E402
import openai  # noqa: E402

import main  # noqa: E402

_message_ids = itertools.count(1)


class FakeMessage:
    """Минимальная замена aiogram Message для обработчиков из main.py."""

    def __init__(
        self,
        chat_id=1,
        user_id=None,
        text=None,
        photo=None,
        document=None,
        caption=None,
        message_id=None,
        thread_id=None,
        is_topic=False,
        fail_send=False,
    ):
        self.message_id = message_id or next(_message_ids)
        self.chat = NS(id=chat_id)
        self.from_user = NS(id=user_id if user_id is not None else chat_id)
        self.text = text
        self.photo = photo
        self.document = document
        self.caption = caption
        self.message_thread_id = thread_id
        self.is_topic_message = is_topic
        self.fail_send = fail_send
        self.replies = []

    async def answer(self, text, **kwargs):
        if self.fail_send:
            self.fail_send = False
            raise RuntimeError("telegram send failed")
        self.replies.append(text)


class FakeTelegram:
    def __init__(self):
        self.actions = []
        self.files = {}          # file_id -> (bytes, file_size из get_file)
        self.get_file_calls = 0
        self.download_timeouts = []
        self.fail_actions = False

    def add_file(self, file_id, data, size=None):
        self.files[file_id] = (data, len(data) if size is None else size)

    async def send_chat_action(self, **kwargs):
        if self.fail_actions:
            raise RuntimeError("telegram down")
        self.actions.append(kwargs)

    async def get_file(self, file_id, **kwargs):
        self.get_file_calls += 1
        _, size = self.files[file_id]
        return NS(file_id=file_id, file_path=f"photos/{file_id}", file_size=size)

    async def download_file(self, file_path, destination=None, timeout=30, **kwargs):
        self.download_timeouts.append(timeout)
        data, _ = self.files[file_path.split("/", 1)[1]]
        destination.write(data)
        destination.seek(0)
        return destination


class FakeAI:
    """Имитация chat.completions.create. По умолчанию: «ответ на <вопрос>»."""

    def __init__(self):
        self.requests = []
        self.delay = 0.0
        self.error = None
        self.responder = None    # fn(messages) -> (content, finish_reason)

    async def create(self, **kwargs):
        self.requests.append(kwargs)
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.error is not None:
            raise self.error
        if self.responder:
            content, finish = self.responder(kwargs["messages"])
        else:
            last = kwargs["messages"][-1]["content"]
            content = "описание изображения" if isinstance(last, list) else f"ответ на {last}"
            finish = "stop"
        return NS(
            choices=[NS(message=NS(content=content), finish_reason=finish)],
            usage=NS(prompt_tokens=10, completion_tokens=5),
        )

    @property
    def last_messages(self):
        return self.requests[-1]["messages"]


@pytest.fixture(autouse=True)
def clean_state():
    """Каждый тест начинает с пустой памяти, лимитов и флагов."""
    def clear():
        main.chat_histories.clear()
        main.processed_messages.clear()
        main.user_requests.clear()
        main.busy_users.clear()
        main.warned_users.clear()
    clear()
    yield
    clear()


@pytest.fixture(autouse=True)
def tg(monkeypatch):
    fake = FakeTelegram()
    monkeypatch.setattr(main.bot, "send_chat_action", fake.send_chat_action)
    monkeypatch.setattr(main.bot, "get_file", fake.get_file)
    monkeypatch.setattr(main.bot, "download_file", fake.download_file)
    return fake


@pytest.fixture(autouse=True)
def ai(monkeypatch):
    fake = FakeAI()
    monkeypatch.setattr(main.ai.chat.completions, "create", fake.create)
    return fake


@pytest.fixture
def no_limits(monkeypatch):
    """Отключает лимит частоты для тестов, где он не проверяется."""
    monkeypatch.setattr(main, "MIN_INTERVAL", 0)
    monkeypatch.setattr(main, "RATE_LIMIT_REQUESTS", 10**6)


@pytest.fixture
def msg():
    return FakeMessage


def api_error(kind):
    request = httpx.Request("POST", "https://api.provod.ai/v1/chat/completions")
    if kind == "timeout":
        return openai.APITimeoutError(request=request)
    if kind == "connection":
        return openai.APIConnectionError(request=request)
    status = int(kind)
    cls = {429: openai.RateLimitError, 500: openai.InternalServerError}.get(
        status, openai.APIStatusError
    )
    return cls("error", response=httpx.Response(status, request=request), body=None)
