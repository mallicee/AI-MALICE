"""Текстовый чат: ответы, обрезка, одновременность, спам, «печатает...», ошибки."""
import asyncio

import pytest

import main
from conftest import api_error


@pytest.mark.parametrize(
    "content, finish, expected",
    [
        ("текст", "stop", "текст"),
        ("текст", "length", "текст\n\n✂️ Ответ получился слишком длинным"),
        ("текст", "LENGTH", "текст\n\n✂️ Ответ получился слишком длинным"),
        ("", "length", "Ответ не уместился в лимит длины"),
        ("", "stop", "Модель вернула пустой ответ"),
    ],
)
async def test_finish_reason(ai, msg, content, finish, expected):
    ai.responder = lambda messages: (content, finish)
    m = msg(text="вопрос")
    await main.chat(m)
    assert m.replies[-1].startswith(expected)


async def test_long_answer_split(ai, msg):
    ai.responder = lambda messages: ("x" * 9000, "stop")
    m = msg(text="длинно")
    await main.chat(m)
    assert [len(r) for r in m.replies] == [4000, 4000, 1000]


async def test_empty_text_ignored(ai, msg):
    await main.chat(msg(text="   "))
    assert ai.requests == []


async def test_concurrent_requests_single_ai_call(ai, msg):
    ai.delay = 0.2
    await asyncio.gather(*(main.chat(msg(chat_id=5, text=f"q{i}")) for i in range(5)))
    assert len(ai.requests) == 1
    assert main.busy_users == set()


async def test_busy_cleared_after_error(ai, msg):
    ai.error = RuntimeError("boom")
    m = msg(text="q")
    await main.chat(m)
    assert m.replies == ["Произошла ошибка при обработке сообщения. Попробуй ещё раз."]
    assert main.busy_users == set()


async def test_spam_gets_single_warning(ai, msg):
    ai.delay = 0.2
    spam = [msg(chat_id=7, text=f"s{i}") for i in range(30)]
    await asyncio.gather(*(main.chat(m) for m in spam))
    assert len(ai.requests) == 1
    assert sum(len(m.replies) for m in spam) == 2      # 1 ответ + 1 предупреждение


async def test_duplicate_update_skipped(ai, msg):
    first = msg(text="q")
    await main.chat(first)
    dup = msg(text="q", message_id=first.message_id)
    await main.chat(dup)
    assert len(ai.requests) == 1 and dup.replies == []


async def test_typing_in_forum_topic(tg, ai, msg):
    ai.delay = 0.05   # реальный ответ идёт секунды; мгновенный не успевает
    await main.chat(msg(text="q", thread_id=77, is_topic=True))
    assert tg.actions[0]["message_thread_id"] == 77
    assert tg.actions[0]["action"] == "typing"


async def test_typing_failure_does_not_break_answer(tg, msg):
    tg.fail_actions = True
    m = msg(text="q")
    await main.chat(m)
    assert m.replies == ["ответ на q"]


@pytest.mark.parametrize(
    "kind, expected",
    [
        ("429", "Сейчас слишком много запросов к ИИ. Попробуй через минуту."),
        ("timeout", "ИИ сейчас не отвечает. Попробуй чуть позже."),
        ("connection", "ИИ сейчас не отвечает. Попробуй чуть позже."),
        ("402", "ИИ временно недоступен. Попробуй позже."),
        ("500", "ИИ временно недоступен. Попробуй позже."),
    ],
)
async def test_api_errors_user_friendly(ai, msg, kind, expected):
    ai.error = api_error(kind)
    m = msg(text="q")
    await main.chat(m)
    assert m.replies == [expected]
    assert main.busy_users == set()


async def test_usage_log_has_no_text(ai, msg, caplog):
    caplog.set_level("INFO")
    await main.chat(msg(chat_id=9, text="секретный вопрос"))
    usage = [r for r in caplog.messages if r.startswith("Модель:")]
    assert usage and "chat_id=9" in usage[0] and "входных токенов: 10" in usage[0]
    assert not any("секретный" in r for r in caplog.messages)
