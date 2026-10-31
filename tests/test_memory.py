"""Память диалогов: сохранение, контекст, лимит, изоляция, /start, /reset."""
import asyncio
import re

import pytest

import main

pytestmark = pytest.mark.usefixtures("no_limits")


def hist(chat_id, user_id=None):
    return list(main.chat_histories.get((chat_id, user_id or chat_id), []))


def contents(messages):
    return [m["content"] for m in messages]


async def say(msg, chat_id, text, user_id=None, **kwargs):
    m = msg(chat_id=chat_id, user_id=user_id, text=text, **kwargs)
    await main.chat(m)
    return m


async def test_history_saved_and_passed(ai, msg):
    await say(msg, 100, "вопрос 1")
    await say(msg, 100, "вопрос 2")
    assert [m["role"] for m in hist(100)] == ["user", "assistant"] * 2
    assert contents(ai.last_messages) == [
        main.SYSTEM_PROMPT, "вопрос 1", "ответ на вопрос 1", "вопрос 2",
    ]


async def test_codeword_scenario(ai, msg):
    """Сценарий из реального теста: «ЛИМОН-27»."""
    def model(messages):
        last = messages[-1]["content"]
        if "Запомни" in last:
            return "Запомнил!", "stop"
        words = [w for m in messages[:-1] if m["role"] == "user"
                 for w in re.findall(r"[А-ЯЁA-Z]+-\d+", m["content"])]
        return (f"Ты назвал кодовое слово {words[-1]}." if words
                else "Ты не называл кодовое слово."), "stop"

    ai.responder = model
    first = await say(msg, 555, "Запомни кодовое слово ЛИМОН-27")
    assert first.replies == ["Запомнил!"]
    second = await say(msg, 555, "Какое кодовое слово я назвал?")
    assert [m["role"] for m in ai.last_messages] == ["system", "user", "assistant", "user"]
    assert second.replies == ["Ты назвал кодовое слово ЛИМОН-27."]

    await main.reset(msg(chat_id=555))
    third = await say(msg, 555, "Какое кодовое слово я назвал?")
    assert [m["role"] for m in ai.last_messages] == ["system", "user"]
    assert third.replies == ["Ты не называл кодовое слово."]


async def test_history_limit(ai, msg):
    for i in range(1, 16):
        await say(msg, 100, f"вопрос {i}")
    h = hist(100)
    assert len(h) == main.HISTORY_LIMIT == 20
    assert h[0] == {"role": "user", "content": "вопрос 6"}
    assert h[-1]["content"] == "ответ на вопрос 15"
    await say(msg, 100, "вопрос 16")
    assert len(ai.last_messages) == 1 + 20 + 1


async def test_isolation_private_and_group(ai, msg):
    await say(msg, 100, "от A")
    await say(msg, 200, "от B")
    assert contents(ai.last_messages) == [main.SYSTEM_PROMPT, "от B"]
    await say(msg, -500, "группа A", user_id=1)
    await say(msg, -500, "группа B", user_id=2)
    assert contents(ai.last_messages) == [main.SYSTEM_PROMPT, "группа B"]
    assert len(hist(-500, 1)) == len(hist(-500, 2)) == 2


async def test_start_keeps_history(msg):
    await say(msg, 100, "q")
    before = hist(100)
    await main.start(msg(chat_id=100))
    assert hist(100) == before


async def test_reset_only_own_chat(msg):
    await say(msg, 100, "q")
    await say(msg, 200, "q")
    await main.reset(msg(chat_id=100))
    assert hist(100) == [] and len(hist(200)) == 2


async def test_reset_during_request(ai, msg):
    ai.delay = 0.2
    task = asyncio.create_task(say(msg, 300, "долгий вопрос"))
    await asyncio.sleep(0.05)
    await main.reset(msg(chat_id=300))
    await task
    assert hist(300) == []


@pytest.mark.parametrize("failure", ["api", "empty", "send"])
async def test_failed_answers_not_saved(ai, msg, failure):
    from conftest import api_error
    await say(msg, 100, "успешный")
    before = hist(100)
    if failure == "api":
        ai.error = api_error("500")
    elif failure == "empty":
        ai.responder = lambda messages: ("", "stop")
    await say(msg, 100, "неудачный", fail_send=(failure == "send"))
    assert hist(100) == before
    assert main.busy_users == set()


async def test_truncated_saved_without_marker(ai, msg):
    ai.responder = lambda messages: ("обрезанный", "length")
    m = await say(msg, 100, "длинно")
    assert hist(100)[-1]["content"] == "обрезанный"
    assert "✂️" in m.replies[-1]
