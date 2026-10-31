"""Изображения: скачивание, Base64, формат запроса, лимиты, память, логи."""
import base64
import datetime as dt
from types import SimpleNamespace as NS

import pytest
from aiogram import F
from aiogram.types import Chat, Document, Message, PhotoSize

import main
from conftest import api_error

JPEG = b"\xff\xd8\xff\xe0" + b"jpeg-body" * 50
PNG = b"\x89PNG\r\n\x1a\n" + b"png-body" * 50
WEBP = b"RIFF\x00\x00\x00\x00WEBP" + b"webp" * 50
GIF = b"GIF89a" + b"gif" * 50
BIG = main.MAX_IMAGE_BYTES + 1

pytestmark = pytest.mark.usefixtures("no_limits")


def sizes(tg, *specs):
    out = []
    for file_id, data, size in specs:
        tg.add_file(file_id, data, size)
        out.append(NS(file_id=file_id, file_size=size))
    return out


def doc(tg, file_id, data, mime, size=None):
    tg.add_file(file_id, data, size)
    return NS(file_id=file_id, mime_type=mime,
              file_size=len(data) if size is None else size)


def image_part(ai):
    return ai.last_messages[-1]["content"]


async def test_photo_with_caption(tg, ai, msg):
    m = msg(chat_id=1, caption="Что за животное?",
            photo=sizes(tg, ("small", JPEG[:20], 20), ("large", JPEG, len(JPEG))))
    await main.photo(m)
    content = image_part(ai)
    assert [p["type"] for p in content] == ["text", "image_url"]
    assert content[0]["text"] == "Что за животное?"
    url = content[1]["image_url"]["url"]
    assert url.startswith("data:image/jpeg;base64,")
    assert base64.b64decode(url.split(",", 1)[1]) == JPEG     # самый крупный размер
    assert tg.download_timeouts == [main.IMAGE_DOWNLOAD_TIMEOUT]
    assert m.replies == ["описание изображения"]


async def test_photo_memory_keeps_marker_not_base64(tg, ai, msg):
    await main.photo(msg(chat_id=1, caption="Кто это?",
                         photo=sizes(tg, ("p", JPEG, len(JPEG)))))
    assert list(main.chat_histories[(1, 1)]) == [
        {"role": "user", "content": "[Пользователь отправил изображение с подписью] Кто это?"},
        {"role": "assistant", "content": "описание изображения"},
    ]
    await main.chat(msg(chat_id=1, text="А подробнее?"))
    ctx = ai.last_messages
    assert [m["role"] for m in ctx] == ["system", "user", "assistant", "user"]
    assert all(isinstance(m["content"], str) and "base64" not in m["content"] for m in ctx)


BSOD_ANSWER = (
    "На фото синий экран Windows (BSOD). "
    "Код остановки: CRITICAL_PROCESS_DIED."
)


def bsod_model(messages):
    """Имитация модели: видит код только на картинке или в своём прошлом ответе."""
    last = messages[-1]["content"]
    if isinstance(last, list):
        return BSOD_ANSWER, "stop"
    seen = [m["content"] for m in messages[:-1]
            if m["role"] == "assistant" and "CRITICAL_PROCESS_DIED" in m["content"]]
    if seen:
        return "Код ошибки: CRITICAL_PROCESS_DIED.", "stop"
    return "Я не вижу изображения, отправь его ещё раз.", "stop"


@pytest.mark.parametrize("caption", ["Что это за ошибка?", None])
async def test_bsod_followup_remembers_answer(tg, ai, msg, caption):
    """Реальный сценарий: фото BSOD → вопрос о коде ошибки текстом."""
    ai.responder = bsod_model
    photo_msg = msg(chat_id=42, caption=caption,
                    photo=sizes(tg, ("bsod", JPEG, len(JPEG))))
    await main.photo(photo_msg)
    assert photo_msg.replies == [BSOD_ANSWER]

    # 1. ответ модели сохранён, Base64 — нет
    history = list(main.chat_histories[(42, 42)])
    assert history[1] == {"role": "assistant", "content": BSOD_ANSWER}
    assert all("base64" not in m["content"] for m in history)

    # 3. тот же ключ у текстового сообщения в том же чате
    question = msg(chat_id=42, text="Какой код ошибки был указан на фотографии?")
    assert main.history_key(question) == main.history_key(photo_msg) == (42, 42)
    await main.chat(question)

    # 2. прошлый ответ есть в контексте второго запроса
    ctx = ai.last_messages
    assert [m["role"] for m in ctx] == ["system", "user", "assistant", "user"]
    assert ctx[2]["content"] == BSOD_ANSWER
    assert ctx[1]["content"].startswith("[Пользователь отправил изображение")
    assert question.replies == ["Код ошибки: CRITICAL_PROCESS_DIED."]
    # 4. история не очищена, а дополнена
    assert len(main.chat_histories[(42, 42)]) == 4


async def test_bsod_answer_survives_until_history_limit(tg, ai, msg):
    """5. Ответ о фото вытесняется только после HISTORY_LIMIT/2 - 1 новых обменов."""
    ai.responder = bsod_model
    await main.photo(msg(chat_id=43, photo=sizes(tg, ("b", JPEG, len(JPEG)))))
    exchanges_kept = main.HISTORY_LIMIT // 2 - 1          # 9 при лимите 20
    for i in range(exchanges_kept):
        await main.chat(msg(chat_id=43, text=f"вопрос {i}"))
    assert BSOD_ANSWER in [m["content"] for m in main.chat_histories[(43, 43)]]
    await main.chat(msg(chat_id=43, text="ещё вопрос"))
    assert BSOD_ANSWER not in [m["content"] for m in main.chat_histories[(43, 43)]]


async def test_photo_without_caption(tg, ai, msg):
    await main.photo(msg(chat_id=2, photo=sizes(tg, ("p", JPEG, len(JPEG)))))
    assert image_part(ai)[0]["text"] == main.DEFAULT_IMAGE_PROMPT
    assert main.chat_histories[(2, 2)][0]["content"] == \
        "[Пользователь отправил изображение без подписи]"


async def test_photo_too_big_rejected_before_download(tg, ai, msg):
    m = msg(photo=sizes(tg, ("b1", JPEG, BIG), ("b2", JPEG, BIG * 2)))
    await main.photo(m)
    assert m.replies == [main.IMAGE_TOO_BIG]
    assert tg.get_file_calls == 0 and ai.requests == []


@pytest.mark.parametrize("data, mime", [(PNG, "image/png"), (WEBP, "image/webp"),
                                        (JPEG, "image/jpeg")])
async def test_image_documents_accepted(tg, ai, msg, data, mime):
    await main.image_document(msg(document=doc(tg, "d", data, mime)))
    assert image_part(ai)[1]["image_url"]["url"].startswith(f"data:{mime};base64,")


async def test_unsupported_document_type(tg, ai, msg):
    m = msg(document=doc(tg, "g", GIF, "image/gif"))
    await main.image_document(m)
    assert m.replies == [main.IMAGE_UNSUPPORTED]
    assert tg.get_file_calls == 0 and ai.requests == []


async def test_big_document_rejected_before_download(tg, ai, msg):
    m = msg(document=doc(tg, "d", PNG, "image/png", size=BIG))
    await main.image_document(m)
    assert m.replies == [main.IMAGE_TOO_BIG] and tg.get_file_calls == 0


async def test_fake_image_rejected_by_signature(tg, ai, msg):
    m = msg(chat_id=7, document=doc(tg, "x", b"MZ\x90\x00not an image" * 10, "image/jpeg"))
    await main.image_document(m)
    assert m.replies == [main.IMAGE_UNSUPPORTED]
    assert ai.requests == [] and not main.chat_histories.get((7, 7))
    assert main.busy_users == set()


async def test_size_from_get_file_over_limit(tg, ai, msg):
    tg.add_file("late", JPEG, BIG)
    m = msg(photo=[NS(file_id="late", file_size=None)])
    await main.photo(m)
    assert m.replies == [main.IMAGE_TOO_BIG] and ai.requests == []


async def test_empty_download(tg, ai, msg):
    tg.add_file("empty", b"", 0)
    m = msg(photo=[NS(file_id="empty", file_size=0)])
    await main.photo(m)
    assert m.replies == ["Не удалось скачать изображение. Попробуй ещё раз."]


async def test_api_error_keeps_history(tg, ai, msg):
    ai.error = api_error("500")
    m = msg(chat_id=3, photo=sizes(tg, ("p", JPEG, len(JPEG))))
    await main.photo(m)
    assert m.replies == ["ИИ временно недоступен. Попробуй позже."]
    assert (3, 3) not in main.chat_histories or not main.chat_histories[(3, 3)]


async def test_rate_limit_applies_before_download(tg, ai, msg, monkeypatch):
    monkeypatch.setattr(main, "MIN_INTERVAL", 3.0)
    await main.photo(msg(chat_id=9, photo=sizes(tg, ("r1", JPEG, len(JPEG)))))
    calls = tg.get_file_calls
    m = msg(chat_id=9, photo=sizes(tg, ("r2", JPEG, len(JPEG))))
    await main.photo(m)
    assert tg.get_file_calls == calls
    assert m.replies and m.replies[0].startswith("🐢")


async def test_logs_without_base64_or_text(tg, ai, msg, caplog):
    caplog.set_level("INFO")
    await main.photo(msg(caption="Что за животное?",
                         photo=sizes(tg, ("p", JPEG, len(JPEG)))))
    b64 = base64.b64encode(JPEG).decode()[:40]
    assert any(r.startswith("[фото] скачано: image/jpeg") for r in caplog.messages)
    assert not any("base64" in r or b64 in r or "животное" in r for r in caplog.messages)


def test_aiogram_filters_route_correctly():
    chat = Chat(id=1, type="private")
    now = dt.datetime.now()
    photo = Message(message_id=1, date=now, chat=chat,
                    photo=[PhotoSize(file_id="a", file_unique_id="a", width=1, height=1)])
    png = Message(message_id=2, date=now, chat=chat,
                  document=Document(file_id="b", file_unique_id="b", mime_type="image/png"))
    pdf = Message(message_id=3, date=now, chat=chat,
                  document=Document(file_id="c", file_unique_id="c",
                                    mime_type="application/pdf"))
    text = Message(message_id=4, date=now, chat=chat, text="привет")
    image_doc = F.document.mime_type.startswith("image/")

    assert F.photo.resolve(photo) and not F.photo.resolve(text)
    assert image_doc.resolve(png)
    assert not image_doc.resolve(pdf) and not image_doc.resolve(text)
    assert not F.text.resolve(photo)
