
import asyncio
import base64
import io
import logging
import os
import socket
import time
from collections import defaultdict, deque
from pathlib import Path

from aiogram import Bot, Dispatcher, F
from aiogram.filters import CommandStart, Command
from aiogram.types import Message
from aiogram.utils.chat_action import ChatActionSender
from dotenv import load_dotenv
from openai import (
    AsyncOpenAI,
    APIConnectionError,
    APIError,
    RateLimitError,
    Timeout,
)


# Загружаем настройки из файла .env рядом с main.py
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

BOT_TOKEN = os.getenv("BOT_TOKEN")
PROVOD_API_KEY = os.getenv("OPENAI_API_KEY")

if not BOT_TOKEN or not PROVOD_API_KEY:
    raise RuntimeError(
        "Не найдены BOT_TOKEN или OPENAI_API_KEY в файле .env"
    )


# Подключение Telegram
bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()


# Подключение ИИ через Provod.ai.
# max_retries: клиент сам повторяет запрос при временных ошибках
# (обрыв связи, таймаут, 408, 409, 429, 5xx) с паузой между попытками.
# connect=10: зависшее подключение (бывает через VPN) обрывается быстро,
# и повторная попытка начинается раньше, а не через 45 секунд.
ai = AsyncOpenAI(
    api_key=PROVOD_API_KEY,
    base_url="https://api.provod.ai/v1",
    timeout=Timeout(45.0, connect=10.0),
    max_retries=2,
)

MODEL = "gemini-3.1-flash-lite"
MAX_TOKENS = 1500

SYSTEM_PROMPT = (
    "Ты AI MALICE — дружелюбный ИИ-ассистент. "
    "Отвечай на языке пользователя. "
    "Объясняй понятно и по делу. "
    "Помогай с обучением, вопросами "
    "и программированием. "
    "Изображения из предыдущих сообщений повторно не передаются, "
    "но их текстовые описания сохраняются в истории диалога. "
    "Если пользователь задаёт уточняющий вопрос о ранее отправленном "
    "изображении, используй предыдущие сообщения и собственное "
    "описание этого изображения. Не проси отправить фотографию "
    "повторно, если нужная информация уже есть в истории."
)


# Память диалогов (в оперативной памяти, пропадает при перезапуске).
# Ключ — (chat_id, user_id): в личке это один чат, а в группе
# у каждого участника своя история, переписки не смешиваются.
HISTORY_LIMIT = 20  # сообщений (пользователь + ассистент)

HistoryKey = tuple[int, int]
chat_histories: dict[HistoryKey, deque[dict[str, str]]] = {}


def history_key(message: Message) -> HistoryKey:
    user_id = (
        message.from_user.id if message.from_user else message.chat.id
    )
    return message.chat.id, user_id


def get_history(key: HistoryKey) -> deque[dict[str, str]]:
    if key not in chat_histories:
        chat_histories[key] = deque(maxlen=HISTORY_LIMIT)
    return chat_histories[key]


# Изображения. Скачиваются только в память, на диск не пишутся.
# В историю попадает текстовая пометка, а не сама картинка:
# иначе Base64 уходил бы в модель с каждым следующим сообщением.
MAX_IMAGE_BYTES = 10 * 1024 * 1024   # Telegram отдаёт ботам файлы до 20 МБ
IMAGE_DOWNLOAD_TIMEOUT = 30
SUPPORTED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}
DEFAULT_IMAGE_PROMPT = "Опиши, что изображено на этом изображении."

IMAGE_TOO_BIG = (
    f"Изображение слишком большое. Максимум — "
    f"{MAX_IMAGE_BYTES // (1024 * 1024)} МБ."
)
IMAGE_UNSUPPORTED = (
    "Этот формат изображения не поддерживается. "
    "Отправь фото или файл JPEG, PNG или WebP."
)


class ImageError(Exception):
    """Проблема с изображением; текст безопасно показать пользователю."""


def detect_image_type(data: bytes) -> str | None:
    """Определяет формат по сигнатуре файла, а не по словам отправителя."""
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


async def download_image(file_id: str) -> str:
    """Скачивает изображение из Telegram и возвращает data URL (Base64)."""
    file = await bot.get_file(file_id)
    if file.file_size and file.file_size > MAX_IMAGE_BYTES:
        raise ImageError(IMAGE_TOO_BIG)

    buffer = io.BytesIO()
    await bot.download_file(
        file.file_path,
        destination=buffer,
        timeout=IMAGE_DOWNLOAD_TIMEOUT,
    )
    data = buffer.getvalue()

    if not data:
        raise ImageError("Не удалось скачать изображение. Попробуй ещё раз.")
    if len(data) > MAX_IMAGE_BYTES:
        raise ImageError(IMAGE_TOO_BIG)

    mime = detect_image_type(data)
    if mime is None:
        raise ImageError(IMAGE_UNSUPPORTED)

    logging.info("[фото] скачано: %s, %s байт", mime, len(data))
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def image_history_text(caption: str) -> str:
    if caption:
        return f"[Пользователь отправил изображение с подписью] {caption}"
    return "[Пользователь отправил изображение без подписи]"


# Защита от повторной обработки одного и того же сообщения
# (Telegram может прислать апдейт ещё раз, например при конфликте
# двух копий бота за getUpdates)
processed_messages: deque[tuple[int, int]] = deque(maxlen=1000)


def already_processed(message: Message) -> bool:
    key = (message.chat.id, message.message_id)
    if key in processed_messages:
        return True
    processed_messages.append(key)
    return False


# Защита от запуска второй копии бота на этом компьютере.
# Память хранится внутри процесса, поэтому две копии с одним токеном
# делят сообщения между собой и «теряют» историю друг друга.
INSTANCE_LOCK_PORT = 47213
_instance_lock: socket.socket | None = None


def acquire_instance_lock() -> bool:
    global _instance_lock
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):  # Windows
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
    try:
        sock.bind(("127.0.0.1", INSTANCE_LOCK_PORT))
    except OSError:
        sock.close()
        return False
    _instance_lock = sock
    return True


# Ограничение частоты запросов к ИИ (на одного пользователя)
MIN_INTERVAL = 3.0          # секунд между сообщениями
RATE_LIMIT_REQUESTS = 10    # не больше запросов...
RATE_LIMIT_PERIOD = 60.0    # ...за столько секунд

user_requests: dict[int, deque[float]] = defaultdict(deque)
busy_users: set[int] = set()
warned_users: dict[int, float] = {}


def should_warn(user_id: int) -> bool:
    """Не чаще раза в MIN_INTERVAL, чтобы бот не отвечал на каждый спам."""
    now = time.monotonic()
    last = warned_users.get(user_id)

    if last is not None and now - last < MIN_INTERVAL:
        return False

    warned_users[user_id] = now
    return True


def check_rate_limit(user_id: int) -> float:
    """Возвращает, сколько секунд нужно подождать (0 — можно)."""
    now = time.monotonic()
    history = user_requests[user_id]

    while history and now - history[0] >= RATE_LIMIT_PERIOD:
        history.popleft()

    if history and now - history[-1] < MIN_INTERVAL:
        return MIN_INTERVAL - (now - history[-1])

    if len(history) >= RATE_LIMIT_REQUESTS:
        return RATE_LIMIT_PERIOD - (now - history[0])

    history.append(now)
    return 0.0


# Команда /start
@dp.message(CommandStart())
async def start(message: Message):
    await message.answer(
        "🤖 Привет! Я AI MALICE.\n\n"
        "Сейчас я работаю через Gemini 3.1 Flash Lite.\n"
        "Можешь задавать вопросы, просить объяснения "
        "и помогать себе с программированием.\n"
        "Можно отправить фото — с подписью-вопросом или без.\n\n"
        "Напиши сообщение, чтобы начать!\n"
        "Команда /reset очищает контекст текущего диалога."
    )


# Отдельная команда сброса
@dp.message(Command("reset"))
async def reset(message: Message):
    # Убираем историю целиком. Если в этот момент ещё идёт запрос,
    # его ответ попадёт в старый (уже отвязанный) deque и не вернётся.
    chat_histories.pop(history_key(message), None)
    await message.answer(
        "Диалог сброшен. Можешь начать новую тему!"
    )


# Обработка текстовых сообщений
@dp.message(F.text)
async def chat(message: Message):
    text = message.text.strip()

    if not text:
        return

    await ask_ai(message, text)


# Фотографии (Telegram всегда присылает их как JPEG)
@dp.message(F.photo)
async def photo(message: Message):
    # Telegram присылает несколько размеров по возрастанию —
    # берём самый крупный, который укладывается в лимит
    sizes = [
        size for size in message.photo
        if (size.file_size or 0) <= MAX_IMAGE_BYTES
    ]
    if not sizes:
        if should_warn(history_key(message)[1]):
            await message.answer(IMAGE_TOO_BIG)
        return

    caption = (message.caption or "").strip()
    await ask_ai(
        message,
        caption or DEFAULT_IMAGE_PROMPT,
        image_file_id=sizes[-1].file_id,
        history_text=image_history_text(caption),
    )


# Изображения, отправленные файлом (без сжатия)
@dp.message(F.document.mime_type.startswith("image/"))
async def image_document(message: Message):
    document = message.document

    if document.mime_type not in SUPPORTED_IMAGE_TYPES:
        if should_warn(history_key(message)[1]):
            await message.answer(IMAGE_UNSUPPORTED)
        return

    if document.file_size and document.file_size > MAX_IMAGE_BYTES:
        if should_warn(history_key(message)[1]):
            await message.answer(IMAGE_TOO_BIG)
        return

    caption = (message.caption or "").strip()
    await ask_ai(
        message,
        caption or DEFAULT_IMAGE_PROMPT,
        image_file_id=document.file_id,
        history_text=image_history_text(caption),
    )


async def ask_ai(
    message: Message,
    text: str,
    image_file_id: str | None = None,
    history_text: str | None = None,
):
    """Общий путь запроса к ИИ: лимиты, память, «печатает...», ошибки.

    text — что видит модель; history_text — что сохраняется в историю
    (по умолчанию то же самое). Изображение скачивается только после
    проверки лимитов.
    """
    if history_text is None:
        history_text = text

    if already_processed(message):
        logging.warning(
            "Повтор сообщения chat_id=%s message_id=%s — пропущен",
            message.chat.id, message.message_id,
        )
        return

    user_id = (
        message.from_user.id if message.from_user else message.chat.id
    )

    if user_id in busy_users:
        if should_warn(user_id):
            await message.answer(
                "⏳ Я ещё отвечаю на прошлый вопрос. Подожди немного."
            )
        return

    wait = check_rate_limit(user_id)
    if wait > 0:
        if should_warn(user_id):
            await message.answer(
                "🐢 Слишком много сообщений подряд. "
                f"Попробуй через {int(wait) + 1} сек."
            )
        return

    busy_users.add(user_id)

    key = history_key(message)
    history = get_history(key)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *history,
        {"role": "user", "content": text},
    ]

    try:
        # «Печатает...» повторяется каждые несколько секунд,
        # пока ждём ответ от модели
        async with ChatActionSender.typing(
            bot=bot,
            chat_id=message.chat.id,
            message_thread_id=(
                message.message_thread_id
                if message.is_topic_message
                else None
            ),
        ):
            if image_file_id:
                image_url = await download_image(image_file_id)
                messages[-1] = {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text},
                        {
                            "type": "image_url",
                            "image_url": {"url": image_url},
                        },
                    ],
                }

            response = await ai.chat.completions.create(
                model=MODEL,
                messages=messages,
                max_tokens=MAX_TOKENS,
            )

        choice = response.choices[0]
        answer = (choice.message.content or "").strip()
        model_answer = answer  # без служебных пометок — для истории
        # Провайдеры могут присылать "length", "LENGTH" и т. п.
        truncated = (
            str(choice.finish_reason or "").strip().lower() == "length"
        )

        if not answer and truncated:
            answer = (
                "Ответ не уместился в лимит длины. "
                "Попробуй задать вопрос короче или конкретнее."
            )
        elif not answer:
            answer = (
                "Модель вернула пустой ответ. "
                "Попробуй задать вопрос иначе."
            )
        elif truncated:
            answer += (
                "\n\n✂️ Ответ получился слишком длинным и был обрезан. "
                "Попроси продолжить или уточни вопрос."
            )

        # Разбиваем длинные ответы для Telegram
        for start_index in range(0, len(answer), 4000):
            await message.answer(
                answer[start_index:start_index + 4000]
            )

        # В историю попадает только успешный непустой ответ модели,
        # и только после того, как пользователь его получил
        if model_answer:
            history.append({"role": "user", "content": history_text})
            history.append({"role": "assistant", "content": model_answer})

        # Выводим статистику использования, если она есть.
        # Только id и счётчики — без текстов переписки.
        if response.usage:
            logging.info(
                "Модель: %s; chat_id=%s user_id=%s; история: %s; "
                "входных токенов: %s; выходных токенов: %s",
                MODEL,
                key[0],
                key[1],
                len(history),
                response.usage.prompt_tokens,
                response.usage.completion_tokens,
            )

    # Подробности ошибок пишутся только в лог,
    # пользователь видит короткое понятное сообщение
    except ImageError as error:
        logging.warning("[фото] отклонено: %s", error)
        await message.answer(str(error))

    except RateLimitError:
        logging.exception("Provod.ai: превышен лимит запросов")
        await message.answer(
            "Сейчас слишком много запросов к ИИ. "
            "Попробуй через минуту."
        )

    except APIConnectionError:
        logging.exception("Provod.ai: нет связи или таймаут")
        await message.answer(
            "ИИ сейчас не отвечает. Попробуй чуть позже."
        )

    except APIError:
        logging.exception("Ошибка API Provod.ai")
        await message.answer(
            "ИИ временно недоступен. Попробуй позже."
        )

    except Exception:
        logging.exception("Ошибка обработки сообщения")
        await message.answer(
            "Произошла ошибка при обработке сообщения. "
            "Попробуй ещё раз."
        )

    finally:
        busy_users.discard(user_id)


async def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [pid %(process)d] "
               "%(name)s: %(message)s",
    )

    if not acquire_instance_lock():
        print(
            "AI MALICE уже запущен на этом компьютере. "
            "Останови первую копию (Ctrl+C в её окне) и запусти снова."
        )
        return

    print("AI MALICE запускается...")
    print(f"Модель: {MODEL}")
    print("Провайдер: Provod.ai")

    await dp.start_polling(bot)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("AI MALICE остановлен.")
