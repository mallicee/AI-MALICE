
import asyncio
import os
import socket
import ssl
import sys
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv
from openai import (
    AsyncOpenAI,
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
)


# Те же настройки, что и в main.py
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

BASE_URL = "https://api.provod.ai/v1"
MODEL = "gemini-3.1-flash-lite"
TIMEOUT = 30.0

key = os.getenv("OPENAI_API_KEY", "").strip()


def safe(text: object, limit: int = 300) -> str:
    """Обрезает текст и вырезает из него API-ключ, если он туда попал."""
    text = str(text)
    if key:
        text = text.replace(key, "***")
    return text[:limit]


def diagnose_network() -> None:
    """Пошагово проверяет DNS, TCP и TLS, чтобы понять, где зависает."""
    host = urlparse(BASE_URL).hostname
    port = 443
    print("\nДиагностика сети:", host)

    try:
        addresses = sorted({
            info[4][0]
            for info in socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
        })
        print("  DNS:  OK ->", ", ".join(addresses))
    except OSError as error:
        print("  DNS:  ОШИБКА -", type(error).__name__, safe(error))
        return

    try:
        sock = socket.create_connection((host, port), timeout=10)
        print("  TCP:  OK -> подключено к", sock.getpeername()[0])
    except OSError as error:
        print("  TCP:  ОШИБКА -", type(error).__name__, safe(error))
        return

    try:
        context = ssl.create_default_context()
        sock.settimeout(15)
        with context.wrap_socket(sock, server_hostname=host) as tls:
            print("  TLS:  OK ->", tls.version())
    except OSError as error:
        print("  TLS:  ОШИБКА -", type(error).__name__, safe(error))
    finally:
        sock.close()

    proxies = [
        name for name in (
            "HTTPS_PROXY", "https_proxy", "HTTP_PROXY",
            "http_proxy", "ALL_PROXY", "all_proxy",
        )
        if os.getenv(name)
    ]
    print("  Прокси в переменных окружения:", ", ".join(proxies) or "нет")


async def main() -> int:
    if not key:
        print("Не найден API-ключ Provod.ai (OPENAI_API_KEY) в файле .env")
        return 1

    ai = AsyncOpenAI(
        api_key=key,
        base_url=BASE_URL,
        timeout=TIMEOUT,
        max_retries=0,
    )

    print("Запрашиваем каталог моделей Provod.ai...")
    print("Адрес:", BASE_URL)

    try:
        page = await ai.models.list()

    except AuthenticationError as error:
        print("Подключение: ЕСТЬ, но ключ отклонён (HTTP", error.status_code, ")")
        print("Проверь OPENAI_API_KEY в .env.")
        return 1

    except APIStatusError as error:
        print("Подключение: ЕСТЬ, сервер вернул HTTP", error.status_code)
        print("Ответ сервера:", safe(error.message))
        return 1

    except APITimeoutError as error:
        cause = error.__cause__
        print("Подключение: ТАЙМАУТ за", TIMEOUT, "сек.")
        if cause is not None:
            print("Причина:", type(cause).__name__, safe(cause) or "(без текста)")
        diagnose_network()
        return 1

    except APIConnectionError as error:
        cause = error.__cause__
        print("Подключение: НЕТ СВЯЗИ")
        if cause is not None:
            print("Причина:", type(cause).__name__, safe(cause) or "(без текста)")
        diagnose_network()
        return 1

    finally:
        await ai.close()

    models = page.data
    available = [
        model for model in models
        if (model.model_extra or {}).get("available") is True
    ]

    print("Подключение: OK")
    print("Всего моделей в каталоге:", len(models))
    print("Доступных моделей:", len(available))

    ids = {model.id for model in models}
    print(
        f"Модель бота {MODEL}:",
        "есть в каталоге" if MODEL in ids else "НЕ НАЙДЕНА в каталоге",
    )

    if not available:
        print(
            "Каталог не вернул моделей с available=True. "
            "Это не обязательно означает, что все модели требуют оплаты."
        )
    else:
        print("\nМодели, отмеченные как доступные:\n")
        for model in available[:40]:
            name = (model.model_extra or {}).get("name", "не указано")
            print("Название:", name)
            print("Точный ID:", model.id)
            print("-" * 30)

        if len(available) > 40:
            print("Показаны первые 40 моделей.")

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
