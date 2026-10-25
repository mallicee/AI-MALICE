import asyncio
import logging
import os
from pathlib import Path

from aiogram import Bot, Dispatcher, F
from aiogram.filters import CommandStart, Command
from aiogram.types import Message
from dotenv import load_dotenv
from openai import AsyncOpenAI, APIError


# Load settings from .env next to this file.
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()
# PROVOD_API_KEY is preferred. OPENAI_API_KEY is also accepted for compatibility
# with the first local version of this project.
PROVOD_API_KEY = (
    os.getenv("PROVOD_API_KEY") or os.getenv("OPENAI_API_KEY", "")
).strip()

if not BOT_TOKEN:
    raise RuntimeError(
        "BOT_TOKEN is missing. Add your Telegram bot token to the .env file."
    )

if not PROVOD_API_KEY:
    raise RuntimeError(
        "PROVOD_API_KEY is missing. Add your Provod.ai API key to the .env file."
    )

MODEL = "gemini-3.1-flash-lite"

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()

ai = AsyncOpenAI(
    api_key=PROVOD_API_KEY,
    base_url="https://api.provod.ai/v1",
    timeout=45.0,
    max_retries=0,
)

SYSTEM_PROMPT = (
    "Ты AI MALICE — дружелюбный ИИ-ассистент. "
    "Отвечай на языке пользователя, понятно и по делу. "
    "Помогай с общими вопросами, обучением и программированием. "
    "Если не уверен в ответе, честно сообщай об этом."
)


@dp.message(CommandStart())
async def start(message: Message) -> None:
    await message.answer(
        "🤖 Привет! Я AI MALICE.\n\n"
        f"Сейчас я работаю через модель {MODEL}.\n"
        "Задавай вопросы, проси объяснения или помощь с кодом.\n\n"
        "Примечание: пока я не сохраняю историю диалога между сообщениями."
    )


@dp.message(Command("help"))
async def help_command(message: Message) -> None:
    await message.answer(
        "Просто отправь мне текстовый вопрос.\n"
        "Команда /start показывает приветствие.\n\n"
        "В этой версии доступны текстовые сообщения; анализ фотографий "
        "и постоянная память диалога пока не реализованы."
    )


@dp.message(F.text)
async def chat(message: Message) -> None:
    text = (message.text or "").strip()
    if not text:
        return

    await bot.send_chat_action(chat_id=message.chat.id, action="typing")

    try:
        response = await ai.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            max_tokens=500,
        )

        answer = response.choices[0].message.content or ""
        answer = answer.strip()
        if not answer:
            answer = "Модель вернула пустой ответ. Попробуй задать вопрос иначе."

        # Telegram messages have a length limit, so split long replies.
        for offset in range(0, len(answer), 4000):
            await message.answer(answer[offset : offset + 4000])

        if response.usage:
            logging.info(
                "Model=%s input_tokens=%s output_tokens=%s",
                MODEL,
                response.usage.prompt_tokens,
                response.usage.completion_tokens,
            )

    except APIError as error:
        logging.exception("AI provider API request failed")
        await message.answer(
            "Не удалось получить ответ от ИИ. Проверь доступ к модели и баланс "
            "Provod.ai. Подробности доступны в консоли запуска. "
            f"Тип ошибки: {type(error).__name__}."
        )
    except Exception:
        logging.exception("Unexpected error while handling a message")
        await message.answer(
            "Произошла ошибка при обработке сообщения. Попробуй ещё раз."
        )


async def main() -> None:
    logging.basicConfig(level=logging.INFO)
    print("AI MALICE starting...")
    print(f"Model: {MODEL}")
    print("Provider: Provod.ai")
    await dp.start_polling(bot)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("AI MALICE stopped.")
