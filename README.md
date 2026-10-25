# AI MALICE

A Telegram chatbot built with Python, aiogram, and a model accessed through the Provod.ai API.

## Features

- Replies to text messages in Telegram.
- Uses the `gemini-3.1-flash-lite` model through Provod.ai.
- Loads credentials from a local `.env` file rather than hard-coding them.
- Logs token usage when the API returns usage statistics.

## Current limitations

- The bot does not retain conversation history between messages.
- Photo/image analysis is not included in this version.
- The bot runs only while the Python process is running.

## Requirements

- Windows, macOS, or Linux
- Python 3.13 recommended
- A Telegram bot token from [@BotFather](https://t.me/BotFather)
- A Provod.ai API key and access to the selected model

API usage may incur charges. Check the provider's balance and pricing before use.

## Setup on Windows

Open Command Prompt in the project folder and run:

```bat
py -3.13 -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
copy .env.example .env
notepad .env
```

In `.env`, replace the placeholders with your own values:

```env
BOT_TOKEN=your_real_telegram_bot_token
PROVOD_API_KEY=your_real_provod_api_key
```

Save `.env`, then start the bot:

```bat
python main.py
```

Open your bot in Telegram, press **Start**, and send a text message. Keep the terminal open while you use the bot.

## Setup on macOS or Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` with your own values and run:

```bash
python main.py
```

## Security

- Never commit `.env`, API keys, or Telegram bot tokens.
- `.env` is excluded by `.gitignore`; `.env.example` contains placeholders only and is safe to publish.
- If a real token or key is ever pushed to a public repository, revoke/rotate it immediately. Removing it in a later commit does not make the exposed secret safe again.

## Model and provider

The model ID is set in `main.py` as `gemini-3.1-flash-lite`, and the API base URL is `https://api.provod.ai/v1`. If your provider account's access or model IDs change, update these settings accordingly.
