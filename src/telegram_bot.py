import logging
from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

from . import config
from . import rag_core

# Настройка логирования для отслеживания работы бота
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Отправляет приветственное сообщение при команде /start."""
    user = update.effective_user
    await update.message.reply_html(
        rf"Привет, {user.mention_html()}! Я ваш личный ассистент по архиву разговоров. "
        "Спросите меня о чем-то, и я поищу ответ в записях.",
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Отправляет инструкцию по использованию при команде /help."""
    await update.message.reply_text(
        "Как это работает:\n"
        "1. Новые записи разговоров автоматически обрабатываются и добавляются в мою базу знаний.\n"
        "2. Вы задаете мне вопрос в этом чате.\n"
        "3. Я ищу наиболее релевантную информацию во всех разговорах.\n"
        "4. Я присылаю вам ответ, сгенерированный на основе найденной информации, и указываю, из каких разговоров она была взята."
    )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обрабатывает текстовые сообщения от пользователя, задающего вопрос."""
    question = update.message.text
    user_id = update.effective_user.id
    logging.info(f"Получен вопрос от пользователя {user_id}: '{question}'")

    # Сообщаем пользователю, что мы начали обработку
    thinking_message = await update.message.reply_text("Ищу информацию в архиве...")

    try:
        # Выполняем основной RAG-запрос
        result = rag_core.query_rag(question)
        answer = result.get("answer", "Не удалось сгенерировать ответ.")
        sources = result.get("sources", [])

        # Формируем итоговый ответ
        if sources:
            # Используем Markdown для форматирования
            source_list = "\n".join(f"• `{source}`" for source in sources)
            response_text = f"{answer}\n\n*Источники:*\n{source_list}"
        else:
            response_text = answer

        # Заменяем "думающее" сообщение на итоговый ответ
        await thinking_message.edit_text(response_text, parse_mode=ParseMode.MARKDOWN)

    except Exception as e:
        logging.error(f"Ошибка при обработке запроса от {user_id}: {e}", exc_info=True)
        await thinking_message.edit_text("К сожалению, при обработке вашего запроса произошла ошибка. Попробуйте еще раз позже.")


def main() -> None:
    """Основная функция для запуска Telegram-бота."""
    logging.info("--- Запуск Telegram-бота ---")

    if not config.TELEGRAM_BOT_TOKEN or config.TELEGRAM_BOT_TOKEN == "YOUR_TELEGRAM_BOT_TOKEN_HERE":
        logging.critical("Токен Telegram-бота не установлен в файле src/config.py. Бот не может быть запущен.")
        return

    # Инициализация RAG-системы (модели для поиска и генерации)
    try:
        logging.info("Инициализация RAG-системы. Это может занять некоторое время...")
        rag_core.initialize_rag()
        logging.info("RAG-система успешно инициализирована.")
    except Exception as e:
        logging.critical(f"Не удалось инициализировать RAG-систему: {e}", exc_info=True)
        logging.critical("Бот не будет запущен из-за ошибки инициализации.")
        return

    # Создание и настройка приложения бота
    application = Application.builder().token(config.TELEGRAM_BOT_TOKEN).build()

    # Регистрация обработчиков команд и сообщений
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Запуск бота в режиме опроса
    logging.info("Бот успешно запущен и готов принимать сообщения.")
    application.run_polling()
    logging.info("Работа бота завершена.")


if __name__ == '__main__':
    # Для запуска бота выполните команду из корневой папки проекта:
    # python -m src.telegram_bot
    main()
