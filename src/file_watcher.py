import os
import time
import logging
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from . import config
from . import transcriber
from . import rag_core

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class NewFileHandler(FileSystemEventHandler):
    """Класс для обработки событий создания нового файла."""
    def on_created(self, event):
        """
        Вызывается при создании нового файла в отслеживаемой папке.
        """
        if event.is_directory:
            return

        # Проверяем, что расширение файла соответствует заданному в конфиге
        if event.src_path.endswith(config.WATCH_FILE_EXTENSION):
            logging.info(f"Обнаружен новый файл: {event.src_path}")

            # Даем файлу немного времени на полную запись, чтобы избежать проблем с доступом
            time.sleep(2)

            try:
                # 1. Транскрибация аудиофайла
                transcript_path, text = transcriber.transcribe_audio(event.src_path)

                if text is None or transcript_path is None:
                    logging.error(f"Транскрибация файла {event.src_path} не удалась. Пропускаем.")
                    return

                # 2. Добавление полученного текста в векторную базу знаний
                # В качестве 'source' используем имя исходного аудиофайла
                source_filename = os.path.basename(event.src_path)
                rag_core.add_text_to_db(text, source_file=source_filename)

                logging.info(f"Файл {source_filename} успешно обработан и добавлен в базу знаний.")

            except Exception as e:
                logging.error(f"Произошла непредвиденная ошибка при обработке файла {event.src_path}: {e}", exc_info=True)


def start_watching():
    """
    Основная функция для запуска сервиса мониторинга.
    """
    logging.info("--- Запуск сервиса мониторинга файлов ---")

    # Инициализация всех моделей перед началом работы
    try:
        logging.info("Инициализация моделей. Это может занять некоторое время...")
        transcriber.load_whisper_model()
        rag_core.initialize_rag()
        logging.info("Все модели успешно инициализированы.")
    except Exception as e:
        logging.critical(f"Критическая ошибка при инициализации моделей: {e}", exc_info=True)
        logging.critical("Сервис мониторинга не может быть запущен. Проверьте конфигурацию и доступность моделей.")
        return

    path = config.AUDIO_FILES_PATH
    if not os.path.exists(path):
        logging.warning(f"Папка для мониторинга '{path}' не найдена. Создаю ее.")
        os.makedirs(path)

    event_handler = NewFileHandler()
    observer = Observer()
    observer.schedule(event_handler, path, recursive=False) # recursive=False, т.к. нас не интересуют подпапки

    logging.info(f"Начинаю мониторинг папки: {path}")
    observer.start()

    try:
        while True:
            time.sleep(5)
    except KeyboardInterrupt:
        observer.stop()
        logging.info("Получен сигнал KeyboardInterrupt. Остановка сервиса мониторинга.")

    observer.join()
    logging.info("Сервис мониторинга файлов остановлен.")

# Этот блок позволяет запускать скрипт напрямую
# python -m src.file_watcher
if __name__ == '__main__':
    start_watching()
