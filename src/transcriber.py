import os
import whisper
from pydub import AudioSegment
import logging
from . import config

# Настройка логирования для вывода информации о процессе
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Глобальная переменная для хранения загруженной модели
model = None

def load_whisper_model():
    """
    Загружает модель Whisper в память согласно конфигурации.
    Это ресурсоемкая операция, поэтому она вынесена в отдельную функцию.
    """
    global model
    if model is None:
        try:
            logging.info(f"Загрузка модели Whisper: {config.WHISPER_MODEL}...")
            # Загружаем модель. При наличии CUDA можно будет использовать GPU.
            model = whisper.load_model(config.WHISPER_MODEL)
            logging.info("Модель Whisper успешно загружена.")
        except Exception as e:
            logging.error(f"Не удалось загрузить модель Whisper: {e}")
            raise e  # Пробрасываем ошибку выше, чтобы приложение остановилось

def transcribe_audio(audio_path: str) -> tuple[str | None, str | None]:
    """
    Транскрибирует один аудиофайл.

    Процесс:
    1. Конвертирует .m4a в .wav (16kHz, моно).
    2. Использует Whisper для транскрибации.
    3. Сохраняет результат в .txt файл.
    4. Удаляет временный .wav файл.

    Args:
        audio_path (str): Путь к исходному аудиофайлу.

    Returns:
        Кортеж (путь к .txt файлу, транскрибированный текст) или (None, None) в случае ошибки.
    """
    if model is None:
        logging.error("Модель Whisper не загружена. Вызовите load_whisper_model() перед транскрибацией.")
        return None, None

    if not os.path.exists(audio_path):
        logging.error(f"Аудиофайл не найден: {audio_path}")
        return None, None

    wav_path = ""
    try:
        logging.info(f"Начало обработки файла: {audio_path}")

        base_name = os.path.basename(audio_path)
        file_name_without_ext = os.path.splitext(base_name)[0]

        # 1. Конвертация в WAV
        audio = AudioSegment.from_file(audio_path, format="m4a")
        audio = audio.set_frame_rate(16000).set_channels(1)

        # Создаем временный wav-файл в той же папке, где лежит исходный
        wav_path = os.path.join(os.path.dirname(audio_path), f"{file_name_without_ext}.wav")
        audio.export(wav_path, format="wav")
        logging.info(f"Файл успешно конвертирован в: {wav_path}")

        # 2. Транскрибация
        logging.info(f"Транскрибация файла с помощью модели '{config.WHISPER_MODEL}'...")
        # fp16=False рекомендуется для работы на CPU
        result = model.transcribe(wav_path, fp16=False)
        transcribed_text = result['text'].strip()
        logging.info("Транскрибация успешно завершена.")

        # 3. Сохранение результата
        if not os.path.exists(config.TRANSCRIPTS_PATH):
            os.makedirs(config.TRANSCRIPTS_PATH)

        transcript_filename = f"{file_name_without_ext}.txt"
        transcript_filepath = os.path.join(config.TRANSCRIPTS_PATH, transcript_filename)

        with open(transcript_filepath, 'w', encoding='utf-8') as f:
            f.write(transcribed_text)
        logging.info(f"Транскрипция сохранена в файл: {transcript_filepath}")

        return transcript_filepath, transcribed_text

    except Exception as e:
        logging.error(f"Произошла ошибка при обработке файла {audio_path}: {e}", exc_info=True)
        return None, None
    finally:
        # 4. Очистка: удаление временного .wav файла
        if os.path.exists(wav_path):
            try:
                os.remove(wav_path)
                logging.info(f"Временный файл {wav_path} удален.")
            except OSError as e:
                logging.error(f"Не удалось удалить временный файл {wav_path}: {e}")
