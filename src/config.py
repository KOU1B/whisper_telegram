# config.py
# Этот файл содержит все основные настройки проекта.
# Отредактируйте его перед запуском.

# --- Настройки Telegram-бота ---
# Вставьте сюда токен, полученный от @BotFather
TELEGRAM_BOT_TOKEN = "YOUR_TELEGRAM_BOT_TOKEN_HERE"

# --- Настройки путей ---
# Папка, куда телефон будет загружать аудиофайлы
AUDIO_FILES_PATH = "audio_files"

# Папка для хранения векторной базы данных
DB_PATH = "db"

# Папка для сохранения текстовых расшифровок
TRANSCRIPTS_PATH = "transcripts"

# --- Настройки моделей ---
# Модель Whisper для транскрибации (tiny, base, small, medium, large)
WHISPER_MODEL = "small"

# --- Настройки RAG ---
# Модель для генерации эмбеддингов (векторов)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Настройки для Llama
# Укажите здесь полный путь к файлу модели Llama в формате GGUF
# Например: "/home/user/models/llama-2-7b-chat.Q4_K_M.gguf"
# Модель нужно скачать отдельно.
LLAMA_MODEL_PATH = "/path/to/your/llama/model.gguf"
LLAMA_N_CTX = 2048 # Размер контекстного окна модели
LLAMA_N_GPU_LAYERS = 0 # Количество слоев для выгрузки на GPU (установите > 0, если есть NVIDIA GPU)

# --- Настройки обработки файлов ---
# Расширение файлов для отслеживания
WATCH_FILE_EXTENSION = ".m4a"
