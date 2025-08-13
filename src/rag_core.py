import logging
import chromadb
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter

from . import config

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Глобальные переменные для моделей и клиента БД
llm = None
embedding_model = None
chroma_client = None
collection = None

def initialize_rag():
    """
    Инициализирует все компоненты RAG:
    - Модель для эмбеддингов
    - Клиент векторной БД ChromaDB
    - Языковую модель Llama
    """
    global embedding_model, chroma_client, collection, llm

    # 1. Загрузка модели для создания эмбеддингов
    try:
        logging.info(f"Загрузка embedding-модели: {config.EMBEDDING_MODEL}...")
        embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)
        logging.info("Embedding-модель успешно загружена.")
    except Exception as e:
        logging.error(f"Не удалось загрузить embedding-модель: {e}")
        raise

    # 2. Инициализация клиента ChromaDB
    try:
        logging.info(f"Инициализация ChromaDB из папки: {config.DB_PATH}...")
        chroma_client = chromadb.PersistentClient(path=config.DB_PATH)
        collection = chroma_client.get_or_create_collection(name="voice_transcripts")
        logging.info("ChromaDB успешно инициализирована.")
    except Exception as e:
        logging.error(f"Не удалось инициализировать ChromaDB: {e}")
        raise

    # 3. Загрузка модели Llama
    try:
        logging.info(f"Загрузка LLM модели: {config.LLAMA_MODEL_PATH}...")
        llm = Llama(
            model_path=config.LLAMA_MODEL_PATH,
            n_ctx=config.LLAMA_N_CTX,
            n_gpu_layers=config.LLAMA_N_GPU_LAYERS,
            verbose=False
        )
        logging.info("LLM модель успешно загружена.")
    except Exception as e:
        logging.error(f"Не удалось загрузить LLM модель. Убедитесь, что путь в config.py указан верно. Ошибка: {e}")
        raise

def add_text_to_db(text: str, source_file: str):
    """
    Разбивает текст на чанки, векторизует и сохраняет в ChromaDB.
    """
    if not all([embedding_model, collection]):
        logging.error("RAG не инициализирован. Невозможно добавить текст.")
        return

    logging.info(f"Добавление в базу текста из файла: {source_file}")

    # Разбиваем текст на чанки
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)

    # Создаем эмбеддинги для каждого чанка
    embeddings = embedding_model.encode(chunks).tolist()

    # Создаем метаданные и ID
    metadatas = [{"source": source_file} for _ in chunks]
    ids = [f"{source_file}_{i}" for i in range(len(chunks))]

    # Добавляем в базу
    collection.add(
        embeddings=embeddings,
        documents=chunks,
        metadatas=metadatas,
        ids=ids
    )
    logging.info(f"Текст из {source_file} успешно добавлен в базу. Чанков: {len(chunks)}")

def query_rag(question: str) -> dict:
    """
    Выполняет RAG-запрос: находит релевантные документы и генерирует ответ.
    """
    if not all([llm, embedding_model, collection]):
        logging.error("RAG не инициализирован. Невозможно выполнить запрос.")
        return {"answer": "Система RAG не инициализирована.", "sources": []}

    logging.info(f"Получен новый запрос: {question}")

    # 1. Поиск релевантных чанков в ChromaDB
    question_embedding = embedding_model.encode(question).tolist()
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=5 # Количество извлекаемых чанков
    )

    retrieved_docs = results['documents'][0]
    retrieved_sources = list(set([meta['source'] for meta in results['metadatas'][0]]))

    if not retrieved_docs:
        return {"answer": "Извините, я не нашел информации по вашему вопросу в базе знаний.", "sources": []}

    context = "\n\n".join(retrieved_docs)

    # 2. Формирование промпта для LLM
    prompt = f"""
Ты — ИИ-ассистент, отвечающий на вопросы на основе предоставленных отрывков из записей телефонных разговоров.
Твоя задача — дать точный и краткий ответ, основываясь ИСКЛЮЧИТЕЛЬНО на приведенном ниже контексте.
Не добавляй информацию, которой нет в тексте. Если ответ в тексте отсутствует, скажи "Информация не найдена".

Контекст из разговоров:
---
{context}
---

Вопрос: {question}

Ответ:
"""

    # 3. Генерация ответа
    logging.info("Генерация ответа с помощью LLM...")
    output = llm(
        prompt,
        max_tokens=512,
        stop=["Вопрос:", "\n"],
        echo=False
    )

    answer = output['choices'][0]['text'].strip()
    logging.info(f"Сгенерирован ответ: {answer}")

    return {"answer": answer, "sources": retrieved_sources}
