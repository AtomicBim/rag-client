"""
Конфигурационные параметры для RAG-системы.
Обновлено с оптимизациями для лучшей производительности и точности.
"""
import os
import logging

# === Параметры Qdrant ===
QDRANT_HOST: str = os.getenv("QDRANT_HOST", "192.168.42.188")
QDRANT_PORT: int = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "internal_regulations_v3")

# Оптимизированные параметры HNSW индекса
HNSW_CONFIG = {
    "m": int(os.getenv("HNSW_M", "24")),  # Увеличено для лучшей точности
    "ef_construct": int(os.getenv("HNSW_EF_CONSTRUCT", "300")),  # Оптимизировано для качества
    "full_scan_threshold": int(os.getenv("HNSW_FULL_SCAN_THRESHOLD", "5000"))
}

# === Параметры модели эмбеддингов ===
EMBEDDING_MODEL_PATH: str = os.getenv(
    "EMBEDDING_MODEL_PATH", 
    "BAAI/bge-m3"  # Обновлено на BGE-M3 для лучшего качества
)
EMBEDDING_DIMENSION: int = int(os.getenv("EMBEDDING_DIMENSION", "1024"))

# === Оптимизированные параметры обработки документов ===
DOCS_ROOT_PATH: str = os.getenv("DOCS_ROOT_PATH", "./rag-source")
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "800"))  # Уменьшено для лучшей точности
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "150"))  # Пропорционально уменьшено
BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "256"))  # Уменьшено для стабильности

# === Внешние API ===
OPENAI_API_ENDPOINT: str = os.getenv(
    "OPENAI_API_ENDPOINT", 
    "http://192.168.45.79:8000/generate_answer"
)

# === Параметры логирования ===
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT: str = os.getenv(
    "LOG_FORMAT",
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# === Параметры сервиса эмбеддингов ===
EMBEDDING_SERVICE_HOST: str = os.getenv("EMBEDDING_SERVICE_HOST", "0.0.0.0")
EMBEDDING_SERVICE_PORT: int = int(os.getenv("EMBEDDING_SERVICE_PORT", "8001"))
CORS_ORIGINS: str = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:8000")

# === Новые параметры поиска и оптимизации ===
SEARCH_LIMIT: int = int(os.getenv("SEARCH_LIMIT", "20"))  # Больше кандидатов для лучшей точности
RERANK_TOP_K: int = int(os.getenv("RERANK_TOP_K", "5"))   # Финальные результаты после переранжирования
SEARCH_THRESHOLD: float = float(os.getenv("SEARCH_THRESHOLD", "0.7"))  # Порог релевантности

# === Флаги функциональности ===
ENABLE_HYBRID_SEARCH: bool = os.getenv("ENABLE_HYBRID_SEARCH", "false").lower() == "true"
ENABLE_RERANKING: bool = os.getenv("ENABLE_RERANKING", "false").lower() == "true"
CACHE_EMBEDDINGS: bool = os.getenv("CACHE_EMBEDDINGS", "true").lower() == "true"

# === Специальные разделители для улучшенного чанкинга ===
CHUNK_SEPARATORS = [
    "\n\n\n",  # Разделы документа
    "\n\n",    # Параграфы
    "\n",      # Строки
    ".",       # Предложения
    "!",
    "?",
    ";",
    " ",       # Слова
    ""         # Символы (последний резерв)
]

def validate_config() -> bool:
    """Проверка конфигурации на корректность."""
    errors = []
    
    # Проверка папки с документами
    if not os.path.exists(DOCS_ROOT_PATH):
        errors.append(f"Папка с документами не найдена: {DOCS_ROOT_PATH}")
    
    # Проверка модели
    if not os.path.exists(EMBEDDING_MODEL_PATH):
        errors.append(f"Модель эмбеддингов не найдена: {EMBEDDING_MODEL_PATH}")
    
    # Проверка числовых параметров
    if CHUNK_SIZE <= 0:
        errors.append(f"Некорректный размер чанка: {CHUNK_SIZE}")
    
    if CHUNK_OVERLAP < 0 or CHUNK_OVERLAP >= CHUNK_SIZE:
        errors.append(f"Некорректное перекрытие чанков: {CHUNK_OVERLAP}")
    
    if EMBEDDING_DIMENSION <= 0:
        errors.append(f"Некорректная размерность вектора: {EMBEDDING_DIMENSION}")
    
    if BATCH_SIZE <= 0:
        errors.append(f"Некорректный размер батча: {BATCH_SIZE}")
    
    if errors:
        print("\n❌ Ошибки конфигурации:")
        for error in errors:
            print(f"  - {error}")
        return False
    
    return True

def get_config_summary() -> dict:
    """Получение сводки конфигурации."""
    return {
        "qdrant": {
            "host": QDRANT_HOST,
            "port": QDRANT_PORT,
            "collection": COLLECTION_NAME
        },
        "embedding": {
            "model_path": EMBEDDING_MODEL_PATH,
            "dimension": EMBEDDING_DIMENSION
        },
        "documents": {
            "root_path": DOCS_ROOT_PATH,
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "batch_size": BATCH_SIZE
        },
        "api": {
            "openai_endpoint": OPENAI_API_ENDPOINT
        }
    }

def setup_logging(name: str = __name__) -> logging.Logger:
    """Настройка единого логирования для всех модулей."""
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL.upper()),
        format=LOG_FORMAT,
        force=True
    )
    return logging.getLogger(name)