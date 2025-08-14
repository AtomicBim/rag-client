"""
Конфигурационные параметры для RAG-системы.
"""
import os

# === Параметры Qdrant ===
QDRANT_HOST: str = os.getenv("QDRANT_HOST", "192.168.42.188")
QDRANT_PORT: int = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "internal_regulations_v2")

# === Параметры модели эмбеддингов ===
EMBEDDING_MODEL_PATH: str = os.getenv(
    "EMBEDDING_MODEL_PATH", 
    "./local_model/sbert_large_mt_nlu_ru"
)
EMBEDDING_DIMENSION: int = int(os.getenv("EMBEDDING_DIMENSION", "1024"))

# === Параметры обработки документов ===
DOCS_ROOT_PATH: str = os.getenv("DOCS_ROOT_PATH", "./rag-source")
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))
BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "512"))

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