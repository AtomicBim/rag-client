"""
Оптимизированные параметры для RAG-системы 2025.
Включает лучшие практики для RTX 3060 4GB и hybrid retrieval.
"""
import os
import logging
from typing import List, Dict, Any

# === Параметры Qdrant ===
QDRANT_HOST: str = os.getenv("QDRANT_HOST", "192.168.42.188")
QDRANT_PORT: int = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "hybrid_rag_v4")

# Оптимизированные параметры HNSW индекса для hybrid search
HNSW_CONFIG = {
    "m": int(os.getenv("HNSW_M", "32")),  # Увеличено для лучшей точности
    "ef_construct": int(os.getenv("HNSW_EF_CONSTRUCT", "400")),  # Оптимизировано 
    "full_scan_threshold": int(os.getenv("HNSW_FULL_SCAN_THRESHOLD", "10000")),
    "max_indexing_threads": int(os.getenv("HNSW_MAX_THREADS", "4"))
}

# === Оптимальные модели эмбеддингов для RTX 3060 4GB ===
EMBEDDING_MODELS = {
    "primary": {
        "path": os.getenv("EMBEDDING_MODEL_PATH", "nomic-ai/nomic-embed-text-v1.5"),
        "dimension": 768,
        "memory_usage": "~300MB",  # Оптимально для 4GB VRAM
        "description": "Лучший выбор для RTX 3060 4GB"
    },
    "fallback": {
        "path": "sentence-transformers/all-MiniLM-L6-v2", 
        "dimension": 384,
        "memory_usage": "~90MB",
        "description": "Резервная модель при нехватке VRAM"
    },
    "multilingual": {
        "path": "intfloat/multilingual-e5-small",
        "dimension": 384, 
        "memory_usage": "~120MB",
        "description": "Для многоязычных документов"
    }
}

# Текущая модель
EMBEDDING_MODEL_PATH: str = EMBEDDING_MODELS["primary"]["path"]
EMBEDDING_DIMENSION: int = EMBEDDING_MODELS["primary"]["dimension"]

# === Оптимизированные параметры чанкинга (2025) ===
DOCS_ROOT_PATH: str = os.getenv("DOCS_ROOT_PATH", "./rag-source")

# Оптимальные размеры чанков для различных типов контента
CHUNKING_STRATEGIES = {
    "default": {
        "chunk_size": 256,  # Токены, оптимально для большинства случаев
        "chunk_overlap": 64,  # 25% перекрытие
        "description": "Универсальная стратегия"
    },
    "technical": {
        "chunk_size": 512,  # Для технической документации
        "chunk_overlap": 128,
        "description": "Техническая документация, API"
    },
    "legal": {
        "chunk_size": 384,  # Для правовых документов
        "chunk_overlap": 96,
        "description": "Юридические документы"
    },
    "faq": {
        "chunk_size": 128,  # Для коротких Q&A
        "chunk_overlap": 32,
        "description": "FAQ, короткие ответы"
    }
}

# Текущая стратегия
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", str(CHUNKING_STRATEGIES["default"]["chunk_size"])))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", str(CHUNKING_STRATEGIES["default"]["chunk_overlap"])))
BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "128"))  # Уменьшено для 4GB VRAM

# === Улучшенные разделители для чанкинга ===
CHUNK_SEPARATORS = [
    "\n\n\n",  # Разделы документа
    "\n\n",    # Параграфы
    "\n",      # Строки
    ". ",      # Предложения
    "! ",
    "? ",
    "; ",
    ", ",      # Части предложений
    " ",       # Слова
    ""         # Символы (последний резерв)
]

# === Параметры гибридного поиска ===
HYBRID_SEARCH_CONFIG = {
    "enabled": True,
    "dense_weight": 0.7,  # Вес dense embeddings
    "sparse_weight": 0.3,  # Вес sparse (BM25) поиска
    "fusion_method": "rrf",  # reciprocal rank fusion
    "rrf_k": 60  # RRF parameter
}

# === Параметры поиска и переранжирования ===
SEARCH_CONFIG = {
    "limit": int(os.getenv("SEARCH_LIMIT", "30")),  # Больше кандидатов
    "rerank_top_k": int(os.getenv("RERANK_TOP_K", "5")),  # Финальные результаты
    "threshold": float(os.getenv("SEARCH_THRESHOLD", "0.65")),  # Понижен порог
    "enable_reranking": True,
    "rerank_model": "cross-encoder/ms-marco-MiniLM-L-6-v2"  # Легкая модель
}

# === Sparse поиск (BM25) параметры ===
BM25_CONFIG = {
    "k1": 1.2,  # Стандартное значение
    "b": 0.75,  # Стандартное значение
    "epsilon": 0.25,  # Сглаживание для коротких документов
    "min_term_freq": 1,
    "remove_stopwords": True,
    "language": "russian"
}

# === Кеширование и оптимизация ===
CACHE_CONFIG = {
    "enable_embedding_cache": True,
    "cache_size_mb": 512,  # Ограничено для 4GB VRAM
    "cache_ttl_seconds": 3600,
    "enable_query_cache": True,
    "query_cache_size": 1000
}

# === GPU оптимизация ===
GPU_CONFIG = {
    "device": "cuda" if os.getenv("FORCE_CPU", "false").lower() != "true" else "cpu",
    "memory_fraction": 0.8,  # 80% VRAM для RTX 3060 4GB
    "enable_mixed_precision": True,
    "batch_inference": True,
    "max_batch_size": 64  # Ограничено для 4GB VRAM
}

# === API конфигурация ===
OPENAI_API_ENDPOINT: str = os.getenv(
    "OPENAI_API_ENDPOINT", 
    "http://192.168.45.79:8000/generate_answer"
)

EMBEDDING_SERVICE_HOST: str = os.getenv("EMBEDDING_SERVICE_HOST", "0.0.0.0")
EMBEDDING_SERVICE_PORT: int = int(os.getenv("EMBEDDING_SERVICE_PORT", "8001"))
CORS_ORIGINS: str = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:8000")

# === Логирование ===
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT: str = os.getenv(
    "LOG_FORMAT",
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# === Мониторинг и метрики ===
METRICS_CONFIG = {
    "enable_metrics": True,
    "track_latency": True,
    "track_memory": True,
    "track_accuracy": True,
    "metrics_port": 8002
}

def get_optimal_model_for_vram(available_vram_gb: float) -> Dict[str, Any]:
    """Выбор оптимальной модели в зависимости от доступной VRAM."""
    if available_vram_gb >= 6:
        return EMBEDDING_MODELS["primary"]
    elif available_vram_gb >= 2:
        return EMBEDDING_MODELS["multilingual"] 
    else:
        return EMBEDDING_MODELS["fallback"]

def get_chunking_strategy(content_type: str = "default") -> Dict[str, Any]:
    """Получение стратегии чанкинга для типа контента."""
    return CHUNKING_STRATEGIES.get(content_type, CHUNKING_STRATEGIES["default"])

def validate_config() -> bool:
    """Проверка конфигурации на корректность."""
    errors = []
    
    # Проверка папки с документами
    if not os.path.exists(DOCS_ROOT_PATH):
        errors.append(f"Папка с документами не найдена: {DOCS_ROOT_PATH}")
    
    # Проверка числовых параметров
    if CHUNK_SIZE <= 0:
        errors.append(f"Некорректный размер чанка: {CHUNK_SIZE}")
    
    if CHUNK_OVERLAP < 0 or CHUNK_OVERLAP >= CHUNK_SIZE:
        errors.append(f"Некорректное перекрытие чанков: {CHUNK_OVERLAP}")
    
    if EMBEDDING_DIMENSION <= 0:
        errors.append(f"Некорректная размерность вектора: {EMBEDDING_DIMENSION}")
    
    # Проверка hybrid search параметров
    total_weight = HYBRID_SEARCH_CONFIG["dense_weight"] + HYBRID_SEARCH_CONFIG["sparse_weight"]
    if abs(total_weight - 1.0) > 0.01:
        errors.append(f"Сумма весов hybrid search должна быть 1.0, получено: {total_weight}")
    
    if errors:
        print("\n❌ Ошибки конфигурации:")
        for error in errors:
            print(f"  - {error}")
        return False
    
    return True

def get_config_summary() -> Dict[str, Any]:
    """Получение полной сводки конфигурации."""
    return {
        "qdrant": {
            "host": QDRANT_HOST,
            "port": QDRANT_PORT,
            "collection": COLLECTION_NAME,
            "hnsw_config": HNSW_CONFIG
        },
        "embedding": {
            "model_path": EMBEDDING_MODEL_PATH,
            "dimension": EMBEDDING_DIMENSION,
            "available_models": EMBEDDING_MODELS
        },
        "chunking": {
            "size": CHUNK_SIZE,
            "overlap": CHUNK_OVERLAP,
            "batch_size": BATCH_SIZE,
            "strategies": CHUNKING_STRATEGIES
        },
        "hybrid_search": HYBRID_SEARCH_CONFIG,
        "search": SEARCH_CONFIG,
        "bm25": BM25_CONFIG,
        "gpu": GPU_CONFIG,
        "cache": CACHE_CONFIG,
        "api": {
            "openai_endpoint": OPENAI_API_ENDPOINT,
            "embedding_service": f"{EMBEDDING_SERVICE_HOST}:{EMBEDDING_SERVICE_PORT}"
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