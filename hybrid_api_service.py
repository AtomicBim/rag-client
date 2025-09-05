"""
API сервис для гибридного RAG поиска с оптимизациями 2025 года.
"""
import asyncio
import time
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, status, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import optimized_config as config
from hybrid_retrieval_service import hybrid_retriever, HybridSearchResult, SearchResult

logger = config.setup_logging(__name__)

# Pydantic модели для API
class SearchRequest(BaseModel):
    """Запрос на поиск."""
    query: str = Field(
        ...,
        description="Поисковый запрос",
        min_length=1,
        max_length=1000,
        example="Как настроить безопасность сервера?"
    )
    top_k: int = Field(
        default=5,
        description="Количество результатов",
        ge=1,
        le=20
    )
    search_type: str = Field(
        default="hybrid",
        description="Тип поиска: dense, sparse, hybrid",
        regex="^(dense|sparse|hybrid)$"
    )
    enable_reranking: bool = Field(
        default=True,
        description="Включить переранжирование результатов"
    )

class SearchResultResponse(BaseModel):
    """Ответ с результатом поиска."""
    text: str = Field(..., description="Текст найденного фрагмента")
    score: float = Field(..., description="Релевантность (0-1)")
    source_file: str = Field(..., description="Исходный файл")
    chunk_index: int = Field(..., description="Номер фрагмента в документе")
    retrieval_method: str = Field(..., description="Метод поиска")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Дополнительные метаданные")

class SearchResponse(BaseModel):
    """Полный ответ поиска."""
    query: str = Field(..., description="Исходный запрос")
    results: List[SearchResultResponse] = Field(..., description="Результаты поиска")
    total_results: int = Field(..., description="Общее количество результатов")
    search_time_ms: float = Field(..., description="Время поиска в миллисекундах")
    search_type: str = Field(..., description="Использованный тип поиска")
    performance_metrics: Optional[Dict[str, float]] = Field(
        None, 
        description="Детальные метрики производительности"
    )

class EmbeddingRequest(BaseModel):
    """Запрос на создание эмбеддинга."""
    text: str = Field(
        ...,
        description="Текст для векторизации",
        min_length=1,
        max_length=5000
    )

class EmbeddingResponse(BaseModel):
    """Ответ с эмбеддингом."""
    embedding: List[float] = Field(..., description="Векторное представление")
    dimension: int = Field(..., description="Размерность вектора")
    processing_time_ms: float = Field(..., description="Время обработки")

class HealthResponse(BaseModel):
    """Ответ проверки состояния."""
    status: str = Field(..., description="Статус сервиса")
    version: str = Field(..., description="Версия сервиса")
    models_loaded: Dict[str, bool] = Field(..., description="Статус загрузки моделей")
    system_info: Dict[str, Any] = Field(..., description="Информация о системе")
    config_summary: Dict[str, Any] = Field(..., description="Краткая информация о конфигурации")

class MetricsResponse(BaseModel):
    """Метрики производительности."""
    total_requests: int = Field(..., description="Общее количество запросов")
    average_response_time_ms: float = Field(..., description="Среднее время ответа")
    successful_requests: int = Field(..., description="Успешные запросы")
    failed_requests: int = Field(..., description="Неудачные запросы")
    memory_usage_mb: float = Field(..., description="Использование памяти")
    gpu_memory_usage_mb: Optional[float] = Field(None, description="Использование GPU памяти")

# Глобальные метрики
class ServiceMetrics:
    def __init__(self):
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.response_times = []
        self.start_time = time.time()
    
    def record_request(self, response_time_ms: float, success: bool):
        self.total_requests += 1
        self.response_times.append(response_time_ms)
        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1
        
        # Ограничиваем размер массива времен ответа
        if len(self.response_times) > 1000:
            self.response_times = self.response_times[-500:]
    
    def get_average_response_time(self) -> float:
        return sum(self.response_times) / len(self.response_times) if self.response_times else 0.0

metrics = ServiceMetrics()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Управление жизненным циклом приложения."""
    logger.info("🚀 Запуск сервиса гибридного поиска...")
    
    # Инициализация всех компонентов
    try:
        # Инициализация моделей
        if not hybrid_retriever.initialize_models():
            logger.error("❌ Не удалось инициализировать модели")
            exit(1)
        
        # Подготовка BM25 корпуса
        if not await prepare_bm25_corpus():
            logger.warning("⚠️ BM25 корпус не подготовлен, sparse поиск недоступен")
        
        logger.info("✅ Все компоненты инициализированы успешно")
        
    except Exception as e:
        logger.error(f"❌ Критическая ошибка инициализации: {e}")
        exit(1)
    
    yield
    
    # Очистка при завершении
    logger.info("🔄 Завершение работы сервиса...")

async def prepare_bm25_corpus():
    """Подготовка BM25 корпуса в фоновом режиме."""
    try:
        return hybrid_retriever.prepare_bm25_corpus()
    except Exception as e:
        logger.error(f"Ошибка подготовки BM25 корпуса: {e}")
        return False

# Создание FastAPI приложения
app = FastAPI(
    title="Hybrid RAG Search API",
    description="Высокопроизводительный API для гибридного поиска с поддержкой dense и sparse методов",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Настройка CORS
allowed_origins = config.CORS_ORIGINS.split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

def get_system_info() -> Dict[str, Any]:
    """Получение информации о системе."""
    import psutil
    import torch
    
    system_info = {
        "cpu_percent": psutil.cpu_percent(),
        "memory_percent": psutil.virtual_memory().percent,
        "memory_available_gb": psutil.virtual_memory().available / (1024**3),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available()
    }
    
    if torch.cuda.is_available():
        system_info.update({
            "cuda_version": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_memory_total_gb": torch.cuda.get_device_properties(0).total_memory / (1024**3),
            "gpu_memory_allocated_gb": torch.cuda.memory_allocated(0) / (1024**3),
            "gpu_memory_cached_gb": torch.cuda.memory_reserved(0) / (1024**3)
        })
    
    return system_info

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    Проверка состояния сервиса и всех компонентов.
    """
    try:
        models_status = {
            "embedding_model": hybrid_retriever.embedding_model is not None,
            "dense_retriever": hybrid_retriever.dense_retriever.qdrant_client is not None,
            "sparse_retriever": hybrid_retriever.bm25_retriever.bm25 is not None,
            "rerank_model": hybrid_retriever.rerank_model is not None
        }
        
        all_models_loaded = all(models_status.values())
        status_text = "healthy" if all_models_loaded else "degraded"
        
        return HealthResponse(
            status=status_text,
            version="2.0.0",
            models_loaded=models_status,
            system_info=get_system_info(),
            config_summary=config.get_config_summary()
        )
        
    except Exception as e:
        logger.error(f"Ошибка проверки состояния: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Ошибка проверки состояния сервиса"
        )

@app.get("/metrics", response_model=MetricsResponse, tags=["System"])
async def get_metrics():
    """
    Получение метрик производительности сервиса.
    """
    try:
        import psutil
        import torch
        
        memory_usage = psutil.Process().memory_info().rss / (1024**2)  # MB
        gpu_memory = None
        
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated(0) / (1024**2)  # MB
        
        return MetricsResponse(
            total_requests=metrics.total_requests,
            average_response_time_ms=metrics.get_average_response_time(),
            successful_requests=metrics.successful_requests,
            failed_requests=metrics.failed_requests,
            memory_usage_mb=memory_usage,
            gpu_memory_usage_mb=gpu_memory
        )
        
    except Exception as e:
        logger.error(f"Ошибка получения метрик: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Ошибка получения метрик"
        )

@app.post("/search", response_model=SearchResponse, tags=["Search"])
async def hybrid_search(request: SearchRequest):
    """
    Выполнение гибридного поиска по документам.
    
    Поддерживает три типа поиска:
    - **dense**: Поиск по векторным эмбеддингам
    - **sparse**: Поиск по BM25 (ключевые слова)  
    - **hybrid**: Комбинация dense + sparse (рекомендуется)
    """
    start_time = time.time()
    
    try:
        logger.info(f"Поисковый запрос: '{request.query}' (тип: {request.search_type})")
        
        # Валидация запроса
        if not request.query.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Поисковый запрос не может быть пустым"
            )
        
        # Выполнение поиска в зависимости от типа
        if request.search_type == "hybrid":
            # Гибридный поиск
            search_result = await hybrid_retriever.hybrid_search(
                query=request.query,
                top_k=request.top_k
            )
            
            performance_metrics = {
                "total_time_ms": search_result.total_time_ms,
                "dense_time_ms": search_result.dense_time_ms,
                "sparse_time_ms": search_result.sparse_time_ms,
                "fusion_time_ms": search_result.fusion_time_ms,
                "rerank_time_ms": search_result.rerank_time_ms
            }
            
            results = search_result.results
            
        elif request.search_type == "dense":
            # Только dense поиск
            query_vector = hybrid_retriever.embedding_model.encode(request.query).tolist()
            results = hybrid_retriever.dense_retriever.search(
                query_vector=query_vector,
                top_k=request.top_k
            )
            performance_metrics = {"search_method": "dense_only"}
            
        elif request.search_type == "sparse":
            # Только sparse поиск
            results = hybrid_retriever.bm25_retriever.search(
                query=request.query,
                top_k=request.top_k
            )
            performance_metrics = {"search_method": "sparse_only"}
            
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Неподдерживаемый тип поиска"
            )
        
        # Переранжирование (если включено)
        if request.enable_reranking and request.search_type != "hybrid":
            rerank_start = time.time()
            results = hybrid_retriever.rerank_results(request.query, results)
            performance_metrics["rerank_time_ms"] = (time.time() - rerank_start) * 1000
        
        # Формирование ответа
        response_results = [
            SearchResultResponse(
                text=result.text,
                score=result.score,
                source_file=result.source_file,
                chunk_index=result.chunk_index,
                retrieval_method=result.retrieval_method,
                metadata=result.metadata
            )
            for result in results[:request.top_k]
        ]
        
        total_time = (time.time() - start_time) * 1000
        
        # Запись метрик
        metrics.record_request(total_time, True)
        
        return SearchResponse(
            query=request.query,
            results=response_results,
            total_results=len(response_results),
            search_time_ms=total_time,
            search_type=request.search_type,
            performance_metrics=performance_metrics
        )
        
    except HTTPException:
        # Переброс HTTP исключений
        metrics.record_request((time.time() - start_time) * 1000, False)
        raise
        
    except Exception as e:
        metrics.record_request((time.time() - start_time) * 1000, False)
        logger.error(f"Ошибка поиска: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Внутренняя ошибка поиска: {str(e)}"
        )

@app.post("/embed", response_model=EmbeddingResponse, tags=["Embedding"])
async def create_embedding(request: EmbeddingRequest):
    """
    Создание векторного представления для текста.
    """
    start_time = time.time()
    
    try:
        if not request.text.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Текст не может быть пустым"
            )
        
        if hybrid_retriever.embedding_model is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Embedding модель не загружена"
            )
        
        # Создание эмбеддинга
        embedding = hybrid_retriever.embedding_model.encode(request.text).tolist()
        
        processing_time = (time.time() - start_time) * 1000
        
        return EmbeddingResponse(
            embedding=embedding,
            dimension=len(embedding),
            processing_time_ms=processing_time
        )
        
    except HTTPException:
        raise
        
    except Exception as e:
        logger.error(f"Ошибка создания эмбеддинга: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка создания эмбеддинга: {str(e)}"
        )

@app.get("/collections/info", tags=["System"])
async def get_collection_info():
    """
    Получение информации о коллекции Qdrant.
    """
    try:
        if hybrid_retriever.dense_retriever.qdrant_client is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Qdrant клиент не инициализирован"
            )
        
        collection_info = hybrid_retriever.dense_retriever.qdrant_client.get_collection(
            collection_name=config.COLLECTION_NAME
        )
        
        return {
            "collection_name": config.COLLECTION_NAME,
            "vectors_count": collection_info.vectors_count,
            "points_count": collection_info.points_count,
            "status": collection_info.status,
            "config": {
                "vector_size": collection_info.config.params.vectors.size,
                "distance": collection_info.config.params.vectors.distance
            }
        }
        
    except Exception as e:
        logger.error(f"Ошибка получения информации о коллекции: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Ошибка получения информации о коллекции"
        )

if __name__ == "__main__":
    logger.info("Запуск API сервиса гибридного поиска...")
    uvicorn.run(
        "hybrid_api_service:app",
        host=config.EMBEDDING_SERVICE_HOST,
        port=config.EMBEDDING_SERVICE_PORT,
        log_level="info",
        access_log=True,
        reload=False
    )