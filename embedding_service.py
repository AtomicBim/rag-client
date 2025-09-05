import uvicorn
import torch
import config
from typing import List
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

logger = config.setup_logging(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class EmbeddingService:
    """Singleton класс для управления моделью эмбеддингов."""
    
    _instance = None
    _model = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def initialize_model(self) -> bool:
        """Инициализация модели эмбеддингов."""
        if self._model is not None:
            return True
            
        try:
            logger.info(f"Загрузка embedding-модели на устройство: {DEVICE.upper()}")
            self._model = SentenceTransformer(config.EMBEDDING_MODEL_PATH, device=DEVICE)
            logger.info("✅ Модель успешно загружена")
            return True
        except Exception as e:
            logger.error(f"❌ Критическая ошибка при загрузке модели: {e}")
            return False
    
    def create_embedding(self, text: str) -> List[float]:
        """Создание векторного представления для текста."""
        if self._model is None:
            raise RuntimeError("Модель не инициализирована")
            
        try:
            embedding = self._model.encode(text, show_progress_bar=False)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Ошибка создания эмбеддинга: {e}")
            raise
    
    def get_model_info(self) -> dict:
        """Получение информации о модели."""
        if self._model is None:
            return {"status": "not_initialized"}
            
        return {
            "status": "initialized",
            "device": DEVICE,
            "model_path": config.EMBEDDING_MODEL_PATH,
            "model_name": getattr(self._model, "_model_name", "unknown")
        }

# Инициализация сервиса
embedding_service = EmbeddingService()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Управление жизненным циклом приложения."""
    # Инициализация при запуске
    if not embedding_service.initialize_model():
        logger.error("Не удалось инициализировать модель")
        exit(1)
    
    yield
    
    # Очистка при завершении
    logger.info("Завершение работы сервиса")

# FastAPI приложение
app = FastAPI(
    title="Embedding Generation Service",
    description="Сервис для создания векторных представлений текста",
    version="1.0.0",
    lifespan=lifespan
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

class TextRequest(BaseModel):
    """Модель запроса для создания эмбеддинга."""
    text: str = Field(
        ...,
        description="Текст для векторизации",
        min_length=1,
        max_length=10000,
        example="Пример текста для векторизации"
    )

class EmbeddingResponse(BaseModel):
    """Модель ответа с векторным представлением."""
    embedding: List[float] = Field(
        ...,
        description="Векторное представление текста"
    )
    dimension: int = Field(
        ...,
        description="Размерность вектора"
    )

class HealthResponse(BaseModel):
    """Модель ответа проверки состояния."""
    status: str = Field(..., description="Состояние сервиса")
    model_info: dict = Field(..., description="Информация о модели")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Проверка состояния сервиса.
    """
    model_info = embedding_service.get_model_info()
    return {
        "status": "healthy" if model_info["status"] == "initialized" else "unhealthy",
        "model_info": model_info
    }

@app.post("/create_embedding", 
          response_model=EmbeddingResponse,
          status_code=status.HTTP_200_OK)
async def create_embedding(request: TextRequest):
    """
    Принимает текст и возвращает его векторное представление.
    Поддерживает тексты длиной до 10,000 символов.
    """
    
    try:
        logger.info(f"Создание эмбеддинга для текста длиной {len(request.text)} символов")
        embedding = embedding_service.create_embedding(request.text)
        
        return {
            "embedding": embedding,
            "dimension": len(embedding)
        }
        
    except Exception as e:
        logger.error(f"Ошибка при создании эмбеддинга: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка обработки запроса: {str(e)}"
        )

if __name__ == "__main__":
    logger.info(f"Запуск сервиса эмбеддингов на {config.EMBEDDING_SERVICE_HOST}:{config.EMBEDDING_SERVICE_PORT}")
    uvicorn.run(
        app, 
        host=config.EMBEDDING_SERVICE_HOST, 
        port=config.EMBEDDING_SERVICE_PORT,
        log_level="info",
        access_log=True
    )