import uvicorn
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

# --- Конфигурация ---
# Путь к вашей локальной модели
MODEL_PATH = "C:/Users/r.grigoriev/Desktop/rag-client/local_model/ru-en-RoSBERTa"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Загрузка embedding-модели на устройство: {DEVICE.upper()}")
try:
    embedding_model = SentenceTransformer(MODEL_PATH, device=DEVICE)
    print("✅ Модель успешно загружена.")
except Exception as e:
    print(f"❌ Критическая ошибка при загрузке модели: {e}")
    exit()

# --- FastAPI Приложение ---
app = FastAPI(title="Embedding Generation Service")

class TextRequest(BaseModel):
    text: str

class EmbeddingResponse(BaseModel):
    embedding: list[float]

@app.post("/create_embedding", response_model=EmbeddingResponse)
async def create_embedding(request: TextRequest):
    """
    Принимает текст и возвращает его векторное представление (embedding).
    """
    try:
        embedding = embedding_model.encode(request.text).tolist()
        return {"embedding": embedding}
    except Exception as e:
        # Логирование ошибки было бы здесь очень полезно
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Запускайте на IP, доступном в вашей локальной сети, например 0.0.0.0
    uvicorn.run(app, host="0.0.0.0", port=8001)