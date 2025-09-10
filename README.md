# RAG-клиент для обработки и индексации документов

## 📋 Описание проекта

RAG-клиент — это система для автоматической индексации и поиска информации в корпоративных документах с использованием технологии Retrieval-Augmented Generation (RAG). Система состоит из трех основных компонентов:

1. **Модуль индексации документов** (`upload_docs.py`) - обработка и индексация документов
2. **Сервис эмбеддингов** (`embedding_service.py`) - веб-API для создания векторных представлений текста
3. **Модуль конфигурации** (`config.py`) - централизованное управление настройками

## 🏗️ Архитектура системы

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Документы      │    │   RAG-клиент    │    │ Векторная БД    │
│  (.docx, .pdf,  │────│   (Python)      │────│   (Qdrant)      │
│   .doc)         │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                │
                        ┌─────────────────┐
                        │ Модель эмбед.   │
                        │ (Sentence-      │
                        │  Transformers)  │
                        └─────────────────┘
```

## 📁 Структура проекта

```
rag-client/
├── config.py                    # Конфигурационные параметры
├── embedding_service.py         # FastAPI веб-сервис эмбеддингов
├── upload_docs.py              # Модуль индексации документов
├── requirements.txt            # Зависимости Python
├── indexing_state.json        # Файл состояния (создается автоматически)
├── local_model/               # Папка для модели эмбеддингов
│   └── multilingual-e5-large/ # Модель Sentence-Transformers
└── rag-source/                # Папка с документами для индексации
    ├── category1/             # Категории документов
    └── category2/
```

## 🚀 Быстрый старт

### 1. Установка зависимостей

```bash
# Установка Python-пакетов
pip install -r requirements.txt

# На Windows может понадобиться Visual C++ Redistributable:
# https://aka.ms/vs/16/release/vc_redist.x64.exe
```

### 2. Настройка окружения

Создайте файл `.env` (опционально) или установите переменные окружения:

```env
# Подключение к Qdrant
QDRANT_HOST=192.168.42.188
QDRANT_PORT=6333
COLLECTION_NAME=internal_regulations_v2

# Путь к модели эмбеддингов
EMBEDDING_MODEL_PATH=./local_model/multilingual-e5-large
EMBEDDING_DIMENSION=1024

# Папка с документами
DOCS_ROOT_PATH=./rag-source

# Параметры обработки
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
BATCH_SIZE=512

# API эндпоинт для генерации ответов
OPENAI_API_ENDPOINT=http://192.168.45.79:8000/generate_answer

# Настройки сервиса эмбеддингов
EMBEDDING_SERVICE_HOST=0.0.0.0
EMBEDDING_SERVICE_PORT=8001
```

### 3. Подготовка модели эмбеддингов

Скачайте модель `multilingual-e5-large` в папку `./local_model/`:

```bash
# Создайте папку для модели
mkdir -p local_model

# Скачайте модель (автоматически при первом запуске) или
# Поместите готовую модель в local_model/multilingual-e5-large/
```

### 4. Подготовка документов

Поместите документы в папку `rag-source/`, организовав их по категориям:

```
rag-source/
├── regulations/       # Внутренние регламенты
│   ├── document1.docx
│   └── document2.pdf
├── procedures/        # Процедуры
│   └── procedure1.doc
└── policies/          # Политики
    └── policy1.pdf
```

### 5. Запуск системы

#### Индексация документов:
```bash
python upload_docs.py
```

#### Запуск сервиса эмбеддингов:
```bash
python embedding_service.py
```

## 📖 Подробное описание модулей

### config.py

Центральный модуль конфигурации, который управляет всеми настройками системы через переменные окружения.

#### Основные функции:

- **`validate_config()`** - Проверяет корректность всех настроек перед запуском
- **`get_config_summary()`** - Возвращает сводку текущих настроек
- **`setup_logging()`** - Настраивает единое логирование для всех модулей

#### Группы параметров:

**Qdrant (векторная БД):**
- `QDRANT_HOST` - Хост сервера Qdrant
- `QDRANT_PORT` - Порт подключения
- `COLLECTION_NAME` - Имя коллекции векторов

**Модель эмбеддингов:**
- `EMBEDDING_MODEL_PATH` - Путь к модели Sentence-Transformers
- `EMBEDDING_DIMENSION` - Размерность векторов (1024 для multilingual-e5-large)

**Обработка документов:**
- `DOCS_ROOT_PATH` - Корневая папка с документами
- `CHUNK_SIZE` - Размер текстовых фрагментов (в символах)
- `CHUNK_OVERLAP` - Перекрытие между фрагментами
- `BATCH_SIZE` - Размер батча для загрузки векторов

### embedding_service.py

FastAPI веб-сервис для создания векторных представлений текста. Реализует REST API для взаимодействия с моделью эмбеддингов.

#### Архитектурные особенности:

- **Singleton паттерн** для `EmbeddingService` - гарантирует одну инициализированную модель
- **Автоматическое управление жизненным циклом** через `lifespan` контекст
- **CORS поддержка** для веб-интеграции
- **Автоматическое определение устройства** (GPU/CPU)

#### API Эндпоинты:

**`GET /health`** - Проверка состояния сервиса
```json
{
  "status": "healthy",
  "model_info": {
    "status": "initialized",
    "device": "cuda",
    "model_path": "./local_model/multilingual-e5-large",
    "model_name": "sentence-transformers/multilingual-E5-large"
  }
}
```

**`POST /create_embedding`** - Создание эмбеддинга для текста
```json
// Запрос:
{
  "text": "Пример текста для векторизации"
}

// Ответ:
{
  "embedding": [0.1, -0.2, 0.3, ...],
  "dimension": 1024
}
```

#### Класс EmbeddingService:

- **`initialize_model()`** - Загружает модель Sentence-Transformers на доступное устройство
- **`create_embedding(text)`** - Создает векторное представление для входного текста
- **`get_model_info()`** - Возвращает информацию о состоянии модели

### upload_docs.py

Основной модуль для индексации документов. Выполняет полный цикл обработки: от извлечения текста до сохранения векторов в Qdrant.

#### Архитектура:

**Класс DocumentIndexer** - главный класс для индексации:

- **`initialize_qdrant()`** - Подключение к векторной БД, создание коллекции при необходимости
- **`initialize_embedding_model()`** - Загрузка модели эмбеддингов
- **`process_document(file_path)`** - Полная обработка одного документа
- **`delete_old_document_records(filename)`** - Удаление старых записей при переиндексации
- **`create_embeddings_batch(chunks, filename, category)`** - Создание батча векторов
- **`upload_points_to_qdrant(points)`** - Загрузка векторов в Qdrant батчами

#### Поддерживаемые форматы документов:

**`.docx` файлы:**
- Используется библиотека `unstructured` с поддержкой таблиц
- Функция: `extract_text_from_docx(file_path)`

**`.pdf` файлы:**
- Используется `pypdf` для извлечения текста
- Функция: `extract_text_from_pdf(file_path)`  
- Поддержка многостраничных документов

**`.doc` файлы (только Windows):**
- Автоматическая конвертация в `.docx` через `win32com.client`
- Функция: `convert_doc_to_docx(file_path)`
- Требует установленный Microsoft Word
- Оригинальный `.doc` файл удаляется после конвертации

#### Умная система индексации:

**Файл состояния (`indexing_state.json`):**
```json
{
  "/path/to/doc1.docx": 1694123456.789,
  "/path/to/doc2.pdf": 1694123500.123
}
```

- **`load_state()`** - Загрузка последнего состояния индексации
- **`save_state(state)`** - Сохранение нового состояния
- **`find_changed_files(docs_path, state)`** - Поиск измененных файлов

Система отслеживает время модификации файлов и обрабатывает только новые или измененные документы.

#### Процесс обработки документа:

1. **Удаление старых записей** - очистка векторов предыдущей версии
2. **Извлечение текста** - в зависимости от формата файла
3. **Разбиение на фрагменты** - используется `RecursiveCharacterTextSplitter`
4. **Создание эмбеддингов** - векторизация каждого фрагмента
5. **Сохранение в Qdrant** - загрузка векторов с метаданными

#### Метаданные векторов:

```python
payload = {
    "text": "Содержимое фрагмента документа...",
    "source_file": "document.docx", 
    "category": "regulations"  # Папка, в которой находится файл
}
```

## ⚙️ Конфигурация и настройки

### Переменные окружения

| Параметр | Значение по умолчанию | Описание |
|----------|---------------------|----------|
| `QDRANT_HOST` | 192.168.42.188 | IP-адрес сервера Qdrant |
| `QDRANT_PORT` | 6333 | Порт Qdrant |
| `COLLECTION_NAME` | internal_regulations_v2 | Имя коллекции векторов |
| `EMBEDDING_MODEL_PATH` | ./local_model/multilingual-e5-large | Путь к модели |
| `EMBEDDING_DIMENSION` | 1024 | Размерность векторов |
| `DOCS_ROOT_PATH` | ./rag-source | Папка с документами |
| `CHUNK_SIZE` | 1000 | Размер текстового фрагмента |
| `CHUNK_OVERLAP` | 200 | Перекрытие фрагментов |
| `BATCH_SIZE` | 512 | Размер батча для Qdrant |
| `OPENAI_API_ENDPOINT` | http://192.168.45.79:8000/generate_answer | Эндпоинт для ответов |
| `EMBEDDING_SERVICE_HOST` | 0.0.0.0 | Хост сервиса эмбеддингов |
| `EMBEDDING_SERVICE_PORT` | 8001 | Порт сервиса эмбеддингов |
| `CORS_ORIGINS` | http://localhost:3000,http://localhost:8000 | Разрешенные CORS источники |
| `LOG_LEVEL` | INFO | Уровень логирования |

### Оптимальные значения параметров:

**CHUNK_SIZE (размер фрагмента):**
- 500-800 символов - для коротких документов
- 1000-1500 символов - для средних документов (рекомендуется)
- 2000+ символов - для больших документов с сложной структурой

**CHUNK_OVERLAP (перекрытие):**
- 10-20% от CHUNK_SIZE - стандартное значение
- 200 символов - универсальное значение для большинства случаев

**BATCH_SIZE:**
- 128-256 - для слабых серверов
- 512 - оптимально для большинства случаев  
- 1024+ - для мощных серверов с большим объемом RAM

## 🐳 Docker развертывание

**Примечание:** В текущей версии проекта отсутствуют готовые Docker конфигурационные файлы. Ниже представлены рекомендуемые конфигурации для контейнеризации системы.

### Создание Dockerfile

Создайте файл `Dockerfile` в корне проекта:

```dockerfile
FROM python:3.11-slim

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Создание рабочей директории
WORKDIR /app

# Копирование файлов зависимостей
COPY requirements.txt .

# Установка Python зависимостей
RUN pip install --no-cache-dir -r requirements.txt

# Копирование исходного кода
COPY . .

# Создание директории для модели и документов
RUN mkdir -p local_model rag-source

# Открытие порта для сервиса эмбеддингов
EXPOSE 8001

# Команда по умолчанию
CMD ["python", "embedding_service.py"]
```

### Docker Compose конфигурация

Создайте файл `docker-compose.yml`:

```yaml
version: '3.8'

services:
  # Векторная база данных Qdrant
  qdrant:
    image: qdrant/qdrant:v1.9.0
    container_name: qdrant
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - qdrant_storage:/qdrant/storage
    environment:
      - QDRANT__SERVICE__HTTP_PORT=6333
      - QDRANT__SERVICE__GRPC_PORT=6334
    networks:
      - rag_network

  # RAG сервис эмбеддингов
  rag-embedding-service:
    build: .
    container_name: rag-embedding-service
    ports:
      - "8001:8001"
    environment:
      - QDRANT_HOST=qdrant
      - QDRANT_PORT=6333
      - COLLECTION_NAME=internal_regulations_v2
      - EMBEDDING_SERVICE_HOST=0.0.0.0
      - EMBEDDING_SERVICE_PORT=8001
      - EMBEDDING_MODEL_PATH=/app/local_model/multilingual-e5-large
      - DOCS_ROOT_PATH=/app/rag-source
    volumes:
      - ./local_model:/app/local_model
      - ./rag-source:/app/rag-source
      - ./indexing_state.json:/app/indexing_state.json
    depends_on:
      - qdrant
    networks:
      - rag_network
    command: python embedding_service.py

  # Инициализация и индексация документов
  rag-indexer:
    build: .
    container_name: rag-indexer
    environment:
      - QDRANT_HOST=qdrant
      - QDRANT_PORT=6333
      - COLLECTION_NAME=internal_regulations_v2
      - EMBEDDING_MODEL_PATH=/app/local_model/multilingual-e5-large
      - DOCS_ROOT_PATH=/app/rag-source
    volumes:
      - ./local_model:/app/local_model
      - ./rag-source:/app/rag-source
      - ./indexing_state.json:/app/indexing_state.json
    depends_on:
      - qdrant
    networks:
      - rag_network
    command: python upload_docs.py
    restart: "no"  # Запускается один раз для индексации

volumes:
  qdrant_storage:

networks:
  rag_network:
    driver: bridge
```

### Команды для развертывания

```bash
# Сборка и запуск всех сервисов
docker-compose up --build

# Запуск в фоновом режиме
docker-compose up -d

# Просмотр логов
docker-compose logs -f rag-embedding-service
docker-compose logs -f rag-indexer

# Перезапуск конкретного сервиса
docker-compose restart rag-embedding-service

# Остановка всех сервисов
docker-compose down

# Остановка с удалением volumes
docker-compose down -v
```

### Multi-stage Dockerfile (оптимизированный)

Для production использования создайте оптимизированный Dockerfile:

```dockerfile
# Этап сборки
FROM python:3.11-slim as builder

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Production этап
FROM python:3.11-slim

# Создание пользователя без прав root
RUN groupadd -r raguser && useradd -r -g raguser raguser

WORKDIR /app

# Копирование установленных пакетов
COPY --from=builder /root/.local /root/.local

# Копирование исходного кода
COPY --chown=raguser:raguser . .

# Создание необходимых директорий
RUN mkdir -p local_model rag-source && \
    chown -R raguser:raguser /app

# Переключение на непривилегированного пользователя
USER raguser

# Обновление PATH для pip packages
ENV PATH=/root/.local/bin:$PATH

EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

CMD ["python", "embedding_service.py"]
```

## 🔧 Устранение неполадок

### Проблемы с установкой

**Ошибка при установке torch/sentence-transformers на Windows:**
```bash
# Установите Microsoft Visual C++ Redistributable
# Скачать: https://aka.ms/vs/16/release/vc_redist.x64.exe

# Затем переустановите torch:
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cu118  # Для GPU
pip install torch --index-url https://download.pytorch.org/whl/cpu    # Для CPU
```

**Проблемы с pywin32 (Windows):**
```bash
# Установка с правами администратора:
pip install pywin32==306

# Или альтернативный способ:
conda install pywin32
```

### Проблемы с Qdrant

**Подключение отклонено:**
- Проверьте доступность Qdrant: `telnet QDRANT_HOST QDRANT_PORT`
- Убедитесь в корректности параметров `QDRANT_HOST` и `QDRANT_PORT`
- Проверьте firewall и сетевые настройки

**Коллекция не создается:**
- Проверьте размерность векторов (`EMBEDDING_DIMENSION`)
- Убедитесь в наличии прав на запись в Qdrant
- Проверите логи Qdrant на наличие ошибок

### Проблемы с документами

**Документы не обрабатываются:**
- Проверьте путь к папке: `DOCS_ROOT_PATH`
- Убедитесь в поддержке формата: `.docx`, `.pdf`, `.doc`
- Проверьте права доступа к файлам
- Исключите временные файлы (начинающиеся с `~`)

**Ошибки извлечения текста:**
- Для `.doc` файлов требуется Microsoft Word на Windows
- PDF файлы могут содержать только изображения (не извлекаемый текст)
- Поврежденные файлы могут вызывать ошибки

### Производительность

**Медленная индексация:**
- Уменьшите `BATCH_SIZE` при недостатке памяти
- Используйте GPU для ускорения (CUDA)
- Оптимизируйте `CHUNK_SIZE` под ваши документы

**Высокое использование памяти:**
- Уменьшите `BATCH_SIZE`
- Используйте меньшую модель эмбеддингов
- Обрабатывайте документы по частям

## 📊 Мониторинг и логирование

### Структура логов

Все модули используют единую систему логирования:

```python
# Пример лог-сообщений:
INFO - Загрузка embedding-модели на устройство: CUDA
INFO - ✅ Модель успешно загружена  
INFO - ——— Инициализация системы ———
INFO - -> Обработка: document.docx
INFO - ✅ Файл обработан, создано 15 векторов
ERROR - ❌ Критическая ошибка при загрузке модели: [детали]
WARNING - ⚠️ Ошибка при удалении старых записей: [детали]
```

### Настройка уровня логирования

```bash
# Установка через переменную окружения:
export LOG_LEVEL=DEBUG  # DEBUG, INFO, WARNING, ERROR

# Или в коде:
import config
logger = config.setup_logging(__name__)
```

### Мониторинг API

**Health check эндпоинт:**
```bash
curl http://localhost:8001/health
```

**Мониторинг через Docker:**
```bash
# Просмотр статуса контейнеров
docker-compose ps

# Мониторинг ресурсов
docker stats
```

## 🔐 Безопасность

### Рекомендации по безопасности

1. **Сетевая безопасность:**
   - Ограничьте доступ к Qdrant только для доверенных сетей
   - Используйте VPN или firewall для защиты портов
   - Настройте CORS только для необходимых доменов

2. **Аутентификация:**
   - Добавьте API ключи для доступа к эндпоинтам
   - Используйте HTTPS в production окружении
   - Регулярно ротируйте ключи доступа

3. **Данные:**
   - Не индексируйте документы с чувствительной информацией
   - Регулярно создавайте бэкапы Qdrant
   - Очищайте временные файлы

### Пример добавления аутентификации

```python
# В embedding_service.py
from fastapi import HTTPException, Depends, status
from fastapi.security import HTTPBearer

security = HTTPBearer()

async def verify_token(token: str = Depends(security)):
    if token.credentials != "your-secret-token":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token"
        )
    return token

@app.post("/create_embedding", dependencies=[Depends(verify_token)])
async def create_embedding(request: TextRequest):
    # Код обработки
    pass
```

## 🚀 Развертывание в production

### Рекомендации для production

1. **Использование внешней БД:**
   - Развертывание Qdrant на отдельном сервере
   - Настройка кластера Qdrant для высокой доступности
   - Регулярные бэкапы данных

2. **Масштабирование:**
   - Использование load balancer для API
   - Горизонтальное масштабирование сервисов
   - Мониторинг производительности

3. **CI/CD pipeline:**
   - Автоматическое тестирование при изменениях
   - Автоматическое развертывание через Docker
   - Версионирование модели и данных

### Пример production docker-compose.yml

```yaml
version: '3.8'

services:
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - rag-embedding-service
    networks:
      - rag_network

  rag-embedding-service:
    build: .
    deploy:
      replicas: 3
    environment:
      - QDRANT_HOST=qdrant-cluster
      - LOG_LEVEL=WARNING
    networks:
      - rag_network
    restart: unless-stopped

  qdrant:
    image: qdrant/qdrant:latest
    volumes:
      - /data/qdrant:/qdrant/storage
    environment:
      - QDRANT__CLUSTER__ENABLED=true
    networks:
      - rag_network
    restart: unless-stopped
```

## 📈 Оптимизация производительности

### Рекомендации по оптимизации

1. **Модель эмбеддингов:**
   - Используйте GPU для ускорения обработки
   - Рассмотрите более легкие модели для простых задач
   - Кэшируйте эмбеддинги для часто используемых текстов

2. **Индексация:**
   - Обрабатывайте документы в несколько потоков
   - Используйте инкрементальную индексацию
   - Оптимизируйте размер чанков под ваши данные

3. **Qdrant:**
   - Настройте индексы для улучшения поиска
   - Используйте SSD для хранения данных
   - Мониторьте использование памяти

## 📚 Дополнительные ресурсы

- [Документация Qdrant](https://qdrant.tech/documentation/)
- [Sentence Transformers](https://www.sbert.net/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [Langchain Text Splitters](https://python.langchain.com/docs/modules/data_connection/document_transformers/)

---

## ✨ Заключение

RAG-клиент представляет собой полнофункциональную систему для индексации и поиска в корпоративных документах. Система спроектирована с учетом масштабируемости, производительности и простоты использования.

Основные преимущества:
- **Умная индексация** - обработка только измененных файлов
- **Многоформатная поддержка** - .docx, .pdf, .doc файлы  
- **Веб-API** - готовый REST интерфейс для интеграции
- **Гибкая конфигурация** - настройка через переменные окружения
- **Production-ready** - готовность к промышленному использованию

Для вопросов и поддержки обращайтесь к документации компонентов или создавайте issue в репозитории проекта.