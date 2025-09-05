import os
import uuid
import config
import json
import torch
import sys
from typing import List, Dict, Optional
from pathlib import Path
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
from unstructured.partition.docx import partition_docx
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models.models import PointStruct
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re
from pypdf import PdfReader

# Файл для хранения состояния индексации
STATE_FILE = "indexing_state.json"
SUPPORTED_EXTENSIONS = {".docx", ".pdf", ".doc"}

logger = config.setup_logging(__name__)

@dataclass
class DocumentProcessingResult:
    """Результат обработки документа."""
    success: bool
    chunks_count: int = 0
    error_message: str = ""

class DocumentPreprocessor:
    """Класс для предобработки и очистки текста документов."""
    
    def __init__(self):
        # Паттерны для очистки текста
        self.patterns = {
            'extra_whitespace': re.compile(r'\s+'),
            'page_numbers': re.compile(r'^\d+\s*$', re.MULTILINE),
            'headers_footers': re.compile(r'^(Стр\.|Страница|Page)\s*\d+.*$', re.MULTILINE),
            'email_pattern': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            'phone_pattern': re.compile(r'\+?[1-9]\d{1,14}'),
            'url_pattern': re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        }
    
    def clean_text(self, text: str) -> str:
        """Очистка и нормализация текста."""
        if not text:
            return ""
            
        # Удаляем лишние пробелы и переносы строк
        text = self.patterns['extra_whitespace'].sub(' ', text)
        
        # Удаляем номера страниц
        text = self.patterns['page_numbers'].sub('', text)
        
        # Удаляем колонтитулы
        text = self.patterns['headers_footers'].sub('', text)
        
        # Нормализуем кавычки
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        
        return text.strip()
    
    def extract_metadata(self, text: str, filename: str) -> dict:
        """Извлечение метаданных для улучшения поиска."""
        return {
            'length': len(text),
            'word_count': len(text.split()),
            'has_emails': bool(self.patterns['email_pattern'].search(text)),
            'has_phones': bool(self.patterns['phone_pattern'].search(text)),
            'has_urls': bool(self.patterns['url_pattern'].search(text)),
            'filename': filename,
            'language': 'ru'  # Можно добавить автоопределение языка
        }
    
class DocumentIndexer:
    """Класс для индексации документов в векторную БД."""
    
    def __init__(self):
        self.qdrant_client: Optional[QdrantClient] = None
        self.embedding_model: Optional[SentenceTransformer] = None
        self.preprocessor = DocumentPreprocessor()
        
        # Оптимизированный text splitter с улучшенными разделителями
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE, 
            chunk_overlap=config.CHUNK_OVERLAP, 
            separators=config.CHUNK_SEPARATORS,
            length_function=len,
            is_separator_regex=False,
            keep_separator=True,  # Сохраняем разделители для контекста
            add_start_index=True  # Добавляем позицию в исходном тексте
        )
    
    def initialize_qdrant(self) -> bool:
        """Инициализация клиента Qdrant."""
        try:
            self.qdrant_client = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)
            
            if not self.qdrant_client.collection_exists(collection_name=config.COLLECTION_NAME):
                logger.info(f"Коллекция '{config.COLLECTION_NAME}' не найдена. Создание новой...")
                self.qdrant_client.create_collection(
                    collection_name=config.COLLECTION_NAME,
                    vectors_config=models.VectorParams(
                        size=config.EMBEDDING_DIMENSION, 
                        distance=models.Distance.COSINE
                    ),
                )
                logger.info("✅ Коллекция успешно создана.")
                
            logger.info(f"✅ Подключение к Qdrant: {config.QDRANT_HOST}:{config.QDRANT_PORT}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Не удалось подключиться к Qdrant: {e}")
            return False
    
    def initialize_embedding_model(self) -> bool:
        """Инициализация модели эмбеддингов."""
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.embedding_model = SentenceTransformer(config.EMBEDDING_MODEL_PATH, device=device)
            logger.info(f"✅ Модель загружена на: {device.upper()}")
            return True
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки модели: {e}")
            logger.error("Возможные причины:")
            logger.error("1. Отсутствует Microsoft Visual C++ Redistributable")
            logger.error("2. Неверный путь к модели")
            logger.error("3. Недостаточно памяти")
            return False
    
    def delete_old_document_records(self, filename: str) -> None:
        """Удаление старых записей документа из Qdrant."""
        if not self.qdrant_client:
            logger.error("❌ Клиент Qdrant не инициализирован")
            return
            
        try:
            self.qdrant_client.delete(
                collection_name=config.COLLECTION_NAME,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[models.FieldCondition(
                            key="source_file", 
                            match=models.MatchValue(value=filename)
                        )]
                    )
                )
            )
            logger.info(f"  - ✅ Старые записи для '{filename}' удалены.")
        except Exception as e:
            logger.warning(f"  - ⚠️ Ошибка при удалении старых записей: {e}")
    
    def create_embeddings_batch(self, chunks: List[str], filename: str, category: str, metadata: dict = None) -> List[PointStruct]:
        """Создание батча векторов для загрузки в Qdrant с улучшенными метаданными."""
        if not self.embedding_model:
            raise ValueError("Модель эмбеддингов не инициализирована")
            
        # Оптимизированное создание эмбеддингов
        embeddings = self.embedding_model.encode(
            chunks, 
            show_progress_bar=False,
            batch_size=32,
            normalize_embeddings=True  # Нормализация для cosine similarity
        )
        points = []
        
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            # Расширенные метаданные для каждого чанка
            payload = {
                "text": chunk,
                "source_file": filename,
                "category": category,
                "chunk_index": i,
                "chunk_length": len(chunk),
                "chunk_word_count": len(chunk.split())
            }
            
            # Добавляем метаданные документа
            if metadata:
                payload.update({
                    "doc_length": metadata.get("length", 0),
                    "doc_word_count": metadata.get("word_count", 0),
                    "has_emails": metadata.get("has_emails", False),
                    "has_phones": metadata.get("has_phones", False),
                    "language": metadata.get("language", "ru")
                })
            
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding.tolist(),
                payload=payload
            )
            points.append(point)
            
        return points
    
    def upload_points_to_qdrant(self, points: List[PointStruct]) -> bool:
        """Загрузка точек в Qdrant батчами."""
        if not self.qdrant_client:
            logger.error("❌ Клиент Qdrant не инициализирован")
            return False
            
        if not points:
            logger.warning("⚠️ Список точек пуст, нечего загружать")
            return True
            
        try:
            batch_size = config.BATCH_SIZE
            logger.info(f"Начинаем загрузку {len(points)} точек батчами по {batch_size}")
            
            for i in range(0, len(points), batch_size):
                batch_points = points[i:i+batch_size]
                logger.info(f"  - Загрузка батча ({len(batch_points)} векторов)...")
                
                # Проверяем структуру точек в батче
                for point in batch_points:
                    if not isinstance(point, PointStruct):
                        logger.error(f"❌ Некорректный тип точки: {type(point)}")
                        return False
                
                self.qdrant_client.upsert(
                    collection_name=config.COLLECTION_NAME, 
                    points=batch_points,
                    wait=True
                )
                
            logger.info(f"✅ Успешно загружено {len(points)} точек в Qdrant")
            return True
            
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки векторов в Qdrant: {e}")
            logger.error(f"Тип ошибки: {type(e).__name__}")
            return False
    
    def process_document(self, file_path: str) -> DocumentProcessingResult:
        """Обработка одного документа."""
        filename = os.path.basename(file_path)
        extension = Path(file_path).suffix.lower()
        
        # Удаление старых записей
        self.delete_old_document_records(filename)
        
        # Извлечение текста
        try:
            if extension == ".doc":
                # Конвертируем .doc в .docx и удаляем оригинал
                converted_path = convert_doc_to_docx(file_path)
                if not converted_path:
                    return DocumentProcessingResult(
                        success=False, 
                        error_message="Не удалось конвертировать .doc файл"
                    )
                document_text = extract_text_from_docx(converted_path)
                # Обновляем filename для корректного отображения в логах
                filename = os.path.basename(converted_path)
            elif extension == ".docx":
                document_text = extract_text_from_docx(file_path)
            elif extension == ".pdf":
                document_text = extract_text_from_pdf(file_path)
            else:
                return DocumentProcessingResult(
                    success=False, 
                    error_message=f"Неподдерживаемый формат файла: {extension}"
                )
                
            if not document_text:
                return DocumentProcessingResult(
                    success=False, 
                    error_message="Не удалось извлечь текст из документа"
                )
            
            # Предобработка текста
            cleaned_text = self.preprocessor.clean_text(document_text)
            if not cleaned_text:
                return DocumentProcessingResult(
                    success=False, 
                    error_message="Текст пуст после предобработки"
                )
            
            # Разбиение на чанки с улучшенным алгоритмом
            chunks = self.text_splitter.split_text(cleaned_text)
            if not chunks:
                return DocumentProcessingResult(
                    success=False, 
                    error_message="Не удалось разбить текст на чанки"
                )
            
            # Создание эмбеддингов с дополнительными метаданными
            category = os.path.basename(os.path.dirname(file_path))
            metadata = self.preprocessor.extract_metadata(cleaned_text, filename)
            points = self.create_embeddings_batch(chunks, filename, category, metadata)
            
            # Загрузка в Qdrant
            if not self.upload_points_to_qdrant(points):
                return DocumentProcessingResult(
                    success=False, 
                    error_message="Ошибка загрузки в Qdrant"
                )
            
            return DocumentProcessingResult(
                success=True, 
                chunks_count=len(chunks)
            )
            
        except Exception as e:
            return DocumentProcessingResult(
                success=False, 
                error_message=str(e)
            )

def load_state() -> Dict[str, float]:
    """Загружает состояние индексации из файла."""
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r', encoding='utf-8') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                logger.warning("Ошибка чтения файла состояния. Начинаем с пустого состояния.")
                return {}
    return {}

def save_state(state: Dict[str, float]) -> None:
    """Сохраняет состояние индексации в файл."""
    try:
        with open(STATE_FILE, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=4, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Ошибка сохранения состояния: {e}")

def convert_doc_to_docx(file_path: str) -> Optional[str]:
    """Конвертирует .doc в .docx и удаляет оригинальный .doc файл."""
    if sys.platform != "win32":
        logger.warning(f"  - ⚠️ Конверсия .doc доступна только на Windows: {os.path.basename(file_path)}")
        return None
    
    try:
        import win32com.client as win32
    except ImportError:
        logger.error("  - ❌ win32com.client не доступен. Установите pywin32 для конвертации .doc файлов.")
        return None
    
    word = None
    try:
        abs_path_doc = os.path.abspath(file_path)
        abs_path_docx = os.path.splitext(abs_path_doc)[0] + ".docx"
        
        # Если .docx уже существует, удаляем .doc и возвращаем путь к .docx
        if os.path.exists(abs_path_docx):
            logger.info(f"  - ✅ .docx уже существует, удаляем .doc: {os.path.basename(file_path)}")
            os.remove(abs_path_doc)
            return abs_path_docx
            
        logger.info(f"  - 🔄 Конвертация .doc → .docx: {os.path.basename(file_path)}")
        
        word = win32.Dispatch("Word.Application")
        word.visible = False
        doc = word.Documents.Open(abs_path_doc)
        doc.SaveAs(abs_path_docx, FileFormat=12)
        doc.Close()
        
        # Удаляем оригинальный .doc файл после успешной конвертации
        os.remove(abs_path_doc)
        logger.info(f"  - ✅ Конвертация завершена, .doc файл удален: {os.path.basename(file_path)}")
        
        return abs_path_docx
        
    except Exception as e:
        logger.error(f"  - ❌ Ошибка конвертации {os.path.basename(file_path)}: {e}")
        return None
    finally:
        if word:
            try:
                word.Quit()
            except:
                pass  # Игнорируем ошибки при закрытии Word

def extract_text_from_docx(file_path: str) -> Optional[str]:
    """Извлекает элементы из .docx файла с помощью unstructured."""
    try:
        elements = partition_docx(filename=file_path, infer_table_structure=True)
        text_content = "\n\n".join([str(el) for el in elements])
        return text_content.strip() if text_content else None
    except Exception as e:
        logger.error(f"  - ❌ Ошибка извлечения текста из {os.path.basename(file_path)}: {e}")
        return None


def extract_text_from_pdf(file_path: str) -> Optional[str]:
    """Извлекает текст из .pdf файла."""
    try:
        reader = PdfReader(file_path)
        text_parts = []
        
        for page_num, page in enumerate(reader.pages, 1):
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text.strip())
                
        return "\n\n".join(text_parts) if text_parts else None
        
    except Exception as e:
        logger.error(f"  - ❌ Ошибка извлечения из PDF {os.path.basename(file_path)}: {e}")
        return None

def find_changed_files(docs_path: str, state: Dict[str, float]) -> List[str]:
    """Находит файлы, которые были созданы или изменены."""
    files_to_process = []
    
    for root, _, files in os.walk(docs_path):
        for filename in files:
            # Пропускаем временные файлы
            if filename.startswith('~'):
                continue
            
            file_path = os.path.join(root, filename)
            extension = Path(file_path).suffix.lower()
            
            # Проверяем поддерживаемые форматы
            if extension not in SUPPORTED_EXTENSIONS:
                continue
                
            file_mod_time = os.path.getmtime(file_path)
            
            # Если файл новый или изменился
            if state.get(file_path) != file_mod_time:
                files_to_process.append(file_path)
                logger.info(f"  - 🔄 В очереди на обработку: {os.path.basename(filename)} (изменен)")
    
    return files_to_process

def main() -> None:
    """Основная функция для умной индексации документов."""
    indexer = DocumentIndexer()
    
    logger.info("——— Инициализация системы ———")
    
    # Инициализация Qdrant
    if not indexer.initialize_qdrant():
        return
    
    # Проверка папки с документами
    absolute_docs_path = os.path.abspath(config.DOCS_ROOT_PATH)
    if not os.path.isdir(absolute_docs_path):
        logger.error(f"❌ Папка не найдена по пути '{absolute_docs_path}'.")
        return
    
    # Определение измененных файлов
    logger.info("——— Проверка состояния документов ———")
    state = load_state()
    files_to_process = find_changed_files(absolute_docs_path, state)
    
    if not files_to_process:
        logger.info("✅ Все документы актуальны. Обновление не требуется.")
        return
    
    logger.info(f"——— Обнаружено {len(files_to_process)} новых/измененных файлов для обработки ———")
    
    # Инициализация модели эмбеддингов
    if not indexer.initialize_embedding_model():
        return
    
    # Обработка файлов
    all_points = []
    new_state = state.copy()
    successful_files = 0
    
    for file_path in files_to_process:
        filename = os.path.basename(file_path)
        logger.info(f"-> Обработка: {filename}")
        
        result = indexer.process_document(file_path)
        
        if result.success:
            # Для .doc файлов, которые были конвертированы, нужно обновить состояние для .docx
            extension = Path(file_path).suffix.lower()
            if extension == ".doc":
                # .doc файл был удален, состояние обновляем для .docx
                docx_path = os.path.splitext(file_path)[0] + ".docx"
                if os.path.exists(docx_path):
                    new_state[docx_path] = os.path.getmtime(docx_path)
                # Удаляем запись о .doc файле из состояния
                new_state.pop(file_path, None)
            else:
                # Обновляем состояние для обычных файлов
                new_state[file_path] = os.path.getmtime(file_path)
            
            successful_files += 1
            logger.info(f"  - ✅ Файл обработан, создано {result.chunks_count} векторов.")
        else:
            logger.error(f"  - ❌ Ошибка обработки {filename}: {result.error_message}")
    
    if successful_files > 0:
        # Сохраняем состояние только для успешно обработанных файлов
        save_state(new_state)
        logger.info(f"✅ Файл состояния '{STATE_FILE}' обновлен.")
        logger.info(f"✅ Локальная индексация завершена: {successful_files}/{len(files_to_process)} файлов обработано успешно.")
    else:
        logger.warning("⚠️ Ни одного файла не было обработано успешно.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("⚠️ Программа прервана пользователем.")
    except Exception as e:
        logger.error(f"❌ Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()