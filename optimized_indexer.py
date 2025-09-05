"""
Оптимизированный индексатор документов для RAG системы 2025.
Включает adaptive chunking, метаданные и эффективную обработку.
"""
import os
import uuid
import json
import torch
import sys
import hashlib
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
import optimized_config as config

# Импорты для обработки документов
from sentence_transformers import SentenceTransformer
from unstructured.partition.docx import partition_docx
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models.models import PointStruct
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import TextSplitter
import re
from pypdf import PdfReader
import tiktoken

logger = config.setup_logging(__name__)

# Константы
STATE_FILE = "optimized_indexing_state.json"
SUPPORTED_EXTENSIONS = {".docx", ".pdf", ".doc", ".txt", ".md"}

@dataclass
class ChunkMetadata:
    """Расширенные метаданные для чанка."""
    chunk_index: int
    chunk_hash: str
    chunk_length: int
    word_count: int
    sentence_count: int
    language: str
    content_type: str
    semantic_summary: Optional[str] = None
    keywords: Optional[List[str]] = None
    reading_time_seconds: int = 0

@dataclass
class DocumentMetadata:
    """Метаданные документа."""
    filename: str
    file_path: str
    file_size: int
    file_hash: str
    content_type: str
    language: str
    total_chunks: int
    processing_time: float
    last_modified: float
    extraction_method: str
    word_count: int
    page_count: int = 0

@dataclass
class ProcessingResult:
    """Результат обработки документа."""
    success: bool
    document_metadata: Optional[DocumentMetadata] = None
    chunks_processed: int = 0
    error_message: str = ""
    warnings: List[str] = None

class AdaptiveChunker:
    """Адаптивный чанкер с учетом типа контента."""
    
    def __init__(self):
        # Инициализация tokenizer для подсчета токенов
        try:
            self.tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
        except:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        # Специализированные splitter'ы
        self.splitters = self._create_splitters()
        
        # Паттерны для определения типа контента
        self.content_patterns = {
            "code": re.compile(r'(def |class |import |function|var |const |let )', re.IGNORECASE),
            "table": re.compile(r'(\|.*\||\t.*\t|<table>)', re.IGNORECASE),
            "list": re.compile(r'(^\d+\.|^[•\-\*]\s|^[a-z]\)|^\([a-z]\))', re.MULTILINE),
            "heading": re.compile(r'^#+\s|^[A-ZА-Я][^.!?]*$', re.MULTILINE),
            "legal": re.compile(r'(статья|пункт|раздел|глава|§|article|section)', re.IGNORECASE),
            "technical": re.compile(r'(API|HTTP|JSON|XML|SQL|URL|UUID)', re.IGNORECASE)
        }
    
    def _create_splitters(self) -> Dict[str, TextSplitter]:
        """Создание специализированных text splitter'ов."""
        splitters = {}
        
        for strategy_name, strategy_config in config.CHUNKING_STRATEGIES.items():
            # Конвертируем токены в символы (приблизительно)
            chunk_size_chars = strategy_config["chunk_size"] * 4  # ~4 символа на токен
            overlap_chars = strategy_config["chunk_overlap"] * 4
            
            splitters[strategy_name] = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size_chars,
                chunk_overlap=overlap_chars,
                separators=config.CHUNK_SEPARATORS,
                length_function=len,
                is_separator_regex=False,
                keep_separator=True,
                add_start_index=True
            )
        
        return splitters
    
    def detect_content_type(self, text: str) -> str:
        """Определение типа контента для выбора стратегии."""
        if not text:
            return "default"
        
        # Подсчет совпадений для каждого типа
        type_scores = {}
        total_length = len(text)
        
        for content_type, pattern in self.content_patterns.items():
            matches = len(pattern.findall(text))
            # Нормализуем по длине текста
            type_scores[content_type] = matches / (total_length / 1000 + 1)
        
        # Находим тип с максимальным скором
        best_type = max(type_scores.keys(), key=lambda k: type_scores[k])
        
        # Если скор слишком низкий, используем default
        if type_scores[best_type] < 0.1:
            return "default"
        
        # Маппинг на стратегии чанкинга
        strategy_mapping = {
            "legal": "legal",
            "technical": "technical", 
            "code": "technical",
            "table": "technical",
            "list": "faq",
            "heading": "default"
        }
        
        return strategy_mapping.get(best_type, "default")
    
    def count_tokens(self, text: str) -> int:
        """Подсчет токенов в тексте."""
        try:
            return len(self.tokenizer.encode(text))
        except:
            # Fallback: приблизительный подсчет
            return len(text) // 4
    
    def create_chunks(self, text: str, filename: str) -> Tuple[List[str], str]:
        """Создание чанков с адаптивной стратегией."""
        if not text:
            return [], "default"
        
        # Определение типа контента
        content_type = self.detect_content_type(text)
        
        # Выбор соответствующего splitter'а
        splitter = self.splitters.get(content_type, self.splitters["default"])
        
        # Создание чанков
        chunks = splitter.split_text(text)
        
        # Фильтрация слишком коротких чанков
        min_tokens = 10
        filtered_chunks = []
        
        for chunk in chunks:
            token_count = self.count_tokens(chunk)
            if token_count >= min_tokens:
                filtered_chunks.append(chunk)
            else:
                logger.debug(f"Пропущен короткий чанк ({token_count} токенов)")
        
        logger.info(f"Создано {len(filtered_chunks)} чанков (стратегия: {content_type})")
        return filtered_chunks, content_type
    
    def create_chunk_metadata(
        self, 
        chunk: str, 
        chunk_index: int, 
        content_type: str,
        filename: str
    ) -> ChunkMetadata:
        """Создание метаданных для чанка."""
        
        # Базовые метрики
        chunk_hash = hashlib.md5(chunk.encode()).hexdigest()
        word_count = len(chunk.split())
        sentence_count = len([s for s in chunk.split('.') if s.strip()])
        
        # Определение языка (упрощенно)
        russian_chars = len(re.findall(r'[а-яё]', chunk.lower()))
        english_chars = len(re.findall(r'[a-z]', chunk.lower()))
        language = "ru" if russian_chars > english_chars else "en"
        
        # Время чтения (200 слов в минуту)
        reading_time = max(1, int(word_count / 200 * 60))
        
        # Извлечение ключевых слов (простой метод)
        words = chunk.lower().split()
        common_words = {'и', 'в', 'на', 'с', 'по', 'для', 'от', 'до', 'при', 'что', 'как', 
                       'this', 'that', 'with', 'from', 'they', 'have', 'will', 'been'}
        keywords = [word for word in set(words) if len(word) > 3 and word not in common_words][:10]
        
        return ChunkMetadata(
            chunk_index=chunk_index,
            chunk_hash=chunk_hash,
            chunk_length=len(chunk),
            word_count=word_count,
            sentence_count=sentence_count,
            language=language,
            content_type=content_type,
            keywords=keywords,
            reading_time_seconds=reading_time
        )

class DocumentProcessor:
    """Улучшенный процессор документов."""
    
    def __init__(self):
        self.chunker = AdaptiveChunker()
        
        # Паттерны для очистки текста
        self.cleaning_patterns = [
            (re.compile(r'\s+'), ' '),  # Множественные пробелы
            (re.compile(r'\n\s*\n\s*\n'), '\n\n'),  # Множественные переводы строк
            (re.compile(r'^(Стр\.|Страница)\s*\d+', re.MULTILINE), ''),  # Номера страниц
        ]
    
    def clean_text(self, text: str) -> str:
        """Улучшенная очистка текста."""
        if not text:
            return ""
        
        # Применение паттернов очистки
        for pattern, replacement in self.cleaning_patterns:
            text = pattern.sub(replacement, text)
        
        # Нормализация кавычек и символов
        replacements = {
            '"': '"', '"': '"', ''': "'", ''': "'",
            '…': '...', '–': '-', '—': '-',
            '\u00a0': ' ',  # Неразрывный пробел
            '\ufeff': '',   # BOM
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        return text.strip()
    
    def extract_text_from_docx(self, file_path: str) -> Tuple[Optional[str], str]:
        """Извлечение текста из DOCX с улучшенной обработкой."""
        try:
            elements = partition_docx(
                filename=file_path, 
                infer_table_structure=True,
                strategy="auto"
            )
            
            text_parts = []
            for element in elements:
                element_text = str(element).strip()
                if element_text:
                    text_parts.append(element_text)
            
            full_text = "\n\n".join(text_parts)
            return self.clean_text(full_text), "unstructured"
            
        except Exception as e:
            logger.error(f"Ошибка извлечения из DOCX {os.path.basename(file_path)}: {e}")
            return None, "error"
    
    def extract_text_from_pdf(self, file_path: str) -> Tuple[Optional[str], str]:
        """Извлечение текста из PDF с метаданными."""
        try:
            reader = PdfReader(file_path)
            text_parts = []
            
            for page_num, page in enumerate(reader.pages, 1):
                page_text = page.extract_text()
                if page_text.strip():
                    # Добавляем маркер страницы для лучшего чанкинга
                    text_parts.append(f"[Страница {page_num}]\n{page_text.strip()}")
            
            full_text = "\n\n".join(text_parts)
            return self.clean_text(full_text), "pypdf"
            
        except Exception as e:
            logger.error(f"Ошибка извлечения из PDF {os.path.basename(file_path)}: {e}")
            return None, "error"
    
    def extract_text_from_txt(self, file_path: str) -> Tuple[Optional[str], str]:
        """Извлечение текста из TXT файлов."""
        try:
            # Попытка определить кодировку
            encodings = ['utf-8', 'utf-8-sig', 'windows-1251', 'cp1252']
            
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        text = f.read()
                    return self.clean_text(text), f"text_{encoding}"
                except UnicodeDecodeError:
                    continue
            
            logger.error(f"Не удалось определить кодировку для {file_path}")
            return None, "encoding_error"
            
        except Exception as e:
            logger.error(f"Ошибка чтения TXT файла {file_path}: {e}")
            return None, "error"
    
    def calculate_file_hash(self, file_path: str) -> str:
        """Вычисление хеша файла для отслеживания изменений."""
        try:
            hasher = hashlib.md5()
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except:
            return ""
    
    def process_document(self, file_path: str) -> ProcessingResult:
        """Главный метод обработки документа."""
        start_time = datetime.now()
        filename = os.path.basename(file_path)
        extension = Path(file_path).suffix.lower()
        warnings = []
        
        logger.info(f"Обработка: {filename}")
        
        # Проверка поддерживаемых форматов
        if extension not in SUPPORTED_EXTENSIONS:
            return ProcessingResult(
                success=False,
                error_message=f"Неподдерживаемый формат: {extension}"
            )
        
        # Извлечение текста
        extracted_text = None
        extraction_method = ""
        
        try:
            if extension == ".docx":
                extracted_text, extraction_method = self.extract_text_from_docx(file_path)
            elif extension == ".pdf":
                extracted_text, extraction_method = self.extract_text_from_pdf(file_path)
            elif extension in [".txt", ".md"]:
                extracted_text, extraction_method = self.extract_text_from_txt(file_path)
            elif extension == ".doc":
                # TODO: Добавить обработку .doc файлов
                warnings.append("Формат .doc требует дополнительной настройки")
                return ProcessingResult(
                    success=False,
                    error_message="Формат .doc пока не поддерживается в этой версии",
                    warnings=warnings
                )
            
            if not extracted_text:
                return ProcessingResult(
                    success=False,
                    error_message="Не удалось извлечь текст из документа"
                )
            
            # Создание чанков
            chunks, content_type = self.chunker.create_chunks(extracted_text, filename)
            
            if not chunks:
                return ProcessingResult(
                    success=False,
                    error_message="Не удалось создать чанки из текста"
                )
            
            # Создание метаданных документа
            file_stats = os.stat(file_path)
            processing_time = (datetime.now() - start_time).total_seconds()
            
            document_metadata = DocumentMetadata(
                filename=filename,
                file_path=file_path,
                file_size=file_stats.st_size,
                file_hash=self.calculate_file_hash(file_path),
                content_type=content_type,
                language="ru",  # TODO: автоопределение
                total_chunks=len(chunks),
                processing_time=processing_time,
                last_modified=file_stats.st_mtime,
                extraction_method=extraction_method,
                word_count=len(extracted_text.split()),
                page_count=extracted_text.count("[Страница") if "[Страница" in extracted_text else 0
            )
            
            return ProcessingResult(
                success=True,
                document_metadata=document_metadata,
                chunks_processed=len(chunks),
                warnings=warnings if warnings else None
            )
            
        except Exception as e:
            return ProcessingResult(
                success=False,
                error_message=f"Ошибка обработки: {str(e)}",
                warnings=warnings if warnings else None
            )

class OptimizedIndexer:
    """Главный класс оптимизированного индексатора."""
    
    def __init__(self):
        self.qdrant_client: Optional[QdrantClient] = None
        self.embedding_model: Optional[SentenceTransformer] = None
        self.processor = DocumentProcessor()
        
    def initialize(self) -> bool:
        """Инициализация всех компонентов."""
        logger.info("Инициализация оптимизированного индексатора...")
        
        # Инициализация Qdrant
        if not self._initialize_qdrant():
            return False
        
        # Инициализация модели эмбеддингов
        if not self._initialize_embedding_model():
            return False
        
        logger.info("✅ Индексатор готов к работе")
        return True
    
    def _initialize_qdrant(self) -> bool:
        """Инициализация Qdrant с оптимизированными настройками."""
        try:
            self.qdrant_client = QdrantClient(
                host=config.QDRANT_HOST, 
                port=config.QDRANT_PORT
            )
            
            # Проверка/создание коллекции
            if not self.qdrant_client.collection_exists(config.COLLECTION_NAME):
                logger.info(f"Создание новой коллекции: {config.COLLECTION_NAME}")
                
                self.qdrant_client.create_collection(
                    collection_name=config.COLLECTION_NAME,
                    vectors_config=models.VectorParams(
                        size=config.EMBEDDING_DIMENSION,
                        distance=models.Distance.COSINE,
                        hnsw_config=models.HnswConfigDiff(
                            m=config.HNSW_CONFIG["m"],
                            ef_construct=config.HNSW_CONFIG["ef_construct"],
                            full_scan_threshold=config.HNSW_CONFIG["full_scan_threshold"],
                            max_indexing_threads=config.HNSW_CONFIG["max_indexing_threads"]
                        )
                    ),
                    optimizers_config=models.OptimizersConfigDiff(
                        default_segment_number=4,
                        max_segment_size=None,
                        memmap_threshold=None,
                        indexing_threshold=20000,
                        flush_interval_sec=5,
                        max_optimization_threads=2
                    )
                )
                logger.info("✅ Коллекция создана с оптимизированными настройками")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Ошибка инициализации Qdrant: {e}")
            return False
    
    def _initialize_embedding_model(self) -> bool:
        """Инициализация модели эмбеддингов с GPU оптимизациями."""
        try:
            device = config.GPU_CONFIG["device"]
            if device == "cuda" and torch.cuda.is_available():
                # Настройка GPU памяти для RTX 3060 4GB
                torch.cuda.set_per_process_memory_fraction(
                    config.GPU_CONFIG["memory_fraction"]
                )
                torch.cuda.empty_cache()
            
            self.embedding_model = SentenceTransformer(
                config.EMBEDDING_MODEL_PATH,
                device=device
            )
            
            # Включение mixed precision для экономии VRAM
            if config.GPU_CONFIG["enable_mixed_precision"] and device == "cuda":
                self.embedding_model.half()
            
            logger.info(f"✅ Embedding модель загружена на {device}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки модели: {e}")
            
            # Попытка загрузки fallback модели
            try:
                fallback_config = config.get_optimal_model_for_vram(2.0)  # 2GB fallback
                logger.info(f"Попытка загрузки fallback модели: {fallback_config['path']}")
                
                self.embedding_model = SentenceTransformer(
                    fallback_config["path"],
                    device="cpu"
                )
                logger.info("✅ Fallback модель загружена на CPU")
                return True
                
            except Exception as e2:
                logger.error(f"❌ Ошибка загрузки fallback модели: {e2}")
                return False

if __name__ == "__main__":
    # Пример использования
    indexer = OptimizedIndexer()
    if indexer.initialize():
        print("Индексатор готов к работе!")
    else:
        print("Ошибка инициализации индексатора")