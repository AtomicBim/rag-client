"""
Сервис гибридного поиска (Dense + Sparse) для RAG системы.
Реализует best practices для 2025 года.
"""
import asyncio
import numpy as np
import torch
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import optimized_config as config

# Импорты для разных компонентов
from sentence_transformers import SentenceTransformer, CrossEncoder
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

logger = config.setup_logging(__name__)

@dataclass
class SearchResult:
    """Результат поиска с метриками."""
    text: str
    score: float
    source_file: str
    chunk_index: int
    retrieval_method: str  # "dense", "sparse", "hybrid"
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class HybridSearchResult:
    """Результат гибридного поиска."""
    results: List[SearchResult]
    total_time_ms: float
    dense_time_ms: float
    sparse_time_ms: float
    fusion_time_ms: float
    rerank_time_ms: float = 0.0

class TextPreprocessor:
    """Предобработка текста для sparse поиска."""
    
    def __init__(self, language: str = "russian"):
        self.language = language
        self._setup_nltk()
        self.stopwords_set = set(stopwords.words('russian') + stopwords.words('english'))
        
        # Паттерны для очистки
        self.clean_patterns = [
            (re.compile(r'[^\w\s]'), ' '),  # Убираем пунктуацию
            (re.compile(r'\s+'), ' '),      # Множественные пробелы
            (re.compile(r'\d+'), ' '),      # Числа (опционально)
        ]
    
    def _setup_nltk(self):
        """Загрузка NLTK ресурсов."""
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
        except LookupError:
            logger.info("Загружаем NLTK ресурсы...")
            nltk.download('punkt')
            nltk.download('stopwords')
    
    def preprocess(self, text: str, remove_stopwords: bool = True) -> str:
        """Предобработка текста для BM25."""
        if not text:
            return ""
            
        # Приведение к нижнему регистру
        text = text.lower()
        
        # Применение паттернов очистки
        for pattern, replacement in self.clean_patterns:
            text = pattern.sub(replacement, text)
        
        # Токенизация
        tokens = word_tokenize(text, language=self.language)
        
        # Удаление стоп-слов
        if remove_stopwords:
            tokens = [token for token in tokens if token not in self.stopwords_set]
        
        return ' '.join(tokens)
    
    def tokenize(self, text: str) -> List[str]:
        """Токенизация для BM25."""
        processed = self.preprocess(text, remove_stopwords=config.BM25_CONFIG["remove_stopwords"])
        return processed.split()

class BM25Retriever:
    """Sparse retriever на основе BM25."""
    
    def __init__(self):
        self.preprocessor = TextPreprocessor(config.BM25_CONFIG["language"])
        self.bm25: Optional[BM25Okapi] = None
        self.documents: List[str] = []
        self.document_metadata: List[Dict[str, Any]] = []
        
    def fit(self, documents: List[str], metadata: List[Dict[str, Any]]):
        """Обучение BM25 на корпусе документов."""
        logger.info(f"Создание BM25 индекса для {len(documents)} документов...")
        
        self.documents = documents
        self.document_metadata = metadata
        
        # Токенизация документов
        tokenized_docs = []
        for doc in documents:
            tokens = self.preprocessor.tokenize(doc)
            tokenized_docs.append(tokens)
        
        # Создание BM25 индекса
        self.bm25 = BM25Okapi(
            tokenized_docs,
            k1=config.BM25_CONFIG["k1"],
            b=config.BM25_CONFIG["b"],
            epsilon=config.BM25_CONFIG["epsilon"]
        )
        
        logger.info("✅ BM25 индекс создан")
    
    def search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """Поиск с использованием BM25."""
        if self.bm25 is None:
            logger.warning("BM25 не инициализирован")
            return []
        
        # Токенизация запроса
        query_tokens = self.preprocessor.tokenize(query)
        
        # Получение оценок BM25
        scores = self.bm25.get_scores(query_tokens)
        
        # Сортировка по убыванию оценки
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Только релевантные результаты
                metadata = self.document_metadata[idx]
                results.append(SearchResult(
                    text=self.documents[idx],
                    score=float(scores[idx]),
                    source_file=metadata.get('source_file', 'unknown'),
                    chunk_index=metadata.get('chunk_index', 0),
                    retrieval_method="sparse",
                    metadata=metadata
                ))
        
        return results

class DenseRetriever:
    """Dense retriever на основе векторных embeddings."""
    
    def __init__(self):
        self.qdrant_client: Optional[QdrantClient] = None
        
    def initialize(self) -> bool:
        """Инициализация подключения к Qdrant."""
        try:
            self.qdrant_client = QdrantClient(
                host=config.QDRANT_HOST, 
                port=config.QDRANT_PORT
            )
            
            # Проверка существования коллекции
            collections = self.qdrant_client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if config.COLLECTION_NAME not in collection_names:
                logger.warning(f"Коллекция {config.COLLECTION_NAME} не найдена")
                return False
                
            logger.info("✅ Dense retriever инициализирован")
            return True
            
        except Exception as e:
            logger.error(f"❌ Ошибка инициализации dense retriever: {e}")
            return False
    
    def search(self, query_vector: List[float], top_k: int = 10) -> List[SearchResult]:
        """Поиск с использованием векторного сходства."""
        if self.qdrant_client is None:
            logger.warning("Qdrant клиент не инициализирован")
            return []
        
        try:
            # Поиск в Qdrant
            search_results = self.qdrant_client.search(
                collection_name=config.COLLECTION_NAME,
                query_vector=query_vector,
                limit=top_k,
                with_payload=True,
                score_threshold=config.SEARCH_CONFIG["threshold"]
            )
            
            results = []
            for result in search_results:
                payload = result.payload
                results.append(SearchResult(
                    text=payload.get('text', ''),
                    score=float(result.score),
                    source_file=payload.get('source_file', 'unknown'),
                    chunk_index=payload.get('chunk_index', 0),
                    retrieval_method="dense",
                    metadata=payload
                ))
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Ошибка dense поиска: {e}")
            return []

class RankFusion:
    """Fusion методы для комбинирования результатов."""
    
    @staticmethod
    def reciprocal_rank_fusion(
        dense_results: List[SearchResult],
        sparse_results: List[SearchResult],
        k: int = 60
    ) -> List[SearchResult]:
        """Reciprocal Rank Fusion (RRF)."""
        
        # Создание словаря для хранения скоров
        fused_scores = defaultdict(float)
        result_map = {}
        
        # Обработка dense результатов
        for rank, result in enumerate(dense_results):
            key = f"{result.source_file}_{result.chunk_index}"
            rrf_score = 1.0 / (k + rank + 1)
            fused_scores[key] += config.HYBRID_SEARCH_CONFIG["dense_weight"] * rrf_score
            result_map[key] = result
        
        # Обработка sparse результатов
        for rank, result in enumerate(sparse_results):
            key = f"{result.source_file}_{result.chunk_index}"
            rrf_score = 1.0 / (k + rank + 1)
            fused_scores[key] += config.HYBRID_SEARCH_CONFIG["sparse_weight"] * rrf_score
            
            if key not in result_map:
                result_map[key] = result
        
        # Сортировка по финальному скору
        sorted_items = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Создание финальных результатов
        fused_results = []
        for key, score in sorted_items:
            result = result_map[key]
            fused_result = SearchResult(
                text=result.text,
                score=float(score),
                source_file=result.source_file,
                chunk_index=result.chunk_index,
                retrieval_method="hybrid",
                metadata=result.metadata
            )
            fused_results.append(fused_result)
        
        return fused_results
    
    @staticmethod
    def weighted_score_fusion(
        dense_results: List[SearchResult],
        sparse_results: List[SearchResult]
    ) -> List[SearchResult]:
        """Weighted score fusion."""
        
        result_scores = defaultdict(float)
        result_map = {}
        
        # Нормализация dense скоров
        if dense_results:
            max_dense_score = max(r.score for r in dense_results)
            for result in dense_results:
                key = f"{result.source_file}_{result.chunk_index}"
                normalized_score = result.score / max_dense_score if max_dense_score > 0 else 0
                result_scores[key] += config.HYBRID_SEARCH_CONFIG["dense_weight"] * normalized_score
                result_map[key] = result
        
        # Нормализация sparse скоров  
        if sparse_results:
            max_sparse_score = max(r.score for r in sparse_results)
            for result in sparse_results:
                key = f"{result.source_file}_{result.chunk_index}"
                normalized_score = result.score / max_sparse_score if max_sparse_score > 0 else 0
                result_scores[key] += config.HYBRID_SEARCH_CONFIG["sparse_weight"] * normalized_score
                
                if key not in result_map:
                    result_map[key] = result
        
        # Сортировка и создание результатов
        sorted_items = sorted(result_scores.items(), key=lambda x: x[1], reverse=True)
        
        fused_results = []
        for key, score in sorted_items:
            result = result_map[key]
            fused_result = SearchResult(
                text=result.text,
                score=float(score),
                source_file=result.source_file,
                chunk_index=result.chunk_index,
                retrieval_method="hybrid",
                metadata=result.metadata
            )
            fused_results.append(fused_result)
        
        return fused_results

class HybridRetriever:
    """Главный класс для гибридного поиска."""
    
    def __init__(self):
        self.embedding_model: Optional[SentenceTransformer] = None
        self.rerank_model: Optional[CrossEncoder] = None
        self.dense_retriever = DenseRetriever()
        self.bm25_retriever = BM25Retriever()
        self.rank_fusion = RankFusion()
        
    def initialize_models(self) -> bool:
        """Инициализация всех моделей."""
        logger.info("Инициализация моделей гибридного поиска...")
        
        try:
            # Инициализация embedding модели
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.embedding_model = SentenceTransformer(
                config.EMBEDDING_MODEL_PATH, 
                device=device
            )
            logger.info(f"✅ Embedding модель загружена на {device}")
            
            # Инициализация reranker (опционально)
            if config.SEARCH_CONFIG["enable_reranking"]:
                self.rerank_model = CrossEncoder(
                    config.SEARCH_CONFIG["rerank_model"],
                    device=device
                )
                logger.info("✅ Reranker модель загружена")
            
            # Инициализация dense retriever
            if not self.dense_retriever.initialize():
                logger.warning("Dense retriever не инициализирован")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Ошибка инициализации моделей: {e}")
            return False
    
    def prepare_bm25_corpus(self) -> bool:
        """Подготовка BM25 корпуса из Qdrant."""
        logger.info("Подготовка BM25 корпуса...")
        
        if self.dense_retriever.qdrant_client is None:
            logger.error("Qdrant клиент не инициализирован")
            return False
        
        try:
            # Получение всех документов из Qdrant
            scroll_result = self.dense_retriever.qdrant_client.scroll(
                collection_name=config.COLLECTION_NAME,
                limit=10000,  # Увеличиваем для больших коллекций
                with_payload=True
            )
            
            documents = []
            metadata = []
            
            for point in scroll_result[0]:
                payload = point.payload
                documents.append(payload.get('text', ''))
                metadata.append({
                    'source_file': payload.get('source_file', 'unknown'),
                    'chunk_index': payload.get('chunk_index', 0),
                    'category': payload.get('category', 'unknown')
                })
            
            # Обучение BM25
            self.bm25_retriever.fit(documents, metadata)
            logger.info(f"✅ BM25 корпус подготовлен ({len(documents)} документов)")
            return True
            
        except Exception as e:
            logger.error(f"❌ Ошибка подготовки BM25 корпуса: {e}")
            return False
    
    def rerank_results(self, query: str, results: List[SearchResult]) -> List[SearchResult]:
        """Переранжирование результатов с помощью cross-encoder."""
        if not self.rerank_model or len(results) <= 1:
            return results
        
        try:
            # Подготовка пар (query, document)
            pairs = [(query, result.text) for result in results]
            
            # Получение скоров reranking
            rerank_scores = self.rerank_model.predict(pairs)
            
            # Обновление скоров и сортировка
            for i, score in enumerate(rerank_scores):
                results[i].score = float(score)
            
            return sorted(results, key=lambda x: x.score, reverse=True)
            
        except Exception as e:
            logger.error(f"❌ Ошибка reranking: {e}")
            return results
    
    async def hybrid_search(self, query: str, top_k: int = 5) -> HybridSearchResult:
        """Основной метод гибридного поиска."""
        import time
        
        start_time = time.time()
        
        # Создание embedding для запроса
        if self.embedding_model is None:
            raise RuntimeError("Embedding модель не инициализирована")
        
        query_vector = self.embedding_model.encode(query).tolist()
        
        # Параллельный поиск
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=2) as executor:
            # Dense поиск
            dense_start = time.time()
            dense_future = loop.run_in_executor(
                executor,
                self.dense_retriever.search,
                query_vector,
                config.SEARCH_CONFIG["limit"]
            )
            
            # Sparse поиск
            sparse_start = time.time()
            sparse_future = loop.run_in_executor(
                executor,
                self.bm25_retriever.search,
                query,
                config.SEARCH_CONFIG["limit"]
            )
            
            # Ожидание результатов
            dense_results = await dense_future
            dense_time = (time.time() - dense_start) * 1000
            
            sparse_results = await sparse_future
            sparse_time = (time.time() - sparse_start) * 1000
        
        # Fusion результатов
        fusion_start = time.time()
        fusion_method = config.HYBRID_SEARCH_CONFIG["fusion_method"]
        
        if fusion_method == "rrf":
            fused_results = self.rank_fusion.reciprocal_rank_fusion(
                dense_results, sparse_results, config.HYBRID_SEARCH_CONFIG["rrf_k"]
            )
        else:
            fused_results = self.rank_fusion.weighted_score_fusion(
                dense_results, sparse_results
            )
        
        fusion_time = (time.time() - fusion_start) * 1000
        
        # Ограничение количества результатов
        fused_results = fused_results[:config.SEARCH_CONFIG["limit"]]
        
        # Reranking
        rerank_time = 0.0
        if config.SEARCH_CONFIG["enable_reranking"]:
            rerank_start = time.time()
            fused_results = self.rerank_results(query, fused_results)
            rerank_time = (time.time() - rerank_start) * 1000
        
        # Финальная выборка
        final_results = fused_results[:top_k]
        
        total_time = (time.time() - start_time) * 1000
        
        return HybridSearchResult(
            results=final_results,
            total_time_ms=total_time,
            dense_time_ms=dense_time,
            sparse_time_ms=sparse_time,
            fusion_time_ms=fusion_time,
            rerank_time_ms=rerank_time
        )

# Singleton instance
hybrid_retriever = HybridRetriever()