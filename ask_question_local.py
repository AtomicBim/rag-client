import sys
import requests
import config
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

class LocalRAGClient:
    """
    Класс клиента для системы RAG.
    Принимает предварительно инициализированные модели для работы.
    """
    # ИСПРАВЛЕНИЕ: Конструктор класса в Python должен называться init
    def __init__(self, embedding_model: SentenceTransformer, qdrant_client: QdrantClient):
        """
        Инициализирует клиент с моделью для эмбеддингов и клиентом Qdrant.
        """
        self.embedding_model = embedding_model
        self.qdrant_client = qdrant_client
        print("✅ Локальный клиент успешно создан и готов к работе.\n")

    def ask(self, question: str) -> tuple[str, list]:
        """
        Основной метод для обработки вопроса.
        """
        # Векторизация вопроса
        question_embedding = self.embedding_model.encode(question).tolist()
        
        # Поиск контекста в Qdrant
        search_results = self.qdrant_client.search(
            collection_name=config.COLLECTION_NAME,
            query_vector=question_embedding,
            limit=SEARCH_LIMIT,
            with_payload=True
        )

        if not search_results:
            return "Релевантный контекст не найден.", []

        # Формирование контекста и источников
        context = "\n---\n".join([result.payload['text'] for result in search_results])
        sources = sorted(list(set([result.payload['source_file'] for result in search_results])))

        # Отправка запроса на сервер (ВМ)
        print("...Отправка запроса на сервер OpenAI на ВМ...")
        try:
            payload = {"question": question, "context": context}
            response = requests.post(config.OPENAI_API_ENDPOINT, json=payload, timeout=120)
            response.raise_for_status()  # Проверка на HTTP ошибки (4xx, 5xx)
            
            answer = response.json().get("answer", "Сервер вернул пустой ответ.")
            return answer, sources
            
        except requests.exceptions.RequestException as e:
            return f"Сетевая ошибка при обращении к серверу на ВМ: {e}", []
        except Exception as e:
            return f"Произошла непредвиденная ошибка: {e}", []

# --- Основной блок выполнения ---
if __name__ == "__main__":
    try:
        # Загружаем все необходимое
        print("Загрузка модели эмбеддингов (SentenceTransformer)...")
        s_transformer = SentenceTransformer(config.LOCAL_EMBEDDING_MODEL, device='cpu')
        
        print(f"Подключение к Qdrant по адресу {config.QDRANT_HOST}...")
        q_client = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)

        # Создаем клиент, передавая ему два аргумента, определенных в init
        local_client = LocalRAGClient(embedding_model=s_transformer, qdrant_client=q_client)
        
        print("Введите 'exit' или 'quit' для завершения.")
        print("-" * 50)
        
        # Запускаем цикл диалога
        while True:
            user_question = input("Ваш вопрос: ")
            if user_question.lower() in ['exit', 'quit']:
                break
            
            answer, sources = local_client.ask(user_question)
            
            print("\n✅ ОТВЕТ:")
            print(answer)
            if sources:
                print(f"\nИсточники: {', '.join(sources)}")
            print("-" * 50)

    except Exception as e:
        print(f"❌ КРИТИЧЕСКАЯ ОШИБКА ПРИ ЗАПУСКЕ: {e}")
        sys.exit(1)