import os
import uuid
import torch
import traceback
from unstructured.partition.docx import partition_docx

from qdrant_client import QdrantClient, models
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# --- КОНФИГУРАЦИЯ ---
QDRANT_HOST = "192.168.42.188"
DOCS_ROOT_PATH = "./rag-source"
COLLECTION_NAME = "internal_regulations_v2"
LOCAL_EMBEDDING_MODEL = "C:/Users/r.grigoriev/Desktop/rag-client/local_model/all-mpnet-base-v2"
EMBEDDING_DIMENSION = 768

def extract_text_from_docx(file_path):
    """Извлекает элементы из .docx файла с помощью unstructured."""
    try:
        elements = partition_docx(filename=file_path)
        # Соединяем текстовое представление всех элементов в один документ
        return "\n\n".join([str(el) for el in elements])
    except Exception as e:
        print(f"  - ❌ Ошибка извлечения текста из {os.path.basename(file_path)}: {e}")
        return None

def extract_text_from_doc(file_path):
    """
    Заглушка для обработки .doc файлов.
    Для полноценной работы рекомендуется использовать утилиты вроде 'antiword' или 'libreoffice'.
    """
    print(f"  - ⚠️ Пропуск старого формата .doc: {os.path.basename(file_path)}")
    return None

def get_text_chunks(text):
    """Разбивает текст на чанки."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_text(text)

def main():
    """Основная функция для индексации документов."""
    # --- Шаг 1: Инициализация ---
    print("--- Инициализация клиентов ---")
    try:
        qdrant_client = QdrantClient(host=QDRANT_HOST, port=6333)
        qdrant_client.get_collections()
        print(f"✅ Подключение к Qdrant: {QDRANT_HOST}:6333")
    except Exception as e:
        print(f"❌ ОШИБКА: Не удалось подключиться к Qdrant. {e}")
        return

    print(f"Загрузка локальной embedding-модели...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedding_model = SentenceTransformer(LOCAL_EMBEDDING_MODEL, device=device)
    print(f"✅ Модель загружена на: {device.upper()}")

    # --- Шаг 2: Подготовка коллекции в Qdrant ---
    try:
        print(f"\n--- Пересоздание коллекции '{COLLECTION_NAME}' ---")
        qdrant_client.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(size=EMBEDDING_DIMENSION, distance=models.Distance.COSINE),
        )
        print("✅ Коллекция успешно пересоздана.")
    except Exception as e:
        print(f"❌ ОШИБКА: Не удалось создать коллекцию. {e}")
        return

    # --- Шаг 3: Рекурсивный поиск и обработка документов ---
    print("\n--- Индексация документов ---")
    absolute_docs_path = os.path.abspath(DOCS_ROOT_PATH)
    if not os.path.isdir(absolute_docs_path):
        print(f"❌ ОШИБКА: Папка не найдена по пути '{absolute_docs_path}'.")
        return

    all_points = []
    processed_files_count = 0

    # Рекурсивный обход всех папок и файлов
    for root, dirs, files in os.walk(absolute_docs_path):
        for filename in files:
            if not filename.lower().endswith((".doc", ".docx")):
                continue

            file_path = os.path.join(root, filename)
            relative_path = os.path.relpath(file_path, absolute_docs_path)
            print(f"-> Обработка: {relative_path}")

            document_text = None
            if filename.lower().endswith(".docx"):
                document_text = extract_text_from_docx(file_path)
            elif filename.lower().endswith(".doc"):
                document_text = extract_text_from_doc(file_path)

            if not document_text:
                continue

            chunks = get_text_chunks(document_text)
            if not chunks:
                print("  - Не удалось разбить текст на чанки.")
                continue

            embeddings = embedding_model.encode(chunks, show_progress_bar=False)
            
            # Получаем имя папки, в которой лежит файл, для категории
            category = os.path.basename(os.path.dirname(file_path))

            for chunk, embedding in zip(chunks, embeddings):
                point = models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding.tolist(),
                    payload={"text": chunk, "source_file": filename, "category": category}
                )
                all_points.append(point)
            
            processed_files_count += 1

    if not all_points:
        print("\n⚠️ Документы для индексации не найдены. Проверьте исходную папку.")
        return

    # --- Шаг 4: Пакетная загрузка в Qdrant ---
    print(f"\n--- Загрузка данных в Qdrant ---")
    print(f"Подготовлено {len(all_points)} векторов из {processed_files_count} файлов.")
    try:
        qdrant_client.upsert(
            collection_name=COLLECTION_NAME,
            points=all_points,
            wait=True
        )
        print("✅ Все векторы успешно загружены!")
    except Exception as e:
        print(f"❌ ОШИБКА во время загрузки векторов в Qdrant: {e}")

    # --- Шаг 5: Завершение ---
    print("\n✅ Локальная индексация полностью завершена!")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\n❌❌❌ КРИТИЧЕСКАЯ ОШИБКА! Выполнение скрипта прервано. ❌❌❌")
        traceback.print_exc()