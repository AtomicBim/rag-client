import os
import uuid
import torch
import traceback
import config
import win32com.client as win32
import sys

from unstructured.partition.docx import partition_docx
from qdrant_client import QdrantClient, models
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

def extract_text_from_docx(file_path):
    """Извлекает элементы из .docx файла с помощью unstructured."""
    try:
        elements = partition_docx(filename=file_path, infer_table_structure=True)
        return "\n\n".join([str(el) for el in elements])
    except Exception as e:
        print(f"  - ❌ Ошибка извлечения текста из {os.path.basename(file_path)}: {e}")
        return None


def convert_doc_to_docx(file_path):
    """
    Конвертирует .doc в .docx с помощью Microsoft Word.
    Возвращает путь к новому .docx файлу.
    """
    if sys.platform != "win32":
        print("  - ⚠️ Пропуск .doc файла: автоматическая конвертация поддерживается только на Windows.")
        return None
        
    word = None
    try:
        # Получаем абсолютные пути, чтобы Word не запутался
        abs_path_doc = os.path.abspath(file_path)
        abs_path_docx = abs_path_doc + "x"
        
        print(f"  - ⚙️ Конвертация .doc с помощью MS Word: {os.path.basename(file_path)}")
        
        # Запускаем приложение Word в фоне
        word = win32.Dispatch("Word.Application")
        word.visible = False
        
        # Открываем .doc файл
        doc = word.Documents.Open(abs_path_doc)
        
        # Сохраняем в формате .docx (wdFormatXMLDocument = 12)
        doc.SaveAs(abs_path_docx, FileFormat=12)
        doc.Close()
        
        return abs_path_docx
        
    except Exception as e:
        print(f"  - ❌ Ошибка конвертации файла {os.path.basename(file_path)} через Word: {e}")
        return None
    finally:
        # Убеждаемся, что приложение Word закрыто, даже если была ошибка
        if word:
            word.Quit()


def get_text_chunks(text):
    """Разбивает текст на чанки."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200,
        # Используем разделители, которые часто встречаются в документах
        separators=["\n\n", "\n", ".", " ", ""]
    )
    return text_splitter.split_text(text)


def main():
    """Основная функция для индексации документов."""
    # ... (код инициализации клиентов остается без изменений) ...
    print("--- Инициализация клиентов ---")
    try:
        qdrant_client = QdrantClient(host=config.QDRANT_HOST, port=6333)
        qdrant_client.get_collections()
        print(f"✅ Подключение к Qdrant: {config.QDRANT_HOST}:6333")
    except Exception as e:
        print(f"❌ ОШИБКА: Не удалось подключиться к Qdrant. {e}")
        return

    print(f"Загрузка локальной embedding-модели...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedding_model = SentenceTransformer(config.EMBEDDING_MODEL_PATH, device=device)
    print(f"✅ Модель загружена на: {device.upper()}")

    # --- Шаг 2: Подготовка коллекции в Qdrant ---
    try:
        print(f"\n--- Пересоздание коллекции '{config.COLLECTION_NAME}' ---")
        qdrant_client.recreate_collection(
            collection_name=config.COLLECTION_NAME,
            vectors_config=models.VectorParams(size=config.EMBEDDING_DIMENSION, distance=models.Distance.COSINE),
        )
        print("✅ Коллекция успешно пересоздана.")
    except Exception as e:
        print(f"❌ ОШИБКА: Не удалось создать коллекцию. {e}")
        return
        
    print("\n--- Индексация документов ---")
    absolute_docs_path = os.path.abspath(config.DOCS_ROOT_PATH)
    if not os.path.isdir(absolute_docs_path):
        print(f"❌ ОШИБКА: Папка не найдена по пути '{absolute_docs_path}'.")
        return

    all_points = []
    processed_files_count = 0

    for root, dirs, files in os.walk(absolute_docs_path):
        for filename in files:
            if not filename.lower().endswith((".doc", ".docx")):
                continue
            
            # Игнорируем временные файлы Word
            if filename.startswith('~'):
                continue

            file_path = os.path.join(root, filename)
            relative_path = os.path.relpath(file_path, absolute_docs_path)
            print(f"-> Обработка: {relative_path}")

            document_text = None
            
            # --- ИЗМЕНЕНИЕ: Новая логика обработки ---
            if filename.lower().endswith(".docx"):
                document_text = extract_text_from_docx(file_path)
            elif filename.lower().endswith(".doc"):
                # 1. Конвертируем .doc в .docx
                docx_path = convert_doc_to_docx(file_path)
                
                if docx_path:
                    # 2. Извлекаем текст из нового .docx
                    document_text = extract_text_from_docx(docx_path)
                    # 3. Удаляем временный .docx файл
                    try:
                        os.remove(docx_path)
                    except Exception as e:
                        print(f"  - ⚠️ Не удалось удалить временный файл {docx_path}: {e}")

            if not document_text:
                continue

            chunks = get_text_chunks(document_text)
            if not chunks:
                print("  - Не удалось разбить текст на чанки.")
                continue

            embeddings = embedding_model.encode(chunks, show_progress_bar=False)
            
            category = os.path.basename(os.path.dirname(file_path))

            for chunk, embedding in zip(chunks, embeddings):
                point = models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding.tolist(),
                    payload={"text": chunk, "source_file": filename, "category": category}
                )
                all_points.append(point)
            
            processed_files_count += 1

    # ... (код загрузки в Qdrant и завершения остается без изменений) ...
    if not all_points:
        print("\n⚠️ Документы для индексации не найдены. Проверьте исходную папку.")
        return

    # --- Шаг 4: Пакетная загрузка в Qdrant ---
    print(f"\n--- Загрузка данных в Qdrant ---")
    print(f"Подготовлено {len(all_points)} векторов из {processed_files_count} файлов.")
    try:
        qdrant_client.upsert(
            collection_name=config.COLLECTION_NAME,
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