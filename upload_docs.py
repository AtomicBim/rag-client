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
from pypdf import PdfReader


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
    Конвертирует .doc в .docx с помощью Microsoft Word и СОХРАНЯЕТ результат.
    Возвращает путь к новому .docx файлу.
    """
    if sys.platform != "win32":
        print("  - ⚠️ Пропуск .doc файла: автоматическая конвертация поддерживается только на Windows.")
        return None
        
    word = None
    try:
        abs_path_doc = os.path.abspath(file_path)
        # --- ИЗМЕНЕНИЕ: Убираем временное имя, сохраняем как обычный .docx
        abs_path_docx = os.path.splitext(abs_path_doc)[0] + ".docx"
        
        # Проверяем, не существует ли уже .docx файл, чтобы не делать лишнюю работу
        if os.path.exists(abs_path_docx):
            print(f"  - ℹ️ Файл {os.path.basename(abs_path_docx)} уже существует. Конвертация не требуется.")
            return abs_path_docx
            
        print(f"  - ⚙️ Конвертация .doc с помощью MS Word: {os.path.basename(file_path)}")
        
        word = win32.Dispatch("Word.Application")
        word.visible = False
        doc = word.Documents.Open(abs_path_doc)
        doc.SaveAs(abs_path_docx, FileFormat=12) # 12 = wdFormatXMLDocument
        doc.Close()
        print(f"  - ✅ Файл успешно сконвертирован и сохранен как: {os.path.basename(abs_path_docx)}")
        return abs_path_docx
        
    except Exception as e:
        print(f"  - ❌ Ошибка конвертации файла {os.path.basename(file_path)} через Word: {e}")
        return None
    finally:
        if word:
            word.Quit()


def extract_text_from_pdf(file_path):
    """Извлекает текст из .pdf файла."""
    try:
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text
    except Exception as e:
        print(f"  - ❌ Ошибка извлечения текста из PDF {os.path.basename(file_path)}: {e}")
        return None


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
    # ... (код инициализации без изменений) ...
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

    print("\n--- Индексация документов ---")
    absolute_docs_path = os.path.abspath(DOCS_ROOT_PATH)
    if not os.path.isdir(absolute_docs_path):
        print(f"❌ ОШИБКА: Папка не найдена по пути '{absolute_docs_path}'.")
        return

    all_points = []
    processed_files_count = 0

    # --- ИЗМЕНЕНИЕ: Полностью переработанная логика обхода файлов ---
    for root, dirs, files in os.walk(absolute_docs_path):
        # Сначала конвертируем все .doc файлы в .docx
        # Это нужно сделать до основной обработки, чтобы правильно применять логику для PDF
        for filename in files:
            if filename.lower().endswith(".doc") and not filename.startswith('~'):
                convert_doc_to_docx(os.path.join(root, filename))
        
        # Перечитываем список файлов, так как могли появиться новые .docx
        all_current_files = os.listdir(root)
        word_doc_basenames = {os.path.splitext(f)[0] for f in all_current_files if f.lower().endswith(('.doc', '.docx'))}

        for filename in all_current_files:
            if filename.startswith('~') or filename.lower().endswith(".doc"):
                # Пропускаем временные файлы и старые .doc (т.к. они уже сконвертированы)
                continue
            
            file_path = os.path.join(root, filename)
            relative_path = os.path.relpath(file_path, absolute_docs_path)
            document_text = None
            
            if filename.lower().endswith(".docx"):
                print(f"-> Обработка .docx: {relative_path}")
                document_text = extract_text_from_docx(file_path)
            
            elif filename.lower().endswith(".pdf"):
                pdf_basename = os.path.splitext(filename)[0]
                if pdf_basename not in word_doc_basenames:
                    print(f"-> Обработка PDF: {relative_path}")
                    document_text = extract_text_from_pdf(file_path)
                else:
                    print(f"  - ⚠️ Пропуск PDF файла '{filename}', так как существует одноименный .doc/.docx.")
                    continue

            # Общая логика для обработки извлеченного текста
            if document_text:
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

    if not all_points:
        print("\n⚠️ Документы для индексации не найдены. Проверьте исходную папку.")
        return

    # ... (код загрузки в Qdrant без изменений) ...
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