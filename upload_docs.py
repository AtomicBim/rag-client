import os
import uuid
import torch
import traceback
import config
import win32com.client as win32
import sys
import json # <-- Используем JSON для файла состояния
from datetime import datetime

from unstructured.partition.docx import partition_docx
from qdrant_client import QdrantClient, models
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

# --- НОВОЕ: Имя файла для хранения состояния индексации ---
STATE_FILE = "indexing_state.json"

def load_state():
    """Загружает состояние индексации из файла."""
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return {} # Если файл пуст или поврежден
    return {}

def save_state(state):
    """Сохраняет состояние индексации в файл."""
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=4)

def extract_text_from_docx(file_path):
    """Извлекает элементы из .docx файла с помощью unstructured."""
    try:
        elements = partition_docx(filename=file_path, infer_table_structure=True)
        return "\n\n".join([str(el) for el in elements])
    except Exception as e:
        print(f"  - ❌ Ошибка извлечения текста из {os.path.basename(file_path)}: {e}")
        return None

def convert_doc_to_docx(file_path):
    """Конвертирует .doc в .docx."""
    if sys.platform != "win32":
        return None
    word = None
    try:
        abs_path_doc = os.path.abspath(file_path)
        abs_path_docx = os.path.splitext(abs_path_doc)[0] + ".docx"
        if os.path.exists(abs_path_docx):
            return abs_path_docx
        print(f"  - ⚙️ Конвертация .doc: {os.path.basename(file_path)}")
        word = win32.Dispatch("Word.Application")
        word.visible = False
        doc = word.Documents.Open(abs_path_doc)
        doc.SaveAs(abs_path_docx, FileFormat=12)
        doc.Close()
        return abs_path_docx
    except Exception as e:
        print(f"  - ❌ Ошибка конвертации {os.path.basename(file_path)}: {e}")
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
        print(f"  - ❌ Ошибка извлечения из PDF {os.path.basename(file_path)}: {e}")
        return None

def get_text_chunks(text):
    """Разбивает текст на чанки."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", ".", " ", ""])
    return text_splitter.split_text(text)

def main():
    """Основная функция для умной индексации документов."""
    try:
        print("--- Инициализация клиентов ---")
        qdrant_client = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)
        
        # --- Проверяем/создаем коллекцию один раз в начале ---
        if not qdrant_client.collection_exists(collection_name=config.COLLECTION_NAME):
            print(f"Коллекция '{config.COLLECTION_NAME}' не найдена. Создание новой...")
            qdrant_client.create_collection(
                collection_name=config.COLLECTION_NAME,
                vectors_config=models.VectorParams(size=config.EMBEDDING_DIMENSION, distance=models.Distance.COSINE),
            )
            print("✅ Коллекция успешно создана.")

        print(f"✅ Подключение к Qdrant: {config.QDRANT_HOST}:{config.QDRANT_PORT}")
    except Exception as e:
        print(f"❌ ОШИБКА: Не удалось подключиться к Qdrant или подготовить коллекцию. {e}")
        return
    
    # --- НОВЫЙ БЛОК: Определение измененных файлов ---
    print("\n--- Проверка состояния документов ---")
    state = load_state()
    files_to_process = []
    
    absolute_docs_path = os.path.abspath(config.DOCS_ROOT_PATH)
    if not os.path.isdir(absolute_docs_path):
        print(f"❌ ОШИБКА: Папка не найдена по пути '{absolute_docs_path}'.")
        return

    # Предварительная конвертация .doc в .docx
    for root, _, files in os.walk(absolute_docs_path):
        for filename in files:
            if filename.lower().endswith(".doc") and not filename.startswith('~'):
                convert_doc_to_docx(os.path.join(root, filename))

    # Основной обход для определения изменений
    for root, _, files in os.walk(absolute_docs_path):
        for filename in files:
            if filename.startswith('~') or filename.lower().endswith(".doc"):
                continue

            file_path = os.path.join(root, filename)
            file_mod_time = os.path.getmtime(file_path)
            
            # Если файл новый или изменился, добавляем в очередь на обработку
            if state.get(file_path) != file_mod_time:
                files_to_process.append(file_path)
                print(f"  - 🔄 В очереди на обработку: {os.path.basename(filename)} (изменен)")
    
    if not files_to_process:
        print("\n✅ Все документы актуальны. Обновление не требуется.")
        return
        
    print(f"\n--- Обнаружено {len(files_to_process)} новых/измененных файлов для обработки ---")

    # --- Загружаем модель только если есть что обрабатывать ---
    print(f"\nЗагрузка локальной embedding-модели...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedding_model = SentenceTransformer(config.EMBEDDING_MODEL_PATH, device=device)
    print(f"✅ Модель загружена на: {device.upper()}")
    
    all_points = []
    new_state = state.copy()

    for file_path in files_to_process:
        filename = os.path.basename(file_path)
        print(f"\n-> Обработка: {filename}")
        
        # --- НОВЫЙ ШАГ: Удаление старых данных для этого файла из Qdrant ---
        print(f"  - Удаление старых записей для '{filename}' из Qdrant...")
        try:
            qdrant_client.delete(
                collection_name=config.COLLECTION_NAME,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[models.FieldCondition(key="source_file", match=models.MatchValue(value=filename))]
                    )
                )
            )
            print("  - ✅ Старые записи удалены.")
        except Exception as e:
            print(f"  - ⚠️ Ошибка при удалении старых записей (возможно, их и не было): {e}")

        # --- Стандартная обработка файла ---
        document_text = None
        if filename.lower().endswith(".docx"):
            document_text = extract_text_from_docx(file_path)
        elif filename.lower().endswith(".pdf"):
            document_text = extract_text_from_pdf(file_path)

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
            
            # Обновляем время модификации в новом состоянии
            new_state[file_path] = os.path.getmtime(file_path)
            print(f"  - ✅ Файл обработан, создано {len(embeddings)} векторов.")

    # --- Загрузка всех НОВЫХ векторов в Qdrant батчами ---
    if not all_points:
        print("\n⚠️ Не удалось создать векторы из измененных файлов.")
        return

    print(f"\n--- Загрузка {len(all_points)} новых векторов в Qdrant ---")
    BATCH_SIZE = 512
    try:
        for i in range(0, len(all_points), BATCH_SIZE):
            batch_points = all_points[i:i+BATCH_SIZE]
            print(f"  - Загрузка батча ({len(batch_points)} векторов)...")
            qdrant_client.upsert(collection_name=config.COLLECTION_NAME, points=batch_points, wait=True)
        
        print("\n✅ Новые векторы успешно загружены!")
        
        # --- Сохраняем новое состояние ТОЛЬКО после успешной загрузки ---
        save_state(new_state)
        print(f"✅ Файл состояния '{STATE_FILE}' обновлен.")

    except Exception as e:
        print(f"❌ ОШИБКА во время загрузки векторов в Qdrant: {e}")
        print(f"\n⚠️ Загрузка не удалась. Файл состояния '{STATE_FILE}' НЕ был обновлен. Попробуйте запустить снова.")

    print("\n✅ Локальная индексация полностью завершена!")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\n❌❌❌ КРИТИЧЕСКАЯ ОШИБКА! Выполнение скрипта прервано. ❌❌❌")
        traceback.print_exc()