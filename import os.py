import os
print("Файл существует:", os.path.exists(r"docmind-rag\Анализ от Нейронаналитика.docx"))
print("Права на чтение:", os.access(r"docmind-rag\Анализ от Нейронаналитика.docx", os.R_OK))