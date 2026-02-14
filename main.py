#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DocMind Local RAG - Мультиагентная система для анализа Word документов
Запуск приложения
"""

import sys
import os
from pathlib import Path

# Добавляем текущую директорию в путь
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def check_dependencies():
    """Проверка установленных зависимостей"""
    required_packages = [
        'sentence_transformers',
        'chromadb',
        'docx',
        'openai',
        'fastapi',
        'uvicorn',
     #   'pyyaml',
        'nltk',
        'sklearn'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print("❌ Отсутствуют зависимости:")
        for package in missing:
            print(f"   - {package}")
        print("\nУстановите их командой:")
        print(f"pip install {' '.join(missing)}")
        return False
    
    print("✅ Все зависимости установлены")
    return True

def check_lm_studio():
    """Проверка подключения к LM Studio"""
    try:
        from openai import OpenAI
        client = OpenAI(
            base_url="http://localhost:1234/v1",
            api_key="not-needed"
        )
        
        response = client.chat.completions.create(
            model="local-model",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=10
        )
        print("✅ LM Studio доступна")
        return True
    except Exception as e:
        print("⚠️ LM Studio не запущена или недоступна")
        print("   Запустите LM Studio и загрузите модель")
        print("   URL: http://localhost:1234/v1")
        return False

def main():
    """Главная функция"""
    print("=" * 50)
    print("📚 DocMind Local RAG System")
    print("Мультиагентная система для анализа Word документов")
    print("=" * 50)
    
    # Проверка зависимостей
    if not check_dependencies():
        sys.exit(1)
    
    # Проверка LM Studio
    check_lm_studio()
    
    # Создание необходимых директорий
    Path("./uploads").mkdir(exist_ok=True)
    Path("./vector_db").mkdir(exist_ok=True)
    
    # Запуск веб-интерфейса
    print("\n🚀 Запуск веб-интерфейса...")
    print("🌐 Откройте браузер и перейдите по адресу: http://localhost:8000")
    print("=" * 50)
    
    from web_interface import start_server
    start_server()

if __name__ == "__main__":
    main()