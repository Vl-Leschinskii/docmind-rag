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
        ('sentence-transformers', 'sentence_transformers'),
        ('chromadb', 'chromadb'),
        ('python-docx', 'docx'),
        ('openai', 'openai'),
        ('fastapi', 'fastapi'),
        ('uvicorn', 'uvicorn'),
        ('pyyaml', 'yaml'),
        ('nltk', 'nltk'),
        ('scikit-learn', 'sklearn'),
        ('numpy', 'numpy'),
        ('requests', 'requests')
    ]
    
    missing = []
    installed = []
    
    for pip_name, import_name in required_packages:
        try:
            __import__(import_name)
            installed.append(pip_name)
            print(f"✅ {pip_name} -> импорт {import_name} OK")
        except ImportError as e:
            missing.append(pip_name)
            print(f"❌ {pip_name} -> ошибка: {e}")
    
    if missing:
        print("\n" + "="*50)
        print("❌ ОТСУТСТВУЮТ ЗАВИСИМОСТИ:")
        for package in missing:
            print(f"   - {package}")
        print("\n" + "="*50)
        print("Установите их командой:")
        print(f"pip install {' '.join(missing)}")
        print("="*50)
        return False
    else:
        print("\n" + "="*50)
        print("✅ ВСЕ ЗАВИСИМОСТИ УСТАНОВЛЕНЫ!")
        print(f"   Установлено пакетов: {len(installed)}")
        print("="*50)
        return True

def check_lm_studio():
    """Проверка подключения к LM Studio (упрощенная)"""
    print("🔍 Проверка подключения к LM Studio...")
    try:
        import requests
        # Попробуем просто получить список моделей — это самый надежный тест
        response = requests.get("http://localhost:1234/v1/models", timeout=5)
        if response.status_code == 200:
            print(f"✅ LM Studio доступна. Статус: {response.status_code}")
            return True
        else:
            print(f"⚠️ LM Studio вернула статус {response.status_code}")
            return False
    except ImportError:
        print("❌ Библиотека requests не установлена")
        return False
    except requests.exceptions.ConnectionError:
        print("❌ Ошибка подключения: порт 1234 не отвечает")
        return False
    except Exception as e:
        print(f"❌ Неизвестная ошибка: {e}")
        return False

#def check_lm_studio():
#    """Проверка подключения к LM Studio"""
#    try:
#        from openai import OpenAI
#        client = OpenAI(
#            base_url="http://localhost:1234/v1",
#            api_key="not-needed"
#        )
#        
#        response = client.chat.completions.create(
#            model="local-model",
#            messages=[{"role": "user", "content": "Hello"}],
#            max_tokens=10
#        )
#        print("✅ LM Studio доступна")
#        return True
#    except Exception as e:
#        print("⚠️ LM Studio не запущена или недоступна")
#        print("   Запустите LM Studio и загрузите модель")
#        print("   URL: http://localhost:1234/v1")
#        return False

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
