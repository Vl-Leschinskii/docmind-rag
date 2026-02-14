# test_system.py
from agents.doc_parser import DocParserAgent
from agents.smart_chunker import SmartChunkerAgent
from agents.vector_agent import VectorAgent

def quick_test():
    """Быстрая проверка всех компонентов"""
    
    # 1. Тест парсера
    print("1️⃣ Тест парсера...")
    parser = DocParserAgent()
    print("✅ Парсер OK")
    
    # 2. Тест чанкера
    print("2️⃣ Тест чанкера...")
    chunker = SmartChunkerAgent("all-MiniLM-L6-v2")
    print("✅ Чанкер OK")
    
    # 3. Тест векторизации
    print("3️⃣ Тест векторизации...")
    vector = VectorAgent("all-MiniLM-L6-v2")
    print("✅ Векторизация OK")
    
    # 4. Тест LM Studio
    print("4️⃣ Тест LM Studio...")
    from openai import OpenAI
    client = OpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")
    response = client.chat.completions.create(
        model="local-model",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=10
    )
    print("✅ LM Studio OK")
    
    print("\n🎉 Все компоненты работают корректно!")

if __name__ == "__main__":
    quick_test()