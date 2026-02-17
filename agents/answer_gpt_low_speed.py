# agents/answer_gpt.py - ПОЛНОСТЬЮ НОВАЯ ВЕРСИЯ
import requests
import json
import time
from typing import List, Dict, Any, Optional

class AnswerGPTAgent:
    """
    Агент для генерации ответов через LM Studio
    Использует прямые HTTP запросы, без библиотеки openai
    """
    
    def __init__(self, 
                 api_base: str = "http://localhost:1234/v1", 
                 model: str = "local-model", 
                 temperature: float = 0.2, 
                 max_tokens: int = 500,
                 timeout: int = 180):
        
        self.api_base = api_base.rstrip('/')
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.session = requests.Session()
        
        # Настройка заголовков
        self.session.headers.update({
            "Content-Type": "application/json"
        })
        
        self.system_prompt = """Ты - ассистент по анализу документов. 
        Используй предоставленные фрагменты документа для ответа.
        Обязательно указывай номера глав и разделов в ответе.
        Если информации недостаточно - так и скажи.
        Отвечай на том же языке, на котором задан вопрос."""
        
        print(f"✅ AnswerGPTAgent инициализирован (прямые HTTP запросы)")
        print(f"   API: {self.api_base}, Модель: {self.model}")
    
    def check_connection(self) -> bool:
        """Проверка подключения к LM Studio"""
        try:
            response = self.session.get(
                f"{self.api_base}/models",
                timeout=5
            )
            
            if response.status_code == 200:
                data = response.json()
                models = []
                if 'data' in data:
                    models = [m['id'] for m in data['data']]
                elif 'models' in data:
                    models = data['models']
                
                print(f"✅ LM Studio доступна. Модели: {models}")
                
                # Если наша модель не указана, используем первую доступную
                if self.model == "local-model" and models:
                    self.model = models[0]
                    print(f"   Автоматически выбрана модель: {self.model}")
                
                return True
            else:
                print(f"❌ LM Studio вернула статус {response.status_code}")
                return False
                
        except requests.exceptions.ConnectionError:
            print(f"❌ Не удалось подключиться к {self.api_base}")
            print("   Убедитесь, что LM Studio запущена и сервер активен")
            return False
        except Exception as e:
            print(f"❌ Ошибка проверки подключения: {e}")
            return False
    
    def generate_answer(self, query: str, context_chunks: List[Dict]) -> str:
        """Генерация ответа через прямой HTTP запрос"""
        
        if not context_chunks:
            return "Нет контекста для генерации ответа."
        
        # Форматируем контекст
        context_text = self._format_context(context_chunks)
        
        # Формируем запрос в формате OpenAI API
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system", 
                    "content": self.system_prompt
                },
                {
                    "role": "user", 
                    "content": f"""
Вопрос: {query}

Контекст из документа:
{context_text}

Ответь на вопрос, используя только предоставленный контекст.
Укажи источники (глава, раздел) в ответе.
"""
                }
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False
        }
        
        # Пробуем отправить запрос
        for attempt in range(2):  # Две попытки
            try:
                response = self.session.post(
                    f"{self.api_base}/chat/completions",
                    json=payload,
                    timeout=self.timeout
                )
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Извлекаем ответ из разных форматов
                    if 'choices' in data and len(data['choices']) > 0:
                        if 'message' in data['choices'][0]:
                            return data['choices'][0]['message']['content']
                        elif 'text' in data['choices'][0]:
                            return data['choices'][0]['text']
                    
                    return "Получен пустой ответ от модели."
                    
                elif response.status_code == 404:
                    return f"Модель '{self.model}' не найдена. Проверьте название модели в LM Studio."
                else:
                    error_text = response.text[:200]
                    return f"Ошибка API (статус {response.status_code}): {error_text}"
                    
            except requests.exceptions.Timeout:
                if attempt == 0:
                    print(f"⏱️ Таймаут, пробую еще раз...")
                    time.sleep(2)
                    continue
                return "Таймаут при обращении к LM Studio. Модель слишком медленная или сервер перегружен."
                
            except requests.exceptions.ConnectionError:
                return "Ошибка подключения к LM Studio. Сервер не доступен."
                
            except Exception as e:
                print(f"❌ Ошибка при запросе: {e}")
                if attempt == 0:
                    continue
                return f"Ошибка при генерации ответа: {str(e)}"
        
        return "Не удалось получить ответ после нескольких попыток."
    
    def _format_context(self, chunks: List[Dict]) -> str:
        """Форматирование контекста с ограничением размера"""
        formatted = []
        total_length = 0
        MAX_CONTEXT_LENGTH = 2000  # Ограничиваем контекст
        
        for i, chunk in enumerate(chunks, 1):
            # Получаем текст чанка
            chunk_text = ""
            if isinstance(chunk, dict):
                chunk_text = chunk.get('text', '')
                if not chunk_text:
                    chunk_text = chunk.get('content', '')
            else:
                chunk_text = str(chunk)
            
            if not chunk_text:
                continue
            
            # Получаем метаданные
            metadata = chunk.get('metadata', {}) if isinstance(chunk, dict) else {}
            
            # Формируем источник
            source_info = []
            if metadata.get('chapter_title'):
                source_info.append(f"Глава: {metadata['chapter_title']}")
            if metadata.get('section_title'):
                source_info.append(f"Раздел: {metadata['section_title']}")
            
            source_str = f"[{', '.join(source_info)}]" if source_info else ""
            
            # Обрезаем слишком длинные чанки
            if len(chunk_text) > 500:
                chunk_text = chunk_text[:500] + "..."
            
            # Формируем фрагмент
            fragment = f"Фрагмент {i} {source_str}:\n{chunk_text}"
            
            # Проверяем общую длину
            if total_length + len(fragment) > MAX_CONTEXT_LENGTH:
                break
            
            formatted.append(fragment)
            total_length += len(fragment)
        
        return "\n\n".join(formatted)
    
    def __repr__(self):
        return f"AnswerGPTAgent(api={self.api_base}, model={self.model})"