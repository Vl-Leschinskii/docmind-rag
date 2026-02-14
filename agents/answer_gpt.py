from openai import OpenAI
from typing import List, Dict, Any

class AnswerGPTAgent:
    """Агент для генерации ответов через LM Studio"""
    
    def __init__(self, api_base="http://localhost:1234/v1", 
                 model="local-model", temperature=0.3, max_tokens=1000):
        self.client = OpenAI(
            base_url=api_base,
            api_key="not-needed"
        )
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        self.system_prompt = """Ты - ассистент по анализу документов. 
        Используй предоставленные фрагменты документа для ответа.
        Обязательно указывай номера глав и разделов в ответе.
        Если информации недостаточно - так и скажи.
        Отвечай на том же языке, на котором задан вопрос."""
    
    def generate_answer(self, query: str, context_chunks: List[Dict]) -> str:
        """Генерация ответа на основе контекста"""
        context_text = self._format_context(context_chunks)
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"""
Вопрос: {query}

Контекст из документа:
{context_text}

Ответь на вопрос, используя только предоставленный контекст.
Укажи источники (глава, раздел) в ответе.
"""}
        ]
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"❌ Ошибка генерации ответа: {str(e)}"
    
    def _format_context(self, chunks: List[Dict]) -> str:
        """Форматирование контекста"""
        formatted = []
        for i, chunk in enumerate(chunks, 1):
            source_info = []
            if 'chapter_title' in chunk.get('metadata', {}):
                source_info.append(f"Глава: {chunk['metadata']['chapter_title']}")
            if 'section_title' in chunk.get('metadata', {}):
                source_info.append(f"Раздел: {chunk['metadata']['section_title']}")
            
            source_str = f"[{', '.join(source_info)}]" if source_info else ""
            formatted.append(f"Фрагмент {i} {source_str}:\n{chunk['text']}")
        
        return "\n\n".join(formatted)