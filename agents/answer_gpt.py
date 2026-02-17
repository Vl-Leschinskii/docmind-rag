# agents/answer_gpt_async.py
import httpx
import asyncio
import json
from typing import List, Dict, Any, Optional

class AnswerGPTAgentAsync:
    """
    Асинхронный агент для максимальной производительности
    """
    
    def __init__(self, 
                 api_base: str = "http://localhost:1234/v1", 
                 model: str = "local-model"):
        
        self.api_base = api_base.rstrip('/')
        self.model = model
        
        # Асинхронный клиент
        self.client = httpx.AsyncClient(
            timeout=120.0,
            limits=httpx.Limits(max_keepalive_connections=10, max_connections=20),
            headers={"Content-Type": "application/json"},
            http2=True
        )
    
    async def generate_answer_async(self, query: str, context_chunks: List[Dict]) -> str:
        """Асинхронная генерация ответа"""
        
        context_text = self._format_context(context_chunks)
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "Ты - ассистент по анализу документов."},
                {"role": "user", "content": f"Контекст: {context_text}\n\nВопрос: {query}"}
            ],
            "temperature": 0.3,
            "max_tokens": 1000
        }
        
        try:
            response = await self.client.post(
                f"{self.api_base}/chat/completions",
                json=payload
            )
            
            if response.status_code == 200:
                data = response.json()
                return data['choices'][0]['message']['content']
            else:
                return f"Ошибка: {response.status_code}"
                
        except Exception as e:
            return f"Ошибка: {str(e)}"
    
    def _format_context(self, chunks: List[Dict]) -> str:
        """Форматирование контекста"""
        return "\n\n".join([chunk.get('text', '') for chunk in chunks[:5]])
    
    async def close(self):
        """Закрытие клиента"""
        await self.client.aclose()