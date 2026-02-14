# agents/smart_chunker.py (исправленная версия)
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import nltk
import traceback
from typing import List, Dict, Any

# Скачиваем ресурсы NLTK
try:
    nltk.download('punkt', quiet=True)
    from nltk.tokenize import sent_tokenize
except Exception as e:
    print(f"⚠️ Ошибка загрузки NLTK: {e}")
    def sent_tokenize(text):
        # Простая токенизация по предложениям
        return [s.strip() + '.' for s in text.split('.') if s.strip()]

class SmartChunkerAgent:
    """Агент для интеллектуального разделения текста на чанки"""
    
    def __init__(self, embedding_model="all-MiniLM-L6-v2", chunk_size=500, overlap=50):
        self.chunk_size = chunk_size
        self.overlap = overlap
        
        print(f"📦 Инициализация чанкера с моделью: {embedding_model}")
        
        try:
            self.embedder = SentenceTransformer(embedding_model)
            print(f"✅ Модель эмбеддингов загружена успешно")
        except Exception as e:
            print(f"⚠️ Ошибка загрузки {embedding_model}: {e}")
            print(f"📦 Пробую fallback модель: all-MiniLM-L6-v2")
            try:
                self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
                print(f"✅ Fallback модель загружена")
            except Exception as e2:
                print(f"❌ Критическая ошибка загрузки модели: {e2}")
                raise
    
    def semantic_chunking(self, text: str) -> List[str]:
        """Адаптивное разделение текста"""
        try:
            if not text or not isinstance(text, str):
                print(f"⚠️ Пустой или невалидный текст")
                return []
            
            if len(text) < self.chunk_size:
                return [text]
            
            # Разбиваем на предложения
            try:
                sentences = sent_tokenize(text)
            except Exception as e:
                print(f"⚠️ Ошибка токенизации: {e}")
                sentences = text.split('. ')
            
            print(f"📊 Получено предложений: {len(sentences)}")
            
            chunks = []
            current_chunk = []
            current_size = 0
            
            for i, sentence in enumerate(sentences):
                sentence = sentence.strip()
                if not sentence:
                    continue
                    
                sentence_size = len(sentence)
                
                if current_size + sentence_size > self.chunk_size and current_chunk:
                    # Сохраняем текущий чанк
                    chunk_text = ' '.join(current_chunk)
                    chunks.append(chunk_text)
                    print(f"  ➕ Чанк {len(chunks)}: {len(chunk_text)} символов")
                    
                    # Создаем перекрытие
                    overlap_chunk = []
                    overlap_size = 0
                    for s in reversed(current_chunk):
                        if overlap_size + len(s) < self.overlap:
                            overlap_chunk.insert(0, s)
                            overlap_size += len(s)
                        else:
                            break
                    
                    current_chunk = overlap_chunk
                    current_size = overlap_size
                
                current_chunk.append(sentence)
                current_size += sentence_size
            
            # Добавляем последний чанк
            if current_chunk:
                chunk_text = ' '.join(current_chunk)
                chunks.append(chunk_text)
                print(f"  ➕ Последний чанк {len(chunks)}: {len(chunk_text)} символов")
            
            print(f"✅ Чанкование завершено. Всего чанков: {len(chunks)}")
            return chunks
            
        except Exception as e:
            print(f"❌ Ошибка в semantic_chunking: {e}")
            traceback.print_exc()
            return [text]  # Возвращаем весь текст как один чанк в случае ошибки
    
    def split_by_semantics(self, text: str) -> List[str]:
        """Разделение по семантическим границам"""
        try:
            if not text or len(text) < 100:  # Слишком короткий текст
                return [text]
            
            sentences = sent_tokenize(text)
            if len(sentences) <= 1:
                return [text]
            
            print(f"🔬 Семантическое разделение {len(sentences)} предложений...")
            
            # Получаем эмбеддинги
            try:
                embeddings = self.embedder.encode(sentences)
                print(f"   Эмбеддинги получены: {embeddings.shape}")
            except Exception as e:
                print(f"⚠️ Ошибка получения эмбеддингов: {e}")
                return self.semantic_chunking(text)
            
            # Ищем точки разрыва
            breaks = [0]
            for i in range(1, len(embeddings)):
                similarity = cosine_similarity(
                    embeddings[i-1].reshape(1, -1),
                    embeddings[i].reshape(1, -1)
                )[0][0]
                
                if similarity < 0.6:  # Порог семантического разрыва
                    breaks.append(i)
                    print(f"   Разрыв после предложения {i} (схожесть: {similarity:.3f})")
            
            breaks.append(len(sentences))
            
            # Формируем чанки
            chunks = []
            for i in range(len(breaks)-1):
                chunk = ' '.join(sentences[breaks[i]:breaks[i+1]])
                if chunk.strip():
                    chunks.append(chunk)
            
            print(f"✅ Семантическое разделение дало {len(chunks)} чанков")
            return chunks if chunks else [text]
            
        except Exception as e:
            print(f"❌ Ошибка в split_by_semantics: {e}")
            traceback.print_exc()
            return self.semantic_chunking(text)