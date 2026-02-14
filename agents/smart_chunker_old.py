from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import nltk
from typing import List, Dict, Any

nltk.download('punkt', quiet=True)
try:
    from nltk.tokenize import sent_tokenize
except:
    def sent_tokenize(text):
        return [s.strip() for s in text.split('.') if s.strip()]

class SmartChunkerAgent:
    """Агент для интеллектуального разделения текста на чанки"""
    
    def __init__(self, embedding_model="all-MiniLM-L6-v2", chunk_size=500, overlap=50):
        self.chunk_size = chunk_size
        self.overlap = overlap
        
        try:
            print(f"📦 Загрузка модели эмбеддингов: {embedding_model}")
            self.embedder = SentenceTransformer(embedding_model)
            print(f"✅ Модель эмбеддингов загружена")
        except Exception as e:
            print(f"⚠️ Ошибка загрузки {embedding_model}: {e}")
            print(f"📦 Использую fallback модель: all-MiniLM-L6-v2")
            self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
    
    def semantic_chunking(self, text: str) -> List[str]:
        """Адаптивное разделение текста"""
        if not text or len(text) < self.chunk_size:
            return [text]
        
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence_size = len(sentence)
            
            if current_size + sentence_size > self.chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                
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
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def split_by_semantics(self, text: str) -> List[str]:
        """Разделение по семантическим границам"""
        sentences = sent_tokenize(text)
        if len(sentences) <= 1:
            return [text]
        
        try:
            embeddings = self.embedder.encode(sentences)
            breaks = [0]
            
            for i in range(1, len(embeddings)):
                similarity = cosine_similarity(
                    embeddings[i-1].reshape(1, -1),
                    embeddings[i].reshape(1, -1)
                )[0][0]
                
                if similarity < 0.6:
                    breaks.append(i)
            
            breaks.append(len(sentences))
            chunks = []
            
            for i in range(len(breaks)-1):
                chunk = ' '.join(sentences[breaks[i]:breaks[i+1]])
                if chunk.strip():
                    chunks.append(chunk)
            
            return chunks if chunks else [text]
            
        except Exception as e:
            print(f"⚠️ Ошибка семантического разделения: {e}")
            return self.semantic_chunking(text)