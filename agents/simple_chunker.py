# agents/simple_chunker.py (временная замена)
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import nltk
from typing import List, Dict, Any

nltk.download('punkt', quiet=True)

class SimpleChunkerAgent:
    """Упрощенный чанкер без семантики - для тестирования"""
    
    def __init__(self, embedding_model=None, chunk_size=500, overlap=50):
        self.chunk_size = chunk_size
        self.overlap = overlap
        print("📦 Используется SIMPLE чанкер (без эмбеддингов)")
    
    def split_by_semantics(self, text: str) -> List[str]:
        """Простое разделение по размеру"""
        if not text:
            return []
        
        words = text.split()
        chunks = []
        current_chunk = []
        current_size = 0
        
        for word in words:
            word_size = len(word) + 1  # +1 для пробела
            
            if current_size + word_size > self.chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                
                # Перекрытие
                overlap_words = current_chunk[-self.overlap//10:] if self.overlap > 0 else []
                current_chunk = overlap_words
                current_size = sum(len(w) + 1 for w in overlap_words)
            
            current_chunk.append(word)
            current_size += word_size
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        print(f"✅ Simple чанкер создал {len(chunks)} чанков")
        return chunks