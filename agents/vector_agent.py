from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional
import numpy as np

class VectorAgent:
    """Агент для векторизации и поиска в векторной БД"""
    
    def __init__(self, embedding_model="all-MiniLM-L6-v2", 
                 use_gpu=False, batch_size=16, db_path="./vector_db"):
        self.batch_size = batch_size
        
        try:
            self.embedder = SentenceTransformer(embedding_model)
            print(f"✅ Загружена модель эмбеддингов: {embedding_model}")
        except Exception as e:
            print(f"⚠️ Ошибка загрузки модели: {e}")
            self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = None
    
    def create_index(self, chunks: List[str], metadata: List[Dict]) -> Any:
        """Создание векторного индекса"""
        try:
            self.client.delete_collection("document_chunks")
        except:
            pass
        
        self.collection = self.client.create_collection(
            name="document_chunks",
            metadata={"hnsw:space": "cosine"}
        )
        
        # Пакетная обработка
        for i in range(0, len(chunks), self.batch_size):
            batch_chunks = chunks[i:i+self.batch_size]
            batch_meta = metadata[i:i+self.batch_size]
            batch_ids = [f"chunk_{j}" for j in range(i, i+len(batch_chunks))]
            
            embeddings = self.embedder.encode(batch_chunks)
            
            self.collection.add(
                embeddings=embeddings.tolist(),
                documents=batch_chunks,
                metadatas=batch_meta,
                ids=batch_ids
            )
        
        print(f"✅ Индекс создан. Чанков: {len(chunks)}")
        return self.collection
    
    def hierarchical_search(self, query: str, top_k: int = 5, 
                           filters: Optional[Dict] = None) -> List[Dict]:
        """Поиск по векторной БД"""
        if not self.collection:
            raise ValueError("Индекс не создан. Сначала вызовите create_index()")
        
        query_embedding = self.embedder.encode(query)
        
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            where=filters
        )
        
        chunks = []
        for i in range(len(results['documents'][0])):
            chunks.append({
                'text': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'id': results['ids'][0][i],
                'distance': results['distances'][0][i] if 'distances' in results else None
            })
        
        return chunks