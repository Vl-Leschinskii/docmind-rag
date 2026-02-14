import os
import json
from typing import List, Dict, Any, Optional

from agents.doc_parser import DocParserAgent
from agents.smart_chunker import SmartChunkerAgent

from agents.vector_agent import VectorAgent
from agents.answer_gpt import AnswerGPTAgent
from agents.validator import ValidatorAgent

class RAGOrchestrator:
    """Оркестратор мультиагентной RAG системы"""
    
    def __init__(self, config_path: str = r"config.yaml"):
        import yaml
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        self.config = config
        
        # Инициализация агентов
        self.agents = {
            'parser': DocParserAgent(),
            'chunker': SmartChunkerAgent(
                embedding_model=config['embedding_model'],
                chunk_size=config['chunk_size'],
                overlap=config['overlap_size']
            ),
            'vector': VectorAgent(
                embedding_model=config['embedding_model'],
                use_gpu=config['use_gpu'],
                batch_size=config['batch_size'],
                db_path=config['vector_db_path']
            ),
            'generator': AnswerGPTAgent(
                api_base=config['lm_studio_url'],
                model=config['generation_model'],
                temperature=config['temperature'],
                max_tokens=config['max_tokens']
            ),
            'validator': ValidatorAgent()
        }
        
        self.doc_structure = None
        self.is_indexed = False
    
    def process_document(self, docx_path: str) -> Dict[str, Any]:
        """Полный пайплайн обработки документа"""
        print(f"\n📄 Начало обработки документа: {docx_path}")
        
        # 1. Парсинг
        print("🔍 Парсинг структуры...")
        self.doc_structure = self.agents['parser'].parse_with_hierarchy(docx_path)
        
        # 2. Чанкование
        print("✂️ Разделение на чанки...")
        chunks = []
        metadata = []
        
        for chapter in self.doc_structure['chapters']:
            if chapter.get('content'):
                chapter_chunks = self.agents['chunker'].split_by_semantics(chapter['content'])
                chunks.extend(chapter_chunks)
                
                for chunk in chapter_chunks:
                    metadata.append({
                        'chapter_id': chapter['id'],
                        'chapter_title': chapter['title'],
                        'level': 1,
                        'type': 'chapter'
                    })
            
            for section in chapter.get('sections', []):
                if section.get('content'):
                    section_chunks = self.agents['chunker'].split_by_semantics(section['content'])
                    chunks.extend(section_chunks)
                    
                    for chunk in section_chunks:
                        metadata.append({
                            'chapter_id': chapter['id'],
                            'chapter_title': chapter['title'],
                            'section_id': section['id'],
                            'section_title': section['title'],
                            'level': 2,
                            'type': 'section'
                        })
        
        # 3. Индексация
        print("🔗 Создание векторного индекса...")
        self.agents['vector'].create_index(chunks, metadata)
        self.is_indexed = True
        
        print(f"✅ Документ обработан. Глав: {len(self.doc_structure['chapters'])}, Чанков: {len(chunks)}")
        
        return {
            'structure': self.doc_structure,
            'chunks_count': len(chunks),
            'chapters_count': len(self.doc_structure['chapters'])
        }
    
    def query_document(self, question: str, chapter_filter: Optional[str] = None) -> Dict[str, Any]:
        """Обработка запроса пользователя"""
        if not self.is_indexed:
            return {"error": "Сначала загрузите и обработайте документ"}
        
        print(f"\n❓ Вопрос: {question}")
        
        # 1. Поиск
        print("🔎 Поиск релевантных чанков...")
        filters = {"chapter_id": chapter_filter} if chapter_filter else None
        chunks = self.agents['vector'].hierarchical_search(question, top_k=5, filters=filters)
        
        # 2. Генерация
        print("🤖 Генерация ответа...")
        answer = self.agents['generator'].generate_answer(question, chunks)
        
        # 3. Валидация
        print("✅ Валидация ответа...")
        validated = self.agents['validator'].validate(answer, chunks)
        
        return validated
    
    def get_document_structure(self) -> Dict:
        """Получение структуры документа"""
        return self.doc_structure