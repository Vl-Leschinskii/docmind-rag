# orchestrator.py
import os
import json
import yaml
from typing import List, Dict, Any, Optional

# Импорты агентов
from agents.doc_parser import DocParserAgent
from agents.smart_chunker import SmartChunkerAgent
from agents.vector_agent import VectorAgent
from agents.answer_gpt import AnswerGPTAgent
from agents.validator import ValidatorAgent

class RAGOrchestrator:
    """Оркестратор мультиагентной RAG системы"""
    
    def __init__(self, config_path: str = "config.yaml"):
        print(f"🔄 RAGOrchestrator.__init__({config_path})")
        
        # Загрузка конфигурации
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        print(f"✅ Конфигурация загружена: {self.config['embedding_model']}")
        
        # Инициализация агентов
        print("🔄 Инициализация агентов...")
        self.agents = {}
        
        try:
            self.agents['parser'] = DocParserAgent()
            print("  ✅ ParserAgent")
        except Exception as e:
            print(f"  ❌ ParserAgent: {e}")
        
        try:
            self.agents['chunker'] = SmartChunkerAgent(
                embedding_model=self.config['embedding_model'],
                chunk_size=self.config['chunk_size'],
                overlap=self.config['overlap_size']
            )
            print("  ✅ ChunkerAgent")
        except Exception as e:
            print(f"  ❌ ChunkerAgent: {e}")
        
        try:
            self.agents['vector'] = VectorAgent(
                embedding_model=self.config['embedding_model'],
                use_gpu=self.config['use_gpu'],
                batch_size=self.config['batch_size'],
                db_path=self.config['vector_db_path']
            )
            print("  ✅ VectorAgent")
        except Exception as e:
            print(f"  ❌ VectorAgent: {e}")
        
        try:
            self.agents['generator'] = AnswerGPTAgent(
                api_base=self.config['lm_studio_url'],
                model=self.config['generation_model'],
                temperature=self.config['temperature'],
                max_tokens=self.config['max_tokens']
            )
            print("  ✅ GeneratorAgent")
        except Exception as e:
            print(f"  ❌ GeneratorAgent: {e}")
        
        try:
            self.agents['validator'] = ValidatorAgent()
            print("  ✅ ValidatorAgent")
        except Exception as e:
            print(f"  ❌ ValidatorAgent: {e}")
        
        self.doc_structure = None
        self.is_indexed = False
        print("✅ RAGOrchestrator инициализирован")
    
    def process_document(self, docx_path: str) -> Dict[str, Any]:
        """
        Полный пайплайн обработки документа
        """
        print(f"\n📄 Начало обработки документа: {docx_path}")
        
        try:
            # 1. ПАРСИНГ - извлекаем структуру
            print("🔍 Парсинг структуры...")
            self.doc_structure = self.agents['parser'].parse_with_hierarchy(docx_path)
            print(f"✅ Найдено глав: {len(self.doc_structure['chapters'])}")
            
            # 2. ЧАНКОВАНИЕ - разбиваем на смысловые фрагменты
            print("✂️ Разделение на чанки...")
            chunks = []
            metadata = []
            
            for chapter_idx, chapter in enumerate(self.doc_structure['chapters']):
                print(f"  Обработка главы {chapter_idx + 1}: {chapter['title'][:50]}...")
                
                # Обработка контента главы
                if chapter.get('content'):
                    print(f"    Контент главы: {len(chapter['content'])} символов")
                    chapter_chunks = self.agents['chunker'].split_by_semantics(chapter['content'])
                    print(f"    Получено чанков из главы: {len(chapter_chunks)}")
                    
                    for chunk in chapter_chunks:
                        chunks.append(chunk)
                        metadata.append({
                            'chapter_id': chapter['id'],
                            'chapter_title': chapter['title'],
                            'level': 1,
                            'type': 'chapter'
                        })
                
                # Обработка разделов внутри главы
                for section_idx, section in enumerate(chapter.get('sections', [])):
                    print(f"    Обработка раздела {section_idx + 1}: {section['title'][:50]}...")
                    
                    if section.get('content'):
                        section_chunks = self.agents['chunker'].split_by_semantics(section['content'])
                        print(f"      Получено чанков из раздела: {len(section_chunks)}")
                        
                        for chunk in section_chunks:
                            chunks.append(chunk)
                            metadata.append({
                                'chapter_id': chapter['id'],
                                'chapter_title': chapter['title'],
                                'section_id': section['id'],
                                'section_title': section['title'],
                                'level': 2,
                                'type': 'section'
                            })
            
            print(f"📊 Всего собрано чанков: {len(chunks)}")
            
            # Проверка на пустые чанки
            if len(chunks) == 0:
                print("⚠️ Внимание: не создано ни одного чанка!")
                # Создаем один общий чанк
                all_text = ""
                for chapter in self.doc_structure['chapters']:
                    if chapter.get('content'):
                        all_text += chapter['content'] + "\n"
                
                if all_text:
                    chunks = [all_text[:self.config['chunk_size']]]
                    metadata = [{
                        'chapter_id': 'all',
                        'chapter_title': 'Весь документ',
                        'level': 0,
                        'type': 'full'
                    }]
                    print(f"✅ Создан один общий чанк")
            
            # 3. ИНДЕКСАЦИЯ - создаем векторный индекс
            print("🔗 Создание векторного индекса...")
            self.agents['vector'].create_index(chunks, metadata)
            self.is_indexed = True
            
            print(f"✅ Документ обработан. Глав: {len(self.doc_structure['chapters'])}, Чанков: {len(chunks)}")
            
            return {
                'structure': self.doc_structure,
                'chunks_count': len(chunks),
                'chapters_count': len(self.doc_structure['chapters'])
            }
            
        except Exception as e:
            print(f"❌ Ошибка при обработке документа: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def query_document(self, question: str, chapter_filter: Optional[str] = None) -> Dict[str, Any]:
        """
        Обработка запроса пользователя
        """
        print(f"\n❓ Вопрос: {question}")
        
        if not self.is_indexed:
            return {
                "error": "Сначала загрузите и обработайте документ",
                "answer": "Документ не загружен. Пожалуйста, сначала загрузите документ."
            }
        
        try:
            # 1. ПОИСК - находим релевантные чанки
            print("🔎 Поиск релевантных чанков...")
            filters = {"chapter_id": chapter_filter} if chapter_filter else None
            chunks = self.agents['vector'].hierarchical_search(question, top_k=5, filters=filters)
            print(f"   Найдено чанков: {len(chunks)}")
            
            if not chunks:
                return {
                    "answer": "По вашему запросу ничего не найдено в документе.",
                    "sources": [],
                    "confidence": 0,
                    "warnings": ["Ничего не найдено"]
                }
            
            # 2. ГЕНЕРАЦИЯ - создаем ответ на основе найденных чанков
            print("🤖 Генерация ответа...")
            answer = self.agents['generator'].generate_answer(question, chunks)
            
            # 3. ВАЛИДАЦИЯ - проверяем качество ответа
            print("✅ Валидация ответа...")
            validated = self.agents['validator'].validate(answer, chunks)
            
            return validated
            
        except Exception as e:
            print(f"❌ Ошибка при обработке запроса: {e}")
            import traceback
            traceback.print_exc()
            return {
                "error": str(e),
                "answer": f"Произошла ошибка при обработке запроса: {str(e)}",
                "sources": [],
                "confidence": 0
            }
    
    def get_document_structure(self) -> Dict:
        """
        Получение структуры документа
        """
        return self.doc_structure
    
    def get_status(self) -> Dict:
        """
        Получение статуса системы
        """
        return {
            "is_indexed": self.is_indexed,
            "has_structure": self.doc_structure is not None,
            "chapters_count": len(self.doc_structure['chapters']) if self.doc_structure else 0,
            "agents": list(self.agents.keys())
        }