
import os
import json
from typing import List, Dict, Any, Optional

from agents.doc_parser import DocParserAgent
#from agents.smart_chunker import SmartChunkerAgent
from agents.simple_chunker import SimpleChunkerAgent


from agents.vector_agent import VectorAgent
from agents.answer_gpt import AnswerGPTAgent
from agents.validator import ValidatorAgent

class RAGOrchestrator:

    """Оркестратор мультиагентной RAG системы"""
    
    def __init__(self, config_path: str = "config.yaml"):
        import yaml
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        self.config = config
        
        # Инициализация агентов
        self.agents = {
            'parser': DocParserAgent(),
            'chunker': SimpleChunkerAgent(chunk_size=50, overlap=5),
           #  'chunker': SmartChunkerAgent(
           #     embedding_model=config['embedding_model'],
           #     chunk_size=config['chunk_size'],
           #     overlap=config['overlap_size']
           # ),
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
    
    try:
        # 1. Парсинг
        print("🔍 Парсинг структуры...")
        self.doc_structure = self.agents['parser'].parse_with_hierarchy(docx_path)
        print(f"✅ Найдено глав: {len(self.doc_structure['chapters'])}")
        
        # 2. Чанкование
        print("✂️ Разделение на чанки...")
        chunks = []
        metadata = []
        
        for chapter_idx, chapter in enumerate(self.doc_structure['chapters']):
            print(f"  Обработка главы {chapter_idx + 1}: {chapter['title'][:50]}...")
            
            if chapter.get('content'):
                print(f"    Контент главы: {len(chapter['content'])} символов")
                chapter_chunks = self.agents['chunker'].split_by_semantics(chapter['content'])
                print(f"    Получено чанков из главы: {len(chapter_chunks)}")
                
                chunks.extend(chapter_chunks)
                for chunk in chapter_chunks:
                    metadata.append({
                        'chapter_id': chapter['id'],
                        'chapter_title': chapter['title'],
                        'level': 1,
                        'type': 'chapter'
                    })
            
            for section_idx, section in enumerate(chapter.get('sections', [])):
                print(f"    Обработка раздела {section_idx + 1}: {section['title'][:50]}...")
                
                if section.get('content'):
                    section_chunks = self.agents['chunker'].split_by_semantics(section['content'])
                    print(f"      Получено чанков из раздела: {len(section_chunks)}")
                    
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
        
        print(f"📊 Всего собрано чанков: {len(chunks)}")
        
        if len(chunks) == 0:
            print("⚠️ Внимание: не создано ни одного чанка!")
            # Создаем хотя бы один чанк из всего документа
            all_text = ' '.join([chapter.get('content', '') for chapter in self.doc_structure['chapters']])
            if all_text:
                chunks = [all_text]
                metadata = [{'chapter_id': 'all', 'chapter_title': 'Весь документ', 'level': 0, 'type': 'full'}]
                print(f"✅ Создан один общий чанк из {len(all_text)} символов")
        
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
        
    except Exception as e:
        print(f"❌ Ошибка при обработке документа: {e}")
        import traceback
        traceback.print_exc()
        raise