import re
from typing import List, Dict, Any

class ValidatorAgent:
    """Агент для валидации ответов"""
    
    def __init__(self, confidence_threshold=0.7):
        self.confidence_threshold = confidence_threshold
    
    def validate(self, answer: str, source_chunks: List[Dict]) -> Dict[str, Any]:
        """Проверка ответа на соответствие источникам"""
        
        validation_result = {
            'answer': answer,
            'sources': self._extract_sources(source_chunks),
            'confidence': self._calculate_confidence(answer, source_chunks),
            'has_citations': self._check_citations(answer),
            'is_grounded': self._check_grounding(answer, source_chunks),
            'warnings': []
        }
        
        if validation_result['confidence'] < self.confidence_threshold:
            validation_result['warnings'].append(
                "⚠️ Низкая уверенность в ответе, рекомендуется проверить факты"
            )
        
        if not validation_result['has_citations']:
            validation_result['warnings'].append(
                "⚠️ Ответ не содержит ссылок на главы/разделы"
            )
        
        if not validation_result['is_grounded']:
            validation_result['warnings'].append(
                "⚠️ Ответ может содержать информацию вне документа"
            )
        
        return validation_result
    
    def _calculate_confidence(self, answer: str, chunks: List[Dict]) -> float:
        """Вычисление уверенности в ответе"""
        if not chunks:
            return 0.0
        
        answer_len = len(answer)
        context_len = sum(len(chunk['text']) for chunk in chunks)
        
        if context_len == 0:
            return 0.0
        
        confidence = min(1.0, answer_len / (context_len * 0.3))
        return round(confidence, 2)
    
    def _check_citations(self, answer: str) -> bool:
        """Проверка наличия цитирований"""
        patterns = [
            r'глав[аы]\s*\d+',
            r'раздел[аы]?\s*[\d\.]+',
            r'стр\.?\s*\d+',
            r'chapter\s*\d+',
            r'section\s*\d+'
        ]
        
        for pattern in patterns:
            if re.search(pattern, answer, re.IGNORECASE):
                return True
        return False
    
    def _check_grounding(self, answer: str, chunks: List[Dict]) -> bool:
        """Проверка привязки к источнику"""
        if not chunks or not answer:
            return False
        
        context_text = ' '.join([chunk['text'] for chunk in chunks]).lower()
        answer_words = set(answer.lower().split())
        context_words = set(context_text.split())
        
        common_words = answer_words.intersection(context_words)
        
        if len(answer_words) > 0:
            grounding_ratio = len(common_words) / len(answer_words)
            return grounding_ratio > 0.3
        
        return False
    
    def _extract_sources(self, chunks: List[Dict]) -> List[Dict]:
        """Извлечение информации об источниках"""
        sources = []
        for chunk in chunks:
            metadata = chunk.get('metadata', {})
            source = {
                'chapter': metadata.get('chapter_title', 'Неизвестно'),
                'section': metadata.get('section_title', 'Неизвестно'),
                'chunk_id': chunk.get('id')
            }
            if source not in sources:
                sources.append(source)
        return sources