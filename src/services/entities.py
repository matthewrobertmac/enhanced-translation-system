# src/services/entities.py
import re
import json
import io
import csv
from typing import List, Dict, Optional
from src.config import ENTITY_TYPES, DEFAULT_ENTITY_GLOSSARY

class EntitiesTracker:
    """
    Service logic for Entity Extraction and Glossary Management.
    Decoupled from UI logic.
    """
    
    def __init__(self, glossary: Optional[Dict] = None):
        # Load glossary from provided dict or defaults
        self.entity_glossary = glossary if glossary is not None else DEFAULT_ENTITY_GLOSSARY.copy()
        self.entity_types = ENTITY_TYPES

    def extract_entities(self, text: str) -> List[Dict]:
        """Extract entities from text using glossary and NER patterns."""
        if not text:
            return []
        
        entities = []
        
        # 1. Extract from glossary
        for term, info in self.entity_glossary.items():
            pattern = re.compile(r'\b' + re.escape(term) + r'\b', re.IGNORECASE)
            matches = pattern.findall(text)
            
            alias_count = 0
            for alias in info.get('aliases', []):
                alias_pattern = re.compile(r'\b' + re.escape(alias) + r'\b', re.IGNORECASE)
                alias_count += len(alias_pattern.findall(text))
            
            total_count = len(matches) + alias_count
            
            if total_count > 0:
                entities.append({
                    'name': term,
                    'type': info.get('type', 'custom'),
                    'count': total_count,
                    'description': info.get('description', ''),
                    'from_glossary': True
                })
        
        # 2. Basic NER patterns (Auto-detect)
        existing_names = {e['name'] for e in entities}
        auto_entities = self._extract_auto_entities(text, existing_names)
        entities.extend(auto_entities)
        
        return entities
    
    def _extract_auto_entities(self, text: str, existing_names: set) -> List[Dict]:
        auto_entities = []
        
        patterns = {
            'person': r'\b([A-Z][a-z]+ [A-Z][a-z]+)\b',
            'location': r'\b(?:in|at|from|to|near) ([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b',
            'organization': r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Inc|Corp|LLC|Ltd|Company|Group|Foundation))\b',
        }
        date_patterns = [
            r'\b(19|20)\d{2}\b',
            r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b',
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b'
        ]

        # Generic Regex
        for e_type, pattern in patterns.items():
            for match in re.finditer(pattern, text):
                val = match.group(1)
                if val not in existing_names and val not in [e['name'] for e in auto_entities]:
                    count = len(re.findall(r'\b' + re.escape(val) + r'\b', text))
                    auto_entities.append({
                        'name': val, 'type': e_type, 'count': count, 
                        'description': f'Auto-detected {e_type}', 'auto_detected': True
                    })

        # Dates
        for pattern in date_patterns:
            for match in re.finditer(pattern, text):
                date_val = match.group(0)
                if date_val not in existing_names and date_val not in [e['name'] for e in auto_entities]:
                    count = len(re.findall(re.escape(date_val), text))
                    auto_entities.append({
                        'name': date_val, 'type': 'date', 'count': count, 
                        'description': 'Auto-detected date', 'auto_detected': True
                    })
        
        return auto_entities
    
    def parse_glossary_file(self, content: str, file_name: str) -> Dict:
        """Parse glossary file content (JSON/CSV/TXT) and return dict."""
        new_glossary = {}
        try:
            if file_name.endswith('.json'):
                new_glossary = json.loads(content)
            elif file_name.endswith('.csv'):
                csv_data = io.StringIO(content)
                reader = csv.DictReader(csv_data)
                for row in reader:
                    term = row.get('term', '').strip()
                    if term:
                        new_glossary[term] = {
                            'type': row.get('type', 'custom'),
                            'description': row.get('description', ''),
                            'aliases': [a.strip() for a in row.get('aliases', '').split(',') if a.strip()]
                        }
            else: # txt
                lines = content.split('\n')
                for line in lines:
                    term = line.strip()
                    if term:
                        new_glossary[term] = {'type': 'custom', 'description': '', 'aliases': []}
            return new_glossary
        except Exception as e:
            print(f"Glossary parse error: {e}")
            return {}

    def update_glossary(self, new_terms: Dict):
        self.entity_glossary.update(new_terms)