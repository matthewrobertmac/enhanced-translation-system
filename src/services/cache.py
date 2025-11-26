# src/services/cache.py
import os
import json
import pickle
import hashlib
import numpy as np
from datetime import datetime
from typing import Optional, Dict

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

class SemanticTranslationCache:
    """
    Cache translations with semantic similarity-based retrieval.
    Stores exact matches, sentence fragments, and embeddings.
    """
    
    def __init__(self, cache_dir: str = ".translation_cache"):
        self.cache_dir = cache_dir
        self.cache = {}  # Full translation cache
        self.sentence_cache = {}  # Sentence-level cache
        self.embeddings = {}  # Store embeddings for similarity search
        self.similarity_threshold = 0.85
        self.embedding_model = None
        self.stats = {
            'hits': 0,
            'misses': 0,
            'partial_hits': 0,
            'similar_hits': 0
        }
        
        os.makedirs(cache_dir, exist_ok=True)
        self.load_cache()
    
    def get_embedding_model(self):
        """Lazy load embedding model."""
        if self.embedding_model is None and SENTENCE_TRANSFORMERS_AVAILABLE:
            # Using a lightweight model for speed
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        return self.embedding_model
    
    def get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get or compute embedding for text."""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            return None
        
        text_hash = self.hash_text(text)
        
        if text_hash not in self.embeddings:
            model = self.get_embedding_model()
            if model:
                self.embeddings[text_hash] = model.encode(text)
        
        return self.embeddings.get(text_hash)
    
    def hash_text(self, text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()
    
    def make_cache_key(self, text: str, context: dict) -> str:
        """Create a unique key combining text hash and context hash."""
        context_str = f"{context.get('source_lang')}_{context.get('target_lang')}_{context.get('audience')}_{context.get('genre')}"
        text_hash = self.hash_text(text)
        return f"{text_hash}_{hashlib.sha256(context_str.encode()).hexdigest()}"
    
    def context_matches(self, cached_context: dict, query_context: dict) -> bool:
        return (
            cached_context.get('source_lang') == query_context.get('source_lang') and
            cached_context.get('target_lang') == query_context.get('target_lang') and
            cached_context.get('audience') == query_context.get('audience') and
            cached_context.get('genre') == query_context.get('genre')
        )
    
    async def get_cached_translation(self, text: str, context: dict) -> Optional[Dict]:
        cache_key = self.make_cache_key(text, context)
        
        # 1. EXACT MATCH
        if cache_key in self.cache:
            self.stats['hits'] += 1
            cached = self.cache[cache_key]
            return {
                'type': 'exact',
                'translation': cached['translation'],
                'speedup': '100x'
            }
        
        # 2. SIMILAR MATCH
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            similar = await self.find_similar_translation(text, context)
            if similar:
                self.stats['similar_hits'] += 1
                return {
                    'type': 'similar',
                    'translation': similar['translation'],
                    'similarity': similar['similarity'],
                    'speedup': '50x'
                }
        
        self.stats['misses'] += 1
        return None

    async def find_similar_translation(self, text: str, context: dict) -> Optional[Dict]:
        if not SENTENCE_TRANSFORMERS_AVAILABLE: return None
        
        query_embedding = self.get_embedding(text)
        if query_embedding is None: return None
        
        best_match = None
        best_similarity = 0.0
        
        for _, cached_data in self.cache.items():
            if not self.context_matches(cached_data['context'], context):
                continue
            
            cached_text = cached_data['source_text']
            cached_embedding = self.get_embedding(cached_text)
            
            if cached_embedding is not None:
                # Cosine similarity
                similarity = float(np.dot(query_embedding, cached_embedding) / 
                                  (np.linalg.norm(query_embedding) * np.linalg.norm(cached_embedding)))
                
                if similarity > self.similarity_threshold and similarity > best_similarity:
                    best_similarity = similarity
                    best_match = {
                        'translation': cached_data['translation'],
                        'similarity': similarity
                    }
        return best_match

    def store_translation(self, text: str, translation: str, context: dict, metadata: Optional[dict] = None):
        cache_key = self.make_cache_key(text, context)
        self.cache[cache_key] = {
            'source_text': text,
            'translation': translation,
            'context': context,
            'metadata': metadata or {},
            'timestamp': datetime.now().isoformat()
        }
        # Auto-save periodically
        if len(self.cache) % 10 == 0:
            self.save_cache()

    def save_cache(self):
        try:
            with open(os.path.join(self.cache_dir, 'translations.pkl'), 'wb') as f:
                pickle.dump(self.cache, f)
            with open(os.path.join(self.cache_dir, 'embeddings.pkl'), 'wb') as f:
                pickle.dump(self.embeddings, f)
            with open(os.path.join(self.cache_dir, 'stats.json'), 'w') as f:
                json.dump(self.stats, f, indent=2)
            return True
        except Exception as e:
            print(f"Cache save failed: {e}")
            return False

    def load_cache(self):
        try:
            cache_file = os.path.join(self.cache_dir, 'translations.pkl')
            if os.path.exists(cache_file):
                with open(cache_file, 'rb') as f:
                    self.cache = pickle.load(f)
            
            emb_file = os.path.join(self.cache_dir, 'embeddings.pkl')
            if os.path.exists(emb_file):
                with open(emb_file, 'rb') as f:
                    self.embeddings = pickle.load(f)
                    
            stats_file = os.path.join(self.cache_dir, 'stats.json')
            if os.path.exists(stats_file):
                with open(stats_file, 'r') as f:
                    self.stats = json.load(f)
        except Exception as e:
            print(f"Cache load failed: {e}")

    def clear_cache(self):
        self.cache = {}
        self.embeddings = {}
        self.stats = {'hits': 0, 'misses': 0, 'partial_hits': 0, 'similar_hits': 0}
        for f in ['translations.pkl', 'embeddings.pkl', 'stats.json']:
            path = os.path.join(self.cache_dir, f)
            if os.path.exists(path): os.remove(path)
    
    def get_stats(self) -> dict:
        total = sum(self.stats.values())
        hit_rate = (total - self.stats['misses']) / max(total, 1) * 100
        return {**self.stats, 'total_entries': len(self.cache), 'hit_rate': hit_rate}
    
    def export_cache(self) -> dict:
        return {'cache': self.cache, 'stats': self.stats}
        
    def import_cache(self, data: dict):
        self.cache.update(data.get('cache', {}))
        self.save_cache()