# src/services/scoring.py
import re
from typing import Dict, Optional
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from src.utils.common import languages_equivalent

# BERTScore (Optional)
try:
    from bert_score import score as bert_score_fn
    BERT_AVAILABLE = True
except Exception:
    BERT_AVAILABLE = False

def compute_bertscore(candidate: str, reference: str) -> Optional[Dict[str, float]]:
    """Compute BERTScore (P/R/F1) if package is available."""
    if not BERT_AVAILABLE:
        return None
    try:
        P, R, F1 = bert_score_fn([candidate], [reference], lang="en", rescale_with_baseline=True)
        return {
            "precision": float(P.mean().item()),
            "recall": float(R.mean().item()),
            "f1": float(F1.mean().item()),
        }
    except Exception as e:
        print(f"BERTScore error: {e}")
        return None

class ConfidenceScorer:
    """Score translation confidence using multiple signals."""
    
    def __init__(self, llm: BaseChatModel):
        self.llm = llm
    
    async def score_translation(
        self, 
        source: str, 
        translation: str, 
        context: dict
    ) -> Dict[str, float]:
        """Generate confidence scores."""
        scores = {}
        
        # 1. Semantic Fidelity (BERTScore if same language / embedding logic)
        if languages_equivalent(context.get('source_lang', ''), context.get('target_lang', '')):
            bs = compute_bertscore(translation, source)
            if bs:
                scores['semantic_fidelity'] = bs['f1']
            else:
                scores['semantic_fidelity'] = 0.75
        else:
            scores['semantic_fidelity'] = 0.75 # Default for cross-lingual without embeddings
        
        # 2. Length Ratio
        src_len = len(source.split())
        tgt_len = len(translation.split())
        ratio = tgt_len / src_len if src_len > 0 else 0
        # Ideal ratio is 0.8-1.2
        scores['length_ratio'] = max(0, 1.0 - min(abs(ratio - 1.0), 0.5) * 2)
        
        # 3. Fluency (LLM-based)
        prompt = f"""Rate the fluency of this {context.get('target_lang', 'target')} text (0.0-1.0).
        Text: {translation[:500]}
        Respond ONLY with a number."""
        
        try:
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            val = float(re.findall(r"0\.\d+|1\.0|0|1", response.content)[0])
            scores['fluency'] = min(max(val, 0.0), 1.0)
        except:
            scores['fluency'] = 0.75
            
        # 4. Terminology
        entities = context.get('entities', [])
        if entities:
            preserved = sum(1 for e in entities if e['name'].lower() in translation.lower())
            scores['terminology'] = preserved / len(entities)
        else:
            scores['terminology'] = 0.9
            
        # Weighted Average
        weights = {'semantic_fidelity': 0.35, 'length_ratio': 0.15, 'fluency': 0.35, 'terminology': 0.15}
        scores['overall'] = sum(scores.get(k, 0.5) * w for k, w in weights.items())
        
        return scores