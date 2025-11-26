# src/utils/common.py
import re
import difflib
from typing import List, Dict, Optional, Tuple

def normalize_lang_label(label: str) -> str:
    """
    Normalize a language label like 'English (US)' -> 'english'.
    """
    if not label:
        return ""
    core = label.split("(")[0].strip().lower()
    return core

def languages_equivalent(src: str, tgt: str) -> bool:
    """
    Check if source and target languages are equivalent (e.g. both English).
    """
    return normalize_lang_label(src) == normalize_lang_label(tgt)

def mode_phrase(src: str, tgt: str) -> str:
    """
    Human-readable short descriptor for UI headers.
    """
    if languages_equivalent(src, tgt):
        return f"{tgt} – refine"
    return f"{src} → {tgt}"

def split_notes(content: str, labels: List[str]) -> Tuple[str, Optional[str], Optional[str]]:
    """
    Split content on the first matching label in labels list (e.g., ["TRANSLATOR NOTES:", "EDITOR NOTES:"]).
    Returns (main_text, found_label, notes_text).
    """
    if not content:
        return "", None, None
        
    for lab in labels:
        if lab in content:
            parts = content.split(lab, 1)
            return parts[0].strip(), lab, parts[1].strip()
    return content.strip(), None, None

def text_similarity(a: str, b: str) -> float:
    """
    Calculate sequence matcher ratio between two strings.
    """
    return difflib.SequenceMatcher(None, a or "", b or "").ratio()

def extract_critical_passages(text: str, issues_list: List[Dict]) -> List[Dict]:
    """
    Extract critical passages that need review based on agent feedback keywords.
    """
    critical = []
    keywords = ['critical', 'warning', 'error', 'ambiguous', 'unclear', 'needs review']
    
    if not text:
        return []

    for issue in issues_list:
        content = issue.get('content', '')
        issue_type = issue.get('type', '')
        agent = issue.get('agent', '')
        
        if any(k in content.lower() for k in keywords):
            sentences = text.split('.')
            for i, sentence in enumerate(sentences):
                if len(sentence.strip()) > 10:
                    critical.append({
                        'passage': sentence.strip() + '.',
                        'issue': content[:200],
                        'agent': agent,
                        'type': issue_type,
                        'sentence_index': i
                    })
    
    return critical[:10]

def sentence_lengths(text: str) -> List[int]:
    """
    Return per-sentence word counts for a text.
    """
    if not text:
        return []
    # Split by punctuation followed by whitespace
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [len(s.split()) for s in sentences if s.strip()]

def nonzero_bins(max_len: int) -> List[int]:
    """
    Generate nice 5-word bins for histograms.
    """
    step = 5
    upper = max(5, ((max_len + step - 1) // step) * step)
    return list(range(0, upper + step, step))