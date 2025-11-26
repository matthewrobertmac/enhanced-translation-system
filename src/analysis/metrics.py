# src/analysis/metrics.py
import re
from typing import List

try:
    import textstat
    TEXTSTAT_AVAILABLE = True
except ImportError:
    TEXTSTAT_AVAILABLE = False

def get_readability_score(text: str) -> float:
    """
    Calculate Flesch Reading Ease score.
    Returns -1.0 if textstat is not available or error occurs.
    """
    if not text or not TEXTSTAT_AVAILABLE:
        return -1.0
    try:
        return textstat.flesch_reading_ease(text)
    except Exception:
        return 0.0

def get_word_count(text: str) -> int:
    """Return simple word count."""
    if not text:
        return 0
    return len(text.split())

def get_sentence_lengths(text: str) -> List[int]:
    """
    Return a list of word counts per sentence.
    Used for histograms.
    """
    if not text:
        return []
    # Split by punctuation followed by whitespace
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [len(s.split()) for s in sentences if s.strip()]