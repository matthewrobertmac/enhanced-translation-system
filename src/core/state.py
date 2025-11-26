# src/core/state.py
from typing import TypedDict, Annotated, List, Dict, Optional
import operator

class TranslationState(TypedDict):
    """
    Represents the state of the translation pipeline as it moves through the graph.
    """
    # --- Inputs ---
    source_text: str
    source_language: str
    target_language: str
    target_audience: str
    genre: str
    
    # --- Translation Versions ---
    literal_translation: str
    cultural_adaptation: str
    tone_adjustment: str
    technical_review_version: str
    literary_polish: str
    final_translation: str
    
    # --- Issue Tracking (Feedback) ---
    literal_issues: List[Dict]
    cultural_issues: List[Dict]
    tone_issues: List[Dict]
    technical_issues: List[Dict]
    literary_issues: List[Dict]
    
    # --- Analysis & Metadata ---
    critical_passages: List[Dict]
    agent_notes: Annotated[List[str], operator.add]  # Appends notes from all agents
    agent_decisions: List[Dict]
    human_feedback: Optional[str]
    revision_count: int
    needs_human_review: bool
    
    # --- Entity Tracking ---
    source_entities: Optional[List[Dict]]
    translated_entities: Optional[List[Dict]]
    entity_preservation_rate: Optional[float]
    
    # --- Validation & Scoring ---
    bertscore_attempts: int
    bertscore_history: List[Dict]
    confidence_scores: Optional[Dict[str, float]]
    
    # --- Planning & Routing ---
    agent_plan: List[str]
    agent_plan_reasoning: Dict[str, str]
    planning_analysis: Dict[str, any]
    skipped_agents: List[str]
    estimated_complexity: str
    current_agent_index: int
    planning_enabled: bool
    
    # --- Performance & Caching ---
    cache_hit: Optional[str]
    cache_speedup: Optional[str]
    
    # --- Timestamps & Config ---
    started_at: str
    completed_at: Optional[str]
    thread_id: Optional[str]