# src/ui/session.py
import streamlit as st
from datetime import datetime
from src.services.cache import SemanticTranslationCache
from src.services.entities import EntitiesTracker

def initialize_session_state():
    """
    Initialize all Streamlit session state variables.
    Ensures singleton instances for services (Cache, Tracker).
    """
    
    # 1. Simple Data Types
    defaults = {
        "translation_state": None,
        "history": [],
        "edited_translation": "",
        "alternatives": [],
        
        # Toggles & Flags
        "enable_entity_tracking": False,
        "enable_entity_awareness": False,
        "planning_enabled": True,
        
        # Force Agents
        "force_cultural_adapter": False,
        "force_tone_specialist": False,
        "force_technical_reviewer": False,
        "force_literary_editor": False,
        "force_bertscore_validator": False,
        
        # Unique Thread ID for LangGraph Checkpointing
        "thread_id": f"thread_{datetime.now().timestamp()}"
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    # 2. Complex Services (Singletons)
    if 'translation_cache' not in st.session_state:
        with st.spinner("Initializing Cache..."):
            st.session_state.translation_cache = SemanticTranslationCache()
            
    if 'entity_tracker' not in st.session_state:
        st.session_state.entity_tracker = EntitiesTracker()