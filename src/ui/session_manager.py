import streamlit as st
from datetime import datetime

# Assuming these classes are available in your backend module
from backend import SemanticTranslationCache, EntitiesTracker

def initialize_session_state():
    """
    Initializes all necessary Streamlit session state variables.
    This ensures that keys exist before they are accessed in the UI.
    """
    
    # Define default values for simple types
    defaults = {
        "translation_state": None,
        "history": [],
        "graph": None,
        "edited_translation": "",
        "alternatives": [], # List of alternative translations
        "enable_entity_tracking": False,
        "enable_entity_awareness": False,
        "planning_enabled": True,
        
        # Configuration flags for forcing specific agents
        "force_cultural_adapter": False,
        "force_tone_specialist": False,
        "force_technical_reviewer": False,
        "force_literary_editor": False,
        "force_bertscore_validator": False,
    }

    # Initialize simple defaults
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

    # Initialize complex objects (Classes)
    # 1. Cache System
    if 'translation_cache' not in st.session_state or st.session_state.translation_cache is None:
        st.session_state.translation_cache = SemanticTranslationCache()
        
    # 2. Entity Tracker
    if 'entity_tracker' not in st.session_state or st.session_state.entity_tracker is None:
        st.session_state.entity_tracker = EntitiesTracker()