# src/config.py

# =====================
# PROMPT GUARDRAILS
# =====================
LANGUAGE_GUARDRAIL = (
    "STRICT LANGUAGE GUARDRAIL:\n"
    "- Reply ONLY in the target language specified.\n"
    "- Do NOT include any words or phrases in other languages.\n"
    "- Any notes, bullets, or headings must also be in the target language.\n"
)

# =====================
# ENTITY CONFIGURATION
# =====================
ENTITY_TYPES = {
    'person': {'emoji': '👤', 'color': '#3b82f6'},
    'location': {'emoji': '📍', 'color': '#10b981'},
    'organization': {'emoji': '🏢', 'color': '#f59e0b'},
    'date': {'emoji': '📅', 'color': '#8b5cf6'},
    'custom': {'emoji': '🏷️', 'color': '#ef4444'}
}

DEFAULT_ENTITY_GLOSSARY = {
    "AI": {
        "type": "custom", 
        "description": "Artificial Intelligence", 
        "aliases": ["A.I.", "artificial intelligence"]
    },
    "LLM": {
        "type": "custom", 
        "description": "Large Language Model", 
        "aliases": ["large language model"]
    },
    "NLP": {
        "type": "custom", 
        "description": "Natural Language Processing", 
        "aliases": ["natural language processing"]
    }
}

# =====================
# AGENT DESCRIPTIONS
# =====================
AGENT_DESCRIPTIONS = {
    "planning": {
        "name": "Workflow Planning Specialist",
        "emoji": "📋",
        "role": "Intelligent Agent Selection & Routing",
        "description": """
        **Primary Responsibility:** Analyze source text and dynamically determine which agents are needed.
        **Key Functions:**
        - Assess text complexity and characteristics
        - Detect technical, cultural, and literary elements
        - Select optimal agent sequence for the task
        - Estimate time and cost savings
        - Provide reasoning for all decisions
        """,
        "expertise": ["Text analysis", "Workflow optimization", "Resource efficiency", "Quality assurance"]
    },
    "literal_translator": {
        "name": "Baseline Specialist",
        "emoji": "🔤",
        "role": "Foundational Accuracy (or faithful baseline in refine mode)",
        "description": """
        **Primary Responsibility:** Provides the baseline pass:
        - Translation mode: create an accurate word-for-word draft.
        - Refine mode (same-language): perform a faithful copy-edit that preserves meaning and structure.
        **Key Functions:**
        - Preserve semantic content and important structure
        - Identify idioms, ambiguities, cultural expressions
        - Preserve technical terms and proper nouns
        - Document challenges for downstream agents
        """,
        "expertise": ["Lexical mapping", "Semantic preservation", "Grammatical structure", "Technical terminology"]
    },
    "cultural_adapter": {
        "name": "Cultural Localization Expert",
        "emoji": "🌍",
        "role": "Cross-Cultural Bridge (or register-localization in refine mode)",
        "description": """
        **Primary Responsibility:** Make content culturally natural in target context.
        - In refine mode, ensure register and references suit the target variety (e.g., US vs UK English).
        """,
        "expertise": ["Localization", "Idiomatic expressions", "Cross-cultural communication", "Audience adaptation"]
    },
    "tone_specialist": {
        "name": "Tone & Voice Consistency Director",
        "emoji": "🎭",
        "role": "Stylistic Harmony & Readability",
        "description": "Ensure consistent tone, voice, pacing, and readability.",
        "expertise": ["Stylistics", "Rhetorical analysis", "Reader psychology", "Narrative voice"]
    },
    "technical_reviewer": {
        "name": "Technical Accuracy Auditor",
        "emoji": "🔬",
        "role": "Precision & Factual Integrity",
        "description": "Verify notation, terminology, measurements, and formats.",
        "expertise": ["Technical writing", "Scientific notation", "Terminology", "Quality assurance"]
    },
    "literary_editor": {
        "name": "Literary Style & Excellence Editor",
        "emoji": "✍️",
        "role": "Publication-Ready Prose Crafting",
        "description": "Elevate prose to publication quality without altering meaning.",
        "expertise": ["Literary criticism", "Creative writing", "Stylistics", "Publishing standards"]
    },
    "finalize": {
        "name": "Master Quality Synthesizer",
        "emoji": "✅",
        "role": "Final Integration & Excellence Assurance",
        "description": "Integrate all contributions and approve the final version.",
        "expertise": ["Editorial judgment", "Synthesis", "Holistic QA"]
    },
    "bertscore_validator": {
        "name": "Semantic Fidelity Validator",
        "emoji": "🎯",
        "role": "Semantic Similarity Assurance (same-language only)",
        "description": """
        **Primary Responsibility:** Ensure semantic fidelity in same-language refinement.
        - Only active when source and target languages are equivalent
        - Validates BERTScore ≥ 0.8 for semantic preservation
        - Iteratively refines output to meet threshold
        **Key Functions:**
        - Compute BERTScore (P/R/F1) against source
        - Identify semantic drift areas
        - Apply targeted refinements without over-editing
        - Preserve meaning while improving clarity
        """,
        "expertise": ["Semantic analysis", "Embedding comparison", "Iterative refinement", "Fidelity validation"]
    }
}