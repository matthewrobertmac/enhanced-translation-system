"""
Enhanced Multi-Agent Translation Workflow with LangGraph and LangSmith
A sophisticated translation system with cultural adaptation, literary editing, comprehensive monitoring, and visuals.

Features:
- 7 specialized translation agents with distinct roles (including BERTScore validator)
- Support for OpenAI and Anthropic models
- Optional LangSmith tracing and monitoring
- Comprehensive agent feedback system
- File upload support (txt, docx, md)
- Multiple export formats (txt, docx, md)
- Critical passage flagging and review
- Safe same-language (e.g., English→English) refinement mode
- BERTScore validation with iterative refinement
- Visualizations: word counts, sentence-length histograms, readability, issue counts, BERTScore bars
- Word clouds: Source, Final, and Difference (words added)
- Entity tracking and network visualization
"""

import streamlit as st
from typing import TypedDict, Annotated, List, Dict, Optional, Literal
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.language_models import BaseChatModel
import operator
from datetime import datetime
import json
import os
import io
import traceback
import re
import collections

# === New: viz imports ===
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# === New: wordcloud (optional) ===
try:
    from wordcloud import WordCloud, STOPWORDS
    WORDCLOUD_AVAILABLE = True
except Exception:
    WORDCLOUD_AVAILABLE = False

# === Entity tracking imports (NEW) ===
try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# ===== Language Guardrail =====
LANGUAGE_GUARDRAIL = (
    "STRICT LANGUAGE GUARDRAIL:\n"
    "- Reply ONLY in the target language specified.\n"
    "- Do NOT include any words or phrases in other languages.\n"
    "- Any notes, bullets, or headings must also be in the target language.\n"
)

# =====================
# ENTITY TRACKER CLASS (NEW - ADDED FOR ENHANCED FUNCTIONALITY)
# =====================
import csv
from collections import Counter, defaultdict

class EntitiesTracker:
    """Entity tracking and visualization system with network graphs"""
    
    def __init__(self):
        self.entity_types = {
            'person': {'emoji': '👤', 'color': '#3b82f6'},
            'location': {'emoji': '📍', 'color': '#10b981'},
            'organization': {'emoji': '🏢', 'color': '#f59e0b'},
            'date': {'emoji': '📅', 'color': '#8b5cf6'},
            'custom': {'emoji': '🏷️', 'color': '#ef4444'}
        }
        
        if 'entity_glossary' not in st.session_state:
            st.session_state.entity_glossary = {}
        
        if 'extracted_entities' not in st.session_state:
            st.session_state.extracted_entities = []
    
    def extract_entities(self, text: str) -> List[Dict]:
        """Extract entities from text using glossary and NER patterns"""
        if not text:
            return []
        
        entities = []
        
        # Extract from glossary
        for term, info in st.session_state.entity_glossary.items():
            pattern = re.compile(r'\b' + re.escape(term) + r'\b', re.IGNORECASE)
            matches = pattern.findall(text)
            
            # Check aliases
            alias_count = 0
            for alias in info.get('aliases', []):
                alias_pattern = re.compile(r'\b' + re.escape(alias) + r'\b', re.IGNORECASE)
                alias_matches = alias_pattern.findall(text)
                alias_count += len(alias_matches)
            
            total_count = len(matches) + alias_count
            
            if total_count > 0:
                entities.append({
                    'name': term,
                    'type': info.get('type', 'custom'),
                    'count': total_count,
                    'description': info.get('description', ''),
                    'from_glossary': True
                })
        
        # Auto-detection patterns
        # Person names
        person_pattern = r'\b([A-Z][a-z]+ [A-Z][a-z]+)\b'
        for match in re.finditer(person_pattern, text):
            name = match.group(1)
            if name not in [e['name'] for e in entities]:
                count = len(re.findall(r'\b' + re.escape(name) + r'\b', text))
                entities.append({
                    'name': name,
                    'type': 'person',
                    'count': count,
                    'description': 'Auto-detected',
                    'auto_detected': True
                })
        
        # Organizations
        org_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Inc|Corp|LLC|Ltd|Company|Group|Foundation))\b'
        for match in re.finditer(org_pattern, text):
            org = match.group(1)
            if org not in [e['name'] for e in entities]:
                count = len(re.findall(r'\b' + re.escape(org) + r'\b', text))
                entities.append({
                    'name': org,
                    'type': 'organization',
                    'count': count,
                    'description': 'Auto-detected',
                    'auto_detected': True
                })
        
        return entities
    
    def upload_glossary(self, file) -> bool:
        """Process uploaded glossary file"""
        try:
            content = file.read()
            
            if file.name.endswith('.json'):
                new_glossary = json.loads(content)
            elif file.name.endswith('.csv'):
                csv_data = io.StringIO(content.decode('utf-8'))
                reader = csv.DictReader(csv_data)
                new_glossary = {}
                for row in reader:
                    term = row.get('term', '').strip()
                    if term:
                        new_glossary[term] = {
                            'type': row.get('type', 'custom'),
                            'description': row.get('description', ''),
                            'aliases': [a.strip() for a in row.get('aliases', '').split(',') if a.strip()]
                        }
            else:
                lines = content.decode('utf-8').split('\n')
                new_glossary = {}
                for line in lines:
                    term = line.strip()
                    if term:
                        new_glossary[term] = {'type': 'custom', 'description': '', 'aliases': []}
            
            st.session_state.entity_glossary.update(new_glossary)
            return True
        except Exception as e:
            st.error(f"Error processing glossary: {str(e)}")
            return False
    
    def visualize_network(self, entities: List[Dict]):
        """Create network visualization using plotly"""
        if not NETWORKX_AVAILABLE or not PLOTLY_AVAILABLE:
            st.warning("Install networkx and plotly for network visualization")
            return
        
        # Create graph
        G = nx.Graph()
        
        # Add nodes
        for entity in entities:
            G.add_node(entity['name'], 
                      type=entity['type'],
                      count=entity['count'],
                      description=entity.get('description', ''))
        
        # Add edges (simplified co-occurrence)
        for i, e1 in enumerate(entities):
            for e2 in entities[i+1:]:
                weight = min(e1['count'], e2['count'])
                G.add_edge(e1['name'], e2['name'], weight=weight)
        
        # Create layout
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        # Create edge traces
        edge_traces = []
        for edge in G.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            trace = go.Scatter(x=[x0, x1, None], y=[y0, y1, None],
                             mode='lines',
                             line=dict(width=0.5, color='#888'),
                             hoverinfo='none')
            edge_traces.append(trace)
        
        # Create node trace
        node_x = []
        node_y = []
        node_text = []
        node_color = []
        node_size = []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            entity = next(e for e in entities if e['name'] == node)
            node_text.append(f"{node}<br>Type: {entity['type']}<br>Count: {entity['count']}")
            node_color.append(self.entity_types[entity['type']]['color'])
            node_size.append(10 + min(entity['count'] * 3, 50))
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            text=[n for n in G.nodes()],
            textposition="top center",
            hoverinfo='text',
            hovertext=node_text,
            marker=dict(
                size=node_size,
                color=node_color,
                line=dict(width=2, color='white')
            )
        )
        
        # Create figure
        fig = go.Figure(data=edge_traces + [node_trace],
                       layout=go.Layout(
                           title='Entity Network',
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=0,l=0,r=0,t=40),
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           height=600
                       ))
        
        st.plotly_chart(fig, use_container_width=True)

# ---- Helpers for same-language paths ----
def normalize_lang_label(label: str) -> str:
    """Normalize a language label like 'English (US)' -> 'english'."""
    core = label.split("(")[0].strip().lower()
    return core

def languages_equivalent(src: str, tgt: str) -> bool:
    return normalize_lang_label(src) == normalize_lang_label(tgt)

def mode_phrase(src: str, tgt: str) -> str:
    """Human-readable short descriptor for UI headers."""
    if languages_equivalent(src, tgt):
        return f"{tgt} – refine"
    return f"{src} → {tgt}"

def split_notes(content: str, labels: List[str]) -> (str, Optional[str], Optional[str]):
    """
    Split content on the first matching label in labels list (e.g., ["TRANSLATOR NOTES:", "EDITOR NOTES:"]).
    Returns (main_text, found_label, notes_text).
    """
    for lab in labels:
        if lab in content:
            parts = content.split(lab, 1)
            return parts[0].strip(), lab, parts[1].strip()
    return content.strip(), None, None

# LangSmith integration (optional)
try:
    from langsmith import Client
    LANGSMITH_AVAILABLE = True
except ImportError:
    LANGSMITH_AVAILABLE = False

# Model imports
try:
    from langchain_openai import ChatOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from langchain_anthropic import ChatAnthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

# Document processing
try:
    from docx import Document
    from docx.shared import Pt, RGBColor
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

# BERTScore (optional; only used in same-language mode)
try:
    from bert_score import score as bert_score_fn
    BERT_AVAILABLE = True
except Exception:
    BERT_AVAILABLE = False


# =====================
# AGENT ROLE DESCRIPTIONS
# =====================
AGENT_DESCRIPTIONS = {
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
    "quality_controller": {
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
        - Maximum 3 refinement attempts

        **Key Functions:**
        - Compute BERTScore (P/R/F1) against source
        - Identify semantic drift areas
        - Apply targeted refinements without over-editing
        - Preserve meaning while improving clarity
        """,
        "expertise": ["Semantic analysis", "Embedding comparison", "Iterative refinement", "Fidelity validation"]
    }
}


# =====================
# State Definition
# =====================
class TranslationState(TypedDict):
    source_text: str
    source_language: str
    target_language: str
    target_audience: str
    genre: str

    # Versions
    literal_translation: str
    cultural_adaptation: str
    tone_adjustment: str
    technical_review_version: str
    literary_polish: str
    final_translation: str

    # Issues
    literal_issues: List[Dict]
    cultural_issues: List[Dict]
    tone_issues: List[Dict]
    technical_issues: List[Dict]
    literary_issues: List[Dict]

    # Critical passages
    critical_passages: List[Dict]

    # Workflow tracking
    agent_notes: Annotated[List[str], operator.add]
    agent_decisions: List[Dict]
    human_feedback: Optional[str]
    revision_count: int
    needs_human_review: bool

    # Entity tracking (NEW - Optional)
    source_entities: Optional[List[Dict]]
    translated_entities: Optional[List[Dict]]
    entity_preservation_rate: Optional[float]

    # BERTScore tracking (NEW)
    bertscore_attempts: int
    bertscore_history: List[Dict]

    # Metadata
    started_at: str
    completed_at: Optional[str]


# =====================
# File Processing Utilities
# =====================

def read_uploaded_file(uploaded_file) -> str:
    """Read content from uploaded file"""
    try:
        file_extension = uploaded_file.name.split('.')[-1].lower()
        if file_extension in ('txt', 'md'):
            return uploaded_file.read().decode('utf-8')
        elif file_extension == 'docx':
            if not DOCX_AVAILABLE:
                st.error("python-docx not installed. Install with: pip install python-docx")
                return ""
            doc = Document(uploaded_file)
            return '\n\n'.join([p.text for p in doc.paragraphs if p.text.strip()])
        else:
            st.error(f"Unsupported file type: {file_extension}")
            return ""
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        return ""


def create_docx_file(text: str, title: str = "Translation") -> io.BytesIO:
    """Create a formatted Word document"""
    if not DOCX_AVAILABLE:
        st.error("python-docx not installed. Install with: pip install python-docx")
        return None

    doc = Document()
    title_paragraph = doc.add_heading(title, level=1)
    title_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER

    timestamp = doc.add_paragraph()
    timestamp.add_run(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}").italic = True
    timestamp.alignment = WD_ALIGN_PARAGRAPH.CENTER

    doc.add_paragraph()
    for para in text.split('\n\n'):
        if para.strip():
            p = doc.add_paragraph(para.strip())
            p.style = 'Normal'
            for run in p.runs:
                run.font.name = 'Calibri'
                run.font.size = Pt(11)

    file_stream = io.BytesIO()
    doc.save(file_stream)
    file_stream.seek(0)
    return file_stream


def create_markdown_file(text: str, title: str = "Translation") -> str:
    return f"""# {title}

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

{text}
"""


def extract_critical_passages(text: str, issues_list: List[Dict]) -> List[Dict]:
    """Extract critical passages that need review based on agent feedback"""
    critical = []
    for issue in issues_list:
        content = issue.get('content', '')
        issue_type = issue.get('type', '')
        agent = issue.get('agent', '')
        if any(k in content.lower() for k in ['critical', 'warning', 'error', 'ambiguous', 'unclear', 'needs review']):
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


# =====================
# Optional: Language Reinforcement
# =====================

def reinforce_language(llm: BaseChatModel, text: str, target_language: str) -> str:
    """Lightweight fixer to ensure output remains in the target language only."""
    try:
        resp = llm.invoke([
            SystemMessage(content="Ensure the following text is in the specified language only, without code-switching."),
            HumanMessage(content=f"{LANGUAGE_GUARDRAIL}\n\nTarget language: {target_language}\n\nText:\n{text}\n\nReturn the corrected text in {target_language}, with no extra commentary.")
        ])
        return resp.content.strip()
    except Exception:
        return text


# =====================
# BERTScore Utility (same-language only)
# =====================

def compute_bertscore(candidate: str, reference: str) -> Optional[Dict[str, float]]:
    """Compute BERTScore (P/R/F1) if package is available; returns None otherwise."""
    if not BERT_AVAILABLE:
        return None
    try:
        # bert-score expects lists of strings
        P, R, F1 = bert_score_fn([candidate], [reference], lang="en", rescale_with_baseline=True)
        return {
            "precision": float(P.mean().item()),
            "recall": float(R.mean().item()),
            "f1": float(F1.mean().item()),
        }
    except Exception:
        return None


# =====================
# Visualization helpers
# =====================

def sentence_lengths(text: str) -> List[int]:
    """Return per-sentence word counts for a text."""
    if not text:
        return []
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [len(s.split()) for s in sentences if s.strip()]

def nonzero_bins(max_len: int) -> List[int]:
    """Nice 5-word bins for histograms."""
    step = 5
    upper = max(5, ((max_len + step - 1) // step) * step)
    return list(range(0, upper + step, step))

# --- New: word frequency + wordcloud helpers ---
_WORD_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ']+")

def tokenize_words(text: str) -> List[str]:
    if not text:
        return []
    return [w.lower() for w in _WORD_RE.findall(text)]

def word_frequencies(text: str, stopwords: Optional[set] = None) -> Dict[str, int]:
    tokens = tokenize_words(text)
    if stopwords is None:
        stopwords = set()
    cnt = collections.Counter(w for w in tokens if w not in stopwords and len(w) > 1)
    return dict(cnt)

def render_wordcloud_from_freq(freqs: Dict[str, int], title: str):
    if not freqs:
        st.info(f"No tokens to display for **{title}**.")
        return
    wc = WordCloud(
        width=1000,
        height=500,
        background_color="white",
        collocations=False
    ).generate_from_frequencies(freqs)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(title)
    st.pyplot(fig)


# =====================
# Enhanced Agent Definitions (translation & refine-aware)
# =====================

class LiteralTranslationAgent:
    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        self.name = "Baseline Specialist"
        self.emoji = "🔤"

    def translate(self, state: TranslationState) -> TranslationState:
        same_lang = languages_equivalent(state['source_language'], state['target_language'])

        system_prompt = (
            "You provide the baseline pass:\n"
            "- Translation mode: accurate, faithful literal translation.\n"
            "- Refine mode: faithful copy-edit that preserves meaning and structure.\n\n"
            + LANGUAGE_GUARDRAIL
        )

        if same_lang:
            user_prompt = f"""Refine the following text in {state['target_language']} without changing its meaning.
{LANGUAGE_GUARDRAIL}

**GOAL (REFINE MODE):**
- Preserve semantics, improve clarity and microstructure
- Flag idioms/ambiguities for downstream agents
- Keep terminology and proper nouns intact
- OUTPUT FORMAT: Provide the refined text first, then add "EDITOR NOTES:" (notes must be in {state['target_language']}).

**TEXT TO REFINE:**
{state['source_text']}

**AUDIENCE**: {state['target_audience']}
**GENRE**: {state.get('genre', 'General')}
"""
            
            # Add entity awareness if enabled (NEW)
            if 'enable_entity_awareness' in st.session_state and st.session_state.enable_entity_awareness:
                if state.get('source_entities'):
                    entity_list = "\n".join([f"- {e['name']} ({e['type']})" for e in state['source_entities'][:15]])
                    user_prompt += f"\n\n**IMPORTANT ENTITIES TO PRESERVE:**\n{entity_list}\n"
        else:
            user_prompt = f"""Translate the following text from {state['source_language']} to {state['target_language']}.
{LANGUAGE_GUARDRAIL}

**CRITICAL INSTRUCTIONS:**
1. Translate with maximum fidelity to the original meaning.
2. Maintain sentence structure initially.
3. Flag idioms/ambiguities for downstream agents.
4. OUTPUT FORMAT: Provide the literal translation first, then "TRANSLATOR NOTES:" (notes must be in {state['target_language']}).

**SOURCE TEXT:**
{state['source_text']}

**TARGET AUDIENCE**: {state['target_audience']}
**GENRE**: {state.get('genre', 'General')}
"""
            
            # Add entity awareness if enabled (NEW)
            if 'enable_entity_awareness' in st.session_state and st.session_state.enable_entity_awareness:
                if state.get('source_entities'):
                    entity_list = "\n".join([f"- {e['name']} ({e['type']})" for e in state['source_entities'][:15]])
                    user_prompt += f"\n\n**IMPORTANT ENTITIES TO PRESERVE:**\n{entity_list}\n"

        try:
            response = self.llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])
            content = response.content

            # Parse notes (supports both translator and editor labels)
            translation, found_label, notes = split_notes(content, ["TRANSLATOR NOTES:", "EDITOR NOTES:"])
            issues = []
            if notes:
                issues.append({
                    "agent": self.name,
                    "type": (found_label[:-1].lower().replace(" ", "_") if found_label else "notes"),
                    "content": notes
                })

            translation = reinforce_language(self.llm, translation, state['target_language'])
            state['literal_translation'] = translation
            state['literal_issues'] = issues
            state['agent_notes'].append(f"{self.emoji} {self.name}: Baseline {'refinement' if same_lang else 'literal translation'} complete")
        except Exception as e:
            st.error(f"Error in Literal step: {str(e)}")
            state['literal_translation'] = state['source_text']
            state['literal_issues'] = [{"agent": self.name, "type": "error", "content": str(e)}]
            state['agent_notes'].append(f"{self.emoji} {self.name}: Error - using source text")
        return state


class CulturalAdaptationAgent:
    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        self.name = "Cultural Localization Expert"
        self.emoji = "🌍"

    def adapt(self, state: TranslationState) -> TranslationState:
        same_lang = languages_equivalent(state['source_language'], state['target_language'])

        system_prompt = "You adapt content for cultural/register naturalness in the target context.\n\n" + LANGUAGE_GUARDRAIL

        if same_lang:
            task_text = (
                "Refine for the target variety's norms (e.g., spelling, idioms, punctuation, register). "
                "Replace region-specific expressions with appropriate target-variety equivalents."
            )
            notes_label = "EDITOR NOTES:"
        else:
            task_text = (
                "Replace source-culture idioms with target equivalents, adapt references, "
                "and adjust communication style for the target audience."
            )
            notes_label = "CULTURAL NOTES:"

        user_prompt = f"""Work on the following text.

{LANGUAGE_GUARDRAIL}

**TARGET CONTEXT**: {state['target_language']} / Audience: {state['target_audience']}
**TEXT:**
{state['literal_translation']}

**YOUR TASKS:**
- {task_text}

**OUTPUT**: Provide the adapted text, then {notes_label} (notes must be in {state['target_language']})."""

        try:
            response = self.llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])
            content = response.content

            adapted, found_label, notes = split_notes(content, ["CULTURAL NOTES:", "EDITOR NOTES:"])
            issues = []
            if notes:
                issues.append({
                    "agent": self.name,
                    "type": (found_label[:-1].lower().replace(" ", "_") if found_label else "notes"),
                    "content": notes
                })

            adapted = reinforce_language(self.llm, adapted, state['target_language'])
            state['cultural_adaptation'] = adapted
            state['cultural_issues'] = issues
            state['agent_notes'].append(f"{self.emoji} {self.name}: {'Register' if same_lang else 'Cultural'} adaptation complete")
        except Exception as e:
            st.error(f"Error in Cultural step: {str(e)}")
            state['cultural_adaptation'] = state['literal_translation']
            state['cultural_issues'] = [{"agent": self.name, "type": "error", "content": str(e)}]
            state['agent_notes'].append(f"{self.emoji} {self.name}: Error - using previous version")
        return state


class ToneConsistencyAgent:
    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        self.name = "Tone & Voice Consistency Director"
        self.emoji = "🎭"

    def adjust_tone(self, state: TranslationState) -> TranslationState:
        system_prompt = "You ensure tone/voice consistency and natural readability.\n\n" + LANGUAGE_GUARDRAIL
        user_prompt = f"""Adjust this text for tone consistency and readability.

{LANGUAGE_GUARDRAIL}

**AUDIENCE**: {state['target_audience']}
**TEXT:**
{state['cultural_adaptation']}

**YOUR TASKS:**
1. Vary sentence length for rhythm
2. Match formality to audience
3. Ensure consistent voice
4. Optimize readability

**OUTPUT**: Provide the adjusted text, then "TONE NOTES:" (notes must be in {state['target_language']})."""

        try:
            response = self.llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])
            content = response.content
            adjusted, found_label, notes = split_notes(content, ["TONE NOTES:"])

            issues = []
            if notes:
                issues.append({"agent": self.name, "type": "tone_notes", "content": notes})

            adjusted = reinforce_language(self.llm, adjusted, state['target_language'])
            state['tone_adjustment'] = adjusted
            state['tone_issues'] = issues
            state['agent_notes'].append(f"{self.emoji} {self.name}: Tone/readability optimized")
        except Exception as e:
            st.error(f"Error in Tone step: {str(e)}")
            state['tone_adjustment'] = state['cultural_adaptation']
            state['tone_issues'] = [{"agent": self.name, "type": "error", "content": str(e)}]
            state['agent_notes'].append(f"{self.emoji} {self.name}: Error - using previous version")
        return state


class TechnicalReviewAgent:
    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        self.name = "Technical Accuracy Auditor"
        self.emoji = "🔬"

    def review(self, state: TranslationState) -> TranslationState:
        system_prompt = "You verify notation, terminology, units, and formatting.\n\n" + LANGUAGE_GUARDRAIL
        user_prompt = f"""Review the following for technical accuracy.

{LANGUAGE_GUARDRAIL}

**TEXT:**
{state['tone_adjustment']}

**YOUR TASKS:**
1. Verify notation/symbols
2. Check terminology
3. Validate measurements/units
4. Ensure number/date/time formatting

**OUTPUT**: Provide the reviewed text, then "TECHNICAL NOTES:" if corrections made (notes must be in {state['target_language']})."""

        try:
            response = self.llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])
            content = response.content
            reviewed, found_label, notes = split_notes(content, ["TECHNICAL NOTES:"])

            issues = []
            if notes:
                issues.append({"agent": self.name, "type": "technical_notes", "content": notes})

            needs_review = "NEEDS_REVIEW" in content or "NEEDS REVIEW" in content

            reviewed = reinforce_language(self.llm, reviewed, state['target_language'])
            state['technical_review_version'] = reviewed
            state['technical_issues'] = issues
            state['needs_human_review'] = needs_review
            state['agent_notes'].append(f"{self.emoji} {self.name}: Technical review completed")
        except Exception as e:
            st.error(f"Error in Technical step: {str(e)}")
            state['technical_review_version'] = state['tone_adjustment']
            state['technical_issues'] = [{"agent": self.name, "type": "error", "content": str(e)}]
            state['needs_human_review'] = False
            state['agent_notes'].append(f"{self.emoji} {self.name}: Error - using previous version")
        return state


class LiteraryEditorAgent:
    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        self.name = "Literary Style & Excellence Editor"
        self.emoji = "✍️"

    def polish(self, state: TranslationState) -> TranslationState:
        system_prompt = "You elevate prose to publication quality without altering meaning.\n\n" + LANGUAGE_GUARDRAIL
        user_prompt = f"""Polish this text to publication quality.

{LANGUAGE_GUARDRAIL}

**TEXT:**
{state['technical_review_version']}
**AUDIENCE**: {state['target_audience']}

**YOUR TASKS:**
- Eliminate awkward phrasing
- Enhance word choice
- Optimize prose rhythm
- Strengthen imagery
- Maintain meaning

**OUTPUT**: Provide the polished text, then "LITERARY NOTES:" (notes must be in {state['target_language']})."""

        try:
            response = self.llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])
            content = response.content

            polished, found_label, notes = split_notes(content, ["LITERARY NOTES:"])
            issues = []
            if notes:
                issues.append({"agent": self.name, "type": "literary_notes", "content": notes})

            polished = reinforce_language(self.llm, polished, state['target_language'])
            state['literary_polish'] = polished
            state['literary_issues'] = issues
            state['agent_notes'].append(f"{self.emoji} {self.name}: Literary polish completed")
        except Exception as e:
            st.error(f"Error in Literary step: {str(e)}")
            state['literary_polish'] = state['technical_review_version']
            state['literary_issues'] = [{"agent": self.name, "type": "error", "content": str(e)}]
            state['agent_notes'].append(f"{self.emoji} {self.name}: Error - using previous version")
        return state


class QualityControlAgent:
    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        self.name = "Master Quality Synthesizer"
        self.emoji = "✅"

    def finalize(self, state: TranslationState) -> TranslationState:
        system_prompt = "You integrate all contributions and output the final publication-ready text.\n\n" + LANGUAGE_GUARDRAIL

        all_issues = (
            state.get('literal_issues', []) +
            state.get('cultural_issues', []) +
            state.get('tone_issues', []) +
            state.get('technical_issues', []) +
            state.get('literary_issues', [])
        )

        user_prompt = f"""Produce the final, publication-ready text.

{LANGUAGE_GUARDRAIL}

**LATEST VERSION:**
{state['literary_polish']}

**OUTPUT**: Provide ONLY the final text without meta-commentary, entirely in {state['target_language']}.
"""

        try:
            response = self.llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])
            content = reinforce_language(self.llm, response.content.strip(), state['target_language'])

            critical_passages = extract_critical_passages(content, all_issues)
            state['final_translation'] = content
            state['completed_at'] = datetime.now().isoformat()
            state['critical_passages'] = critical_passages
            state['agent_notes'].append(f"{self.emoji} {self.name}: Final output approved")
            
            # Extract entities from final translation if tracking is enabled (NEW)
            if 'enable_entity_tracking' in st.session_state and st.session_state.enable_entity_tracking:
                entity_tracker = st.session_state.entity_tracker
                state['translated_entities'] = entity_tracker.extract_entities(content)
                
                # Calculate preservation rate
                if state.get('source_entities'):
                    source_names = {e['name'].lower() for e in state['source_entities']}
                    translated_names = {e['name'].lower() for e in state['translated_entities']}
                    if source_names:
                        state['entity_preservation_rate'] = len(source_names & translated_names) / len(source_names)
        except Exception as e:
            st.error(f"Error in Finalize step: {str(e)}")
            state['final_translation'] = state['literary_polish']
            state['completed_at'] = datetime.now().isoformat()
            state['critical_passages'] = []
            state['agent_notes'].append(f"{self.emoji} {self.name}: Error - using literary version")
        return state


class BERTScoreValidatorAgent:
    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        self.name = "Semantic Fidelity Validator"
        self.emoji = "🎯"
        self.max_attempts = 3
        self.target_score = 0.8

    def validate_and_refine(self, state: TranslationState) -> TranslationState:
        """Validate BERTScore and refine if needed (same-language mode only)"""
        
        # Only run in same-language mode
        if not languages_equivalent(state['source_language'], state['target_language']):
            state['agent_notes'].append(f"{self.emoji} {self.name}: Skipped (cross-language mode)")
            return state
        
        if not BERT_AVAILABLE:
            state['agent_notes'].append(f"{self.emoji} {self.name}: Skipped (bert-score not installed)")
            return state

        source_text = state['source_text']
        current_text = state['final_translation']
        
        # Initialize tracking
        if 'bertscore_attempts' not in state:
            state['bertscore_attempts'] = 0
        if 'bertscore_history' not in state:
            state['bertscore_history'] = []
        
        attempt = 0
        while attempt < self.max_attempts:
            state['bertscore_attempts'] += 1
            attempt += 1
            
            # Compute BERTScore
            scores = compute_bertscore(current_text, source_text)
            
            if scores is None:
                state['agent_notes'].append(f"{self.emoji} {self.name}: BERTScore computation failed")
                break
            
            f1_score = scores['f1']
            state['bertscore_history'].append({
                'attempt': state['bertscore_attempts'],
                'f1': f1_score,
                'precision': scores['precision'],
                'recall': scores['recall']
            })
            
            # Check if threshold met
            if f1_score >= self.target_score:
                state['agent_notes'].append(
                    f"{self.emoji} {self.name}: ✅ BERTScore validated (F1={f1_score:.3f})"
                )
                state['final_translation'] = current_text
                break
            
            # Need refinement
            if attempt >= self.max_attempts:
                state['agent_notes'].append(
                    f"{self.emoji} {self.name}: ⚠️ Max attempts reached (F1={f1_score:.3f} < {self.target_score})"
                )
                state['needs_human_review'] = True
                break
            
            # Request refinement
            state['agent_notes'].append(
                f"{self.emoji} {self.name}: 🔄 Refining (F1={f1_score:.3f} < {self.target_score}, attempt {attempt}/{self.max_attempts})"
            )
            
            system_prompt = (
                "You are a semantic fidelity specialist. Your ONLY goal is to increase semantic similarity "
                "to the source text while maintaining natural language quality.\n\n"
                + LANGUAGE_GUARDRAIL
            )
            
            user_prompt = f"""The current refined text has insufficient semantic similarity to the source (BERTScore F1: {f1_score:.3f}, target: {self.target_score}).

{LANGUAGE_GUARDRAIL}

**SOURCE TEXT (preserve its meaning):**
{source_text}

**CURRENT REFINED TEXT (needs closer alignment):**
{current_text}

**YOUR TASK:**
1. Identify where meaning has drifted from the source
2. Adjust ONLY those areas to restore semantic fidelity
3. DO NOT over-edit - preserve what is already good
4. Maintain natural, fluent language
5. Keep technical terms and proper nouns identical

**CRITICAL:** The goal is semantic similarity, not word-for-word copying. Preserve the SOURCE's meaning using natural language.

**OUTPUT:** Provide ONLY the refined text in {state['target_language']}, with no meta-commentary."""

            try:
                response = self.llm.invoke([
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt)
                ])
                current_text = reinforce_language(self.llm, response.content.strip(), state['target_language'])
                
            except Exception as e:
                st.error(f"Error in BERTScore refinement: {str(e)}")
                state['agent_notes'].append(f"{self.emoji} {self.name}: Error during refinement - {str(e)}")
                break
        
        state['final_translation'] = current_text
        return state


# =====================
# LLM Initialization Helper
# =====================

def initialize_llm(
    provider: Literal["openai", "anthropic"],
    model: str,
    api_key: str,
    temperature: float = 0.3
) -> BaseChatModel:
    if provider == "openai":
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI not available. Install with: pip install langchain-openai")
        return ChatOpenAI(model=model, temperature=temperature, api_key=api_key, timeout=120)
    elif provider == "anthropic":
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("Anthropic not available. Install with: pip install langchain-anthropic")
        return ChatAnthropic(model=model, temperature=temperature, api_key=api_key, timeout=120)
    else:
        raise ValueError(f"Unsupported provider: {provider}")


# =====================
# LangSmith Setup
# =====================

def setup_langsmith(api_key: Optional[str], project_name: str = "translation-pipeline"):
    if not api_key:
        return False
    if not LANGSMITH_AVAILABLE:
        st.warning("LangSmith not installed. Install with: pip install langsmith")
        return False
    try:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = api_key
        os.environ["LANGCHAIN_PROJECT"] = project_name
        Client(api_key=api_key)
        return True
    except Exception as e:
        st.error(f"LangSmith setup failed: {str(e)}")
        return False


# =====================
# Conditional Logic
# =====================

def should_get_human_feedback(state: TranslationState) -> str:
    if state.get('needs_human_review', False):
        return "human_review"
    return "literary_polish"


# =====================
# Graph Construction
# =====================

def build_translation_graph(llm: BaseChatModel) -> StateGraph:
    literal_agent = LiteralTranslationAgent(llm)
    cultural_agent = CulturalAdaptationAgent(llm)
    tone_agent = ToneConsistencyAgent(llm)
    technical_agent = TechnicalReviewAgent(llm)
    literary_agent = LiteraryEditorAgent(llm)
    qc_agent = QualityControlAgent(llm)
    bertscore_agent = BERTScoreValidatorAgent(llm)  # NEW

    workflow = StateGraph(TranslationState)
    workflow.add_node("literal_translation", literal_agent.translate)
    workflow.add_node("cultural_adaptation", cultural_agent.adapt)
    workflow.add_node("tone_adjustment", tone_agent.adjust_tone)
    workflow.add_node("technical_review", technical_agent.review)
    workflow.add_node("literary_polish", literary_agent.polish)
    workflow.add_node("finalize", qc_agent.finalize)
    workflow.add_node("bertscore_validation", bertscore_agent.validate_and_refine)  # NEW

    workflow.set_entry_point("literal_translation")
    workflow.add_edge("literal_translation", "cultural_adaptation")
    workflow.add_edge("cultural_adaptation", "tone_adjustment")
    workflow.add_edge("tone_adjustment", "technical_review")
    workflow.add_conditional_edges(
        "technical_review",
        should_get_human_feedback,
        {"human_review": END, "literary_polish": "literary_polish"}
    )
    workflow.add_edge("literary_polish", "finalize")
    workflow.add_edge("finalize", "bertscore_validation")  # NEW
    workflow.add_edge("bertscore_validation", END)  # NEW

    return workflow.compile()


# =====================
# Streamlit UI
# =====================

def main():
    st.set_page_config(
        page_title="Advanced Translation System",
        page_icon="🌐",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state FIRST (before any UI elements)
    if 'translation_state' not in st.session_state:
        st.session_state.translation_state = None
    if 'graph' not in st.session_state:
        st.session_state.graph = None
    if 'history' not in st.session_state:
        st.session_state.history = []
    if 'edited_translation' not in st.session_state:
        st.session_state.edited_translation = None
    
    # Entity tracker initialization (NEW - must be before sidebar)
    if 'entity_tracker' not in st.session_state:
        st.session_state.entity_tracker = EntitiesTracker()
    if 'enable_entity_tracking' not in st.session_state:
        st.session_state.enable_entity_tracking = False
    if 'enable_entity_awareness' not in st.session_state:
        st.session_state.enable_entity_awareness = False

    # Custom CSS
    st.markdown("""
    <style>
        .agent-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            border-radius: 10px;
            color: white;
            margin: 10px 0;
        }
        .critical-passage {
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 10px;
            margin: 10px 0;
            border-radius: 4px;
        }
        .stTabs [data-baseweb="tab-list"] { gap: 24px; }
        .stTabs [data-baseweb="tab"] { padding: 10px 20px; }
    </style>
    """, unsafe_allow_html=True)

    st.title("🌐 Advanced Multi-Agent Translation System")
    st.caption("Mode: auto-switches to **Refine** if Source ≈ Target (e.g., English → English).")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")

        st.subheader("🤖 Model Provider")
        provider = st.radio(
            "Select Provider",
            ["openai", "anthropic"],
            format_func=lambda x: "OpenAI (GPT-4)" if x == "openai" else "Anthropic (Claude)"
        )

        st.subheader("🔑 API Keys")
        if provider == "openai":
            api_key = st.text_input("OpenAI API Key", type="password", help="Required for translation/refinement")
            model_options = ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"]
        else:
            api_key = st.text_input("Anthropic API Key", type="password", help="Required for translation/refinement")
            model_options = ["claude-sonnet-4-20250514", "claude-3-5-sonnet-20241022", "claude-3-opus-20240229"]

        st.subheader("📊 Monitoring (Optional)")
        enable_langsmith = st.checkbox("Enable LangSmith Tracing", help="Track and monitor agent interactions")
        langsmith_key = None
        langsmith_project = "translation-pipeline"
        if enable_langsmith:
            langsmith_key = st.text_input("LangSmith API Key", type="password")
            langsmith_project = st.text_input("LangSmith Project Name", value="translation-pipeline")
            if langsmith_key:
                if setup_langsmith(langsmith_key, langsmith_project):
                    st.success("✅ LangSmith enabled")
                else:
                    st.warning("⚠️ LangSmith setup failed")

        st.divider()
        st.subheader("🎛️ Model Settings")
        model = st.selectbox("Model", model_options, index=0)
        temperature = st.slider("Temperature", 0.0, 1.0, 0.3, help="Lower = more conservative, Higher = more creative")

        st.divider()
        st.subheader("🌍 Translation Settings")
        source_lang = st.selectbox(
            "Source Language",
            [
                "Ukrainian","Russian","Polish","German","French","Spanish","Italian","Portuguese",
                "English (US)","English (UK)","Romanian","Czech","Slovak","Bulgarian","Dutch","Other",
            ],
            help="Language of the source text"
        )
        target_lang = st.selectbox(
            "Target Language",
            ["English (US)","English (UK)","Spanish","French","German","Other"],
            help="Language to translate into"
        )

        same_lang_note = languages_equivalent(source_lang, target_lang)
        if same_lang_note:
            st.info("Same-language detected: running in **Refine mode** (copy-edit without changing meaning).")

        audience = st.selectbox(
            "Target Audience",
            [
                "General wellness readers","Academic/Technical audience","Business professionals",
                "Literary fiction readers","Young adults","Healthcare professionals","Scientific community"
            ],
            help="Who will read this output?"
        )
        genre = st.selectbox(
            "Content Genre",
            [
                "Wellness/Self-help","Literary Fiction","Academic/Technical",
                "Business/Professional","Scientific/Medical","General Non-fiction"
            ],
            help="Type of content"
        )

        st.divider()
        show_charts = st.checkbox("Show analytics visualizations", value=True)
        
        # Entity Tracking Options (NEW)
        st.divider()
        st.subheader("🎯 Entity Tracking")
        st.session_state.enable_entity_tracking = st.checkbox(
            "Enable entity tracking",
            value=st.session_state.enable_entity_tracking,
            help="Track entities (people, places, organizations) through translation"
        )
        
        if st.session_state.enable_entity_tracking:
            st.session_state.enable_entity_awareness = st.checkbox(
                "Entity-aware agents",
                value=st.session_state.enable_entity_awareness,
                help="Give translation agents awareness of important entities"
            )
            
            # Glossary upload
            uploaded_glossary = st.file_uploader(
                "Upload Entity Glossary",
                type=['json', 'csv', 'txt'],
                help="Upload a glossary of important terms to track"
            )
            
            if uploaded_glossary:
                if st.session_state.entity_tracker.upload_glossary(uploaded_glossary):
                    st.success(f"✅ Glossary loaded")
            
            # Quick add term
            with st.expander("➕ Quick Add Term"):
                new_term = st.text_input("Term name")
                term_type = st.selectbox("Type", ["person", "location", "organization", "date", "custom"])
                term_desc = st.text_input("Description (optional)")
                if st.button("Add to Glossary"):
                    if new_term:
                        st.session_state.entity_glossary[new_term] = {
                            'type': term_type,
                            'description': term_desc,
                            'aliases': []
                        }
                        st.success(f"Added: {new_term}")
                        st.rerun()
            
            st.metric("Glossary Terms", len(st.session_state.entity_glossary))

    # Main content
    st.header(f"📝 Interface · {mode_phrase(source_lang, target_lang)}")

    with st.expander("📚 Learn About Our Translation Agents", expanded=False):
        st.markdown("### The Seven-Agent Pipeline")
        for _, agent_info in AGENT_DESCRIPTIONS.items():
            st.markdown(f"""
            <div class="agent-card">
                <h3>{agent_info['emoji']} {agent_info['name']}</h3>
                <h4>Role: {agent_info['role']}</h4>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(agent_info['description'])
            st.markdown("**Areas of Expertise:**")
            cols = st.columns(len(agent_info['expertise']))
            for idx, ex in enumerate(agent_info['expertise']):
                cols[idx].info(ex)
            st.divider()

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📄 Source Text")
        uploaded_file = st.file_uploader("Upload a file (optional)", type=['txt','docx','md'])
        if uploaded_file:
            file_content = read_uploaded_file(uploaded_file)
            if file_content:
                st.success(f"✅ File loaded: {uploaded_file.name}")
                source_text = st.text_area("Edit source text if needed", value=file_content, height=400)
            else:
                source_text = st.text_area("Enter text", height=400, placeholder="Paste text manually...")
        else:
            source_text = st.text_area(
                "Enter text",
                height=400,
                placeholder="Paste your source text here…",
            )

        col_a, col_b = st.columns([3, 1])
        with col_a:
            translate_button = st.button(
                "🚀 Start Pipeline",
                type="primary",
                disabled=not api_key,
                use_container_width=True
            )
        with col_b:
            if st.button("🗑️ Clear", use_container_width=True):
                st.session_state.translation_state = None
                st.session_state.edited_translation = None
                st.rerun()

        if not api_key:
            st.warning("⚠️ Enter your API key in the sidebar to begin")

        if source_text:
            st.caption(f"📊 {len(source_text)} characters | ~{len(source_text.split())} words")

    with col2:
        st.subheader("🎯 Result")

        if translate_button and source_text:
            try:
                status_container = st.empty()
                progress_container = st.empty()
                with status_container.container():
                    st.info("🔄 Initializing pipeline...")

                llm = initialize_llm(provider=provider, model=model, api_key=api_key, temperature=temperature)
                graph = build_translation_graph(llm)
                st.session_state.graph = graph

                current_time = datetime.now().isoformat()
                initial_state = {
                    "source_text": source_text,
                    "source_language": source_lang,
                    "target_language": target_lang,
                    "target_audience": audience,
                    "genre": genre,
                    "literal_translation": "",
                    "cultural_adaptation": "",
                    "tone_adjustment": "",
                    "technical_review_version": "",
                    "literary_polish": "",
                    "final_translation": "",
                    "literal_issues": [],
                    "cultural_issues": [],
                    "tone_issues": [],
                    "technical_issues": [],
                    "literary_issues": [],
                    "critical_passages": [],
                    "agent_notes": [],
                    "agent_decisions": [],
                    "human_feedback": None,
                    "revision_count": 0,
                    "needs_human_review": False,
                    "bertscore_attempts": 0,
                    "bertscore_history": [],
                    "started_at": current_time,
                    "completed_at": None
                }
                
                # Add entity fields if tracking is enabled (NEW)
                if st.session_state.enable_entity_tracking:
                    entity_tracker = st.session_state.entity_tracker
                    source_entities = entity_tracker.extract_entities(source_text)
                    initial_state['source_entities'] = source_entities
                    initial_state['translated_entities'] = []
                    initial_state['entity_preservation_rate'] = 0.0
                    
                    if source_entities:
                        st.info(f"🎯 Tracking {len(source_entities)} entities through translation")
                else:
                    initial_state['source_entities'] = None
                    initial_state['translated_entities'] = None
                    initial_state['entity_preservation_rate'] = None

                stages = [
                    ("🔤 Baseline pass", 0.14),
                    ("🌍 Cultural/Register adaptation", 0.29),
                    ("🎭 Tone consistency", 0.43),
                    ("🔬 Technical review", 0.57),
                    ("✍️ Literary polish", 0.71),
                    ("✅ Finalize", 0.86),
                    ("🎯 Semantic validation", 1.0)  # NEW
                ]
                for stage_name, progress_val in stages:
                    with status_container.container():
                        st.info(f"🔄 {stage_name}...")
                    with progress_container.container():
                        st.progress(progress_val)

                with status_container.container():
                    st.info("🔄 Running agents...")
                result = graph.invoke(initial_state)

                status_container.empty()
                progress_container.empty()

                st.session_state.translation_state = result
                st.session_state.edited_translation = result.get('final_translation', '')
                st.session_state.history.append({
                    "timestamp": datetime.now().isoformat(),
                    "source": source_text[:100] + "...",
                    "target_lang": target_lang,
                    "model": model,
                    "result": result
                })

                st.success("✅ Pipeline completed!")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Error during processing: {str(e)}")
                st.code(traceback.format_exc())

        if st.session_state.translation_state:
            state = st.session_state.translation_state
            st.markdown("### 📖 Final Output")
            display_text = st.session_state.edited_translation or state.get('final_translation', 'No output yet')

            st.text_area("Publication-Ready Text", display_text, height=400, key="final_display", label_visibility="visible")

            # Downloads
            st.markdown("### 📥 Download Options")
            col_x, col_y, col_z, col_w = st.columns(4)
            final_text = st.session_state.edited_translation or state.get('final_translation', '')

            with col_x:
                st.download_button(
                    "📄 Text (.txt)", final_text,
                    file_name=f"translation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain", use_container_width=True
                )
            with col_y:
                md_content = create_markdown_file(final_text, "Translation")
                st.download_button(
                    "📝 Markdown (.md)", md_content,
                    file_name=f"translation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                    mime="text/markdown", use_container_width=True
                )
            with col_z:
                if DOCX_AVAILABLE:
                    docx_file = create_docx_file(final_text, "Translation")
                    if docx_file:
                        st.download_button(
                            "📘 Word (.docx)", docx_file,
                            file_name=f"translation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.docx",
                            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                            use_container_width=True
                        )
                else:
                    st.button("📘 Word (.docx)", disabled=True, help="Install python-docx: pip install python-docx", use_container_width=True)
            with col_w:
                st.download_button(
                    "📊 Report (.json)", json.dumps(state, indent=2, default=str),
                    file_name=f"translation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json", use_container_width=True
                )

    # Detailed Analysis Tabs
    if st.session_state.translation_state:
        st.divider()
        st.header("🔍 Detailed Analysis")
        state = st.session_state.translation_state

        # Build tabs list dynamically
        tab_names = [
            "✏️ Edit & Review","🔄 Agent Workflow","📊 All Versions",
            "⚠️ Issues & Feedback","📈 Analytics","📚 History"
        ]
        
        # Add entity tab if enabled
        if st.session_state.enable_entity_tracking:
            tab_names.append("🎯 Entities")
        
        # Create tabs
        tabs = st.tabs(tab_names)
        
        # Original tabs
        tab1, tab2, tab3, tab4, tab5, tab6 = tabs[:6]
        
        # Entity tab if enabled
        tab7 = tabs[6] if len(tabs) > 6 else None

        with tab1:
            st.subheader("Edit & Review")
            edited_text = st.text_area(
                "Edit the output as needed",
                value=st.session_state.edited_translation or state.get('final_translation', ''),
                height=300, key="edit_translation_area", label_visibility="visible"
            )
            if edited_text != st.session_state.edited_translation:
                st.session_state.edited_translation = edited_text
                st.success("✅ Edits saved")

            col_reset, col_apply = st.columns([1, 1])
            with col_reset:
                if st.button("↺ Reset to Original", use_container_width=True):
                    st.session_state.edited_translation = state.get('final_translation', '')
                    st.rerun()
            with col_apply:
                if st.button("💾 Save Final Version", type="primary", use_container_width=True):
                    state['final_translation'] = st.session_state.edited_translation
                    st.success("✅ Final version saved!")

            st.divider()
            st.markdown("#### 🚩 Critical Passages")
            critical_passages = state.get('critical_passages', [])
            if critical_passages:
                st.info(f"Found {len(critical_passages)} passage(s) that may need attention")
                for idx, p in enumerate(critical_passages, 1):
                    st.markdown(f"""
                    <div class="critical-passage">
                        <strong>Passage {idx}</strong> - Flagged by: {p.get('agent','Unknown')}
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown(f"**Text:** {p.get('passage','N/A')}")
                    st.markdown(f"**Issue:** {p.get('issue','No details available')}")
                    st.markdown(f"**Type:** `{p.get('type','general')}`")
                    with st.expander(f"✏️ Edit Passage {idx}"):
                        new_p = st.text_area(f"Edit passage {idx}", value=p.get('passage',''), key=f"edit_passage_{idx}", height=100)
                        if st.button(f"Apply Edit to Passage {idx}", key=f"apply_{idx}"):
                            old = p.get('passage','')
                            if old in st.session_state.edited_translation:
                                st.session_state.edited_translation = st.session_state.edited_translation.replace(old, new_p)
                                st.success(f"✅ Passage {idx} updated")
                                st.rerun()
                    st.divider()
            else:
                st.success("✅ No critical passages flagged")

            review_notes = st.text_area("📝 Review Notes", placeholder="Enter notes/feedback…", height=150, key="review_notes")
            if st.button("💾 Save Review Notes"):
                if 'review_notes' not in state:
                    state['review_notes'] = []
                state['review_notes'].append({'timestamp': datetime.now().isoformat(), 'notes': review_notes})
                st.success("✅ Review notes saved")

        with tab2:
            st.subheader("Agent Workflow Progress")
            for note in state.get('agent_notes', []):
                st.success(f"✓ {note}")
            st.divider()
            if state.get('started_at') and state.get('completed_at'):
                try:
                    start = datetime.fromisoformat(state['started_at'])
                    end = datetime.fromisoformat(state['completed_at'])
                    st.metric("Total Processing Time", f"{(end - start).total_seconds():.1f} seconds")
                except:
                    st.info("Processing time not available")

        with tab3:
            st.subheader("All Versions")
            versions = [
                ("1️⃣ Baseline", state.get('literal_translation','')),
                ("2️⃣ Cultural/Register", state.get('cultural_adaptation','')),
                ("3️⃣ Tone", state.get('tone_adjustment','')),
                ("4️⃣ Technical", state.get('technical_review_version','')),
                ("5️⃣ Literary", state.get('literary_polish','')),
                ("6️⃣ Final", state.get('final_translation','')),
            ]
            for title, content in versions:
                with st.expander(title, expanded=False):
                    # FIX: non-empty label + hide it
                    st.text_area(f"{title} content", content, height=200, key=f"version_{title}", label_visibility="collapsed")

        with tab4:
            st.subheader("Issues & Feedback")
            all_issues = (
                state.get('literal_issues', []) +
                state.get('cultural_issues', []) +
                state.get('tone_issues', []) +
                state.get('technical_issues', []) +
                state.get('literary_issues', [])
            )
            if all_issues:
                for issue in all_issues:
                    with st.expander(f"{issue.get('agent','Unknown')} - {issue.get('type','general')}"):
                        st.markdown(issue.get('content',''))
            else:
                st.info("✅ No issues flagged")
            if state.get('needs_human_review'):
                st.warning("⚠️ Flagged for human review")
                feedback = st.text_area("Provide feedback for revision", placeholder="Describe needed changes…")
                if st.button("Submit Feedback"):
                    st.info("Feedback captured (extend workflow to loop back if desired).")

        with tab5:
            st.subheader("Analytics")

            # Quick metrics
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                source_words = len(state['source_text'].split())
                final_words = len(state.get('final_translation','').split())
                st.metric("Word Count", f"{final_words}", delta=f"{final_words - source_words}")
            with col_b:
                st.metric("Agents Involved", len(state.get('agent_notes', [])))
            with col_c:
                total_issues = sum(len(state.get(k, [])) for k in ['literal_issues','cultural_issues','tone_issues','technical_issues','literary_issues'])
                st.metric("Total Issues Addressed", total_issues)

            if show_charts:
                st.divider()

                # ---------- Visuals: Source vs Final word count ----------
                with st.expander("Word Count Overview", expanded=True):
                    try:
                        df_counts = pd.DataFrame(
                            {"Text": ["Source", "Final"], "Words": [source_words, final_words]}
                        ).set_index("Text")
                        st.bar_chart(df_counts)
                    except Exception:
                        st.warning("Could not render word count chart.")

                st.divider()

                # ---------- Visuals: Sentence length distributions ----------
                with st.expander("Sentence Length Distribution (words per sentence)", expanded=False):
                    src_lens = sentence_lengths(state.get('source_text', ''))
                    fin_lens = sentence_lengths(state.get('final_translation', ''))
                    max_len_for_bins = max([0] + src_lens + fin_lens)
                    bins = nonzero_bins(max_len_for_bins)

                    cols = st.columns(2)
                    with cols[0]:
                        st.caption("Source")
                        try:
                            fig1, ax1 = plt.subplots()
                            ax1.hist(src_lens, bins=bins)
                            ax1.set_xlabel("Words per sentence")
                            ax1.set_ylabel("Frequency")
                            ax1.set_title("Source")
                            st.pyplot(fig1)
                        except Exception:
                            st.warning("Could not render source histogram.")

                    with cols[1]:
                        st.caption("Final")
                        try:
                            fig2, ax2 = plt.subplots()
                            ax2.hist(fin_lens, bins=bins)
                            ax2.set_xlabel("Words per sentence")
                            ax2.set_ylabel("Frequency")
                            ax2.set_title("Final")
                            st.pyplot(fig2)
                        except Exception:
                            st.warning("Could not render final histogram.")

                st.divider()

                # ---------- New: Word Clouds (Source, Final, Added Words) ----------
                with st.expander("Word Clouds (Before / After / Difference)", expanded=True):
                    if not WORDCLOUD_AVAILABLE:
                        st.info("Install `wordcloud` to enable this: `pip install wordcloud`")
                    else:
                        # Build stopword set: wordcloud's STOPWORDS plus a few common extras
                        sw = set(STOPWORDS) | {"—", "–", "'", """, """, "…"}

                        src_text = state.get('source_text', '')
                        fin_text = state.get('final_translation', '')
                        src_freq = word_frequencies(src_text, stopwords=sw)
                        fin_freq = word_frequencies(fin_text, stopwords=sw)

                        # Difference: positive deltas (words added or increased)
                        diff_freq = {}
                        for w, f_cnt in fin_freq.items():
                            s_cnt = src_freq.get(w, 0)
                            delta = f_cnt - s_cnt
                            if delta > 0:
                                diff_freq[w] = delta

                        cols_wc = st.columns(1)
                        # To keep layout simple, stack them vertically for readability
                        st.caption("Before (Source)")
                        try:
                            render_wordcloud_from_freq(src_freq, "Source Word Cloud")
                        except Exception:
                            st.warning("Could not render source word cloud.")

                        st.caption("After (Final)")
                        try:
                            render_wordcloud_from_freq(fin_freq, "Final Word Cloud")
                        except Exception:
                            st.warning("Could not render final word cloud.")

                        st.caption("Difference (Added Words)")
                        try:
                            render_wordcloud_from_freq(diff_freq, "Added Words Word Cloud")
                        except Exception:
                            st.warning("Could not render difference word cloud.")

                st.divider()

                # ---------- Optional readability (textstat) ----------
                with st.expander("Readability (Flesch Reading Ease)", expanded=False):
                    try:
                        import textstat
                        fre_src = textstat.flesch_reading_ease(state.get('source_text', ''))
                        fre_fin = textstat.flesch_reading_ease(state.get('final_translation', ''))
                        c1, c2 = st.columns(2)
                        c1.metric("Source", f"{fre_src:.1f}")
                        c2.metric("Final", f"{fre_fin:.1f}", delta=f"{fre_fin - fre_src:+.1f}")
                    except Exception:
                        st.info("Install `textstat` for readability: `pip install textstat`")

                st.divider()

                # ---------- BERTScore bars (only for same-language refine runs) ----------
                if languages_equivalent(state['source_language'], state['target_language']):
                    with st.expander("BERTScore (same-language refine mode)", expanded=True):
                        if not BERT_AVAILABLE:
                            st.info("Install `bert-score` to enable this metric: `pip install bert-score`")
                        else:
                            refs = state.get('source_text', '').strip()
                            cands = state.get('final_translation', '').strip()
                            if refs and cands:
                                bs = compute_bertscore(candidate=cands, reference=refs)
                                if bs:
                                    try:
                                        fig3, ax3 = plt.subplots()
                                        ax3.barh(["Precision", "Recall", "F1"], [bs["precision"], bs["recall"], bs["f1"]])
                                        ax3.set_xlim(0, 1)
                                        ax3.set_xlabel("Score")
                                        ax3.set_title("BERTScore")
                                        st.pyplot(fig3)
                                    except Exception:
                                        st.warning("Could not render BERTScore chart.")
                                else:
                                    st.warning("BERTScore could not be computed.")
                            else:
                                st.info("Provide both source and final text to compute BERTScore.")
                        
                        # BERTScore refinement history (NEW)
                        if state.get('bertscore_history'):
                            st.divider()
                            st.markdown("#### 🎯 BERTScore Refinement History")
                            
                            history_df = pd.DataFrame(state['bertscore_history'])
                            
                            try:
                                fig_history, ax_history = plt.subplots()
                                ax_history.plot(history_df['attempt'], history_df['f1'], 'o-', label='F1', linewidth=2)
                                ax_history.plot(history_df['attempt'], history_df['precision'], 's-', label='Precision', alpha=0.7)
                                ax_history.plot(history_df['attempt'], history_df['recall'], '^-', label='Recall', alpha=0.7)
                                ax_history.axhline(y=0.8, color='r', linestyle='--', label='Target (0.8)')
                                ax_history.set_xlabel('Refinement Attempt')
                                ax_history.set_ylabel('Score')
                                ax_history.set_title('BERTScore Evolution')
                                ax_history.legend()
                                ax_history.grid(True, alpha=0.3)
                                ax_history.set_ylim(0, 1)
                                st.pyplot(fig_history)
                                
                                st.caption(f"Total attempts: {len(history_df)} | Final F1: {history_df.iloc[-1]['f1']:.3f}")
                            except Exception:
                                st.warning("Could not render BERTScore refinement history chart.")
                else:
                    st.caption("BERTScore is only shown for same-language refine runs.")

                st.divider()

                # ---------- Issue counts by stage ----------
                with st.expander("Issues by Stage", expanded=False):
                    issue_counts = {
                        "Literal": len(state.get('literal_issues', [])),
                        "Cultural/Register": len(state.get('cultural_issues', [])),
                        "Tone": len(state.get('tone_issues', [])),
                        "Technical": len(state.get('technical_issues', [])),
                        "Literary": len(state.get('literary_issues', [])),
                    }
                    try:
                        df_issues = pd.DataFrame(
                            {"Stage": list(issue_counts.keys()), "Count": list(issue_counts.values())}
                        ).set_index("Stage")
                        st.bar_chart(df_issues)
                    except Exception:
                        st.warning("Could not render issues chart.")
            else:
                st.info("Visualizations are disabled.")

        with tab6:
            st.subheader("History")
            if st.session_state.history:
                for i, item in enumerate(reversed(st.session_state.history), 1):
                    with st.expander(f"Run {len(st.session_state.history) - i + 1} – {item['timestamp'][:19]}"):
                        st.write(f"**Model:** {item.get('model','Unknown')}")
                        st.write(f"**Target:** {item.get('target_lang','Unknown')}")
                        st.write(f"**Source Preview:** {item['source']}")
                        if st.button(f"Load this output", key=f"load_{i}"):
                            st.session_state.translation_state = item['result']
                            st.session_state.edited_translation = item['result'].get('final_translation','')
                            st.rerun()
            else:
                st.info("No history yet")
        
        # Entity Tab (NEW - only if enabled)
        if st.session_state.enable_entity_tracking and tab7:
            with tab7:
                st.subheader("🎯 Entity Analysis & Tracking")
                
                if state:
                    entity_tracker = st.session_state.entity_tracker
                    
                    # Extract entities if not already done
                    if 'source_entities' not in state or state['source_entities'] is None:
                        state['source_entities'] = entity_tracker.extract_entities(state.get('source_text', ''))
                    if 'translated_entities' not in state or state['translated_entities'] is None:
                        state['translated_entities'] = entity_tracker.extract_entities(state.get('final_translation', ''))
                    
                    # Entity comparison
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("### 📖 Source Entities")
                        source_entities = state.get('source_entities', [])
                        
                        if source_entities:
                            st.metric("Total Entities", len(source_entities))
                            
                            # Group by type
                            entity_types = {}
                            for entity in source_entities:
                                if entity['type'] not in entity_types:
                                    entity_types[entity['type']] = []
                                entity_types[entity['type']].append(entity)
                            
                            for entity_type, entities in entity_types.items():
                                emoji = entity_tracker.entity_types[entity_type]['emoji']
                                with st.expander(f"{emoji} {entity_type.capitalize()} ({len(entities)})"):
                                    for entity in entities[:20]:
                                        badge = "📚" if entity.get('from_glossary') else "🔮"
                                        st.write(f"• **{entity['name']}** {badge} (×{entity['count']})")
                                        if entity.get('description') and entity['description'] != 'Auto-detected':
                                            st.caption(entity['description'])
                        else:
                            st.info("No entities detected in source")
                    
                    with col2:
                        st.markdown("### 📝 Translated Entities")
                        translated_entities = state.get('translated_entities', [])
                        
                        if translated_entities:
                            st.metric("Total Entities", len(translated_entities))
                            
                            # Group by type
                            entity_types = {}
                            for entity in translated_entities:
                                if entity['type'] not in entity_types:
                                    entity_types[entity['type']] = []
                                entity_types[entity['type']].append(entity)
                            
                            for entity_type, entities in entity_types.items():
                                emoji = entity_tracker.entity_types[entity_type]['emoji']
                                with st.expander(f"{emoji} {entity_type.capitalize()} ({len(entities)})"):
                                    for entity in entities[:20]:
                                        badge = "📚" if entity.get('from_glossary') else "🔮"
                                        st.write(f"• **{entity['name']}** {badge} (×{entity['count']})")
                                        if entity.get('description') and entity['description'] != 'Auto-detected':
                                            st.caption(entity['description'])
                        else:
                            st.info("No entities detected in translation")
                    
                    # Preservation analysis
                    if source_entities and translated_entities:
                        st.divider()
                        st.markdown("### 📊 Entity Preservation Analysis")
                        
                        source_names = {e['name'].lower() for e in source_entities}
                        translated_names = {e['name'].lower() for e in translated_entities}
                        
                        preserved = source_names & translated_names
                        lost = source_names - translated_names
                        added = translated_names - source_names
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            preservation_rate = len(preserved) / len(source_names) * 100 if source_names else 0
                            st.metric("Preservation Rate", f"{preservation_rate:.1f}%")
                        
                        with col2:
                            st.metric("Preserved", len(preserved))
                        
                        with col3:
                            st.metric("Lost", len(lost))
                        
                        with col4:
                            st.metric("Added", len(added))
                        
                        # Details
                        if lost:
                            with st.expander(f"⚠️ Lost Entities ({len(lost)})", expanded=len(lost) <= 5):
                                for name in list(lost)[:30]:
                                    entity = next((e for e in source_entities if e['name'].lower() == name), None)
                                    if entity:
                                        emoji = entity_tracker.entity_types[entity['type']]['emoji']
                                        st.write(f"{emoji} {entity['name']}")
                        
                        if added:
                            with st.expander(f"➕ Added Entities ({len(added)})"):
                                for name in list(added)[:30]:
                                    entity = next((e for e in translated_entities if e['name'].lower() == name), None)
                                    if entity:
                                        emoji = entity_tracker.entity_types[entity['type']]['emoji']
                                        st.write(f"{emoji} {entity['name']}")
                        
                        # Network visualization
                        st.divider()
                        st.markdown("### 🌐 Entity Network Visualization")
                        
                        viz_choice = st.radio(
                            "Select entities to visualize",
                            ["Source Entities", "Translated Entities", "All Entities"],
                            horizontal=True
                        )
                        
                        if viz_choice == "Source Entities":
                            viz_entities = source_entities[:30]
                        elif viz_choice == "Translated Entities":
                            viz_entities = translated_entities[:30]
                        else:
                            # Combine but limit total
                            all_entities = {}
                            for e in source_entities + translated_entities:
                                if e['name'] not in all_entities:
                                    all_entities[e['name']] = e
                            viz_entities = list(all_entities.values())[:30]
                        
                        if viz_entities:
                            entity_tracker.visualize_network(viz_entities)
                        
                        # Comparison chart
                        if PLOTLY_AVAILABLE:
                            st.divider()
                            st.markdown("### 📈 Entity Frequency Comparison")
                            
                            # Create comparison data
                            all_entity_names = {}
                            for e in source_entities:
                                all_entity_names[e['name']] = {'source': e['count'], 'translated': 0, 'type': e['type']}
                            for e in translated_entities:
                                if e['name'] in all_entity_names:
                                    all_entity_names[e['name']]['translated'] = e['count']
                                else:
                                    all_entity_names[e['name']] = {'source': 0, 'translated': e['count'], 'type': e['type']}
                            
                            # Sort by total count and take top 20
                            sorted_entities = sorted(all_entity_names.items(), 
                                                   key=lambda x: x[1]['source'] + x[1]['translated'], 
                                                   reverse=True)[:20]
                            
                            if sorted_entities:
                                df_data = []
                                for name, counts in sorted_entities:
                                    df_data.append({
                                        'Entity': name,
                                        'Source': counts['source'],
                                        'Translated': counts['translated']
                                    })
                                
                                df = pd.DataFrame(df_data)
                                
                                # Create grouped bar chart
                                fig = go.Figure()
                                fig.add_trace(go.Bar(
                                    name='Source',
                                    y=df['Entity'],
                                    x=df['Source'],
                                    orientation='h',
                                    marker_color='#3b82f6'
                                ))
                                fig.add_trace(go.Bar(
                                    name='Translated',
                                    y=df['Entity'],
                                    x=df['Translated'],
                                    orientation='h',
                                    marker_color='#10b981'
                                ))
                                
                                fig.update_layout(
                                    title='Top 20 Entities: Source vs Translated',
                                    xaxis_title='Occurrences',
                                    barmode='group',
                                    height=600,
                                    margin=dict(l=150)
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Complete a translation to see entity analysis")


if __name__ == "__main__":
    main()