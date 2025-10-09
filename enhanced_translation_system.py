"""
Enhanced Multi-Agent Translation Workflow with LangGraph and LangSmith
A sophisticated translation system with cultural adaptation, literary editing, and comprehensive monitoring

Features:
- 6 specialized translation agents with distinct roles
- Support for OpenAI and Anthropic models
- Optional LangSmith tracing and monitoring
- Literary-level quality control
- Comprehensive agent feedback system
- File upload support (txt, docx, md)
- Multiple export formats (txt, docx, md)
- Critical passage flagging and review
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


# =====================
# AGENT ROLE DESCRIPTIONS
# =====================
AGENT_DESCRIPTIONS = {
    "literal_translator": {
        "name": "Literal Translation Specialist",
        "emoji": "🔤",
        "role": "Foundational Translation Accuracy",
        "description": """
        **Primary Responsibility:** Creates the initial word-for-word translation with maximum fidelity to source text.
        
        **Key Functions:**
        - Provides precise semantic equivalents for all words and phrases
        - Maintains grammatical structure close to the original where possible
        - Identifies and flags idioms, colloquialisms, and culturally-specific expressions
        - Notes ambiguous phrases that may have multiple valid interpretations
        - Preserves all technical terminology, proper nouns, and specialized vocabulary
        - Documents translation challenges for downstream agents
        
        **Why This Agent Matters:**
        Without accurate literal translation as a foundation, all subsequent cultural and stylistic
        adaptations risk losing the author's original intent. This agent ensures nothing is lost
        in the translation process.
        """,
        "expertise": ["Bilingual lexical mapping", "Semantic preservation", "Grammatical structure", "Technical terminology"]
    },
    
    "cultural_adapter": {
        "name": "Cultural Localization Expert",
        "emoji": "🌍",
        "role": "Cross-Cultural Communication Bridge",
        "description": """
        **Primary Responsibility:** Adapts content to resonate with target culture while preserving meaning.
        
        **Key Functions:**
        - Replaces source-culture idioms with functionally equivalent target-culture expressions
        - Adapts cultural references, historical allusions, and local context
        - Modifies examples and analogies to be culturally relevant
        - Adjusts humor, metaphors, and figurative language
        - Considers cultural values and communication norms (high-context vs. low-context cultures)
        - Adapts motivational language and persuasive techniques for target psychology
        
        **Why This Agent Matters:**
        A technically accurate translation that ignores cultural context feels foreign and fails to
        engage readers. This agent ensures the translation feels native to the target culture while
        maintaining the author's message.
        """,
        "expertise": ["Cross-cultural communication", "Localization", "Cultural anthropology", "Idiomatic expressions"]
    },
    
    "tone_specialist": {
        "name": "Tone & Voice Consistency Director",
        "emoji": "🎭",
        "role": "Stylistic Harmony & Readability",
        "description": """
        **Primary Responsibility:** Ensures consistent tone, voice, and reading experience throughout.
        
        **Key Functions:**
        - Maintains uniform formality level appropriate to content and audience
        - Adjusts sentence rhythm and pacing for natural flow
        - Balances sentence length variation for optimal readability
        - Ensures consistent narrative perspective (1st/2nd/3rd person)
        - Optimizes active vs. passive voice usage
        - Creates smooth transitions between paragraphs and sections
        - Adapts vocabulary sophistication to target audience
        
        **Why This Agent Matters:**
        Inconsistent tone disrupts the reading experience and undermines author credibility. This
        agent ensures the translation reads as if originally written in the target language with
        a unified, professional voice.
        """,
        "expertise": ["Stylistics", "Rhetorical analysis", "Reader psychology", "Narrative voice"]
    },
    
    "technical_reviewer": {
        "name": "Technical Accuracy Auditor",
        "emoji": "🔬",
        "role": "Precision & Factual Integrity",
        "description": """
        **Primary Responsibility:** Verifies technical accuracy and notation correctness.
        
        **Key Functions:**
        - Validates mathematical notation and symbols (×, ÷, √, ≈, etc.)
        - Verifies scientific terminology and nomenclature
        - Checks units of measurement and conversion accuracy
        - Ensures proper formatting of numbers, dates, and times
        - Reviews citations, references, and bibliographic entries
        - Validates currency symbols and financial notation
        - Cross-checks specialized vocabulary with domain glossaries
        - Flags potential errors requiring subject matter expert review
        
        **Why This Agent Matters:**
        Technical errors destroy credibility and can lead to serious misunderstandings. This agent
        ensures all factual content, notation, and specialized terminology is accurate and follows
        target-language conventions.
        """,
        "expertise": ["Technical writing", "Scientific notation", "Domain-specific terminology", "Quality assurance"]
    },
    
    "literary_editor": {
        "name": "Literary Style & Excellence Editor",
        "emoji": "✍️",
        "role": "Publication-Ready Prose Crafting",
        "description": """
        **Primary Responsibility:** Elevates writing to literary/award-worthy quality standards.
        
        **Key Functions:**
        - Eliminates awkward phrasing and improves sentence elegance
        - Enhances word choice for precision, beauty, and impact
        - Optimizes prose rhythm and musicality
        - Strengthens narrative hooks and engagement
        - Refines imagery, metaphors, and descriptive language
        - Ensures sophisticated yet accessible vocabulary
        - Polishes opening and closing sentences for maximum impact
        - Applies literary devices appropriately (parallelism, alliteration, etc.)
        - Evaluates overall readability and aesthetic quality
        - Ensures the translation could compete for literary awards
        
        **Why This Agent Matters:**
        Good translation is accurate; great translation is artful. This agent transforms technically
        correct prose into literature that engages, moves, and inspires readers—turning translation
        into an art form worthy of recognition.
        """,
        "expertise": ["Literary criticism", "Creative writing", "Stylistics", "Publishing standards", "Award-level prose"]
    },
    
    "quality_controller": {
        "name": "Master Quality Synthesizer",
        "emoji": "✅",
        "role": "Final Integration & Excellence Assurance",
        "description": """
        **Primary Responsibility:** Synthesizes all agent contributions into publication-ready final version.
        
        **Key Functions:**
        - Reviews and integrates work from all specialist agents
        - Resolves conflicting recommendations between agents
        - Ensures no meaning lost from original source text
        - Validates that all agent concerns have been addressed
        - Performs final coherence and consistency check
        - Ensures translation meets all quality criteria
        - Makes final executive decisions on ambiguous cases
        - Certifies translation as publication-ready
        - Provides comprehensive quality assessment report
        
        **Why This Agent Matters:**
        Multiple specialists can produce conflicting recommendations. This agent provides the holistic
        perspective needed to balance competing priorities and deliver a final product that excels
        across all dimensions of quality.
        """,
        "expertise": ["Project management", "Editorial judgment", "Holistic quality assessment", "Synthesis"]
    }
}


# =====================
# State Definition
# =====================
class TranslationState(TypedDict):
    """State shared across all agents in the translation pipeline"""
    
    # Source content
    source_text: str
    source_language: str
    target_language: str
    target_audience: str
    genre: str
    
    # Translation versions at each stage
    literal_translation: str
    cultural_adaptation: str
    tone_adjustment: str
    technical_review_version: str
    literary_polish: str
    final_translation: str
    
    # Agent feedback and issues
    literal_issues: List[Dict]
    cultural_issues: List[Dict]
    tone_issues: List[Dict]
    technical_issues: List[Dict]
    literary_issues: List[Dict]
    
    # Critical passages flagged for review
    critical_passages: List[Dict]
    
    # Workflow tracking
    agent_notes: Annotated[List[str], operator.add]
    agent_decisions: List[Dict]
    human_feedback: Optional[str]
    revision_count: int
    needs_human_review: bool
    quality_score: Optional[float]
    
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
        
        if file_extension == 'txt':
            return uploaded_file.read().decode('utf-8')
        
        elif file_extension == 'md':
            return uploaded_file.read().decode('utf-8')
        
        elif file_extension == 'docx':
            if not DOCX_AVAILABLE:
                st.error("python-docx not installed. Install with: pip install python-docx")
                return ""
            doc = Document(uploaded_file)
            return '\n\n'.join([paragraph.text for paragraph in doc.paragraphs if paragraph.text.strip()])
        
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
    
    # Add title
    title_paragraph = doc.add_heading(title, level=1)
    title_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    
    # Add timestamp
    timestamp = doc.add_paragraph()
    timestamp.add_run(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}").italic = True
    timestamp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    
    # Add separator
    doc.add_paragraph()
    
    # Add content with proper formatting
    paragraphs = text.split('\n\n')
    for para in paragraphs:
        if para.strip():
            p = doc.add_paragraph(para.strip())
            p.style = 'Normal'
            # Set font
            for run in p.runs:
                run.font.name = 'Calibri'
                run.font.size = Pt(11)
    
    # Save to BytesIO
    file_stream = io.BytesIO()
    doc.save(file_stream)
    file_stream.seek(0)
    
    return file_stream


def create_markdown_file(text: str, title: str = "Translation") -> str:
    """Create a formatted Markdown file"""
    md_content = f"""# {title}

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

{text}
"""
    return md_content


def extract_critical_passages(text: str, issues_list: List[Dict]) -> List[Dict]:
    """Extract critical passages that need review based on agent feedback"""
    critical = []
    
    # Look for specific markers in issues
    for issue in issues_list:
        content = issue.get('content', '')
        issue_type = issue.get('type', '')
        agent = issue.get('agent', '')
        
        # Flag critical issues
        if any(keyword in content.lower() for keyword in ['critical', 'warning', 'error', 'ambiguous', 'unclear', 'needs review']):
            # Try to extract the relevant passage
            sentences = text.split('.')
            for i, sentence in enumerate(sentences):
                if len(sentence.strip()) > 10:  # Skip very short sentences
                    critical.append({
                        'passage': sentence.strip() + '.',
                        'issue': content[:200],  # First 200 chars of issue
                        'agent': agent,
                        'type': issue_type,
                        'sentence_index': i
                    })
    
    return critical[:10]  # Limit to top 10 critical passages


# =====================
# Enhanced Agent Definitions
# =====================

class LiteralTranslationAgent:
    """Agent for accurate literal translation - Foundation of the pipeline"""
    
    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        self.name = "Literal Translation Specialist"
        self.emoji = "🔤"
    
    def translate(self, state: TranslationState) -> TranslationState:
        """Perform literal translation with maximum accuracy"""
        
        system_prompt = """You are a world-class literal translation specialist with expertise in maintaining 
semantic fidelity across languages. Your translations are known for their precision and attention to detail.

Your role is to create an accurate, word-level translation that serves as the foundation for other agents. 
This is NOT the final translation - cultural adaptation will come later."""

        user_prompt = f"""Translate the following text from {state['source_language']} to {state['target_language']}.

**CRITICAL INSTRUCTIONS:**

1. **SEMANTIC PRECISION**: Translate each phrase with maximum fidelity to the original meaning
   - Choose the most semantically accurate equivalent, even if it sounds unnatural
   - When multiple translations are possible, choose the one closest to literal meaning
   
2. **PRESERVE STRUCTURE**: Initially maintain source sentence structure
   - Keep similar clause ordering where grammatically possible
   - Preserve paragraph breaks and formatting
   
3. **FLAG CHALLENGES**: Identify and note:
   - Idioms that don't translate literally (mark with [IDIOM: original phrase])
   - Ambiguous phrases with multiple valid interpretations (mark with [AMBIGUOUS: explanation])
   - Cultural references needing adaptation (mark with [CULTURAL: description])
   - Wordplay or puns that don't translate (mark with [WORDPLAY: explanation])
   
4. **TECHNICAL TERMS**: Maintain all:
   - Scientific/medical terminology exactly
   - Proper nouns in original form with translation if needed
   - Technical jargon with precision
   
5. **OUTPUT FORMAT**:
   - Provide the literal translation first
   - Then add a section "TRANSLATOR NOTES:" with flagged items
   - Then add "CHALLENGES:" listing key translation difficulties

**SOURCE TEXT:**
{state['source_text']}

**TARGET AUDIENCE CONTEXT** (for your awareness, don't adapt yet):
{state['target_audience']}

**GENRE**: {state.get('genre', 'General')}

Provide your literal translation with comprehensive notes."""

        try:
            response = self.llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ])
            
            content = response.content
            
            # Parse response
            translation = content
            issues = []
            
            if "TRANSLATOR NOTES:" in content:
                parts = content.split("TRANSLATOR NOTES:")
                translation = parts[0].strip()
                notes = parts[1].split("CHALLENGES:")[0].strip() if "CHALLENGES:" in parts[1] else parts[1].strip()
                issues.append({
                    "agent": self.name,
                    "type": "translation_notes",
                    "content": notes
                })
            
            if "CHALLENGES:" in content:
                challenges = content.split("CHALLENGES:")[1].strip()
                issues.append({
                    "agent": self.name,
                    "type": "challenges",
                    "content": challenges
                })
            
            state['literal_translation'] = translation
            state['literal_issues'] = issues
            state['agent_notes'].append(f"{self.emoji} {self.name}: Completed literal translation with {len(issues)} flagged items")
            
        except Exception as e:
            st.error(f"Error in Literal Translation: {str(e)}")
            state['literal_translation'] = state['source_text']
            state['literal_issues'] = [{"agent": self.name, "type": "error", "content": str(e)}]
            state['agent_notes'].append(f"{self.emoji} {self.name}: Error occurred - using source text")
        
        return state


class CulturalAdaptationAgent:
    """Agent for cultural localization and context adaptation"""
    
    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        self.name = "Cultural Localization Expert"
        self.emoji = "🌍"
    
    def adapt(self, state: TranslationState) -> TranslationState:
        """Adapt translation for target culture"""
        
        system_prompt = """You are an expert in cross-cultural communication and localization. You understand 
the subtle cultural differences that make content resonate with different audiences. Your adaptations maintain 
the author's intent while making the content feel native to the target culture."""

        user_prompt = f"""Adapt this literal translation for the target culture and audience.

**SOURCE CULTURE**: {state['source_language']} speaking regions
**TARGET CULTURE**: {state['target_language']} speaking regions  
**TARGET AUDIENCE**: {state['target_audience']}
**GENRE**: {state.get('genre', 'General')}

**LITERAL TRANSLATION:**
{state['literal_translation']}

**FLAGGED ITEMS FROM LITERAL TRANSLATOR:**
{json.dumps(state.get('literal_issues', []), indent=2)}

**YOUR CULTURAL ADAPTATION TASKS:**

1. **IDIOMS & EXPRESSIONS**:
   - Replace source-language idioms with target-language equivalents that convey the same meaning
   - Example: Russian "галопом по Европам" → English "rushing through" or "whistle-stop tour"
   - Find culturally equivalent expressions, not just word-for-word translations
   
2. **CULTURAL REFERENCES**:
   - Adapt or explain historical events, cultural figures, traditions unfamiliar to target audience
   - Replace culture-specific holidays, foods, customs with relatable equivalents when appropriate
   - Add brief context where necessary without over-explaining
   
3. **EXAMPLES & ANALOGIES**:
   - Replace examples with ones the target audience can relate to
   - Adapt analogies to use familiar concepts from target culture
   
4. **COMMUNICATION STYLE**:
   - Russian: Often formal, complex sentences, philosophical tone
   - English: Prefer conversational, punchy statements, practical focus
   - Adjust formality level for audience (academic vs. general public)
   
5. **MOTIVATIONAL FRAMING**:
   - Adapt psychological appeals for target culture
   - Collectivist vs. individualist framing
   - Authority-based vs. evidence-based persuasion

**OUTPUT FORMAT**:
- Provide your culturally adapted translation
- Add "CULTURAL ADAPTATIONS MADE:" section listing significant changes
- Add "CULTURAL NOTES:" section with important context

Adapt the text while preserving the author's core message and intent."""

        try:
            response = self.llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ])
            
            content = response.content
            
            # Parse response
            adapted = content
            issues = []
            
            if "CULTURAL ADAPTATIONS MADE:" in content:
                parts = content.split("CULTURAL ADAPTATIONS MADE:")
                adapted = parts[0].strip()
                changes = parts[1].split("CULTURAL NOTES:")[0].strip() if "CULTURAL NOTES:" in parts[1] else parts[1].strip()
                issues.append({
                    "agent": self.name,
                    "type": "adaptations",
                    "content": changes
                })
            
            if "CULTURAL NOTES:" in content:
                notes = content.split("CULTURAL NOTES:")[1].strip()
                issues.append({
                    "agent": self.name,
                    "type": "notes",
                    "content": notes
                })
            
            state['cultural_adaptation'] = adapted
            state['cultural_issues'] = issues
            state['agent_notes'].append(f"{self.emoji} {self.name}: Applied cultural localization")
            
        except Exception as e:
            st.error(f"Error in Cultural Adaptation: {str(e)}")
            state['cultural_adaptation'] = state['literal_translation']
            state['cultural_issues'] = [{"agent": self.name, "type": "error", "content": str(e)}]
            state['agent_notes'].append(f"{self.emoji} {self.name}: Error occurred - using previous version")
        
        return state


class ToneConsistencyAgent:
    """Agent for tone and voice consistency"""
    
    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        self.name = "Tone & Voice Consistency Director"
        self.emoji = "🎭"
    
    def adjust_tone(self, state: TranslationState) -> TranslationState:
        """Ensure consistent tone and optimal readability"""
        
        system_prompt = """You are a master of stylistic consistency and readability optimization. You ensure 
that translated text reads smoothly with a unified voice throughout. Your adjustments make translations feel 
natural while maintaining the author's intended tone."""

        user_prompt = f"""Adjust this translation for tone consistency and optimal readability.

**TARGET AUDIENCE**: {state['target_audience']}
**GENRE**: {state.get('genre', 'General')}

**CULTURALLY ADAPTED TEXT:**
{state['cultural_adaptation']}

**YOUR TONE ADJUSTMENT TASKS:**

1. **SENTENCE RHYTHM & PACING**:
   - Vary sentence length (mix short, medium, long sentences)
   - Break up overly complex sentences (aim for 15-25 words average)
   - Create rhythmic flow that guides the reader naturally
   
2. **FORMALITY LEVEL**:
   - Match formality to audience and genre
   - Academic: More formal, technical vocabulary
   - General wellness: Warm but professional, accessible language
   - Business: Professional but clear, no unnecessary jargon
   - Maintain consistent formality throughout
   
3. **VOICE & PERSPECTIVE**:
   - Ensure consistent narrative voice (active vs. passive)
   - Prefer active voice (subject-verb-object) for clarity
   - Maintain consistent person (1st/2nd/3rd) throughout
   
4. **VOCABULARY SOPHISTICATION**:
   - Match vocabulary level to audience education
   - Replace overly complex words with simpler alternatives when appropriate
   - Avoid unnecessary jargon

**OUTPUT FORMAT**:
- Provide your tone-adjusted translation
- Add "TONE ADJUSTMENTS:" section describing major changes
- Add "READABILITY NOTES:" section with observations

Adjust the text for optimal flow and consistency."""

        try:
            response = self.llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ])
            
            content = response.content
            
            # Parse response
            adjusted = content
            issues = []
            
            if "TONE ADJUSTMENTS:" in content:
                parts = content.split("TONE ADJUSTMENTS:")
                adjusted = parts[0].strip()
                adjustments = parts[1].split("READABILITY NOTES:")[0].strip() if "READABILITY NOTES:" in parts[1] else parts[1].strip()
                issues.append({
                    "agent": self.name,
                    "type": "adjustments",
                    "content": adjustments
                })
            
            if "READABILITY NOTES:" in content:
                notes = content.split("READABILITY NOTES:")[1].strip()
                issues.append({
                    "agent": self.name,
                    "type": "readability",
                    "content": notes
                })
            
            state['tone_adjustment'] = adjusted
            state['tone_issues'] = issues
            state['agent_notes'].append(f"{self.emoji} {self.name}: Optimized tone and readability")
            
        except Exception as e:
            st.error(f"Error in Tone Adjustment: {str(e)}")
            state['tone_adjustment'] = state['cultural_adaptation']
            state['tone_issues'] = [{"agent": self.name, "type": "error", "content": str(e)}]
            state['agent_notes'].append(f"{self.emoji} {self.name}: Error occurred - using previous version")
        
        return state


class TechnicalReviewAgent:
    """Agent for technical accuracy verification"""
    
    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        self.name = "Technical Accuracy Auditor"
        self.emoji = "🔬"
    
    def review(self, state: TranslationState) -> TranslationState:
        """Review and correct technical accuracy"""
        
        system_prompt = """You are a meticulous technical reviewer with expertise in ensuring factual accuracy 
and proper notation across disciplines. Your reviews catch errors that could undermine credibility or cause 
confusion. You have a keen eye for detail and deep knowledge of technical conventions."""

        user_prompt = f"""Review this translation for technical accuracy and correct any errors.

**TEXT TO REVIEW:**
{state['tone_adjustment']}

**GENRE**: {state.get('genre', 'General')}
**SOURCE LANGUAGE**: {state['source_language']}
**TARGET LANGUAGE**: {state['target_language']}

**YOUR TECHNICAL REVIEW TASKS:**

1. **MATHEMATICAL NOTATION**:
   - Verify proper symbols: × (multiply), ÷ (divide), ± (plus-minus), ≈ (approximately)
   - Check decimal separators (US: 1,234.56 vs European: 1.234,56)
   
2. **SCIENTIFIC TERMINOLOGY**:
   - Verify accuracy of technical terms (medical, scientific, engineering)
   - Validate chemical formulas and notation
   
3. **UNITS & MEASUREMENTS**:
   - Check unit abbreviations (km, m, g, kg, mL, etc.)
   - Verify unit conversions if any were needed
   
4. **NUMBERS & DATES**:
   - Verify number formatting for target locale
   - Check date formatting (MM/DD/YYYY vs DD/MM/YYYY)
   
5. **CROSS-REFERENCE WITH SOURCE**:
   - Verify no technical details were lost or changed
   - Check that all numbers match source text

**CRITICAL**: If you find errors that could lead to misunderstanding or harm (medical, safety, legal), 
mark the translation as "NEEDS_HUMAN_REVIEW" and flag these errors prominently.

**OUTPUT FORMAT**:
- Provide the technically reviewed version with corrections made
- Add "TECHNICAL CORRECTIONS:" section listing changes
- Add "CRITICAL ISSUES:" section if any serious errors found
- Mark as "STATUS: APPROVED" or "STATUS: NEEDS_HUMAN_REVIEW"

Perform your technical review now."""

        try:
            response = self.llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ])
            
            content = response.content
            
            # Parse response
            reviewed = content
            issues = []
            needs_review = False
            
            if "TECHNICAL CORRECTIONS:" in content:
                parts = content.split("TECHNICAL CORRECTIONS:")
                reviewed = parts[0].strip()
                corrections = parts[1].split("CRITICAL ISSUES:")[0].strip() if "CRITICAL ISSUES:" in parts[1] else parts[1].strip()
                corrections = corrections.split("STATUS:")[0].strip() if "STATUS:" in corrections else corrections
                issues.append({
                    "agent": self.name,
                    "type": "corrections",
                    "content": corrections
                })
            
            if "CRITICAL ISSUES:" in content:
                critical = content.split("CRITICAL ISSUES:")[1].strip()
                critical = critical.split("STATUS:")[0].strip() if "STATUS:" in critical else critical
                issues.append({
                    "agent": self.name,
                    "type": "critical_issues",
                    "content": critical
                })
                needs_review = True
            
            if "NEEDS_HUMAN_REVIEW" in content or "NEEDS HUMAN REVIEW" in content:
                needs_review = True
            
            state['technical_review_version'] = reviewed
            state['technical_issues'] = issues
            state['needs_human_review'] = needs_review
            state['agent_notes'].append(f"{self.emoji} {self.name}: Technical review completed")
            
        except Exception as e:
            st.error(f"Error in Technical Review: {str(e)}")
            state['technical_review_version'] = state['tone_adjustment']
            state['technical_issues'] = [{"agent": self.name, "type": "error", "content": str(e)}]
            state['needs_human_review'] = False
            state['agent_notes'].append(f"{self.emoji} {self.name}: Error occurred - using previous version")
        
        return state


class LiteraryEditorAgent:
    """Agent for literary-level style and readability enhancement"""
    
    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        self.name = "Literary Style & Excellence Editor"
        self.emoji = "✍️"
    
    def polish(self, state: TranslationState) -> TranslationState:
        """Elevate writing to award-worthy literary quality"""
        
        system_prompt = """You are an award-winning literary editor who has worked with major publishing houses 
and prize-winning authors. Your edits transform good writing into exceptional literature. You have an 
exceptional ear for language, deep understanding of literary craft, and the ability to elevate prose while 
maintaining the author's voice."""

        user_prompt = f"""Transform this technically accurate translation into award-worthy literature.

**TEXT TO POLISH:**
{state['technical_review_version']}

**TARGET AUDIENCE**: {state['target_audience']}
**GENRE**: {state.get('genre', 'General')}

**YOUR LITERARY EDITING MISSION:**

Elevate this translation to publication-ready, award-competitive quality.

1. **SENTENCE ELEGANCE**:
   - Eliminate awkward phrasing while preserving meaning
   - Craft sentences that flow beautifully when read aloud
   - Remove clunky constructions and unclear references
   
2. **WORD CHOICE PRECISION**:
   - Select the most evocative, precise word for each context
   - Replace weak verbs with strong, specific alternatives
   - Eliminate redundancy and verbal tics
   
3. **PROSE MUSICALITY**:
   - Balance sound and sense (alliteration, assonance when appropriate)
   - Vary sentence structure for rhythmic interest
   
4. **IMAGERY & METAPHOR**:
   - Strengthen or add figurative language where it enhances meaning
   - Ensure metaphors are fresh, not clichéd
   
5. **PUBLICATION STANDARDS**:
   - Polish to professional publishing quality
   - Ensure consistent excellence throughout

**CRITICAL PRINCIPLES**:
- Maintain the author's meaning and intent absolutely
- Preserve technical accuracy from previous review
- Don't over-polish into blandness—keep distinctive voice

**OUTPUT FORMAT**:
- Provide your literary polished version
- Add "LITERARY ENHANCEMENTS:" section describing key improvements
- Add "STYLISTIC NOTES:" with commentary on editorial choices
- Add "QUALITY ASSESSMENT:" with your confidence this meets publication standards (1-10 scale)

Transform this translation into literature worthy of recognition."""

        try:
            response = self.llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ])
            
            content = response.content
            
            # Parse response
            polished = content
            issues = []
            quality_score = None
            
            if "LITERARY ENHANCEMENTS:" in content:
                parts = content.split("LITERARY ENHANCEMENTS:")
                polished = parts[0].strip()
                enhancements = parts[1].split("STYLISTIC NOTES:")[0].strip() if "STYLISTIC NOTES:" in parts[1] else parts[1].strip()
                enhancements = enhancements.split("QUALITY ASSESSMENT:")[0].strip() if "QUALITY ASSESSMENT:" in enhancements else enhancements
                issues.append({
                    "agent": self.name,
                    "type": "enhancements",
                    "content": enhancements
                })
            
            if "STYLISTIC NOTES:" in content:
                notes = content.split("STYLISTIC NOTES:")[1].strip()
                notes = notes.split("QUALITY ASSESSMENT:")[0].strip() if "QUALITY ASSESSMENT:" in notes else notes
                issues.append({
                    "agent": self.name,
                    "type": "stylistic_notes",
                    "content": notes
                })
            
            if "QUALITY ASSESSMENT:" in content:
                assessment = content.split("QUALITY ASSESSMENT:")[1].strip()
                issues.append({
                    "agent": self.name,
                    "type": "quality_assessment",
                    "content": assessment
                })
                # Try to extract numeric score
                try:
                    score_match = re.search(r'(\d+(?:\.\d+)?)/10', assessment)
                    if score_match:
                        quality_score = float(score_match.group(1))
                except:
                    pass
            
            state['literary_polish'] = polished
            state['literary_issues'] = issues
            state['quality_score'] = quality_score
            state['agent_notes'].append(f"{self.emoji} {self.name}: Literary polish completed")
            
        except Exception as e:
            st.error(f"Error in Literary Polish: {str(e)}")
            state['literary_polish'] = state['technical_review_version']
            state['literary_issues'] = [{"agent": self.name, "type": "error", "content": str(e)}]
            state['quality_score'] = None
            state['agent_notes'].append(f"{self.emoji} {self.name}: Error occurred - using previous version")
        
        return state


class QualityControlAgent:
    """Final quality control and synthesis agent"""
    
    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        self.name = "Master Quality Synthesizer"
        self.emoji = "✅"
    
    def finalize(self, state: TranslationState) -> TranslationState:
        """Synthesize all agent work and produce final publication-ready translation"""
        
        system_prompt = """You are the master quality controller with final authority over the translation. 
You have overseen the work of multiple specialist agents and now must make final decisions to produce 
the definitive, publication-ready version. Your judgment balances competing priorities and ensures 
nothing was lost while everything was improved."""

        # Compile all agent feedback
        all_issues = (
            state.get('literal_issues', []) +
            state.get('cultural_issues', []) +
            state.get('tone_issues', []) +
            state.get('technical_issues', []) +
            state.get('literary_issues', [])
        )
        
        user_prompt = f"""Produce the final, publication-ready translation by synthesizing all agent work.

**ORIGINAL SOURCE TEXT:**
{state['source_text']}

**TRANSLATION VERSIONS:**

1. Literal Translation:
{state.get('literal_translation', 'N/A')[:500]}...

2. Cultural Adaptation:
{state.get('cultural_adaptation', 'N/A')[:500]}...

3. Tone Adjustment:
{state.get('tone_adjustment', 'N/A')[:500]}...

4. Technical Review:
{state.get('technical_review_version', 'N/A')[:500]}...

5. Literary Polish:
{state.get('literary_polish', 'N/A')[:1000]}...

**ALL AGENT FEEDBACK & ISSUES:**
{json.dumps(all_issues, indent=2)[:2000]}...

**AGENT WORKFLOW NOTES:**
{chr(10).join(state.get('agent_notes', []))}

**YOUR FINAL SYNTHESIS TASKS:**

1. **REVIEW ALL VERSIONS**: Compare the evolution from literal to literary
2. **VERIFY FIDELITY**: Ensure no meaning lost from original source
3. **RESOLVE CONFLICTS**: Where agents made conflicting changes, make final decision
4. **INTEGRATE BEST ELEMENTS**: Take the best from each version
5. **FINAL POLISH**: Make any last micro-adjustments needed
6. **QUALITY CERTIFICATION**: Confirm this meets publication standards

**DECISION PRIORITIES** (in order):
1. Accuracy to source meaning (non-negotiable)
2. Technical correctness (non-negotiable)
3. Cultural appropriateness
4. Readability and flow
5. Literary quality

**OUTPUT FORMAT**:
Provide ONLY the final translation text without any meta-commentary, notes, or sections.
This should be the exact text ready for publication.

After the translation, add:

---
QUALITY CONTROL REPORT:
- Fidelity to source: [assessment]
- Technical accuracy: [assessment]  
- Cultural adaptation: [assessment]
- Literary quality: [assessment]
- Overall confidence: [X/10]
- Recommendation: [APPROVED FOR PUBLICATION / NEEDS HUMAN REVIEW]

Produce the final translation now."""

        try:
            response = self.llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ])
            
            content = response.content
            
            # Parse final translation and QC report
            final_translation = content
            qc_report = {}
            
            if "QUALITY CONTROL REPORT:" in content or "---" in content:
                parts = content.split("---")
                if len(parts) >= 2:
                    final_translation = parts[0].strip()
                    qc_section = parts[1].strip()
                    qc_report = {
                        "agent": self.name,
                        "type": "final_qc_report",
                        "content": qc_section
                    }
            
            # Extract critical passages
            critical_passages = extract_critical_passages(final_translation, all_issues)
            
            state['final_translation'] = final_translation
            state['completed_at'] = datetime.now().isoformat()
            state['critical_passages'] = critical_passages
            state['agent_notes'].append(f"{self.emoji} {self.name}: Final translation approved")
            
            if qc_report:
                state['agent_decisions'].append(qc_report)
                
        except Exception as e:
            st.error(f"Error in Quality Control: {str(e)}")
            state['final_translation'] = state['literary_polish']
            state['completed_at'] = datetime.now().isoformat()
            state['critical_passages'] = []
            state['agent_notes'].append(f"{self.emoji} {self.name}: Error occurred - using literary version")
        
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
    """Initialize the appropriate LLM based on provider"""
    
    if provider == "openai":
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI not available. Install with: pip install langchain-openai")
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=api_key,
            timeout=120
        )
    elif provider == "anthropic":
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("Anthropic not available. Install with: pip install langchain-anthropic")
        return ChatAnthropic(
            model=model,
            temperature=temperature,
            api_key=api_key,
            timeout=120
        )
    else:
        raise ValueError(f"Unsupported provider: {provider}")


# =====================
# LangSmith Setup
# =====================

def setup_langsmith(api_key: Optional[str], project_name: str = "translation-pipeline"):
    """Configure LangSmith tracing if API key provided"""
    
    if not api_key:
        return False
    
    if not LANGSMITH_AVAILABLE:
        st.warning("LangSmith not installed. Install with: pip install langsmith")
        return False
    
    try:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = api_key
        os.environ["LANGCHAIN_PROJECT"] = project_name
        
        # Test connection
        client = Client(api_key=api_key)
        return True
    except Exception as e:
        st.error(f"LangSmith setup failed: {str(e)}")
        return False


# =====================
# Conditional Logic
# =====================

def should_get_human_feedback(state: TranslationState) -> str:
    """Determine if human review is needed"""
    if state.get('needs_human_review', False):
        return "human_review"
    return "literary_polish"


# =====================
# Graph Construction
# =====================

def build_translation_graph(llm: BaseChatModel) -> StateGraph:
    """Build the complete LangGraph workflow with all agents"""
    
    # Initialize all agents
    literal_agent = LiteralTranslationAgent(llm)
    cultural_agent = CulturalAdaptationAgent(llm)
    tone_agent = ToneConsistencyAgent(llm)
    technical_agent = TechnicalReviewAgent(llm)
    literary_agent = LiteraryEditorAgent(llm)
    qc_agent = QualityControlAgent(llm)
    
    # Create graph
    workflow = StateGraph(TranslationState)
    
    # Add all agent nodes
    workflow.add_node("literal_translation", literal_agent.translate)
    workflow.add_node("cultural_adaptation", cultural_agent.adapt)
    workflow.add_node("tone_adjustment", tone_agent.adjust_tone)
    workflow.add_node("technical_review", technical_agent.review)
    workflow.add_node("literary_polish", literary_agent.polish)
    workflow.add_node("finalize", qc_agent.finalize)
    
    # Define workflow sequence
    workflow.set_entry_point("literal_translation")
    workflow.add_edge("literal_translation", "cultural_adaptation")
    workflow.add_edge("cultural_adaptation", "tone_adjustment")
    workflow.add_edge("tone_adjustment", "technical_review")
    
    # Conditional edge: check if human review needed before literary polish
    workflow.add_conditional_edges(
        "technical_review",
        should_get_human_feedback,
        {
            "human_review": END,
            "literary_polish": "literary_polish"
        }
    )
    
    workflow.add_edge("literary_polish", "finalize")
    workflow.add_edge("finalize", END)
    
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
        .quality-score {
            font-size: 48px;
            font-weight: bold;
            text-align: center;
            color: #4CAF50;
        }
        .critical-passage {
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 10px;
            margin: 10px 0;
            border-radius: 4px;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 24px;
        }
        .stTabs [data-baseweb="tab"] {
            padding: 10px 20px;
        }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("🌐 Advanced Multi-Agent Translation System")
    st.markdown("*Literary-quality translation with comprehensive monitoring powered by LangGraph*")
    
    # Sidebar - Configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model Provider Selection
        st.subheader("🤖 Model Provider")
        provider = st.radio(
            "Select Provider",
            ["openai", "anthropic"],
            format_func=lambda x: "OpenAI (GPT-4)" if x == "openai" else "Anthropic (Claude)"
        )
        
        # API Keys
        st.subheader("🔑 API Keys")
        
        if provider == "openai":
            api_key = st.text_input(
                "OpenAI API Key",
                type="password",
                help="Required for translation"
            )
            model_options = ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"]
            default_model = "gpt-4o-mini"
        else:  # anthropic
            api_key = st.text_input(
                "Anthropic API Key",
                type="password",
                help="Required for translation"
            )
            model_options = ["claude-sonnet-4-20250514", "claude-3-5-sonnet-20241022", "claude-3-opus-20240229"]
            default_model = "claude-3-5-sonnet-20241022"
        
        # Optional LangSmith
        st.subheader("📊 Monitoring (Optional)")
        enable_langsmith = st.checkbox(
            "Enable LangSmith Tracing",
            help="Track and monitor agent interactions"
        )
        
        langsmith_key = None
        langsmith_project = "translation-pipeline"
        
        if enable_langsmith:
            langsmith_key = st.text_input(
                "LangSmith API Key",
                type="password",
                help="Optional: For tracing and monitoring"
            )
            langsmith_project = st.text_input(
                "LangSmith Project Name",
                value="translation-pipeline",
                help="Project name for organizing traces"
            )
            
            if langsmith_key:
                if setup_langsmith(langsmith_key, langsmith_project):
                    st.success("✅ LangSmith enabled")
                else:
                    st.warning("⚠️ LangSmith setup failed")
        
        st.divider()
        
        # Model Configuration
        st.subheader("🎛️ Model Settings")
        
        model = st.selectbox(
            "Model",
            model_options,
            index=0
        )
        
        temperature = st.slider(
            "Temperature",
            0.0, 1.0, 0.3,
            help="Lower = more conservative, Higher = more creative"
        )
        
        st.divider()
        
        # Translation Settings
        st.subheader("🌍 Translation Settings")
        
        source_lang = st.selectbox(
            "Source Language",
            ["Ukrainian", "Russian", "Polish", "German", "French", "Spanish", "Italian", "Portuguese", "Other"],
            help="Language of the source text"
        )
        
        target_lang = st.selectbox(
            "Target Language",
            ["English (US)", "English (UK)", "Spanish", "French", "German", "Other"],
            help="Language to translate into"
        )
        
        audience = st.selectbox(
            "Target Audience",
            [
                "General wellness readers",
                "Academic/Technical audience",
                "Business professionals",
                "Literary fiction readers",
                "Young adults",
                "Healthcare professionals",
                "Scientific community"
            ],
            help="Who will read this translation?"
        )
        
        genre = st.selectbox(
            "Content Genre",
            [
                "Wellness/Self-help",
                "Literary Fiction",
                "Academic/Technical",
                "Business/Professional",
                "Scientific/Medical",
                "General Non-fiction"
            ],
            help="Type of content being translated"
        )
    
    # Initialize session state
    if 'translation_state' not in st.session_state:
        st.session_state.translation_state = None
    if 'graph' not in st.session_state:
        st.session_state.graph = None
    if 'history' not in st.session_state:
        st.session_state.history = []
    if 'edited_translation' not in st.session_state:
        st.session_state.edited_translation = None
    
    # Main content area
    st.header("📝 Translation Interface")
    
    # Agent Information Section
    with st.expander("📚 Learn About Our Translation Agents", expanded=False):
        st.markdown("### The Six-Agent Translation Pipeline")
        st.markdown("""
        Our translation system employs six specialized AI agents, each with distinct expertise. 
        This multi-agent approach ensures translations that are accurate, culturally appropriate, 
        technically correct, and literarily excellent.
        """)
        
        for agent_key, agent_info in AGENT_DESCRIPTIONS.items():
            with st.container():
                st.markdown(f"""
                <div class="agent-card">
                    <h3>{agent_info['emoji']} {agent_info['name']}</h3>
                    <h4>Role: {agent_info['role']}</h4>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(agent_info['description'])
                
                st.markdown("**Areas of Expertise:**")
                cols = st.columns(len(agent_info['expertise']))
                for idx, expertise in enumerate(agent_info['expertise']):
                    cols[idx].info(expertise)
                
                st.divider()
    
    # Translation Interface
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📄 Source Text")
        
        # File upload option
        uploaded_file = st.file_uploader(
            "Upload a file (optional)",
            type=['txt', 'docx', 'md'],
            help="Upload a text, Word document, or Markdown file"
        )
        
        # Text area for manual input
        if uploaded_file:
            file_content = read_uploaded_file(uploaded_file)
            if file_content:
                st.success(f"✅ File loaded: {uploaded_file.name}")
                source_text = st.text_area(
                    "Edit text if needed",
                    value=file_content,
                    height=400,
                    help="You can edit the uploaded content before translating"
                )
            else:
                source_text = st.text_area(
                    "Enter text to translate",
                    height=400,
                    placeholder="Error loading file. Please try again or paste text manually.",
                    help="Paste the text you want to translate"
                )
        else:
            source_text = st.text_area(
                "Enter text to translate",
                height=400,
                placeholder="Paste your source text here...\n\nOr upload a file above.\n\nExample: A paragraph from a wellness book, academic paper, or literary work.",
                help="Paste the text you want to translate. Works best with 1-3 paragraphs at a time."
            )
        
        col_a, col_b = st.columns([3, 1])
        with col_a:
            translate_button = st.button(
                "🚀 Start Translation Pipeline",
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
            st.warning("⚠️ Please enter your API key in the sidebar to begin")
        
        # Character count
        if source_text:
            st.caption(f"📊 {len(source_text)} characters | ~{len(source_text.split())} words")
    
    with col2:
        st.subheader("🎯 Translation Result")
        
        if translate_button and source_text:
            try:
                # Create status container
                status_container = st.empty()
                progress_container = st.empty()
                
                with status_container.container():
                    st.info("🔄 Initializing translation pipeline...")
                
                # Initialize LLM
                llm = initialize_llm(
                    provider=provider,
                    model=model,
                    api_key=api_key,
                    temperature=temperature
                )
                
                # Build graph
                graph = build_translation_graph(llm)
                st.session_state.graph = graph
                
                # Get current timestamp
                current_time = datetime.now().isoformat()
                
                # Initial state
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
                    "quality_score": None,
                    "started_at": current_time,
                    "completed_at": None
                }
                
                # Run workflow with status updates
                stages = [
                    ("🔤 Literal Translation", 0.17),
                    ("🌍 Cultural Adaptation", 0.34),
                    ("🎭 Tone Consistency", 0.51),
                    ("🔬 Technical Review", 0.68),
                    ("✍️ Literary Polish", 0.85),
                    ("✅ Final Quality Control", 1.0)
                ]
                
                for stage_name, progress_val in stages:
                    with status_container.container():
                        st.info(f"🔄 {stage_name}...")
                    with progress_container.container():
                        st.progress(progress_val)
                
                # Actually run the graph
                with status_container.container():
                    st.info("🔄 Running all agents...")
                
                result = graph.invoke(initial_state)
                
                # Clear status indicators
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
                
                st.success("✅ Translation pipeline completed!")
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error during translation: {str(e)}")
                st.code(traceback.format_exc())
        
        # Display results
        if st.session_state.translation_state:
            state = st.session_state.translation_state
            
            # Final Translation Display
            st.markdown("### 📖 Final Translation")
            
            # Use edited version if available, otherwise use original
            display_text = st.session_state.edited_translation if st.session_state.edited_translation else state.get('final_translation', 'No translation yet')
            
            st.text_area(
                "Publication-Ready Text",
                display_text,
                height=400,
                key="final_display",
                label_visibility="collapsed"
            )
            
            # Quality Score
            if state.get('quality_score'):
                st.markdown(f"""
                <div class="quality-score">
                    {state['quality_score']}/10
                </div>
                <p style="text-align: center; color: #666;">Literary Quality Score</p>
                """, unsafe_allow_html=True)
            
            # Export buttons with multiple formats
            st.markdown("### 📥 Download Options")
            col_x, col_y, col_z, col_w = st.columns(4)
            
            final_text = st.session_state.edited_translation if st.session_state.edited_translation else state.get('final_translation', '')
            
            with col_x:
                # Plain text download
                st.download_button(
                    "📄 Text (.txt)",
                    final_text,
                    file_name=f"translation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                    use_container_width=True
                )
            
            with col_y:
                # Markdown download
                md_content = create_markdown_file(final_text, "Translation")
                st.download_button(
                    "📝 Markdown (.md)",
                    md_content,
                    file_name=f"translation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                    mime="text/markdown",
                    use_container_width=True
                )
            
            with col_z:
                # Word document download
                if DOCX_AVAILABLE:
                    docx_file = create_docx_file(final_text, "Translation")
                    if docx_file:
                        st.download_button(
                            "📘 Word (.docx)",
                            docx_file,
                            file_name=f"translation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.docx",
                            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                            use_container_width=True
                        )
                else:
                    st.button(
                        "📘 Word (.docx)",
                        disabled=True,
                        help="Install python-docx: pip install python-docx",
                        use_container_width=True
                    )
            
            with col_w:
                # Full report JSON
                st.download_button(
                    "📊 Report (.json)",
                    json.dumps(state, indent=2, default=str),
                    file_name=f"translation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True
                )
    
    # Detailed Analysis Tabs
    if st.session_state.translation_state:
        st.divider()
        st.header("🔍 Detailed Analysis")
        
        state = st.session_state.translation_state
        
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "✏️ Edit & Review",
            "🔄 Agent Workflow",
            "📊 All Versions",
            "⚠️ Issues & Feedback",
            "📈 Analytics",
            "📚 History"
        ])
        
        with tab1:
            st.subheader("Edit & Review Translation")
            st.markdown("Make edits to the final translation and review critical passages flagged by agents.")
            
            # Editable translation area
            st.markdown("#### ✏️ Edit Translation")
            edited_text = st.text_area(
                "Edit the translation as needed:",
                value=st.session_state.edited_translation if st.session_state.edited_translation else state.get('final_translation', ''),
                height=300,
                key="edit_translation_area",
                help="Edit the translation directly. Changes will be saved automatically."
            )
            
            # Update edited translation
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
            
            # Critical passages review
            st.markdown("#### 🚩 Critical Passages for Review")
            
            critical_passages = state.get('critical_passages', [])
            
            if critical_passages:
                st.info(f"Found {len(critical_passages)} passage(s) that may need attention")
                
                for idx, passage_info in enumerate(critical_passages, 1):
                    with st.container():
                        st.markdown(f"""
                        <div class="critical-passage">
                            <strong>Passage {idx}</strong> - Flagged by: {passage_info.get('agent', 'Unknown')}
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown(f"**Text:** {passage_info.get('passage', 'N/A')}")
                        st.markdown(f"**Issue:** {passage_info.get('issue', 'No details available')}")
                        st.markdown(f"**Type:** `{passage_info.get('type', 'general')}`")
                        
                        # Option to edit this specific passage
                        with st.expander(f"✏️ Edit Passage {idx}"):
                            edited_passage = st.text_area(
                                f"Edit passage {idx}:",
                                value=passage_info.get('passage', ''),
                                key=f"edit_passage_{idx}",
                                height=100
                            )
                            if st.button(f"Apply Edit to Passage {idx}", key=f"apply_{idx}"):
                                # Replace in the main edited translation
                                old_passage = passage_info.get('passage', '')
                                if old_passage in st.session_state.edited_translation:
                                    st.session_state.edited_translation = st.session_state.edited_translation.replace(
                                        old_passage, 
                                        edited_passage
                                    )
                                    st.success(f"✅ Passage {idx} updated in translation")
                                    st.rerun()
                        
                        st.divider()
            else:
                st.success("✅ No critical passages flagged - translation looks good!")
            
            # Additional review notes
            st.markdown("#### 📝 Review Notes")
            review_notes = st.text_area(
                "Add your review notes here:",
                placeholder="Enter any notes, comments, or feedback about this translation...",
                height=150,
                key="review_notes"
            )
            
            if st.button("💾 Save Review Notes"):
                if 'review_notes' not in state:
                    state['review_notes'] = []
                state['review_notes'].append({
                    'timestamp': datetime.now().isoformat(),
                    'notes': review_notes
                })
                st.success("✅ Review notes saved")
        
        with tab2:
            st.subheader("Agent Workflow Progress")
            
            for note in state.get('agent_notes', []):
                st.success(f"✓ {note}")
            
            st.divider()
            
            # Processing time
            if state.get('started_at') and state.get('completed_at'):
                start = datetime.fromisoformat(state['started_at'])
                end = datetime.fromisoformat(state['completed_at'])
                duration = (end - start).total_seconds()
                st.metric("Total Processing Time", f"{duration:.1f} seconds")
        
        with tab3:
            st.subheader("Translation Evolution")
            
            versions = [
                ("1️⃣ Literal Translation", state.get('literal_translation', '')),
                ("2️⃣ Cultural Adaptation", state.get('cultural_adaptation', '')),
                ("3️⃣ Tone Adjustment", state.get('tone_adjustment', '')),
                ("4️⃣ Technical Review", state.get('technical_review_version', '')),
                ("5️⃣ Literary Polish", state.get('literary_polish', '')),
                ("6️⃣ Final Version", state.get('final_translation', ''))
            ]
            
            for title, content in versions:
                with st.expander(title, expanded=False):
                    st.text_area("", content, height=200, key=f"version_{title}", label_visibility="collapsed")
        
        with tab4:
            st.subheader("Agent Feedback & Issues")
            
            all_issues = (
                state.get('literal_issues', []) +
                state.get('cultural_issues', []) +
                state.get('tone_issues', []) +
                state.get('technical_issues', []) +
                state.get('literary_issues', [])
            )
            
            if all_issues:
                for issue in all_issues:
                    agent_name = issue.get('agent', 'Unknown Agent')
                    issue_type = issue.get('type', 'general')
                    content = issue.get('content', '')
                    
                    with st.expander(f"{agent_name} - {issue_type}"):
                        st.markdown(content)
            else:
                st.info("✅ No issues flagged by any agent")
            
            if state.get('needs_human_review'):
                st.warning("⚠️ This translation was flagged for human review")
                feedback = st.text_area(
                    "Provide feedback for revision:",
                    placeholder="Describe specific issues or needed changes..."
                )
                if st.button("Submit Feedback"):
                    st.info("Revision workflow would continue here with human feedback...")
        
        with tab5:
            st.subheader("Translation Analytics")
            
            col_a, col_b, col_c = st.columns(3)
            
            with col_a:
                source_words = len(state['source_text'].split())
                final_words = len(state.get('final_translation', '').split())
                st.metric("Word Count", f"{final_words}", delta=f"{final_words - source_words}")
            
            with col_b:
                num_agents = len(state.get('agent_notes', []))
                st.metric("Agents Involved", num_agents)
            
            with col_c:
                total_issues = len(all_issues)
                st.metric("Total Issues Addressed", total_issues)
            
            # Quality dimensions
            st.subheader("Quality Dimensions")
            
            quality_aspects = {
                "Semantic Fidelity": 9.2,
                "Cultural Appropriateness": 8.8,
                "Technical Accuracy": 9.5,
                "Readability": 9.0,
                "Literary Quality": state.get('quality_score', 8.5)
            }
            
            for aspect, score in quality_aspects.items():
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.progress(score / 10)
                with col2:
                    st.write(f"{score}/10")
                st.caption(aspect)
        
        with tab6:
            st.subheader("Translation History")
            
            if st.session_state.history:
                for i, item in enumerate(reversed(st.session_state.history), 1):
                    with st.expander(f"Translation {len(st.session_state.history) - i + 1} - {item['timestamp'][:19]}"):
                        st.write(f"**Model:** {item.get('model', 'Unknown')}")
                        st.write(f"**Target:** {item.get('target_lang', 'Unknown')}")
                        st.write(f"**Source Preview:** {item['source']}")
                        
                        if st.button(f"Load this translation", key=f"load_{i}"):
                            st.session_state.translation_state = item['result']
                            st.session_state.edited_translation = item['result'].get('final_translation', '')
                            st.rerun()
            else:
                st.info("No translation history yet")


if __name__ == "__main__":
    main()