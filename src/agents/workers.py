# src/agents/workers.py
import json
from datetime import datetime
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.language_models import BaseChatModel

# Imports from our new structure
from src.config import LANGUAGE_GUARDRAIL
from src.core.state import TranslationState
from src.utils.common import split_notes, languages_equivalent, text_similarity, extract_critical_passages
from src.services.cache import SemanticTranslationCache
from src.services.entities import EntitiesTracker
from src.services.scoring import ConfidenceScorer, compute_bertscore

# =====================
# HELPER: Language Reinforcement
# =====================
async def reinforce_language(llm: BaseChatModel, text: str, target_language: str) -> str:
    """Lightweight fixer to ensure output remains in the target language."""
    try:
        resp = await llm.ainvoke([
            SystemMessage(content="Ensure the following text is in the specified language only, without code-switching."),
            HumanMessage(content=f"{LANGUAGE_GUARDRAIL}\n\nTarget language: {target_language}\n\nText:\n{text}\n\nReturn the corrected text in {target_language}, with no extra commentary.")
        ])
        return resp.content.strip()
    except Exception:
        return text

# =====================
# 1. PLANNING AGENT
# =====================
class PlanningAgent:
    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        self.name = "Workflow Planning Specialist"
        self.emoji = "📋"

    async def analyze_and_plan(self, state: TranslationState) -> TranslationState:
        # If plan exists, skip
        if state.get('agent_plan'):
            state['agent_notes'].append(f"{self.emoji} {self.name}: Plan already exists - skipping")
            return state

        # Check manual override
        if not state.get('planning_enabled', True):
            # Return default full plan
            state['agent_plan'] = [
                'literal_translator', 'cultural_adapter', 'tone_specialist',
                'technical_reviewer', 'literary_editor', 'finalize'
            ]
            # Add BERTScore if same language
            if languages_equivalent(state['source_language'], state['target_language']):
                state['agent_plan'].append('bertscore_validator')
                
            state['agent_notes'].append(f"{self.emoji} {self.name}: Planning disabled - running default flow")
            state['current_agent_index'] = 0
            return state

        # Planning Logic
        source_text = state['source_text']
        system_prompt = f"""You are an expert translation workflow planner.
        Analyze the source text and determine which agents are needed.
        
        AVAILABLE AGENTS:
        - literal_translator (Required)
        - cultural_adapter (For idioms/localization)
        - tone_specialist (For readability/voice)
        - technical_reviewer (For terminology/numbers)
        - literary_editor (For creative/prose)
        - finalize (Required)
        - bertscore_validator (Same-language only)

        OUTPUT JSON FORMAT: {{ "required_agents": ["literal_translator", ...], "reasoning": {{...}} }}
        """
        
        user_prompt = f"Analyze this text (Length: {len(source_text)} chars):\n{source_text[:1000]}..."
        
        try:
            response = await self.llm.ainvoke([
                SystemMessage(content=system_prompt), 
                HumanMessage(content=user_prompt)
            ])
            
            # Parse JSON
            content = response.content.strip().replace('```json', '').replace('```', '')
            plan_data = json.loads(content)
            
            required = plan_data.get('required_agents', [])
            
            # Enforce mandatory agents
            if 'literal_translator' not in required: required.insert(0, 'literal_translator')
            if 'finalize' not in required: required.append('finalize')
            
            state['agent_plan'] = required
            state['agent_plan_reasoning'] = plan_data.get('reasoning', {})
            state['agent_notes'].append(f"{self.emoji} {self.name}: Plan created with {len(required)} agents")
            
        except Exception as e:
            # Fallback
            state['agent_plan'] = ['literal_translator', 'finalize']
            state['agent_notes'].append(f"{self.emoji} {self.name}: Planning failed ({str(e)}) - using minimal fallback")
            
        state['current_agent_index'] = 0
        return state

# =====================
# 2. LITERAL TRANSLATOR
# =====================
class LiteralTranslationAgent:
    def __init__(self, llm: BaseChatModel, cache: SemanticTranslationCache, enable_awareness: bool = False):
        self.llm = llm
        self.cache = cache
        self.enable_awareness = enable_awareness
        self.name = "Baseline Specialist"
        self.emoji = "🔤"

    async def translate(self, state: TranslationState) -> TranslationState:
        if state.get('literal_translation'):
            state['current_agent_index'] += 1
            return state

        # 1. Check Cache
        context = {
            'source_lang': state['source_language'],
            'target_lang': state['target_language'],
            'audience': state['target_audience'],
            'genre': state.get('genre', 'General')
        }
        
        if self.cache:
            cached = await self.cache.get_cached_translation(state['source_text'], context)
            if cached and cached['type'] == 'exact':
                state['literal_translation'] = cached['translation']
                state['cache_hit'] = 'exact'
                state['agent_notes'].append(f"⚡ {self.emoji} {self.name}: CACHE HIT (Exact)")
                state['current_agent_index'] += 1
                return state

        # 2. Run Translation
        same_lang = languages_equivalent(state['source_language'], state['target_language'])
        system_prompt = f"You provide the baseline pass.\n{LANGUAGE_GUARDRAIL}"
        
        if same_lang:
            task = "Refine this text preserving meaning. Return literal translation then 'EDITOR NOTES:'"
        else:
            task = f"Translate from {state['source_language']} to {state['target_language']}. Return translation then 'TRANSLATOR NOTES:'"

        user_prompt = f"{task}\n\nTEXT:\n{state['source_text']}\n\nAUDIENCE: {state['target_audience']}"
        
        # Entity Awareness
        if self.enable_awareness and state.get('source_entities'):
            entities = ", ".join([e['name'] for e in state['source_entities'][:10]])
            user_prompt += f"\n\nKeep these entities consistent: {entities}"
        
        try:
            response = await self.llm.ainvoke([
                SystemMessage(content=system_prompt), 
                HumanMessage(content=user_prompt)
            ])
            
            # Process Output
            translation, _, notes = split_notes(response.content, ["TRANSLATOR NOTES:", "EDITOR NOTES:"])
            translation = await reinforce_language(self.llm, translation, state['target_language'])
            
            state['literal_translation'] = translation
            if notes:
                state['literal_issues'] = [{"agent": self.name, "content": notes, "type": "notes"}]
                
            # Save to cache
            if self.cache:
                self.cache.store_translation(state['source_text'], translation, context, {'agent': self.name})
            
            state['agent_notes'].append(f"{self.emoji} {self.name}: Baseline complete")
            
        except Exception as e:
            state['literal_translation'] = state['source_text'] # Fallback
            state['agent_notes'].append(f"{self.emoji} {self.name}: Error - {str(e)}")
            
        state['current_agent_index'] += 1
        return state

# =====================
# 3. CULTURAL ADAPTER
# =====================
class CulturalAdaptationAgent:
    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        self.name = "Cultural Localization Expert"
        self.emoji = "🌍"

    async def adapt(self, state: TranslationState) -> TranslationState:
        if state.get('cultural_adaptation'):
            state['current_agent_index'] += 1
            return state
            
        prev_text = state.get('literal_translation', "")
        
        system_prompt = f"You adapt content for cultural naturalness.\n{LANGUAGE_GUARDRAIL}"
        user_prompt = f"Adapt idioms and cultural references for {state['target_language']}.\n\nTEXT:\n{prev_text}\n\nOutput adapted text then 'CULTURAL NOTES:'"
        
        try:
            response = await self.llm.ainvoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])
            adapted, _, notes = split_notes(response.content, ["CULTURAL NOTES:"])
            state['cultural_adaptation'] = adapted
            if notes: state['cultural_issues'] = [{"agent": self.name, "content": notes, "type": "cultural"}]
            state['agent_notes'].append(f"{self.emoji} {self.name}: Adaptation complete")
        except Exception as e:
            state['cultural_adaptation'] = prev_text
            state['agent_notes'].append(f"{self.emoji} {self.name}: Skipped due to error")

        state['current_agent_index'] += 1
        return state

# =====================
# 4. TONE SPECIALIST
# =====================
class ToneConsistencyAgent:
    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        self.name = "Tone & Voice Consistency Director"
        self.emoji = "🎭"

    async def adjust_tone(self, state: TranslationState) -> TranslationState:
        if state.get('tone_adjustment'):
            state['current_agent_index'] += 1
            return state
            
        # Get latest text
        prev_text = state.get('cultural_adaptation') or state.get('literal_translation') or ""
        
        user_prompt = f"Adjust tone for audience: {state['target_audience']}.\n\nTEXT:\n{prev_text}\n\nOutput text then 'TONE NOTES:'"
        
        try:
            response = await self.llm.ainvoke([
                SystemMessage(content=f"Ensure consistent voice.\n{LANGUAGE_GUARDRAIL}"), 
                HumanMessage(content=user_prompt)
            ])
            adjusted, _, notes = split_notes(response.content, ["TONE NOTES:"])
            state['tone_adjustment'] = adjusted
            if notes: state['tone_issues'] = [{"agent": self.name, "content": notes, "type": "tone"}]
            state['agent_notes'].append(f"{self.emoji} {self.name}: Tone adjusted")
        except Exception:
            state['tone_adjustment'] = prev_text
            
        state['current_agent_index'] += 1
        return state

# =====================
# 5. TECHNICAL REVIEWER
# =====================
class TechnicalReviewAgent:
    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        self.name = "Technical Accuracy Auditor"
        self.emoji = "🔬"

    async def review(self, state: TranslationState) -> TranslationState:
        if state.get('technical_review_version'):
            state['current_agent_index'] += 1
            return state
            
        prev_text = state.get('tone_adjustment') or state.get('cultural_adaptation') or state.get('literal_translation')
        
        user_prompt = f"Check terminology and units.\n\nTEXT:\n{prev_text}\n\nOutput text then 'TECHNICAL NOTES:'"
        
        try:
            response = await self.llm.ainvoke([
                SystemMessage(content=f"Verify technical accuracy.\n{LANGUAGE_GUARDRAIL}"), 
                HumanMessage(content=user_prompt)
            ])
            reviewed, _, notes = split_notes(response.content, ["TECHNICAL NOTES:"])
            state['technical_review_version'] = reviewed
            if notes: state['technical_issues'] = [{"agent": self.name, "content": notes, "type": "technical"}]
            state['agent_notes'].append(f"{self.emoji} {self.name}: Review complete")
        except Exception:
            state['technical_review_version'] = prev_text
            
        state['current_agent_index'] += 1
        return state

# =====================
# 6. LITERARY EDITOR
# =====================
class LiteraryEditorAgent:
    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        self.name = "Literary Style Editor"
        self.emoji = "✍️"

    async def polish(self, state: TranslationState) -> TranslationState:
        if state.get('literary_polish'):
            state['current_agent_index'] += 1
            return state
            
        prev_text = state.get('technical_review_version') or state.get('tone_adjustment') or state.get('literal_translation')
        
        user_prompt = f"Polish prose for publication.\n\nTEXT:\n{prev_text}\n\nOutput text then 'LITERARY NOTES:'"
        
        try:
            response = await self.llm.ainvoke([
                SystemMessage(content=f"Elevate prose quality.\n{LANGUAGE_GUARDRAIL}"), 
                HumanMessage(content=user_prompt)
            ])
            polished, _, notes = split_notes(response.content, ["LITERARY NOTES:"])
            state['literary_polish'] = polished
            if notes: state['literary_issues'] = [{"agent": self.name, "content": notes, "type": "literary"}]
            state['agent_notes'].append(f"{self.emoji} {self.name}: Polished")
        except Exception:
            state['literary_polish'] = prev_text
            
        state['current_agent_index'] += 1
        return state

# =====================
# 7. QUALITY CONTROL (FINALIZE)
# =====================
class QualityControlAgent:
    def __init__(self, llm: BaseChatModel, entity_tracker: EntitiesTracker):
        self.llm = llm
        self.entity_tracker = entity_tracker
        self.name = "Master Quality Synthesizer"
        self.emoji = "✅"

    async def finalize(self, state: TranslationState) -> TranslationState:
        if state.get('final_translation'):
            state['current_agent_index'] += 1
            return state
            
        # Determine latest version
        latest = (state.get('literary_polish') or state.get('technical_review_version') or 
                  state.get('tone_adjustment') or state.get('cultural_adaptation') or 
                  state.get('literal_translation'))
                  
        user_prompt = f"Finalize this text for publication. No meta-comments.\n\nTEXT:\n{latest}"
        
        try:
            response = await self.llm.ainvoke([
                SystemMessage(content=f"Output final text.\n{LANGUAGE_GUARDRAIL}"), 
                HumanMessage(content=user_prompt)
            ])
            final_text = await reinforce_language(self.llm, response.content.strip(), state['target_language'])
            state['final_translation'] = final_text
            state['completed_at'] = datetime.now().isoformat()
            
            # Run Analysis
            # 1. Critical Passages
            all_issues = (state.get('literal_issues', []) + state.get('cultural_issues', []) + 
                          state.get('tone_issues', []) + state.get('technical_issues', []))
            state['critical_passages'] = extract_critical_passages(final_text, all_issues)
            
            # 2. Entity Tracking
            if self.entity_tracker:
                state['translated_entities'] = self.entity_tracker.extract_entities(final_text)
            
            # 3. Confidence Score
            scorer = ConfidenceScorer(self.llm)
            scores = await scorer.score_translation(
                state['source_text'], final_text, 
                {'source_lang': state['source_language'], 'target_lang': state['target_language']}
            )
            state['confidence_scores'] = scores
            
            state['agent_notes'].append(f"{self.emoji} {self.name}: Finalized")
            
        except Exception as e:
            state['final_translation'] = latest
            state['agent_notes'].append(f"{self.emoji} {self.name}: Finalization error - {e}")
            
        state['current_agent_index'] += 1
        return state

# =====================
# 8. BERTSCORE VALIDATOR
# =====================
class BERTScoreValidatorAgent:
    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        self.name = "Semantic Fidelity Validator"
        self.emoji = "🎯"
        self.target_score = 0.8

    async def validate_and_refine(self, state: TranslationState) -> TranslationState:
        # Only for same-language
        if not languages_equivalent(state['source_language'], state['target_language']):
             state['current_agent_index'] += 1
             return state
             
        current = state['final_translation']
        source = state['source_text']
        
        # Initialize history
        if 'bertscore_history' not in state: state['bertscore_history'] = []
        if 'bertscore_attempts' not in state: state['bertscore_attempts'] = 0
        
        scores = compute_bertscore(current, source)
        if not scores:
            state['current_agent_index'] += 1
            return state
            
        f1 = scores['f1']
        state['bertscore_history'].append({'attempt': state['bertscore_attempts'], 'f1': f1})
        state['bertscore_attempts'] += 1
        
        if f1 < self.target_score and state['bertscore_attempts'] < 3:
            state['agent_notes'].append(f"{self.emoji} {self.name}: Refining (F1={f1:.2f})")
            # Refine Logic
            prompt = f"Improve semantic similarity to source (F1: {f1:.2f}).\nSource: {source}\nCurrent: {current}"
            try:
                resp = await self.llm.ainvoke([HumanMessage(content=prompt)])
                state['final_translation'] = resp.content.strip()
            except Exception:
                pass
        else:
            state['agent_notes'].append(f"{self.emoji} {self.name}: Validation passed (F1={f1:.2f})")
            state['current_agent_index'] += 1
            
        return state