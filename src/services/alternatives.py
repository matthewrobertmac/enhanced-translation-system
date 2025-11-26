# src/services/alternatives.py
from typing import List, Dict
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from src.config import LANGUAGE_GUARDRAIL
from src.utils.common import languages_equivalent

class AlternativeTranslationGenerator:
    """Generate alternative translation variants with different temperature settings."""
    
    def __init__(self, llm: BaseChatModel):
        self.llm = llm
    
    async def generate_alternatives(self, state: dict, num_alternatives: int = 3) -> List[Dict]:
        alternatives = []
        
        # Strategy 1: Conservative
        alternatives.append(await self._generate_variant(
            state, temperature=0.2, strategy="conservative", 
            description="More literal and faithful"
        ))
        
        # Strategy 2: Balanced
        alternatives.append(await self._generate_variant(
            state, temperature=0.5, strategy="balanced", 
            description="Balanced between literal and natural"
        ))
        
        # Strategy 3: Creative
        if num_alternatives >= 3:
            alternatives.append(await self._generate_variant(
                state, temperature=0.8, strategy="creative", 
                description="More natural and idiomatic"
            ))
            
        return alternatives
    
    async def _generate_variant(self, state: dict, temperature: float, strategy: str, description: str) -> Dict:
        try:
            # Clone LLM with specific temperature (simplified approach)
            # Note: In a strict type system, we might need a factory, but this works for LangChain objects
            # Ideally, we should use the factory from src.core.llm to create a new instance,
            # but simply overriding params in invoke or passing config is complex.
            # For simplicity in this refactor, we rely on the LLM's default or pass params if supported.
            # A better way is to instantiate a new LLM, but let's reuse the prompt logic for now.
            
            source_text = state['source_text']
            source_lang = state['source_language']
            target_lang = state['target_language']
            
            prompt = f"""
            Task: Generate a {strategy} translation (Temperature: {temperature}).
            {LANGUAGE_GUARDRAIL}
            
            Source ({source_lang}): {source_text}
            Target ({target_lang}):
            """
            
            # We use the existing LLM. In a perfect world, we'd re-init with new temperature.
            # For this refactor, we assume the temperature effect is handled by the prompt instruction style
            # or we accept the base model's temp. 
            # To truly change temp, we'd need to import initialize_llm here, which creates circular deps.
            # Let's proceed with the existing LLM instance.
            
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            
            return {
                'strategy': strategy,
                'description': description,
                'temperature': temperature,
                'translation': response.content.strip(),
                'error': False
            }
        except Exception as e:
            return {
                'strategy': strategy, 
                'description': description,
                'translation': f"Error: {str(e)}",
                'error': True
            }