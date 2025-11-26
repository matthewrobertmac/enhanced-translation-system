# src/agents/workflow.py
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.language_models import BaseChatModel

from src.core.state import TranslationState

# LangGraph Checkpointing Configuration Keys
# These keys are expected by LangGraph's checkpointing system:
# - "configurable": A nested dict containing checkpoint-related settings
# - "thread_id": Unique identifier for the conversation/execution thread
# - "checkpoint_ns": Namespace for checkpoints, used to isolate different workflows
# See: https://langchain-ai.github.io/langgraph/concepts/persistence/
LANGGRAPH_CONFIG_THREAD_ID_KEY = "thread_id"
LANGGRAPH_CONFIG_CHECKPOINT_NS_KEY = "checkpoint_ns"
DEFAULT_THREAD_ID = "default_thread"
DEFAULT_CHECKPOINT_NAMESPACE = "translation"
from src.services.cache import SemanticTranslationCache
from src.services.entities import EntitiesTracker
from src.agents.workers import (
    PlanningAgent, 
    LiteralTranslationAgent, 
    CulturalAdaptationAgent,
    ToneConsistencyAgent,
    TechnicalReviewAgent,
    LiteraryEditorAgent,
    QualityControlAgent,
    BERTScoreValidatorAgent
)

def route_to_next_agent(state: TranslationState) -> str:
    """
    Router function: looks at the 'agent_plan' and 'current_agent_index' 
    to determine which node to go to next.
    """
    agent_plan = state.get('agent_plan', [])
    current_index = state.get('current_agent_index', 0)
    
    # If index exceeds plan length, we are done
    if current_index >= len(agent_plan):
        return END
    
    # Return the name of the next agent in the plan
    return agent_plan[current_index]

class TranslationPipeline:
    def __init__(self, llm: BaseChatModel, cache: SemanticTranslationCache = None, tracker: EntitiesTracker = None):
        self.llm = llm
        self.cache = cache or SemanticTranslationCache()
        self.entity_tracker = tracker or EntitiesTracker()
        self.enable_entity_awareness = True 

    def build_workflow(self):
        workflow = StateGraph(TranslationState)
        
        # 1. Initialize Agents
        planner = PlanningAgent(self.llm)
        literal = LiteralTranslationAgent(self.llm, self.cache, self.enable_entity_awareness)
        cultural = CulturalAdaptationAgent(self.llm)
        tone = ToneConsistencyAgent(self.llm)
        technical = TechnicalReviewAgent(self.llm)
        literary = LiteraryEditorAgent(self.llm)
        qc = QualityControlAgent(self.llm, self.entity_tracker)
        bert_validator = BERTScoreValidatorAgent(self.llm)
        
        # 2. Add Nodes
        # Node names must match what strings are used in 'agent_plan'
        workflow.add_node("planning", planner.analyze_and_plan)
        workflow.add_node("literal_translator", literal.translate)
        workflow.add_node("cultural_adapter", cultural.adapt)
        workflow.add_node("tone_specialist", tone.adjust_tone)
        workflow.add_node("technical_reviewer", technical.review)
        workflow.add_node("literary_editor", literary.polish)
        workflow.add_node("finalize", qc.finalize)
        workflow.add_node("bertscore_validator", bert_validator.validate_and_refine)
        
        # 3. Define Edges
        # Start at planning
        workflow.set_entry_point("planning")
        
        # All nodes route dynamically based on the plan
        nodes = [
            "planning", 
            "literal_translator", 
            "cultural_adapter", 
            "tone_specialist",
            "technical_reviewer",
            "literary_editor",
            "finalize",
            "bertscore_validator"
        ]
        
        for node in nodes:
            workflow.add_conditional_edges(node, route_to_next_agent)
            
        # Compile with memory for checkpointing (state persistence)
        checkpointer = MemorySaver()
        return workflow.compile(checkpointer=checkpointer)

    async def run(self, state: TranslationState):
        """
        Execute the workflow.
        
        The config dictionary uses LangGraph's checkpointing system keys:
        - thread_id: Unique identifier for execution thread (for state persistence)
        - checkpoint_ns: Namespace to isolate checkpoints from different workflows
        
        These keys are wrapped in "configurable" as required by LangGraph.
        """
        app = self.build_workflow()
        
        # Build configuration for LangGraph checkpointing
        # The "configurable" wrapper is required by LangGraph's invoke API
        config = {
            "configurable": {
                LANGGRAPH_CONFIG_THREAD_ID_KEY: state.get("thread_id", DEFAULT_THREAD_ID),
                LANGGRAPH_CONFIG_CHECKPOINT_NS_KEY: DEFAULT_CHECKPOINT_NAMESPACE
            }
        }
        
        return await app.ainvoke(state, config=config)