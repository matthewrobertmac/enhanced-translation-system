# src/core/llm.py
import os
from typing import Optional, Literal
from langchain_core.language_models import BaseChatModel

# Model imports with protection
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

# LangSmith imports
try:
    from langsmith import Client
    LANGSMITH_AVAILABLE = True
except ImportError:
    LANGSMITH_AVAILABLE = False

def initialize_llm(
    provider: Literal["openai", "anthropic"],
    model: str,
    api_key: str,
    temperature: float = 0.3
) -> BaseChatModel:
    """
    Factory function to initialize the LLM based on provider.
    """
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

def setup_langsmith(api_key: Optional[str], project_name: str = "translation-pipeline") -> bool:
    """
    Configure LangSmith tracing if API key is provided.
    """
    if not api_key:
        return False
    if not LANGSMITH_AVAILABLE:
        print("LangSmith not installed. Install with: pip install langsmith")
        return False
    
    try:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = api_key
        os.environ["LANGCHAIN_PROJECT"] = project_name
        Client(api_key=api_key)
        return True
    except Exception as e:
        print(f"LangSmith setup failed: {str(e)}")
        return False