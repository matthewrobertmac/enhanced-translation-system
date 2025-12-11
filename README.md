# enhanced-translation-system

## Introduction

Deployed at: https://advancedtranslationsystem.com

Enhanced Multi-Agent Translation Workflow with LangGraph and LangSmith

A sophisticated translation system with cultural adaptation, literary editing, comprehensive monitoring, and visuals.

✨ Key Features

- 🧠 INTELLIGENT PLANNING AGENT - dynamically selects required agents
- 🧠 7 specialized translation agents with distinct roles (including BERTScore validator)
- ⚡ **SMART SEMANTIC CACHING** - 5-10x speedup on similar content
- 🎯 **CONFIDENCE SCORES** - Multi-metric translation quality assessment
- 🔀 **DIFF VISUALIZATION** - Visual comparison between agent versions
- 🎲 **ALTERNATIVE TRANSLATIONS** - Generate and compare multiple variants
- 🧩 Support for OpenAI and Anthropic models (e.g., GPT-4o, Claude-3.5-Sonnet)
- 📊 Optional LangSmith integration for detailed tracing, monitoring, and reproducibility
- 💬 Comprehensive agent feedback system with issue tracking and human-review flags
- 📁 File upload support (.txt, .docx, .md)
- 📤 Multiple export formats (.txt, .docx, .md)
- 🚨 Critical passage flagging and review
- 🔄 Safe same-language (e.g., English→English) refinement mode
- 🎯 BERTScore validation with iterative refinement
- 📈 Visualizations: word counts, sentence-length histograms, readability, issue counts, BERTScore bars
- ☁️ Word clouds: Source, Final, and Difference (words added)
- 🎯 Entity tracking and network visualization
- 🔊 **TTS AUDIO PLAYBACK** - Listen to translated text aloud via ElevenLabs

## Getting Started

### Setup Guide:

```bash
# 1. Clone the repository
git clone https://github.com/matthewrobertmac/enhanced-translation-system.git
cd enhanced-translation-system

# 2. Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate    # On macOS/Linux
# .\.venv\Scripts\activate   # On Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run Streamlit App
streamlit run app.py
```

### Sidebar (Streamlit UI Component)
When the application is launched using `streamlit run app.py`, Streamlit renders a left-hand **sidebar**.  
This sidebar acts as an interactive control panel where users can configure the translation system before running the pipeline.  
All controls are implemented using Streamlit’s built-in `st.sidebar` interface.

### Sidebar Configuration Steps

1. **Select provider:**
   - `openai`
   - `anthropic`

2. **Paste your API key**

3. **Choose a model:**
   - **OpenAI:** `gpt-4o`, `gpt-4-turbo`, `gpt-3.5-turbo`
   - **Anthropic:** `claude-3-5-sonnet-20241022`, `claude-3-opus-20240229`

4. **Adjust temperature as needed**

The sidebar dynamically updates based on user selections, allowing for flexible experimentation across different model providers and configurations.


## System Architecture

### 1. UI Layer (Streamlit)

- Text input and file uploads (`.txt`, `.md`, `.docx`)  
- Model provider selection (OpenAI / Anthropic)  
- Source / target language selection  
- Audience selection (General / Academic / Business / Literary)  
- Buttons to run, clear state, and download outputs  
- Tabs for analysis, diff view, alternatives, planning, word clouds, and entities  
- Cache statistics & clear-cache button  

### 2. Translation Pipeline (LangGraph)

- Uses `StateGraph` with a shared `TranslationState`  
- `PlanningAgent` analyzes the input and generates an `agent_plan`  
- `route_to_next_agent(state)` routes execution based on:
  - `agent_plan`
  - `current_agent_index`  
- Supports:
  - Dynamic selection of agents  
  - Early stopping when plan is complete  
  - Checkpointing via `MemorySaver`  

### 3. Multi-Agent System

Each “agent” is a specialized LLM-powered module (same base model, different prompts and logic):

- **PlanningAgent**  
  - Analyzes source text  
  - Decides which agents are needed  
  - Builds `agent_plan` & reasoning  

- **LiteralTranslationAgent**  
  - Produces baseline translation or same-language refinement  
  - Uses semantic cache for exact/similar hits  
  - Can be entity-aware (preserve key terms)  

- **CulturalAdaptationAgent**  
  - Adapts idioms, cultural references, and register  
  - Produces `cultural_adaptation` + notes  

- **ToneConsistencyAgent**  
  - Adjusts tone and voice for target audience  
  - Produces `tone_adjustment` + tone notes  

- **TechnicalReviewAgent**  
  - Checks terminology, units, formats, and technical correctness  
  - Produces `technical_review_version` + technical notes  

- **LiteraryEditorAgent**  
  - Polishes style and flow for publication  
  - Produces `literary_polish` + literary notes  

- **QualityControlAgent (Finalizer)**  
  - Chooses the best latest version  
  - Runs final LLM pass to produce `final_translation`  
  - Extracts critical passages based on issues  
  - Runs entity extraction on final text  
  - Computes confidence scores  

- **BERTScoreValidatorAgent**  
  - Only used when source and target languages are equivalent  
  - Computes BERTScore (if available)  
  - Optionally refines the output up to a target F1 threshold  

### 4. Backend Services

- **SemanticTranslationCache (`src/services/cache.py`)**
  - Exact match cache based on (text + context) hash  
  - Semantic-similar match using `SentenceTransformer` embeddings  
  - Stores translations, embeddings, and stats  
  - Provides cache hit rate and supports export/import  

- **EntitiesTracker (`src/services/entities.py`)**
  - Uses a default glossary (`DEFAULT_ENTITY_GLOSSARY`)  
  - Detects:
    - Glossary-defined entities (AI, LLM, NLP, etc.)  
    - Auto-detected persons, locations, organizations, dates  
  - Supports glossary import from JSON/CSV/TXT  

- **ConfidenceScorer (`src/services/scoring.py`)**
  - Combines:
    - Semantic fidelity (BERTScore for same-language, or default)  
    - Length ratio  
    - Fluency (LLM-based rating)  
    - Terminology preservation (entity coverage)  
  - Produces an `overall` confidence score  

- **AlternativeTranslationGenerator (`src/services/alternatives.py`)**
  - Generates multiple variants:
    - Conservative (literal, lower temperature)  
    - Balanced  
    - Creative (more idiomatic)  
  - Uses the same LLM with different prompting strategies  

### 5. Output Layer

- Final translation (editable in UI)  
- 3 alternative variants (if requested)  
- Download options:
  - Plain text (`.txt`)  
  - Markdown (`.md`)  
  - Word document (`.docx`) with title + timestamp  
- Visual analytics:
  - Diff view (baseline vs. final)  
  - Word clouds (source vs. final)  
  - Entity network graph  
  - Confidence score display  

## Project Structure

```bash
project_root/
├── app.py
├── legacy/                       # ← Old pre-refactor modules (kept for reference)
│   ├── entities_tracker_module.py
│   ├── old_backend.py
│   └── old_pipeline_experiments/
│
└── src/
    ├── agents/
    │   ├── workers.py            # All agent classes
    │   └── workflow.py           # LangGraph pipeline + routing
    │
    ├── analysis/
    │   ├── metrics.py            # Readability, stats
    │   └── visualizations.py     # Diff, wordcloud, entity graph
    │
    ├── core/
    │   ├── llm.py                # OpenAI/Anthropic initialization
    │   └── state.py              # TranslationState schema
    │
    ├── data/
    │   └── stopwords.py          # Multilingual stopwords
    │
    ├── services/
    │   ├── cache.py              # Semantic cache
    │   ├── entities.py           # Entity extraction + glossary
    │   ├── scoring.py            # Confidence scoring + BERTScore
    │   └── alternatives.py       # Alternative translation generator
    │
    ├── ui/
    │   ├── session.py            # Streamlit session state
    │   └── styles.py             # Custom CSS
    │
    └── utils/
        ├── common.py             # Helpers (notes, similarity, etc.)
        └── files.py              # TXT/MD/DOCX export
```
