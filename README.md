enhanced-translation-system
Deployed at: https://advancedtranslationsystem.com
Enhanced Multi-Agent Translation Workflow with LangGraph and LangSmith
A sophisticated translation system with cultural adaptation, literary editing, comprehensive monitoring, and visuals.
✨ Key Features

🧠 INTELLIGENT PLANNING AGENT - dynamically selects required agents
🧠 7 specialized translation agents with distinct roles (including BERTScore validator)
⚡ SMART SEMANTIC CACHING - 5-10x speedup on similar content
🎯 CONFIDENCE SCORES - Multi-metric translation quality assessment
🔀 DIFF VISUALIZATION - Visual comparison between agent versions
🎲 ALTERNATIVE TRANSLATIONS - Generate and compare multiple variants
🧩 Support for OpenAI and Anthropic models (e.g., GPT-4o, Claude-3.5-Sonnet)
📊 Optional LangSmith integration for detailed tracing, monitoring, and reproducibility
💬 Comprehensive agent feedback system with issue tracking and human-review flags
📁 File upload support (.txt, .docx, .md)
📤 Multiple export formats (.txt, .docx, .md)
🚨 Critical passage flagging and review
🔄 Safe same-language (e.g., English→English) refinement mode
🎯 BERTScore validation with iterative refinement
📈 Visualizations: word counts, sentence-length histograms, readability, issue counts, BERTScore bars
☁️ Word clouds: Source, Final, and Difference (words added)
🎯 Entity tracking and network visualization
🔊 TTS AUDIO PLAYBACK - Listen to translated text aloud via ElevenLabs

🧰 Installation
1. Clone the repository
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
