A sophisticated AI-powered translation system that brings editorial-level quality to multilingual projects.  
This system goes far beyond literal translation — it performs cultural adaptation, tone harmonization,  
technical verification, and literary editing, producing texts that read as if they were written in the target language.

---

## ✨ Key Features

- 🧠 **Six Specialized Translation Agents**, each with distinct editorial roles:
  1. **Literal Translator** — ensures semantic precision and lexical fidelity  
  2. **Cultural Adapter** — localizes idioms and references for the target audience  
  3. **Tone Director** — maintains stylistic harmony and consistent narrative voice  
  4. **Technical Reviewer** — validates factual, numerical, and scientific correctness  
  5. **Literary Editor** — elevates prose to publication quality  
  6. **Quality Controller** — synthesizes all layers into a final, cohesive output  

- 🔄 **Full Workflow Automation** using [LangGraph](https://github.com/langchain-ai/langgraph)  
- 🧩 **Supports both OpenAI and Anthropic models** (e.g., GPT-4, Claude-3.5-Sonnet)  
- 📊 **Optional LangSmith integration** for detailed tracing, monitoring, and reproducibility  
- 💬 **Agent feedback reports** with issue tracking and human-review flags  
- 📁 **File upload support** (`.txt`, `.docx`, `.md`)  
- 📤 **Multiple export formats** for publishing and archiving  
- 🚨 **Critical passage detection** for linguistically complex or ambiguous areas  

---

## 🧰 Installation

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
streamlit run enhanced-translation-system

