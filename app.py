# app.py
import streamlit as st
import asyncio
from datetime import datetime

# --- Module Imports (The Refactored Structure) ---
from src.config import AGENT_DESCRIPTIONS
from src.ui.styles import apply_custom_styles
from src.ui.session import initialize_session_state
from src.core.llm import initialize_llm, setup_langsmith
from src.agents.workflow import TranslationPipeline
from src.services.alternatives import AlternativeTranslationGenerator
from src.utils.common import mode_phrase, languages_equivalent
from src.utils.files import read_uploaded_file, create_docx_file, create_markdown_file
from src.analysis.visualizations import (
    create_diff_visualization, render_wordcloud, compute_frequencies, 
    render_entity_network
)
# Import stopwords for wordcloud
from src.data.stopwords import MULTILINGUAL_STOPWORDS

def main():
    st.set_page_config(
        page_title="Advanced Translation System",
        page_icon="🌐",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # 1. Initialize State & Styles
    initialize_session_state()
    apply_custom_styles()
    
    st.title("🌐 Advanced Multi-Agent Translation System")
    st.caption("🎯 Modular Refactored Architecture: Agents • Cache • Entities • Analytics")

    # ==========================================
    # 🔄 SIDEBAR CONFIGURATION
    # ==========================================
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model Provider
        provider = st.radio("Provider", ["openai", "anthropic"], format_func=lambda x: x.title())
        api_key = st.text_input(f"{provider.title()} API Key", type="password")
        
        if provider == "openai":
            model = st.selectbox("Model", ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"])
        else:
            model = st.selectbox("Model", ["claude-3-5-sonnet-20241022", "claude-3-opus-20240229"])
            
        temperature = st.slider("Temperature", 0.0, 1.0, 0.3)
        
        st.divider()
        
        # Cache Stats
        stats = st.session_state.translation_cache.get_stats()
        col1, col2 = st.columns(2)
        col1.metric("Hits", stats['hits'] + stats['similar_hits'])
        col2.metric("Rate", f"{stats['hit_rate']:.1f}%")
        if st.button("Clear Cache"):
            st.session_state.translation_cache.clear_cache()
            st.rerun()
            
        st.divider()
        
        # Language Settings
        langs = ["English (US)", "English (UK)", "Spanish", "French", "German", "Chinese (Traditional)", "Japanese", "Korean"]
        source_lang = st.selectbox("Source Language", langs)
        target_lang = st.selectbox("Target Language", langs, index=1)
        
        audience = st.selectbox("Audience", ["General", "Academic", "Business", "Literary"])
        
        st.divider()
        
        # Planning & Toggles
        st.session_state.planning_enabled = st.checkbox("Enable Smart Planning", value=True)
        st.session_state.enable_entity_tracking = st.checkbox("Enable Entity Tracking", value=False)

    # ==========================================
    # 📝 MAIN INTERFACE
    # ==========================================
    col_source, col_result = st.columns(2)
    
    # --- Left Column: Input ---
    with col_source:
        st.subheader("📄 Source Text")
        uploaded = st.file_uploader("Upload file", type=['txt', 'md', 'docx'])
        
        if uploaded:
            content = read_uploaded_file(uploaded)
            source_text = st.text_area("Editor", value=content, height=400)
        else:
            source_text = st.text_area("Editor", height=400, placeholder="Paste text here...")
            
        btn_col1, btn_col2 = st.columns([1, 3])
        with btn_col1:
            start_btn = st.button("🚀 Run", type="primary", disabled=not api_key, use_container_width=True)
        with btn_col2:
            if st.button("🗑️ Clear State", use_container_width=True):
                st.session_state.translation_state = None
                st.session_state.alternatives = []
                st.rerun()

    # --- Right Column: Output & Actions ---
    with col_result:
        st.subheader("🎯 Result")
        
        # 1. PROCESSING LOGIC
        if start_btn and source_text:
            try:
                # Init LLM
                llm = initialize_llm(provider, model, api_key, temperature)
                
                # Init Pipeline (Connecting Services)
                pipeline = TranslationPipeline(
                    llm, 
                    cache=st.session_state.translation_cache,
                    tracker=st.session_state.entity_tracker
                )
                pipeline.enable_entity_awareness = st.session_state.enable_entity_awareness
                
                # Initial State
                initial_state = {
                    "source_text": source_text,
                    "source_language": source_lang,
                    "target_language": target_lang,
                    "target_audience": audience,
                    "genre": "General",
                    "agent_notes": [],
                    "planning_enabled": st.session_state.planning_enabled,
                    "thread_id": st.session_state.thread_id,
                    # Add default empty lists for required fields to avoid KeyError
                    "literal_issues": [], "cultural_issues": [], "tone_issues": [], 
                    "technical_issues": [], "literary_issues": [], "critical_passages": [],
                    "bertscore_history": []
                }
                
                # Entity extraction pre-run (if enabled)
                if st.session_state.enable_entity_tracking:
                    st.session_state.entity_tracker.extract_entities(source_text)
                    initial_state['source_entities'] = st.session_state.entity_tracker.extract_entities(source_text)

                # Run Graph
                with st.status("🤖 Agents working...", expanded=True) as status:
                    st.write("Initializing workflow...")
                    result = asyncio.run(pipeline.run(initial_state))
                    status.update(label="✅ Complete!", state="complete", expanded=False)
                
                st.session_state.translation_state = result
                st.session_state.edited_translation = result.get('final_translation', '')
                st.rerun()
                
            except Exception as e:
                st.error(f"Pipeline Error: {str(e)}")
                import traceback
                st.code(traceback.format_exc())

        # 2. DISPLAY LOGIC
        if st.session_state.translation_state:
            state = st.session_state.translation_state
            
            # Confidence Score
            if state.get('confidence_scores'):
                score = state['confidence_scores']['overall']
                st.info(f"**Confidence Score:** {score:.1%}")
            
            # Output Text Area
            final_text = st.text_area(
                "Final Output", 
                value=st.session_state.edited_translation, 
                height=400
            )
            st.session_state.edited_translation = final_text # Sync edits
            
            # Download Buttons
            d_col1, d_col2, d_col3 = st.columns(3)
            with d_col1:
                st.download_button("Download TXT", final_text, "trans.txt")
            with d_col2:
                md = create_markdown_file(final_text)
                st.download_button("Download MD", md, "trans.md")
            with d_col3:
                docx = create_docx_file(final_text)
                if docx:
                    st.download_button("Download DOCX", docx, "trans.docx", 
                                     mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")

    # ==========================================
    # 📊 ANALYSIS TABS (Bottom Section)
    # ==========================================
    if st.session_state.translation_state:
        st.divider()
        tabs = st.tabs(["📊 Analysis", "🔀 Diff", "🎲 Alternatives", "📋 Planning", "☁️ Word Cloud", "🕸️ Entities"])
        
        state = st.session_state.translation_state
        
        # --- Tab 1: Analysis ---
        with tabs[0]:
            st.markdown("### Agent Workflow")
            for note in state.get('agent_notes', []):
                st.text(f"• {note}")
                
            if state.get('confidence_scores'):
                st.json(state['confidence_scores'])

        # --- Tab 2: Diff ---
        with tabs[1]:
            if state.get('literal_translation'):
                html = create_diff_visualization(
                    state['literal_translation'], 
                    state['final_translation'], 
                    "Baseline", "Final"
                )
                st.markdown(html, unsafe_allow_html=True)
            else:
                st.info("No baseline to compare.")

        # --- Tab 3: Alternatives ---
        with tabs[2]:
            if st.button("Generate 3 Alternatives"):
                with st.spinner("Brainstorming variants..."):
                    llm_alt = initialize_llm(provider, model, api_key, temperature)
                    gen = AlternativeTranslationGenerator(llm_alt)
                    alts = asyncio.run(gen.generate_alternatives(state))
                    st.session_state.alternatives = alts
            
            for alt in st.session_state.alternatives:
                with st.expander(f"{alt['strategy'].title()} ({alt['temperature']})"):
                    st.write(alt['translation'])
                    if st.button(f"Use {alt['strategy']}", key=alt['strategy']):
                        st.session_state.edited_translation = alt['translation']
                        st.rerun()

        # --- Tab 4: Planning ---
        with tabs[3]:
            st.markdown(f"**Complexity:** {state.get('estimated_complexity', 'N/A')}")
            st.json(state.get('agent_plan', []))
            st.json(state.get('agent_plan_reasoning', {}))

        # --- Tab 5: Word Cloud ---
        with tabs[4]:
            # Get stopwords for selected language
            # Handle language mapping simply
            lang_key = source_lang if "English" in source_lang else source_lang.split()[0]
            sw = MULTILINGUAL_STOPWORDS.get(lang_key, set())
            
            col_wc1, col_wc2 = st.columns(2)
            with col_wc1:
                st.caption("Source")
                freqs = compute_frequencies(state['source_text'], sw)
                render_wordcloud(freqs, "Source")
            with col_wc2:
                st.caption("Final")
                freqs = compute_frequencies(state['final_translation'], sw)
                render_wordcloud(freqs, "Final")

        # --- Tab 6: Entities ---
        with tabs[5]:
            if st.session_state.enable_entity_tracking:
                entities = st.session_state.entity_tracker.extract_entities(state['final_translation'])
                render_entity_network(entities)
            else:
                st.warning("Entity tracking is disabled in settings.")

if __name__ == "__main__":
    main()