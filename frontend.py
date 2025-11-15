
# =====================
# Streamlit UI - COMPLETE WITH NEW FEATURES
# =====================
import streamlit as st
import asyncio
from datetime import datetime
from backend import *

async def main():
    st.set_page_config(
        page_title="Advanced Translation System",
        page_icon="🌐",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items=None
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
    if 'thread_id' not in st.session_state:
        st.session_state.thread_id = f"thread_{datetime.now().timestamp()}"
    
    # Cache initialization
    if 'translation_cache' not in st.session_state:
        st.session_state.translation_cache = SemanticTranslationCache()
    
    # Entity tracker initialization
    if 'entity_tracker' not in st.session_state:
        st.session_state.entity_tracker = EntitiesTracker()
    if 'enable_entity_tracking' not in st.session_state:
        st.session_state.enable_entity_tracking = False
    if 'enable_entity_awareness' not in st.session_state:
        st.session_state.enable_entity_awareness = False
    
    # Planning preferences
    if 'planning_enabled' not in st.session_state:
        st.session_state.planning_enabled = True
    if 'force_cultural_adapter' not in st.session_state:
        st.session_state.force_cultural_adapter = False
    if 'force_tone_specialist' not in st.session_state:
        st.session_state.force_tone_specialist = False
    if 'force_technical_reviewer' not in st.session_state:
        st.session_state.force_technical_reviewer = False
    if 'force_literary_editor' not in st.session_state:
        st.session_state.force_literary_editor = False
    if 'force_bertscore_validator' not in st.session_state:
        st.session_state.force_bertscore_validator = False
    
    # NEW: Alternative translations state
    if 'alternatives' not in st.session_state:
        st.session_state.alternatives = []
    
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
        .planning-card {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            padding: 15px;
            border-radius: 8px;
            color: white;
            margin: 10px 0;
        }
        .cache-card {
            background: linear-gradient(135deg, #00c6ff 0%, #0072ff 100%);
            padding: 15px;
            border-radius: 8px;
            color: white;
            margin: 10px 0;
        }
        .confidence-card {
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
            padding: 15px;
            border-radius: 8px;
            color: white;
            margin: 10px 0;
        }
        .stTabs [data-baseweb="tab-list"] { gap: 24px; }
        .stTabs [data-baseweb="tab"] { padding: 10px 20px; }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("🌐 Advanced Multi-Agent Translation System")
    st.caption("🎯 **NEW**: Confidence Scores • Diff Visualization • Alternative Translations • Smart Semantic Caching")
    
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
            model_options = ["claude-3-5-sonnet-20241022", "claude-3-opus-20240229", "claude-3-haiku-20240307"]
        
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
        
        # Cache Controls
        st.subheader("⚡ Smart Cache")
        
        cache_stats = st.session_state.translation_cache.get_stats()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Entries", cache_stats['total_entries'])
        with col2:
            st.metric("Hit Rate", f"{cache_stats['hit_rate']:.1f}%")
        
        col3, col4 = st.columns(2)
        with col3:
            st.metric("Hits", cache_stats['hits'] + cache_stats['similar_hits'] + cache_stats['partial_hits'])
        with col4:
            st.metric("Misses", cache_stats['misses'])
        
        if cache_stats['total_entries'] > 0:
            with st.expander("📊 Cache Details"):
                st.write(f"**Exact hits:** {cache_stats['hits']}")
                st.write(f"**Similar hits:** {cache_stats['similar_hits']}")
                st.write(f"**Partial hits:** {cache_stats['partial_hits']}")
                st.write(f"**Sentence cache:** {cache_stats['sentence_entries']} entries")
                st.write(f"**Embeddings:** {cache_stats['embedding_entries']} stored")
        
        col_save, col_clear = st.columns(2)
        with col_save:
            if st.button("💾 Save", use_container_width=True):
                if st.session_state.translation_cache.save_cache():
                    st.success("✅ Saved")
        with col_clear:
            if st.button("🗑️ Clear", use_container_width=True):
                st.session_state.translation_cache.clear_cache()
                st.success("✅ Cleared")
                st.rerun()
        
        if cache_stats['total_entries'] > 0:
            cache_export = st.session_state.translation_cache.export_cache()
            st.download_button(
                "📥 Export Cache",
                json.dumps(cache_export, indent=2, default=str),
                file_name=f"translation_cache_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )
        
        uploaded_cache = st.file_uploader("📤 Import Cache", type=['json'])
        if uploaded_cache:
            try:
                cache_data = json.load(uploaded_cache)
                st.session_state.translation_cache.import_cache(cache_data)
                st.success("✅ Cache imported")
                st.rerun()
            except Exception as e:
                st.error(f"Import failed: {str(e)}")
        
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            st.warning("⚠️ Install sentence-transformers for semantic caching: `pip install sentence-transformers`")
        
        st.divider()
        
        # Planning Controls
        st.subheader("📋 Intelligent Planning")
        st.session_state.planning_enabled = st.checkbox(
            "Enable Smart Planning",
            value=st.session_state.planning_enabled,
            help="Let AI select optimal agents for each task"
        )
        
        if not st.session_state.planning_enabled:
            st.info("⚠️ Planning disabled - all agents will run")
        else:
            st.success("✅ Smart planning active - optimizing workflow")
            
            with st.expander("🎯 Manual Overrides", expanded=False):
                st.caption("Force inclusion of specific agents:")
                st.session_state.force_cultural_adapter = st.checkbox(
                    "🌍 Force Cultural Adapter",
                    value=st.session_state.force_cultural_adapter
                )
                st.session_state.force_tone_specialist = st.checkbox(
                    "🎭 Force Tone Specialist",
                    value=st.session_state.force_tone_specialist
                )
                st.session_state.force_technical_reviewer = st.checkbox(
                    "🔬 Force Technical Reviewer",
                    value=st.session_state.force_technical_reviewer
                )
                st.session_state.force_literary_editor = st.checkbox(
                    "✍️ Force Literary Editor",
                    value=st.session_state.force_literary_editor
                )
                st.session_state.force_bertscore_validator = st.checkbox(
                    "🎯 Force BERTScore Validator",
                    value=st.session_state.force_bertscore_validator,
                    help="Only applies to same-language refinement"
                )
        
        st.divider()
        LANGUAGE_OPTIONS = [
        "English (US)", "English (UK)", 
        "Spanish", "French", "German", "Italian", 
        "Portuguese", "Romanian", "Polish", "Ukrainian", 
        "Russian", "Czech", "Slovak", "Bulgarian", "Dutch",
        "Other"
        ]
        st.subheader("🌍 Translation Settings")
        source_lang = st.selectbox(
            "Source Language",
            LANGUAGE_OPTIONS,
            help="Language of the source text"
        )
        
        target_lang = st.selectbox(
            "Target Language",
            LANGUAGE_OPTIONS,
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
        
        # Entity Tracking Options
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
            
            uploaded_glossary = st.file_uploader(
                "Upload Entity Glossary",
                type=['json', 'csv', 'txt'],
                help="Upload a glossary of important terms to track"
            )
            
            if uploaded_glossary:
                if st.session_state.entity_tracker.upload_glossary(uploaded_glossary):
                    st.success(f"✅ Glossary loaded")
            
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
        st.markdown("### The Eight-Agent Pipeline (with Intelligent Planning)")
        for agent_key, agent_info in AGENT_DESCRIPTIONS.items():
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
                st.session_state.alternatives = []
                st.session_state.thread_id = f"thread_{datetime.now().timestamp()}"
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
                
                # 1) Initialize LLM
                llm = initialize_llm(
                    provider=provider,
                    model=model,
                    api_key=api_key,
                    temperature=temperature
                )
                
                # 2) Initialize pipeline and connect it to UI-managed objects
                pipeline = TranslationPipeline(llm)
                pipeline.semantic_cache = st.session_state.translation_cache
                pipeline.entity_tracker = st.session_state.entity_tracker
                pipeline.enable_entity_awareness = st.session_state.enable_entity_awareness
                pipeline.enable_entity_tracking = st.session_state.enable_entity_tracking
                
                current_time = datetime.now().isoformat()
                
                # 3) Build initial translation state
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
                    "completed_at": None,
                    "agent_plan": [],
                    "agent_plan_reasoning": {},
                    "planning_analysis": {},
                    "skipped_agents": [],
                    "estimated_complexity": "unknown",
                    "current_agent_index": 0,
                    "planning_enabled": st.session_state.planning_enabled,
                    "cache_hit": None,
                    "cache_speedup": None,
                    "confidence_scores": None
                }
                
                # 4) Optional: entity tracking on the UI side
                if st.session_state.enable_entity_tracking:
                    entity_tracker = st.session_state.entity_tracker
                    source_entities = entity_tracker.extract_entities(source_text)
                    initial_state["source_entities"] = source_entities
                    initial_state["translated_entities"] = []
                    initial_state["entity_preservation_rate"] = 0.0
                    
                    if source_entities:
                        st.info(f"🎯 Tracking {len(source_entities)} entities through translation")
                else:
                    initial_state["source_entities"] = None
                    initial_state["translated_entities"] = None
                    initial_state["entity_preservation_rate"] = None
                
                with status_container.container():
                    st.info("🔄 Running agents...")
                with progress_container.container():
                    st.progress(0.10)
                
                # 5) Run pipeline (backend is async, main() is async, so we can await)
                result = await pipeline.run(initial_state)
                
                # 6) Save state and history
                st.session_state.translation_state = result
                st.session_state.history.append({
                    "timestamp": datetime.now().isoformat(),
                    "source": source_text[:100] + "...",
                    "target_lang": target_lang,
                    "model": model,
                    "result": result
                })
                
                # 7) Show cache impact (if any)
                if result.get("cache_hit"):
                    cache_type = result["cache_hit"]
                    speedup = result.get("cache_speedup", "unknown")
                    st.success(f"✅ Pipeline completed! ⚡ CACHE HIT ({cache_type}) - {speedup} speedup")
                else:
                    st.success("✅ Pipeline completed!")
                
                st.rerun()
            
            except Exception as e:
                st.error(f"❌ Error during processing: {str(e)}")
                st.code(traceback.format_exc())
 
                
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
                
                with status_container.container():
                    st.info("🔄 Planning workflow...")
                with progress_container.container():
                    st.progress(0.05)
                
                with status_container.container():
                    st.info("🔄 Running agents...")
                
                config = {"configurable": {"thread_id": st.session_state.thread_id}}
                result = None
                
                try:
                    result = await graph.ainvoke(initial_state, config)
                except Exception as inner_e:
                    st.warning("Partial failure detected - attempting resume from checkpoint...")
                    state_snapshot = graph.checkpointer.get_tuple(config)
                    if state_snapshot:
                        result = await graph.ainvoke(None, config)
                    else:
                        raise inner_e
                
                status_container.empty()
                progress_container.empty()
                
                st.session_state.translation_state = result
                st.session_state.edited_translation = result.get('final_translation', '') if result else ''
                st.session_state.alternatives = []  # Reset alternatives
                
                st.session_state.history.append({
                    "timestamp": datetime.now().isoformat(),
                    "source": source_text[:100] + "...",
                    "target_lang": target_lang,
                    "model": model,
                    "result": result
                })
                
                # Show cache impact
                if result.get('cache_hit'):
                    cache_type = result['cache_hit']
                    speedup = result.get('cache_speedup', 'unknown')
                    st.success(f"✅ Pipeline completed! ⚡ CACHE HIT ({cache_type}) - {speedup} speedup")
                else:
                    st.success("✅ Pipeline completed!")
                
                st.rerun()
            
            except Exception as e:
                st.error(f"❌ Error during processing: {str(e)}")
                st.code(traceback.format_exc())
        
        if st.session_state.translation_state:
            state = st.session_state.translation_state
            
            # Display cache hit if applicable
            if state.get('cache_hit'):
                st.markdown(f"""
                <div class="cache-card">
                    <h3>⚡ Cache Hit: {state['cache_hit'].upper()}</h3>
                    <p>Speedup: {state.get('cache_speedup', 'significant')}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # NEW: Display confidence scores
            if state.get('confidence_scores'):
                confidence = state['confidence_scores']
                st.markdown(f"""
                <div class="confidence-card">
                    <h3>🎯 Translation Confidence: {confidence['overall']:.0%}</h3>
                    <p>Fluency: {confidence.get('fluency', 0.5):.0%} | Fidelity: {confidence.get('semantic_fidelity', 0.5):.0%} | Terminology: {confidence.get('terminology', 1.0):.0%}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Display planning information
            if state.get('agent_plan'):
                st.markdown("### 📋 Execution Plan")
                
                plan = state.get('agent_plan', [])
                skipped = state.get('skipped_agents', [])
                complexity = state.get('estimated_complexity', 'unknown')
                
                agent_names = {
                    'literal_translator': '🔤 Baseline',
                    'cultural_adapter': '🌍 Cultural',
                    'tone_specialist': '🎭 Tone',
                    'technical_reviewer': '🔬 Technical',
                    'literary_editor': '✍️ Literary',
                    'finalize': '✅ Finalize',
                    'bertscore_validator': '🎯 BERTScore'
                }
                
                col_info1, col_info2, col_info3 = st.columns(3)
                with col_info1:
                    st.metric("Agents Used", f"{len(plan)}/7")
                with col_info2:
                    savings = int((1 - len(plan) / 7) * 100)
                    st.metric("Efficiency Gain", f"{savings}%")
                with col_info3:
                    st.metric("Complexity", complexity.title())
                
                route_display = " → ".join([agent_names.get(a, a) for a in plan])
                st.info(f"**Route:** {route_display}")
                
                if skipped:
                    skipped_display = ", ".join([agent_names.get(a, a) for a in skipped])
                    st.caption(f"⏭️ **Skipped:** {skipped_display}")
            
            st.divider()
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
        
        tab_names = [
            "🎯 Confidence","🔀 Diff Viewer","🎲 Alternatives",
            "⚡ Cache","📋 Planning","✏️ Edit & Review","🔄 Agent Workflow",
            "📊 All Versions","⚠️ Issues & Feedback","📈 Analytics","📚 History"
        ]
        
        if st.session_state.enable_entity_tracking:
            tab_names.append("🎯 Entities")
        
        tabs = st.tabs(tab_names)
        
        # Unpack tabs
        if len(tabs) == 12:
            (tab_confidence, tab_diff, tab_alternatives, tab_cache, tab_planning, 
             tab_edit, tab_workflow, tab_versions, tab_issues, tab_analytics, 
             tab_history, tab_entities) = tabs
        else:
            (tab_confidence, tab_diff, tab_alternatives, tab_cache, tab_planning, 
             tab_edit, tab_workflow, tab_versions, tab_issues, tab_analytics, 
             tab_history) = tabs
            tab_entities = None
        
        # NEW: Confidence Tab
        with tab_confidence:
            st.subheader("🎯 Translation Confidence Scores")
            
            if state.get('confidence_scores'):
                confidence = state['confidence_scores']
                
                # Overall score display
                overall = confidence['overall']
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Overall", f"{overall:.1%}", 
                             help="Weighted average of all metrics")
                with col2:
                    st.metric("Fluency", f"{confidence.get('fluency', 0.5):.1%}",
                             help="Natural language quality")
                with col3:
                    st.metric("Semantic Fidelity", f"{confidence.get('semantic_fidelity', 0.5):.1%}",
                             help="Meaning preservation (BERTScore)")
                with col4:
                    st.metric("Terminology", f"{confidence.get('terminology', 1.0):.1%}",
                             help="Key term preservation")
                
                # Visual gauge
                if PLOTLY_AVAILABLE:
                    st.divider()
                    st.markdown("### 📊 Confidence Visualization")
                    
                    # Gauge chart
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = overall * 100,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Overall Confidence"},
                        gauge = {
                            'axis': {'range': [None, 100]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 50], 'color': "lightgray"},
                                {'range': [50, 75], 'color': "gray"},
                                {'range': [75, 100], 'color': "lightgreen"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 80
                            }
                        }
                    ))
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Bar chart of components
                    st.markdown("### 📈 Component Breakdown")
                    metrics_data = {
                        'Metric': ['Fluency', 'Semantic Fidelity', 'Terminology', 'Length Ratio'],
                        'Score': [
                            confidence.get('fluency', 0.5) * 100,
                            confidence.get('semantic_fidelity', 0.5) * 100,
                            confidence.get('terminology', 1.0) * 100,
                            confidence.get('length_ratio', 1.0) * 100
                        ]
                    }
                    
                    fig2 = go.Figure(data=[
                        go.Bar(
                            x=metrics_data['Metric'],
                            y=metrics_data['Score'],
                            marker_color=['#3b82f6', '#10b981', '#f59e0b', '#8b5cf6']
                        )
                    ])
                    fig2.update_layout(
                        title='Confidence Metrics',
                        yaxis_title='Score (%)',
                        yaxis_range=[0, 100],
                        showlegend=False
                    )
                    st.plotly_chart(fig2, use_container_width=True)
                
                # Interpretation guide
                st.divider()
                st.markdown("### 📖 Score Interpretation")
                
                if overall >= 0.85:
                    st.success("✅ **Excellent** - High confidence translation")
                elif overall >= 0.7:
                    st.info("ℹ️ **Good** - Solid translation with minor areas for review")
                elif overall >= 0.5:
                    st.warning("⚠️ **Fair** - Consider reviewing specific sections")
                else:
                    st.error("❌ **Poor** - Significant revision recommended")
                
                with st.expander("📚 Understanding the Metrics"):
                    st.markdown("""
                    **Overall Confidence:** Weighted combination of all metrics
                    - 35% Semantic Fidelity
                    - 35% Fluency
                    - 15% Terminology
                    - 15% Length Ratio
                    
                    **Fluency:** How natural and well-written the translation reads
                    
                    **Semantic Fidelity:** How well the meaning is preserved (uses BERTScore for same-language)
                    
                    **Terminology:** Percentage of key entities/terms preserved
                    
                    **Length Ratio:** How appropriate the length is (ideal: 0.8-1.2x source)
                    """)
            else:
                st.info("Confidence scores not available for this translation")
        
        # NEW: Diff Viewer Tab
        with tab_diff:
            st.subheader("🔀 Visual Diff Between Agent Versions")
            
            versions = [
                ("Baseline", state.get('literal_translation', '')),
                ("Cultural", state.get('cultural_adaptation', '')),
                ("Tone", state.get('tone_adjustment', '')),
                ("Technical", state.get('technical_review_version', '')),
                ("Literary", state.get('literary_polish', '')),
                ("Final", state.get('final_translation', ''))
            ]
            
            # Filter out empty versions
            available_versions = [(name, text) for name, text in versions if text]
            
            if len(available_versions) >= 2:
                st.markdown("### 📊 Change Rate Summary")
                
                # Calculate change rates between consecutive versions
                change_data = []
                for i in range(len(available_versions) - 1):
                    name1, text1 = available_versions[i]
                    name2, text2 = available_versions[i + 1]
                    change_rate = calculate_change_rate(text1, text2)
                    change_data.append({
                        'Transition': f"{name1} → {name2}",
                        'Change Rate': change_rate
                    })
                
                if change_data:
                    df_changes = pd.DataFrame(change_data)
                    
                    if PLOTLY_AVAILABLE:
                        fig = go.Figure(data=[
                            go.Bar(
                                x=df_changes['Transition'],
                                y=df_changes['Change Rate'],
                                marker_color='#3b82f6'
                            )
                        ])
                        fig.update_layout(
                            title='Change Rate Between Agent Versions',
                            yaxis_title='Change Rate (%)',
                            xaxis_title='Agent Transition',
                            showlegend=False
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.dataframe(df_changes)
                
                st.divider()
                st.markdown("### 🔍 Detailed Comparison")
                
                # Version selector
                col1, col2 = st.columns(2)
                with col1:
                    version1_idx = st.selectbox(
                        "Compare from:",
                        range(len(available_versions)),
                        format_func=lambda i: available_versions[i][0]
                    )
                with col2:
                    version2_idx = st.selectbox(
                        "Compare to:",
                        range(len(available_versions)),
                        index=min(version1_idx + 1, len(available_versions) - 1),
                        format_func=lambda i: available_versions[i][0]
                    )
                
                if version1_idx != version2_idx:
                    name1, text1 = available_versions[version1_idx]
                    name2, text2 = available_versions[version2_idx]
                    
                    change_rate = calculate_change_rate(text1, text2)
                    st.info(f"**Change Rate:** {change_rate:.1f}% of the text was modified")
                    
                    diff_html = create_diff_visualization(text1, text2, name1, name2)
                    st.markdown(diff_html, unsafe_allow_html=True)
                else:
                    st.warning("Please select two different versions to compare")
            else:
                st.info("Need at least 2 versions to show diff")
        
        # NEW: Alternatives Tab
        with tab_alternatives:
            st.subheader("🎲 Alternative Translation Variants")
            
            if not st.session_state.alternatives:
                st.markdown("""
                Generate alternative translations using different strategies:
                - **Conservative**: More literal and faithful
                - **Balanced**: Mix of literal and natural
                - **Creative**: More idiomatic and natural
                """)
                
                num_alts = st.slider("Number of alternatives", 2, 3, 3)
                
                if st.button("🎲 Generate Alternatives", type="primary"):
                    with st.spinner("Generating alternative translations..."):
                        try:
                            llm = initialize_llm(provider=provider, model=model, api_key=api_key, temperature=temperature)
                            generator = AlternativeTranslationGenerator(llm)
                            alternatives = await generator.generate_alternatives(state, num_alts)
                            st.session_state.alternatives = alternatives
                            st.success(f"✅ Generated {len(alternatives)} alternatives")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error generating alternatives: {str(e)}")
            else:
                st.success(f"✅ {len(st.session_state.alternatives)} alternatives generated")
                
                # Display current translation for comparison
                with st.expander("📖 Current Translation (for comparison)", expanded=False):
                    st.text_area(
                        "Current",
                        st.session_state.edited_translation,
                        height=150,
                        key="current_for_comparison",
                        disabled=True
                    )
                
                st.divider()
                
                # Display alternatives
                for idx, alt in enumerate(st.session_state.alternatives, 1):
                    if alt.get('error'):
                        st.error(f"❌ Alternative {idx} ({alt['strategy']}): Failed to generate")
                        continue
                    
                    st.markdown(f"### 🎯 Alternative {idx}: {alt['strategy'].title()}")
                    st.caption(f"📝 {alt['description']} (temperature: {alt['temperature']})")
                    
                    st.text_area(
                        f"Alt {idx}",
                        alt['translation'],
                        height=150,
                        key=f"alt_display_{idx}",
                        label_visibility="collapsed"
                    )
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if st.button(f"✅ Use This Version", key=f"use_alt_{idx}"):
                            st.session_state.edited_translation = alt['translation']
                            st.success("Alternative selected! Scroll up to see it in the Result section.")
                            st.rerun()
                    
                    with col2:
                        if st.button(f"⚖️ Compare with Current", key=f"compare_alt_{idx}"):
                            st.markdown("#### Comparison")
                            current = st.session_state.edited_translation
                            change_rate = calculate_change_rate(current, alt['translation'])
                            st.info(f"Change rate: {change_rate:.1f}%")
                            diff_html = create_diff_visualization(
                                current,
                                alt['translation'],
                                "Current",
                                f"Alternative {idx}"
                            )
                            st.markdown(diff_html, unsafe_allow_html=True)
                    
                    with col3:
                        # Word count comparison
                        current_words = len(st.session_state.edited_translation.split())
                        alt_words = len(alt['translation'].split())
                        word_diff = alt_words - current_words
                        st.metric("Words", alt_words, delta=word_diff)
                    
                    st.divider()
                
                if st.button("🔄 Generate New Alternatives"):
                    st.session_state.alternatives = []
                    st.rerun()
        
        # Cache Tab
        with tab_cache:
            st.subheader("⚡ Smart Cache Performance")
            
            cache_stats = st.session_state.translation_cache.get_stats()
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Entries", cache_stats['total_entries'])
            with col2:
                st.metric("Hit Rate", f"{cache_stats['hit_rate']:.1f}%")
            with col3:
                st.metric("Total Hits", cache_stats['hits'] + cache_stats['similar_hits'] + cache_stats['partial_hits'])
            with col4:
                st.metric("Misses", cache_stats['misses'])
            
            if state.get('cache_hit'):
                st.success(f"✅ This translation used cache: **{state['cache_hit'].upper()}** hit (speedup: {state.get('cache_speedup', 'significant')})")
            else:
                st.info("ℹ️ This translation was processed from scratch (no cache hit)")
            
            st.divider()
            
            st.markdown("### 📊 Cache Breakdown")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Exact Hits", cache_stats['hits'], help="100% match - instant result")
            with col2:
                st.metric("Similar Hits", cache_stats['similar_hits'], help="High similarity - light refinement")
            with col3:
                st.metric("Partial Hits", cache_stats['partial_hits'], help="Some sentences cached")
            
            if cache_stats['total_entries'] > 0:
                st.markdown("### 📦 Cache Contents")
                st.write(f"**Sentence-level cache:** {cache_stats['sentence_entries']} entries")
                st.write(f"**Embeddings stored:** {cache_stats['embedding_entries']}")
                
                if SENTENCE_TRANSFORMERS_AVAILABLE:
                    st.success("✅ Semantic similarity matching enabled")
                else:
                    st.warning("⚠️ Install sentence-transformers for semantic caching")
        
        # Planning Tab
        with tab_planning:
            st.subheader("📋 Planning Analysis & Decisions")
            
            if state.get('agent_plan'):
                reasoning = state.get('agent_plan_reasoning', {})
                analysis = state.get('planning_analysis', {})
                
                st.markdown("#### 📊 Text Analysis")
                if analysis:
                    cols = st.columns(3)
                    if 'length' in analysis:
                        cols[0].metric("Length", f"{analysis['length']} words")
                    if 'complexity_score' in analysis:
                        cols[1].metric("Complexity", f"{analysis.get('complexity_score', 0):.1f}/10")
                    if 'technical_terms_count' in analysis:
                        cols[2].metric("Technical Terms", analysis.get('technical_terms_count', 0))
                    
                    cols2 = st.columns(3)
                    if 'cultural_references_count' in analysis:
                        cols2[0].metric("Cultural Refs", analysis.get('cultural_references_count', 0))
                    if 'tone_consistency' in analysis:
                        cols2[1].metric("Tone", analysis.get('tone_consistency', 'N/A'))
                    if 'literary_elements' in analysis:
                        cols2[2].metric("Literary", analysis.get('literary_elements', 'N/A'))
                
                st.divider()
                st.markdown("#### 🎯 Agent Selection Reasoning")
                
                for agent_key in ['literal_translator', 'cultural_adapter', 'tone_specialist',
                                 'technical_reviewer', 'literary_editor', 'finalize', 'bertscore_validator']:
                    if agent_key in reasoning:
                        reason = reasoning[agent_key]
                        if reason.startswith('✅'):
                            st.success(f"**{agent_key.replace('_', ' ').title()}:** {reason}")
                        else:
                            st.info(f"**{agent_key.replace('_', ' ').title()}:** {reason}")
            else:
                st.info("Planning information not available")
        
        # Edit & Review Tab
        with tab_edit:
            st.subheader("✏️ Edit & Review")
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
        
        # Agent Workflow Tab
        with tab_workflow:
            st.subheader("🔄 Agent Workflow Progress")
            unique_notes = []
            seen = set()
            for note in state.get('agent_notes', []):
                if note not in seen:
                    unique_notes.append(note)
                    seen.add(note)
            
            for note in unique_notes:
                if "CACHE HIT" in note:
                    st.success(f"⚡ {note}")
                else:
                    st.success(f"✓ {note}")
            
            st.divider()
            
            if state.get('started_at') and state.get('completed_at'):
                try:
                    start = datetime.fromisoformat(state['started_at'])
                    end = datetime.fromisoformat(state['completed_at'])
                    duration = (end - start).total_seconds()
                    st.metric("Total Processing Time", f"{duration:.1f} seconds")
                except:
                    st.info("Processing time not available")
        
        # All Versions Tab
        with tab_versions:
            st.subheader("📊 All Versions")
            versions = [
                ("1️⃣ Baseline", state.get('literal_translation','')),
                ("2️⃣ Cultural/Register", state.get('cultural_adaptation','')),
                ("3️⃣ Tone", state.get('tone_adjustment','')),
                ("4️⃣ Technical", state.get('technical_review_version','')),
                ("5️⃣ Literary", state.get('literary_polish','')),
                ("6️⃣ Final", state.get('final_translation','')),
            ]
            
            for title, content in versions:
                if content:
                    with st.expander(title, expanded=False):
                        st.text_area(f"{title} content", content, height=200, key=f"version_{title}", label_visibility="collapsed")
        
        # Issues & Feedback Tab
        with tab_issues:
            st.subheader("⚠️ Issues & Feedback")
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
        
        # Analytics Tab
        with tab_analytics:
            st.subheader("📈 Analytics")
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
                with st.expander("📊 Word Count Overview", expanded=True):
                    try:
                        df_counts = pd.DataFrame(
                            {"Text": ["Source", "Final"], "Words": [source_words, final_words]}
                        ).set_index("Text")
                        st.bar_chart(df_counts)
                    except Exception:
                        st.warning("Could not render word count chart.")
                
                st.divider()
                with st.expander("📏 Sentence Length Distribution (words per sentence)", expanded=False):
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
                with st.expander("☁️ Word Clouds (Before / After / Difference)", expanded=True):
                    if not WORDCLOUD_AVAILABLE:
                        st.info("Install `wordcloud` to enable this: `pip install wordcloud`")
                    else:
                        sw = set(STOPWORDS) | {"—", "–", "'", """, """, "…"}
                        src_text = state.get('source_text', '')
                        fin_text = state.get('final_translation', '')
                        
                        src_freq = word_frequencies(src_text, stopwords=sw)
                        fin_freq = word_frequencies(fin_text, stopwords=sw)
                        
                        diff_freq = {}
                        for w, f_cnt in fin_freq.items():
                            s_cnt = src_freq.get(w, 0)
                            delta = f_cnt - s_cnt
                            if delta > 0:
                                diff_freq[w] = delta
                        
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
                with st.expander("📖 Readability (Flesch Reading Ease)", expanded=False):
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
                if languages_equivalent(state['source_language'], state['target_language']):
                    with st.expander("🎯 BERTScore (same-language refine mode)", expanded=True):
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
                with st.expander("📋 Issues by Stage", expanded=False):
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
        
        # History Tab
        with tab_history:
            st.subheader("📚 History")
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
        
        # Entity Tab (only if enabled)
        if st.session_state.enable_entity_tracking and tab_entities:
            with tab_entities:
                st.subheader("🎯 Entity Analysis & Tracking")
                
                if state:
                    entity_tracker = st.session_state.entity_tracker
                    
                    if 'source_entities' not in state or state['source_entities'] is None:
                        state['source_entities'] = entity_tracker.extract_entities(state.get('source_text', ''))
                    if 'translated_entities' not in state or state['translated_entities'] is None:
                        state['translated_entities'] = entity_tracker.extract_entities(state.get('final_translation', ''))
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("### 📖 Source Entities")
                        source_entities = state.get('source_entities', [])
                        
                        if source_entities:
                            st.metric("Total Entities", len(source_entities))
                            
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
                            all_entities = {}
                            for e in source_entities + translated_entities:
                                if e['name'] not in all_entities:
                                    all_entities[e['name']] = e
                            viz_entities = list(all_entities.values())[:30]
                        
                        if viz_entities:
                            entity_tracker.visualize_network(viz_entities)
                        
                        if PLOTLY_AVAILABLE:
                            st.divider()
                            st.markdown("### 📈 Entity Frequency Comparison")
                            
                            all_entity_names = {}
                            for e in source_entities:
                                all_entity_names[e['name']] = {'source': e['count'], 'translated': 0, 'type': e['type']}
                            for e in translated_entities:
                                if e['name'] in all_entity_names:
                                    all_entity_names[e['name']]['translated'] = e['count']
                                else:
                                    all_entity_names[e['name']] = {'source': 0, 'translated': e['count'], 'type': e['type']}
                            
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
    asyncio.run(main())