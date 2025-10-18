"""
Entities Tracker Module for Translation Workflow
Provides entity extraction, tracking, and visualization capabilities
"""

import streamlit as st
import json
import re
import io
import csv
from typing import Dict, List, Tuple, Optional, Set
from collections import Counter, defaultdict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Optional imports for enhanced functionality
try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False


class EntitiesTracker:
    """Entity tracking and visualization system"""
    
    def __init__(self):
        self.entity_types = {
            'person': {'emoji': '👤', 'color': '#3b82f6'},
            'location': {'emoji': '📍', 'color': '#10b981'},
            'organization': {'emoji': '🏢', 'color': '#f59e0b'},
            'date': {'emoji': '📅', 'color': '#8b5cf6'},
            'custom': {'emoji': '🏷️', 'color': '#ef4444'}
        }
        
        # Initialize session state for glossary if not exists
        if 'entity_glossary' not in st.session_state:
            st.session_state.entity_glossary = self._get_default_glossary()
        
        if 'extracted_entities' not in st.session_state:
            st.session_state.extracted_entities = []
    
    def _get_default_glossary(self) -> Dict:
        """Get default glossary entries"""
        return {
            "AI": {"type": "custom", "description": "Artificial Intelligence", "aliases": ["A.I.", "artificial intelligence"]},
            "LLM": {"type": "custom", "description": "Large Language Model", "aliases": ["large language model"]},
            "NLP": {"type": "custom", "description": "Natural Language Processing", "aliases": ["natural language processing"]}
        }
    
    def extract_entities(self, text: str) -> List[Dict]:
        """
        Extract entities from text using glossary and NER patterns
        """
        if not text:
            return []
        
        entities = []
        text_lower = text.lower()
        
        # Extract from glossary
        for term, info in st.session_state.entity_glossary.items():
            # Create regex pattern for term
            pattern = re.compile(r'\b' + re.escape(term) + r'\b', re.IGNORECASE)
            matches = pattern.findall(text)
            
            # Check aliases
            alias_count = 0
            for alias in info.get('aliases', []):
                alias_pattern = re.compile(r'\b' + re.escape(alias) + r'\b', re.IGNORECASE)
                alias_matches = alias_pattern.findall(text)
                alias_count += len(alias_matches)
            
            total_count = len(matches) + alias_count
            
            if total_count > 0:
                entities.append({
                    'name': term,
                    'type': info['type'],
                    'count': total_count,
                    'description': info.get('description', ''),
                    'from_glossary': True
                })
        
        # Basic NER patterns
        entities.extend(self._extract_auto_entities(text, [e['name'] for e in entities]))
        
        return entities
    
    def _extract_auto_entities(self, text: str, existing_names: List[str]) -> List[Dict]:
        """Extract entities using pattern matching"""
        auto_entities = []
        
        # Extract person names (two capitalized words)
        person_pattern = r'\b([A-Z][a-z]+ [A-Z][a-z]+)\b'
        for match in re.finditer(person_pattern, text):
            name = match.group(1)
            if name not in existing_names and name not in [e['name'] for e in auto_entities]:
                count = len(re.findall(r'\b' + re.escape(name) + r'\b', text))
                auto_entities.append({
                    'name': name,
                    'type': 'person',
                    'count': count,
                    'description': 'Auto-detected person',
                    'auto_detected': True
                })
        
        # Extract locations (after prepositions)
        location_pattern = r'\b(?:in|at|from|to|near) ([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b'
        for match in re.finditer(location_pattern, text):
            location = match.group(1)
            if location not in existing_names and location not in [e['name'] for e in auto_entities]:
                count = len(re.findall(r'\b' + re.escape(location) + r'\b', text))
                auto_entities.append({
                    'name': location,
                    'type': 'location',
                    'count': count,
                    'description': 'Auto-detected location',
                    'auto_detected': True
                })
        
        # Extract organizations (Inc, Corp, LLC, etc.)
        org_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Inc|Corp|LLC|Ltd|Company|Group|Foundation))\b'
        for match in re.finditer(org_pattern, text):
            org = match.group(1)
            if org not in existing_names and org not in [e['name'] for e in auto_entities]:
                count = len(re.findall(r'\b' + re.escape(org) + r'\b', text))
                auto_entities.append({
                    'name': org,
                    'type': 'organization',
                    'count': count,
                    'description': 'Auto-detected organization',
                    'auto_detected': True
                })
        
        # Extract dates
        date_patterns = [
            r'\b(19|20)\d{2}\b',
            r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b',
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b'
        ]
        
        for pattern in date_patterns:
            for match in re.finditer(pattern, text):
                date = match.group(0)
                if date not in existing_names and date not in [e['name'] for e in auto_entities]:
                    count = len(re.findall(re.escape(date), text))
                    auto_entities.append({
                        'name': date,
                        'type': 'date',
                        'count': count,
                        'description': 'Auto-detected date',
                        'auto_detected': True
                    })
        
        return auto_entities
    
    def upload_glossary(self, file) -> bool:
        """Process uploaded glossary file"""
        try:
            content = file.read()
            
            if file.name.endswith('.json'):
                new_glossary = json.loads(content)
            elif file.name.endswith('.csv'):
                # Parse CSV
                csv_data = io.StringIO(content.decode('utf-8'))
                reader = csv.DictReader(csv_data)
                new_glossary = {}
                for row in reader:
                    term = row.get('term', '').strip()
                    if term:
                        new_glossary[term] = {
                            'type': row.get('type', 'custom'),
                            'description': row.get('description', ''),
                            'aliases': [a.strip() for a in row.get('aliases', '').split(',') if a.strip()]
                        }
            else:
                # Plain text - one term per line
                lines = content.decode('utf-8').split('\n')
                new_glossary = {}
                for line in lines:
                    term = line.strip()
                    if term:
                        new_glossary[term] = {'type': 'custom', 'description': '', 'aliases': []}
            
            # Merge with existing glossary
            st.session_state.entity_glossary.update(new_glossary)
            return True
        except Exception as e:
            st.error(f"Error processing glossary: {str(e)}")
            return False
    
    def render_glossary_manager(self):
        """Render glossary management interface"""
        st.subheader("📚 Glossary Management")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # File upload
            uploaded_file = st.file_uploader(
                "Upload Glossary",
                type=['json', 'csv', 'txt'],
                help="JSON format: {\"term\": {\"type\": \"...\", \"description\": \"...\", \"aliases\": [...]}}"
            )
            
            if uploaded_file:
                if self.upload_glossary(uploaded_file):
                    st.success(f"✅ Glossary loaded: {len(st.session_state.entity_glossary)} terms")
        
        with col2:
            st.metric("Total Terms", len(st.session_state.entity_glossary))
            
            # Export glossary
            if st.button("📥 Export Glossary"):
                glossary_json = json.dumps(st.session_state.entity_glossary, indent=2)
                st.download_button(
                    "Download JSON",
                    data=glossary_json,
                    file_name=f"glossary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        
        # Add new term
        with st.expander("➕ Add New Term"):
            col1, col2 = st.columns(2)
            with col1:
                new_term = st.text_input("Term")
                term_type = st.selectbox("Type", list(self.entity_types.keys()))
            with col2:
                description = st.text_input("Description")
                aliases = st.text_input("Aliases (comma-separated)")
            
            if st.button("Add Term", type="primary"):
                if new_term:
                    st.session_state.entity_glossary[new_term] = {
                        'type': term_type,
                        'description': description,
                        'aliases': [a.strip() for a in aliases.split(',') if a.strip()]
                    }
                    st.success(f"Added: {new_term}")
                    st.rerun()
        
        # Display glossary
        if st.session_state.entity_glossary:
            st.subheader("Current Glossary")
            
            # Search filter
            search_term = st.text_input("🔍 Search terms", "")
            
            # Filter glossary
            filtered_glossary = {
                k: v for k, v in st.session_state.entity_glossary.items()
                if search_term.lower() in k.lower() or 
                   search_term.lower() in v.get('description', '').lower()
            }
            
            # Display as table
            if filtered_glossary:
                for term, info in filtered_glossary.items():
                    col1, col2, col3, col4 = st.columns([3, 1, 3, 1])
                    with col1:
                        emoji = self.entity_types[info['type']]['emoji']
                        st.write(f"{emoji} **{term}**")
                    with col2:
                        st.write(f"*{info['type']}*")
                    with col3:
                        st.write(info.get('description', ''))
                    with col4:
                        if st.button("🗑️", key=f"del_{term}"):
                            del st.session_state.entity_glossary[term]
                            st.rerun()
    
    def visualize_entities(self, entities: List[Dict], viz_type: str = "network"):
        """Create entity visualizations"""
        if not entities:
            st.info("No entities to visualize")
            return
        
        if viz_type == "bar":
            self._create_bar_chart(entities)
        elif viz_type == "network" and PLOTLY_AVAILABLE:
            self._create_network_graph(entities)
        elif viz_type == "cloud" and WORDCLOUD_AVAILABLE:
            self._create_word_cloud(entities)
        elif viz_type == "table":
            self._create_table(entities)
        else:
            st.warning(f"Visualization type '{viz_type}' not available. Install required libraries.")
    
    def _create_bar_chart(self, entities: List[Dict]):
        """Create bar chart of entity frequencies"""
        df = pd.DataFrame(entities)
        df = df.sort_values('count', ascending=True)
        
        fig, ax = plt.subplots(figsize=(10, max(6, len(entities) * 0.3)))
        
        colors = [self.entity_types[e['type']]['color'] for e in entities]
        bars = ax.barh(df['name'], df['count'], color=colors)
        
        ax.set_xlabel('Occurrences')
        ax.set_title('Entity Frequencies')
        ax.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2, 
                   f' {int(width)}', ha='left', va='center')
        
        plt.tight_layout()
        st.pyplot(fig)
    
    def _create_network_graph(self, entities: List[Dict]):
        """Create network graph using plotly"""
        if not PLOTLY_AVAILABLE:
            st.error("Install plotly: pip install plotly")
            return
        
        # Create simple co-occurrence network
        G = nx.Graph()
        
        # Add nodes
        for entity in entities:
            G.add_node(entity['name'], 
                      type=entity['type'],
                      count=entity['count'],
                      description=entity.get('description', ''))
        
        # Add edges (simplified - connect all entities)
        for i, e1 in enumerate(entities):
            for e2 in entities[i+1:]:
                weight = min(e1['count'], e2['count'])
                G.add_edge(e1['name'], e2['name'], weight=weight)
        
        # Create layout
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        # Create edge traces
        edge_traces = []
        for edge in G.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            trace = go.Scatter(x=[x0, x1, None], y=[y0, y1, None],
                             mode='lines',
                             line=dict(width=0.5, color='#888'),
                             hoverinfo='none')
            edge_traces.append(trace)
        
        # Create node trace
        node_x = []
        node_y = []
        node_text = []
        node_color = []
        node_size = []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            entity = next(e for e in entities if e['name'] == node)
            node_text.append(f"{node}<br>Type: {entity['type']}<br>Count: {entity['count']}")
            node_color.append(self.entity_types[entity['type']]['color'])
            node_size.append(10 + entity['count'] * 2)
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            text=[n for n in G.nodes()],
            textposition="top center",
            hoverinfo='text',
            hovertext=node_text,
            marker=dict(
                size=node_size,
                color=node_color,
                line=dict(width=2, color='white')
            )
        )
        
        # Create figure
        fig = go.Figure(data=edge_traces + [node_trace],
                       layout=go.Layout(
                           title='Entity Network',
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=0,l=0,r=0,t=40),
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           height=600
                       ))
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _create_word_cloud(self, entities: List[Dict]):
        """Create word cloud visualization"""
        if not WORDCLOUD_AVAILABLE:
            st.error("Install wordcloud: pip install wordcloud")
            return
        
        # Create frequency dict
        freq_dict = {e['name']: e['count'] for e in entities}
        
        # Generate word cloud
        wc = WordCloud(width=800, height=400, 
                      background_color='white',
                      relative_scaling=0.5,
                      min_font_size=10).generate_from_frequencies(freq_dict)
        
        # Display
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        ax.set_title('Entity Word Cloud')
        st.pyplot(fig)
    
    def _create_table(self, entities: List[Dict]):
        """Create detailed entity table"""
        df = pd.DataFrame(entities)
        
        # Add emoji column
        df['Type Icon'] = df['type'].map(lambda t: self.entity_types[t]['emoji'])
        
        # Reorder columns
        columns = ['Type Icon', 'name', 'type', 'count', 'description']
        df = df[columns]
        df.columns = ['', 'Entity', 'Type', 'Count', 'Description']
        
        # Sort by count
        df = df.sort_values('Count', ascending=False)
        
        st.dataframe(df, use_container_width=True, hide_index=True)
    
    def render_entity_tracker(self, text: str = None):
        """Main render method for entity tracker"""
        st.header("🎯 Entity Tracker")
        
        tabs = st.tabs(["📝 Extract", "📊 Visualize", "📚 Glossary", "📈 Analytics"])
        
        with tabs[0]:
            # Text input
            if text:
                input_text = st.text_area("Text to analyze", value=text, height=200)
            else:
                input_text = st.text_area(
                    "Text to analyze", 
                    placeholder="Paste your text here to extract entities...",
                    height=200
                )
            
            if st.button("🔍 Extract Entities", type="primary"):
                if input_text:
                    entities = self.extract_entities(input_text)
                    st.session_state.extracted_entities = entities
                    st.success(f"Extracted {len(entities)} entities")
                else:
                    st.warning("Please enter some text to analyze")
            
            # Display extracted entities
            if st.session_state.extracted_entities:
                st.subheader(f"Found {len(st.session_state.extracted_entities)} Entities")
                
                # Filter by type
                col1, col2 = st.columns([1, 3])
                with col1:
                    entity_types = ['all'] + list(set(e['type'] for e in st.session_state.extracted_entities))
                    filter_type = st.selectbox("Filter by type", entity_types)
                
                # Filter entities
                filtered = st.session_state.extracted_entities
                if filter_type != 'all':
                    filtered = [e for e in filtered if e['type'] == filter_type]
                
                # Display entities
                for entity in filtered:
                    emoji = self.entity_types[entity['type']]['emoji']
                    auto_tag = "🔮" if entity.get('auto_detected') else "📚"
                    
                    col1, col2, col3 = st.columns([3, 1, 2])
                    with col1:
                        st.write(f"{emoji} **{entity['name']}** {auto_tag}")
                    with col2:
                        st.write(f"×{entity['count']}")
                    with col3:
                        st.write(f"*{entity.get('description', '')}*")
        
        with tabs[1]:
            if st.session_state.extracted_entities:
                # Visualization options
                viz_type = st.radio(
                    "Visualization Type",
                    ["bar", "network", "cloud", "table"],
                    horizontal=True
                )
                
                self.visualize_entities(st.session_state.extracted_entities, viz_type)
            else:
                st.info("Extract entities first to visualize")
        
        with tabs[2]:
            self.render_glossary_manager()
        
        with tabs[3]:
            if st.session_state.extracted_entities:
                self._render_analytics(st.session_state.extracted_entities)
            else:
                st.info("Extract entities first to see analytics")
    
    def _render_analytics(self, entities: List[Dict]):
        """Render entity analytics"""
        st.subheader("📊 Entity Analytics")
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Entities", len(entities))
        
        with col2:
            total_occurrences = sum(e['count'] for e in entities)
            st.metric("Total Occurrences", total_occurrences)
        
        with col3:
            unique_types = len(set(e['type'] for e in entities))
            st.metric("Entity Types", unique_types)
        
        with col4:
            auto_detected = sum(1 for e in entities if e.get('auto_detected'))
            st.metric("Auto-detected", auto_detected)
        
        # Type distribution
        st.subheader("Entity Type Distribution")
        type_counts = Counter(e['type'] for e in entities)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Pie chart
        colors = [self.entity_types[t]['color'] for t in type_counts.keys()]
        ax1.pie(type_counts.values(), labels=type_counts.keys(), colors=colors, autopct='%1.1f%%')
        ax1.set_title("Entity Types")
        
        # Top entities by count
        top_entities = sorted(entities, key=lambda x: x['count'], reverse=True)[:10]
        if top_entities:
            names = [e['name'][:20] + '...' if len(e['name']) > 20 else e['name'] for e in top_entities]
            counts = [e['count'] for e in top_entities]
            colors = [self.entity_types[e['type']]['color'] for e in top_entities]
            
            ax2.barh(names[::-1], counts[::-1], color=colors[::-1])
            ax2.set_xlabel("Occurrences")
            ax2.set_title("Top 10 Entities")
            ax2.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Export analytics
        if st.button("📥 Export Analytics"):
            analytics = {
                'summary': {
                    'total_entities': len(entities),
                    'total_occurrences': total_occurrences,
                    'unique_types': unique_types,
                    'auto_detected': auto_detected
                },
                'type_distribution': dict(type_counts),
                'entities': entities,
                'timestamp': datetime.now().isoformat()
            }
            
            st.download_button(
                "Download JSON",
                data=json.dumps(analytics, indent=2),
                file_name=f"entity_analytics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )


def integrate_with_translation(translation_state: dict) -> dict:
    """
    Integrate entity tracking with translation workflow
    
    Args:
        translation_state: Current translation state dictionary
        
    Returns:
        Updated state with entity information
    """
    tracker = EntitiesTracker()
    
    # Extract entities from source text
    source_entities = tracker.extract_entities(translation_state.get('source_text', ''))
    
    # Extract entities from translated text
    translated_entities = tracker.extract_entities(translation_state.get('final_translation', ''))
    
    # Add to translation state
    translation_state['source_entities'] = source_entities
    translation_state['translated_entities'] = translated_entities
    
    # Calculate entity preservation rate
    source_names = {e['name'].lower() for e in source_entities}
    translated_names = {e['name'].lower() for e in translated_entities}
    
    if source_names:
        preservation_rate = len(source_names & translated_names) / len(source_names)
        translation_state['entity_preservation_rate'] = preservation_rate
    
    return translation_state


# Demo usage
if __name__ == "__main__":
    st.set_page_config(page_title="Entity Tracker", page_icon="🎯", layout="wide")
    
    tracker = EntitiesTracker()
    
    # Sample text for demo
    sample_text = """
    John Smith, CEO of TechCorp, announced at the AI Summit in New York that 
    the company will be launching a new product in March 2024. The announcement 
    was made during the keynote speech where Smith discussed the future of 
    artificial intelligence and machine learning. TechCorp, based in San Francisco,
    has been a leader in AI research since 2020.
    """
    
    tracker.render_entity_tracker(sample_text)

