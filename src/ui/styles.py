# src/ui/styles.py
import streamlit as st

def apply_custom_styles():
    """Inject custom CSS into the Streamlit app."""
    st.markdown("""
    <style>
        /* Card styling for agents */
        .agent-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            border-radius: 10px;
            color: white;
            margin: 10px 0;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        
        /* Critical passage highlighting */
        .critical-passage {
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 15px;
            margin: 10px 0;
            border-radius: 4px;
            color: #856404;
        }
        
        /* Planning card */
        .planning-card {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            padding: 15px;
            border-radius: 8px;
            color: white;
            margin: 10px 0;
        }
        
        /* Cache stats card */
        .cache-card {
            background: linear-gradient(135deg, #00c6ff 0%, #0072ff 100%);
            padding: 15px;
            border-radius: 8px;
            color: white;
            margin: 10px 0;
        }
        
        /* Confidence score card */
        .confidence-card {
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
            padding: 15px;
            border-radius: 8px;
            color: white;
            margin: 10px 0;
        }
        
        /* Tab spacing adjustment */
        .stTabs [data-baseweb="tab-list"] { gap: 24px; }
        .stTabs [data-baseweb="tab"] { padding: 10px 20px; }
    </style>
    """, unsafe_allow_html=True)