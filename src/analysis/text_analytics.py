import streamlit as st
import matplotlib.pyplot as plt
import re
import unicodedata
from wordcloud import WordCloud

# Ensure this import path is correct based on your project structure
from multilingual_stopwords import MULTILINGUAL_STOPWORDS

def wordcloud_frequencies(text: str, stopwords: set) -> dict:
    """
    Compute word frequency dictionary after removing stopwords.
    Handles Unicode punctuation, normalizes tokens, and filters properly.
    
    Args:
        text (str): The source text to analyze.
        stopwords (set): A set of words to ignore.
        
    Returns:
        dict: A dictionary mapping words to their frequency counts.
    """
    if not text:
        return {}

    # Normalize accents and unicode punctuation (e.g., convert smart quotes)
    text = unicodedata.normalize("NFKC", text)
    
    # Replace all punctuation (including smart quotes) with spaces
    # Keeps alphanumeric characters, whitespace, and hyphens
    text = re.sub(r"[^\w\s-]", " ", text)
    
    # Convert to lowercase for consistent counting
    text = text.lower()
    
    # Split into tokens
    words = text.split()
    freq = {}
    
    for w in words:
        # Remove leading/trailing hyphens
        w = w.strip("-")
        
        # Skip single characters or empty strings
        if len(w) <= 1:
            continue
            
        # Skip stopwords
        if w in stopwords:
            continue
            
        freq[w] = freq.get(w, 0) + 1
        
    return freq

def render_wordcloud_from_freq(freq_dict: dict, title: str = "Word Cloud"):
    """
    Render a wordcloud in Streamlit based on a frequency dictionary.
    
    Args:
        freq_dict (dict): Dictionary of word frequencies.
        title (str): Title for the chart (currently unused in visualization but good for context).
    """
    if not freq_dict:
        st.info("No data to render word cloud.")
        return

    try:
        # Generate word cloud object
        wc = WordCloud(
            width=800,
            height=400,
            background_color="white",
            colormap="viridis"
        ).generate_from_frequencies(freq_dict)

        # Create Matplotlib figure
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        
        # Render in Streamlit
        st.pyplot(fig)

    except Exception as e:
        st.warning(f"Could not render word cloud: {str(e)}")