"""
Compatibility functions for Streamlit.
"""

import streamlit as st
from packaging import version
import streamlit
from typing import Any

# Get Streamlit version
STREAMLIT_VERSION = version.parse(streamlit.__version__)

def divider():
    """
    Create a horizontal divider with consistent styling.
    """
    if STREAMLIT_VERSION >= version.parse("1.16.0"):
        return st.divider()
    else:
        return st.markdown("---")

def rerun():
    """
    Rerun the Streamlit app.
    """
    if STREAMLIT_VERSION >= version.parse("1.18.0"):
        return st.rerun()
    else:
        return st.experimental_rerun()

def display_file_info(file: Any):
    """
    Display information about an uploaded file.
    
    Args:
        file: The uploaded file object
    """
    st.markdown(f"**Filename:** {file.name}")
    st.markdown(f"**Size:** {file.size / 1024:.2f} KB")
    st.markdown(f"**Type:** {file.type}") 