import streamlit as st
from packaging import version
import streamlit

# Get Streamlit version
STREAMLIT_VERSION = version.parse(streamlit.__version__)

def divider():
    """
    Cross-version compatible divider
    """
    if STREAMLIT_VERSION >= version.parse("1.16.0"):
        return st.divider()
    else:
        return st.markdown("---")

def rerun():
    """
    Cross-version compatible rerun
    """
    if STREAMLIT_VERSION >= version.parse("1.18.0"):
        return st.rerun()
    else:
        return st.experimental_rerun() 