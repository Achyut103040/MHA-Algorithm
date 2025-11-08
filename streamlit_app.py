"""
MHA Flow - Streamlit Web Application Entry Point
This file is the main entry point for Streamlit Cloud deployment.
"""

import sys
from pathlib import Path

# Add the current directory to the path
sys.path.insert(0, str(Path(__file__).parent))

# Import and run the main UI
from mha_toolbox.mha_ui_complete import *

# The streamlit app will run automatically when this file is executed
