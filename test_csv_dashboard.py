"""
CSV Dashboard Test Runner
========================

Quick test script to verify the CSV dashboard functionality
"""

import streamlit as st
import sys
from pathlib import Path
import json
from datetime import datetime

# Add the parent directory to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from mha_toolbox.csv_session_manager import SessionCSVManager, ConvergencePlotter
from mha_toolbox.comprehensive_dashboard import ComprehensiveDashboard

def create_sample_session():
    """Create a sample session for testing"""
    
    csv_manager = SessionCSVManager()
    
    # Initialize a test session
    session_id = csv_manager.initialize_session("Test_Session", "Breast Cancer")
    
    # Add some sample algorithms
    for i, alg_name in enumerate(["PSO", "GA", "SMA"]):
        algorithm_results = {
            'best_fitness': 0.1 + i * 0.05,
            'total_iterations': 50,
            'execution_time': 25.5 + i * 5,
            'statistics': {
                'mean_fitness': 0.15 + i * 0.05,
                'std_fitness': 0.02,
                'total_runs': 3
            }
        }
        
        # Create sample convergence data
        import numpy as np
        np.random.seed(i)
        convergence_curve = []
        current_fitness = 1.0
        for j in range(50):
            current_fitness = max(algorithm_results['best_fitness'], current_fitness - np.random.exponential(0.02))
            convergence_curve.append(current_fitness)
        
        csv_manager.add_algorithm_to_session(alg_name, algorithm_results, convergence_curve)
    
    return csv_manager

def test_csv_dashboard():
    """Test the CSV dashboard functionality"""
    
    st.title("🧪 CSV Dashboard Test")
    
    if st.button("🚀 Create Sample Session"):
        with st.spinner("Creating sample session..."):
            csv_manager = create_sample_session()
            st.session_state.test_csv_manager = csv_manager
            st.success("✅ Sample session created!")
    
    if 'test_csv_manager' in st.session_state:
        st.markdown("---")
        st.markdown("## 📊 **Sample Dashboard**")
        
        # Initialize components
        convergence_plotter = ConvergencePlotter()
        dashboard = ComprehensiveDashboard(st.session_state.test_csv_manager, convergence_plotter)
        
        # Display dashboard
        dashboard.display_main_dashboard()

if __name__ == "__main__":
    test_csv_dashboard()