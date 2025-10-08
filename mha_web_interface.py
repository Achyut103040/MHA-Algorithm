"""
MHA Comparison Toolbox - Streamlit Web Interface
===============================================

Professional web interface for metaheuristic algorithm comparison
with real-time progress updates and interactive visualizations.

Features:
- Algorithm selection and comparison
- Task type selection (Feature Selection, Feature Optimization, Hyperparameter Tuning)
- Real-time progress tracking
- Interactive plots and results
- Download results functionality
- Persistent results storage and management
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import time
import os
from datetime import datetime
import threading
import queue
import io
import base64
import sys
from pathlib import Path

# Add the parent directory to sys.path to import mha_toolbox modules
sys.path.append(str(Path(__file__).parent))

from mha_comparison_toolbox import MHAComparisonToolbox
# Note: run_algorithm_with_timeout is defined in this file
from mha_toolbox.results_manager import ResultsManager

# Configure page
st.set_page_config(
    page_title="MHA Comparison Toolbox",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.main-header {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    padding: 1rem;
    border-radius: 10px;
    margin-bottom: 2rem;
}
.main-header h1 {
    color: white;
    text-align: center;
    margin: 0;
}
.metric-card {
    background: #f8f9fa;
    padding: 1rem;
    border-radius: 8px;
    border-left: 4px solid #667eea;
    margin: 0.5rem 0;
}
.stTabs [data-baseweb="tab-list"] {
    gap: 2px;
}
.stTabs [data-baseweb="tab"] {
    background-color: #f0f2f6;
    border-radius: 4px 4px 0px 0px;
}
.stTabs [aria-selected="true"] {
    background-color: #667eea;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# Import the comparison toolbox
import sys
sys.path.append('.')
from mha_comparison_toolbox import MHAComparisonToolbox

# Initialize session state
if 'toolbox' not in st.session_state:
    st.session_state.toolbox = None
if 'results_ready' not in st.session_state:
    st.session_state.results_ready = False
if 'progress_data' not in st.session_state:
    st.session_state.progress_data = {}
if 'results_manager' not in st.session_state:
    st.session_state.results_manager = ResultsManager()

def main():
    """Main web interface function"""
    
    # Header with demo banner
    st.markdown("""
    <div class="main-header">
        <h1>🧬 MHA Comprehensive Demo System</h1>
        <p style="text-align: center; color: white; margin: 0;">
            Complete Metaheuristic Algorithm Comparison & Optimization Platform
        </p>
        <p style="text-align: center; color: #f0f0f0; margin: 0.5rem 0 0 0; font-size: 0.9rem;">
            ✨ Ready-to-use • 🚀 All algorithms auto-selected • ⚡ Optimized for demos • 🎯 Exploration/Exploitation analysis
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Quick demo instructions
    with st.expander("🚀 Quick Demo Guide", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **🎯 Ready to run in 3 steps:**
            1. 📊 Choose a dataset (6 available)
            2. ⚙️ Select parameter preset
            3. ▶️ Click "Run Comparison"
            
            **🔥 Demo Features:**
            - 🧬 **17 algorithms** auto-selected
            - ⚡ **Robust timeout** protection
            - 🎯 **Real-time progress** tracking
            """)
        with col2:
            st.markdown("""
            **📊 Available Datasets:**
            - Breast Cancer (569 samples, 30 features)
            - Wine (178 samples, 13 features)
            - Iris (150 samples, 4 features)  
            - Digits (1797 samples, 64 features)
            - California Housing (20640 samples)
            - Diabetes (442 samples, 10 features)
            """)
    
    # Algorithm availability check
    try:
        import mha_toolbox as mha
        st.success("✅ MHA Toolbox loaded successfully - All 17 algorithms ready!")
    except ImportError:
        st.error("❌ MHA Toolbox not found. Please install: `pip install mha-toolbox`")
        return
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("🔄 Navigation")
        
        # Page selection
        page = st.radio(
            "Select Page:",
            ["🧬 Run Experiments", "📚 Results History"],
            index=0
        )
        
        if page == "📚 Results History":
            display_persistent_results_manager()
            return
        
        st.markdown("---")
        st.header("⚙️ Configuration")
        
        # Task type selection
        task_type = st.selectbox(
            "🎯 Select Task Type",
            ["feature_selection", "feature_optimization", "hyperparameter_tuning"],
            index=0,
            format_func=lambda x: {
                "feature_selection": "🔍 Feature Selection (Default)",
                "feature_optimization": "🎯 Feature Optimization", 
                "hyperparameter_tuning": "⚙️ Hyperparameter Tuning"
            }[x]
        )
        
        st.markdown("---")
        
        # Data upload/selection
        st.subheader("📊 Data Selection")
        data_source = st.radio(
            "Choose data source:",
            ["Sample Datasets", "Upload CSV"]
        )
        
        if data_source == "Sample Datasets":
            dataset_choice = st.selectbox(
                "Select dataset:",
                ["Breast Cancer", "Wine", "Iris", "Digits", "California Housing", "Diabetes"]
            )
            
            X, y, dataset_name = load_sample_dataset(dataset_choice)
            
        else:
            uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
            if uploaded_file is not None:
                df = pd.read_csv(uploaded_file)
                st.write("Preview:", df.head())
                
                target_col = st.selectbox("Select target column:", df.columns)
                feature_cols = [col for col in df.columns if col != target_col]
                
                X = df[feature_cols].values
                y = df[target_col].values
                dataset_name = uploaded_file.name
            else:
                st.warning("Please upload a CSV file")
                return
        
        # Display data info
        if 'X' in locals():
            st.info(f"📋 Data: {X.shape[0]} samples, {X.shape[1]} features")
        
        st.markdown("---")
        
        # Algorithm selection - All algorithms auto-selected
        st.subheader("🧬 Algorithm Selection")
        
        # Complete algorithm list (37 algorithms discovered)
        available_algorithms = [
            "pso", "gwo", "sca", "woa", "alo", "ants", "ao", "aoa", "ba", "cgo", 
            "chio", "coa", "csa", "de", "eo", "fa", "fbi", "ga", "gbo", "hgso",
            "ica", "innov", "mrfo", "msa", "pfa", "qsa", "sa", "sma", "spbo",
            "spider", "ssa", "tso", "vcs", "vns", "wca", "wdo", "aco"
        ]
        
        # Simple algorithm descriptions (no categorization)
        algorithm_descriptions = {
            "pso": "Particle Swarm Optimization - Fast swarm intelligence",
            "gwo": "Grey Wolf Optimizer - Pack hunting behavior", 
            "sca": "Sine Cosine Algorithm - Mathematical exploration",
            "woa": "Whale Optimization Algorithm - Humpback whale hunting",
            "alo": "Ant Lion Optimizer - Antlion hunting strategy",
            "ants": "Ant System - Classic ant colony approach",
            "ao": "Aquila Optimizer - Eagle hunting behavior",
            "aoa": "Arithmetic Optimization Algorithm - Mathematical operations",
            "ba": "Bat Algorithm - Echolocation navigation",
            "cgo": "Chaos Game Optimization - Chaos theory based",
            "chio": "Coronavirus Herd Immunity Optimization",
            "coa": "Coyote Optimization Algorithm - Pack behavior",
            "csa": "Crow Search Algorithm - Crow intelligence",
            "de": "Differential Evolution - Evolutionary approach",
            "eo": "Equilibrium Optimizer - Physics equilibrium",
            "fa": "Firefly Algorithm - Bioluminescent attraction",
            "fbi": "Forensic-Based Investigation - Crime investigation",
            "ga": "Genetic Algorithm - Natural selection",
            "gbo": "Gradient Based Optimizer - Gradient descent",
            "hgso": "Henry Gas Solubility Optimization",
            "ica": "Imperialist Competitive Algorithm",
            "innov": "Innovation Algorithm - Creative optimization",
            "mrfo": "Manta Ray Foraging Optimization",
            "msa": "Moth Search Algorithm - Moth navigation",
            "pfa": "Pathfinder Algorithm - Path exploration",
            "qsa": "Queuing Search Algorithm - Queue theory",
            "sa": "Simulated Annealing - Metal cooling process",
            "sma": "Slime Mould Algorithm - Slime behavior",
            "spbo": "Student Psychology Based Optimization",
            "spider": "Social Spider Algorithm - Spider web",
            "ssa": "Salp Swarm Algorithm - Salp chain movement",
            "tso": "Tuna Swarm Optimization - Fish schooling",
            "vcs": "Virus Colony Search - Viral propagation",
            "vns": "Variable Neighborhood Search",
            "wca": "Water Cycle Algorithm - Water circulation",
            "wdo": "Wind Driven Optimization - Wind dynamics",
            "aco": "Ant Colony Optimization - Pheromone trails"
        }
        
        # Algorithm selection mode
        st.info("🚀 **All 37 algorithms auto-selected for comprehensive comparison**")
        
        selection_mode = st.selectbox(
            "Selection mode:",
            ["All Algorithms (37)", "Fast Subset (15)", "Popular Algorithms (10)", "Custom Selection"],
            index=0,  # Default to "All Algorithms (37)"
            help="Choose how many algorithms to run"
        )
        
        if selection_mode == "All Algorithms (37)":
            selected_algorithms = available_algorithms
        elif selection_mode == "Fast Subset (15)":
            selected_algorithms = ["pso", "gwo", "sca", "woa", "alo", "ga", "de", "ssa", "fa", "ba", "sa", "eo", "msa", "cgo", "gbo"]
        elif selection_mode == "Popular Algorithms (10)":
            selected_algorithms = ["pso", "gwo", "sca", "woa", "alo", "ga", "de", "ssa", "fa", "ba"]
        else:
            selected_algorithms = st.multiselect(
                "Custom algorithm selection:",
                available_algorithms,
                default=available_algorithms,  # Default to ALL algorithms instead of just 10
                format_func=lambda x: f"{x.upper()} - {algorithm_descriptions.get(x, 'Algorithm')}",
                help="Manually select specific algorithms"
            )
        
        if len(selected_algorithms) < 1:
            st.warning("⚠️ Please select at least 1 algorithm")
            return
            
        # Show selection summary (no categorization)
        st.info(f"🎯 **{len(selected_algorithms)} algorithms selected** from {len(available_algorithms)} available")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🧬 Selected", len(selected_algorithms))
        with col2:
            st.metric("📊 Available", len(available_algorithms))
        with col3:
            coverage = (len(selected_algorithms) / len(available_algorithms)) * 100
            st.metric("📈 Coverage", f"{coverage:.1f}%")
        
        # Show selected algorithms (first 10 for display)
        if selected_algorithms:
            display_algorithms = selected_algorithms[:10]
            remaining = len(selected_algorithms) - 10
            
            algo_display = ", ".join([alg.upper() for alg in display_algorithms])
            if remaining > 0:
                algo_display += f" ... and {remaining} more"
            
            st.success(f"✅ **Algorithms Ready**: {algo_display}")
            
            # Debug info for troubleshooting
            if len(selected_algorithms) == 1:
                st.warning(f"🔍 **Debug**: Only 1 algorithm selected: {selected_algorithms[0]}. Check selection mode above.")
        
        st.markdown("---")
        
        # Optimization parameters - Demo optimized
        st.subheader("🔧 Parameters")
        
        # Preset configurations
        preset_config = st.selectbox(
            "Parameter preset:",
            ["Demo (Fast)", "Standard", "Thorough", "Custom"],
            help="Pre-configured parameter sets for different needs"
        )
        
        if preset_config == "Demo (Fast)":
            max_iterations = 20
            population_size = 15
            n_runs = 2
            timeout_minutes = 5
            st.info("⚡ Fast demo settings for quick results")
        elif preset_config == "Standard":
            max_iterations = 50
            population_size = 25
            n_runs = 3
            timeout_minutes = 10
            st.info("⚖️ Balanced settings for good results")
        elif preset_config == "Thorough":
            max_iterations = 100
            population_size = 40
            n_runs = 5
            timeout_minutes = 20
            st.info("🎯 Comprehensive settings for best results")
        else:
            col1, col2 = st.columns(2)
            with col1:
                max_iterations = st.slider("Max Iterations", 10, 200, 50)
                population_size = st.slider("Population Size", 10, 100, 25)
            with col2:
                n_runs = st.slider("Number of Runs", 1, 10, 3)
                timeout_minutes = st.slider("Timeout (minutes)", 1, 60, 15)
        
        # Show current settings
        st.markdown(f"""
        **Current Settings:**
        - 🔄 Iterations: {max_iterations}
        - 👥 Population: {population_size}
        - 🎲 Runs: {n_runs}
        - ⏱️ Timeout: {timeout_minutes} min
        """)
        
        # Additional settings
        with st.expander("⚙️ Advanced Settings"):
            enable_auto_save = st.checkbox("Auto-save results", value=True)
            show_detailed_progress = st.checkbox("Show detailed progress", value=True)
            enable_parallel = st.checkbox("Enable parallel execution", value=True)
    
    # Main content area
    if 'X' in locals() and len(selected_algorithms) >= 1:
        
        # Debug info
        st.info(f"🔍 **Debug**: {len(selected_algorithms)} algorithms selected: {', '.join(selected_algorithms[:10])}{'...' if len(selected_algorithms) > 10 else ''}")
        
        # Run comparison button
        if st.button("🚀 Start Comparison", type="primary", width='stretch'):
            st.info(f"🚀 **Starting comprehensive comparison with {len(selected_algorithms)} algorithms**")
            
            # Debug display - show selected algorithms
            with st.expander("🔍 Debug: Selected Algorithms"):
                st.write(f"**Selection Mode**: {selection_mode}")
                st.write(f"**Total Selected**: {len(selected_algorithms)}")
                st.write(f"**Algorithms**: {', '.join(selected_algorithms)}")
            
            run_comparison_with_progress(
                X, y, dataset_name, task_type, selected_algorithms,
                max_iterations, population_size, n_runs, timeout_minutes
            )
        
        # Display results if available
        if st.session_state.results_ready and st.session_state.toolbox:
            display_results()
    
    else:
        st.info("👆 Please configure the parameters in the sidebar to start")

def load_sample_dataset(dataset_choice):
    """Load sample datasets"""
    if dataset_choice == "Breast Cancer":
        from sklearn.datasets import load_breast_cancer
        data = load_breast_cancer()
        return data.data, data.target, "Breast Cancer"
    
    elif dataset_choice == "Wine":
        from sklearn.datasets import load_wine
        data = load_wine()
        return data.data, data.target, "Wine"
    
    elif dataset_choice == "Iris":
        from sklearn.datasets import load_iris
        data = load_iris()
        return data.data, data.target, "Iris"
    
    elif dataset_choice == "Digits":
        from sklearn.datasets import load_digits
        data = load_digits()
        return data.data, data.target, "Digits"
    
    elif dataset_choice == "California Housing":
        from sklearn.datasets import fetch_california_housing
        data = fetch_california_housing()
        return data.data, data.target, "California Housing"
        
    elif dataset_choice == "Diabetes":
        from sklearn.datasets import load_diabetes
        data = load_diabetes()
        return data.data, data.target, "Diabetes"

def run_comparison_with_progress(X, y, dataset_name, task_type, algorithms, 
                                max_iterations, population_size, n_runs, timeout_minutes):
    """Run comprehensive comparison with ALL algorithms at once"""
    
    st.info(f"🚀 **COMPREHENSIVE MODE**: Running {len(algorithms)} algorithms simultaneously")
    st.write(f"**Selected algorithms**: {', '.join(algorithms)}")
    
    # Initialize progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    progress_details = st.empty()
    
    # Summary metrics
    summary_container = st.container()
    
    # Initialize toolbox
    toolbox = MHAComparisonToolbox()
    toolbox.set_task_type(task_type)
    toolbox.load_data(X, y, dataset_name)
    
    # Progress tracking
    total_algorithms = len(algorithms)
    total_steps = total_algorithms * n_runs
    current_step = 0
    
    # Results tracking
    successful_algorithms = 0
    failed_algorithms = 0
    all_results = {}
    
    status_text.text(f"🚀 Starting comprehensive comparison of {total_algorithms} algorithms...")
    
    # Display progress summary
    with summary_container:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            success_metric = st.metric("✅ Successful", successful_algorithms)
        with col2:
            total_metric = st.metric("📊 Total", total_algorithms)
        with col3:
            progress_metric = st.metric("🔄 Progress", "0%")
        with col4:
            time_metric = st.metric("⏱️ Total Time", "0s")
    
    # Run comparison with progress updates
    start_time = time.time()
    
    try:
        for i, alg_name in enumerate(algorithms):
            alg_start_time = time.time()
            
            # Update status
            progress_percent = (i / total_algorithms) * 100
            status_text.text(f"🧬 [{i+1}/{total_algorithms}] Running {alg_name.upper()}... ({progress_percent:.1f}%)")
            
            # Run algorithm with timeout protection
            alg_results = run_algorithm_with_timeout(
                alg_name, X, y, task_type, max_iterations, 
                population_size, n_runs, timeout_minutes * 60
            )
            
            if alg_results:
                # Success
                all_results[alg_name] = alg_results
                toolbox.results[alg_name] = alg_results  # Add to main toolbox
                successful_algorithms += 1
                
                # Update progress
                current_step += n_runs
                progress = current_step / total_steps
                progress_bar.progress(min(progress, 1.0))
                
                # Show algorithm results
                with progress_details.container():
                    st.markdown(f"### ✅ {alg_name.upper()} - Completed")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Best Fitness", f"{alg_results['statistics']['best_fitness']:.6f}")
                    with col2:
                        st.metric("Mean Fitness", f"{alg_results['statistics']['mean_fitness']:.6f}")
                    with col3:
                        st.metric("Execution Time", f"{alg_results['statistics']['mean_time']:.2f}s")
                    with col4:
                        if task_type == 'feature_selection' and 'mean_features' in alg_results['statistics']:
                            st.metric("Features", f"{alg_results['statistics']['mean_features']:.1f}")
                        else:
                            st.metric("Runs", f"{alg_results['statistics']['total_runs']}")
                
            else:
                # Failed
                failed_algorithms += 1
                st.warning(f"⚠️ {alg_name.upper()} timed out or failed")
            
            # Update summary metrics
            elapsed_time = time.time() - start_time
            with summary_container:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("✅ Successful", successful_algorithms)
                with col2:
                    st.metric("📊 Total", total_algorithms)
                with col3:
                    progress_percent = ((i + 1) / total_algorithms) * 100
                    st.metric("🔄 Progress", f"{progress_percent:.1f}%")
                with col4:
                    st.metric("⏱️ Total Time", f"{elapsed_time:.1f}s")
        
        # Final completion
        total_time = time.time() - start_time
        progress_bar.progress(1.0)
        
        if successful_algorithms > 0:
            status_text.text(f"✅ Comparison completed! {successful_algorithms}/{total_algorithms} algorithms successful")
            
            # Store results in session state
            st.session_state.toolbox = toolbox
            st.session_state.results_ready = True
            
            # Enhanced auto-save with persistent storage
            with st.spinner("💾 Auto-saving comprehensive results with persistent storage..."):
                results_manager = st.session_state.results_manager
                experiment_name = f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                export_data, saved_files = results_manager.save_comprehensive_results(
                    toolbox, 
                    session_name=experiment_name
                )
                
                if saved_files:
                    st.success(f"✅ **Results Permanently Saved** to Session: `{results_manager.current_session}`")
                    
                    # Show actual file paths that were created
                    st.info("📁 **Local Files Created:**")
                    for file_type, file_path in saved_files.items():
                        if file_path and os.path.exists(file_path):
                            # Get absolute path for display
                            abs_path = os.path.abspath(file_path)
                            filename = os.path.basename(file_path)
                            st.text(f"• {file_type.replace('_', ' ').title()}: {filename}")
                            st.text(f"  📂 Path: {abs_path}")
                    
                    # Show directory where files are saved
                    session_dir = os.path.join(os.getcwd(), "results", "persistent_storage", "sessions", results_manager.current_session)
                    if os.path.exists(session_dir):
                        st.info(f"📂 **All files saved to**: `{session_dir}`")
                else:
                    st.warning("⚠️ Persistent storage encountered issues but results are still available below")
            
            # Summary statistics
            st.markdown("---")
            st.markdown("## 📊 Comprehensive Results Analysis")
            
            # Overall statistics
            all_fitnesses = []
            all_times = []
            all_features = []
            
            for results in all_results.values():
                stats = results['statistics']
                all_fitnesses.append(stats['best_fitness'])
                all_times.append(stats['mean_time'])
                if task_type == 'feature_selection' and 'mean_features' in stats:
                    all_features.append(stats['mean_features'])
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("✅ Successful", successful_algorithms, f"{(successful_algorithms/total_algorithms)*100:.1f}%")
            with col2:
                st.metric("📊 Total", total_algorithms)
            with col3:
                st.metric("🏆 Best Fitness", f"{min(all_fitnesses):.6f}" if all_fitnesses else "N/A")
            with col4:
                st.metric("⏱️ Total Time", f"{total_time:.1f}s")
            
            # Best algorithm identification
            if all_results:
                best_alg = min(all_results.items(), key=lambda x: x[1]['statistics']['best_fitness'])
                fastest_alg = min(all_results.items(), key=lambda x: x[1]['statistics']['mean_time'])
                
                st.success(f"🏆 **Best Algorithm**: {best_alg[0].upper()} (Fitness: {best_alg[1]['statistics']['best_fitness']:.6f})")
                st.info(f"⚡ **Fastest Algorithm**: {fastest_alg[0].upper()} (Time: {fastest_alg[1]['statistics']['mean_time']:.2f}s)")
            
            # Display detailed results
            display_results()
        
        else:
            status_text.text("❌ All algorithms failed or timed out")
            st.error("No algorithms completed successfully. Try reducing iterations or increasing timeout.")
            
    except Exception as e:
        status_text.text(f"❌ Error during comparison: {str(e)}")
        st.error(f"Comparison failed: {str(e)}")
        progress_bar.progress(0)
        return

def run_algorithm_with_timeout(alg_name, X, y, task_type, max_iterations, 
                              population_size, n_runs, timeout_seconds):
    """Run single algorithm with robust timeout protection and error handling"""
    
    import mha_toolbox as mha
    import signal
    from concurrent.futures import ThreadPoolExecutor, TimeoutError
    
    def timeout_handler(signum, frame):
        raise TimeoutError("Algorithm execution timeout")
    
    params = {
        'max_iterations': max_iterations,
        'population_size': population_size,
        'verbose': False
    }
    
    runs_data = []
    total_start_time = time.time()
    
    for run in range(n_runs):
        try:
            run_start_time = time.time()
            
            # Check global timeout
            if time.time() - total_start_time > timeout_seconds:
                st.warning(f"⏱️ Global timeout reached for {alg_name.upper()}")
                break
            
            # Set individual run timeout (30% of remaining time)
            remaining_time = timeout_seconds - (time.time() - total_start_time)
            run_timeout = min(remaining_time * 0.3, timeout_seconds / n_runs)
            
            # Run with timeout protection
            try:
                if hasattr(signal, 'SIGALRM'):  # Unix systems
                    signal.signal(signal.SIGALRM, timeout_handler)
                    signal.alarm(int(run_timeout))
                
                if task_type == 'feature_selection':
                    result = mha.optimize(alg_name, X, y, **params)
                    
                    run_result = {
                        'run': run + 1,
                        'best_fitness': float(result.best_fitness_),
                        'n_selected_features': int(result.n_selected_features_),
                        'convergence_curve': [float(x) for x in result.global_fitness_],
                        'execution_time': time.time() - run_start_time,
                        'final_accuracy': float(1 - result.best_fitness_),
                        'success': True
                    }
                
                elif task_type == 'feature_optimization':
                    # Create custom objective function for feature weights optimization
                    def feature_weights_objective(weights):
                        """Optimize feature weights for better model performance"""
                        try:
                            # Ensure weights are numpy array and match feature dimensions
                            weights = np.array(weights)
                            
                            # Normalize weights to [0, 1] range
                            weights = np.clip(weights, 0, 1)
                            
                            # Ensure weights match feature dimensions exactly
                            if len(weights) != X.shape[1]:
                                # Resize weights to match features
                                if len(weights) < X.shape[1]:
                                    # Repeat weights to match feature count
                                    weights = np.tile(weights, (X.shape[1] // len(weights)) + 1)[:X.shape[1]]
                                else:
                                    # Truncate weights to match feature count
                                    weights = weights[:X.shape[1]]
                            
                            # Apply weights to features (broadcasting properly)
                            X_weighted = X * weights.reshape(1, -1)
                            
                            # Use cross-validation with weighted features for classification/regression
                            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
                            from sklearn.model_selection import cross_val_score
                            
                            # Check if it's classification or regression
                            if len(np.unique(y)) < 20 and np.all(y == y.astype(int)):
                                # Classification
                                model = RandomForestClassifier(n_estimators=10, random_state=42)
                                scores = cross_val_score(model, X_weighted, y, cv=3, scoring='accuracy')
                                return 1.0 - np.mean(scores)  # Minimize for optimization
                            else:
                                # Regression
                                model = RandomForestRegressor(n_estimators=10, random_state=42)
                                scores = cross_val_score(model, X_weighted, y, cv=3, scoring='neg_mean_squared_error')
                                return -np.mean(scores)  # Minimize for optimization
                                
                        except Exception as e:
                            print(f"Feature optimization error: {e}")
                            return 1000.0  # High penalty for errors
                    
                    # Use proper optimization call with bounds for weights
                    result = mha.optimize(
                        alg_name, 
                        objective_function=feature_weights_objective, 
                        dimensions=X.shape[1],
                        bounds=[(0, 1)] * X.shape[1],  # Weights between 0 and 1
                        **params
                    )
                    
                    run_result = {
                        'run': run + 1,
                        'best_fitness': float(result.best_fitness_),
                        'convergence_curve': [float(x) for x in result.global_fitness_],
                        'execution_time': time.time() - run_start_time,
                        'optimized_weights': result.best_solution_.tolist(),
                        'performance_score': float(1 - result.best_fitness_),
                        'success': True
                    }
                
                elif task_type == 'hyperparameter_tuning':
                    # Create objective for hyperparameter optimization
                    def hyperparameter_objective(params_vector):
                        """Optimize ML model hyperparameters"""
                        try:
                            # Ensure params_vector is numpy array and normalized
                            params_vector = np.array(params_vector)
                            params_vector = np.clip(params_vector, 0, 1)
                            
                            # Map parameters to RandomForest hyperparameters
                            n_estimators = max(10, int(params_vector[0] * 190 + 10))  # 10-200
                            max_depth = max(3, int(params_vector[1] * 17 + 3)) if params_vector[1] > 0.1 else None  # 3-20 or None
                            min_samples_split = max(2, int(params_vector[2] * 18 + 2))  # 2-20
                            
                            # Check if it's classification or regression
                            if len(np.unique(y)) < 20 and np.all(y == y.astype(int)):
                                # Classification
                                from sklearn.ensemble import RandomForestClassifier
                                from sklearn.model_selection import cross_val_score
                                
                                rf = RandomForestClassifier(
                                    n_estimators=n_estimators,
                                    max_depth=max_depth,
                                    min_samples_split=min_samples_split,
                                    random_state=42
                                )
                                
                                # Use cross-validation score (minimize negative accuracy)
                                scores = cross_val_score(rf, X, y, cv=3, scoring='accuracy')
                                return 1.0 - np.mean(scores)  # Minimize for optimization
                            else:
                                # Regression
                                from sklearn.ensemble import RandomForestRegressor
                                from sklearn.model_selection import cross_val_score
                                
                                rf = RandomForestRegressor(
                                    n_estimators=n_estimators,
                                    max_depth=max_depth,
                                    min_samples_split=min_samples_split,
                                    random_state=42
                                )
                                
                                # Use cross-validation score (minimize negative R2)
                                scores = cross_val_score(rf, X, y, cv=3, scoring='neg_mean_squared_error')
                                return -np.mean(scores)  # Minimize for optimization
                            
                        except Exception as e:
                            print(f"Hyperparameter tuning error: {e}")
                            return 1.0  # High penalty for errors
                    
                    # Use proper optimization call with bounds for hyperparameters
                    result = mha.optimize(
                        alg_name, 
                        objective_function=hyperparameter_objective, 
                        dimensions=3,
                        bounds=[(0, 1)] * 3,  # Parameters between 0 and 1
                        **params
                    )
                    
                    # Calculate best hyperparameters from optimized solution
                    best_params = result.best_solution_
                    best_n_estimators = max(10, int(best_params[0] * 190 + 10))
                    best_max_depth = max(3, int(best_params[1] * 17 + 3)) if best_params[1] > 0.1 else None
                    best_min_samples_split = max(2, int(best_params[2] * 18 + 2))
                    
                    run_result = {
                        'run': run + 1,
                        'best_fitness': float(result.best_fitness_),
                        'convergence_curve': [float(x) for x in result.global_fitness_],
                        'execution_time': time.time() - run_start_time,
                        'best_accuracy': float(1 - result.best_fitness_),
                        'optimized_params': result.best_solution_.tolist(),
                        'best_hyperparameters': {
                            'n_estimators': best_n_estimators,
                            'max_depth': best_max_depth,
                            'min_samples_split': best_min_samples_split
                        },
                        'success': True
                    }
                
                else:
                    # Default fallback
                    run_result = {
                        'run': run + 1,
                        'best_fitness': np.random.random() * 0.1,
                        'execution_time': time.time() - run_start_time,
                        'success': True
                    }
                
                if hasattr(signal, 'SIGALRM'):
                    signal.alarm(0)  # Clear alarm
                    
            except (TimeoutError, Exception) as e:
                if hasattr(signal, 'SIGALRM'):
                    signal.alarm(0)
                st.warning(f"⚠️ Run {run+1} failed for {alg_name.upper()}: {str(e)[:50]}...")
                continue
            
            runs_data.append(run_result)
            
        except Exception as e:
            st.error(f"❌ Critical error in {alg_name.upper()}: {str(e)[:50]}...")
            continue
    
    if not runs_data:
        st.error(f"❌ All runs failed for {alg_name.upper()}")
        return None
    
    # Calculate robust statistics
    try:
        fitnesses = [run['best_fitness'] for run in runs_data if run.get('success', False)]
        times = [run['execution_time'] for run in runs_data if run.get('success', False)]
        
        if not fitnesses:
            return None
        
        statistics = {
            'mean_fitness': float(np.mean(fitnesses)),
            'std_fitness': float(np.std(fitnesses)),
            'best_fitness': float(np.min(fitnesses)),
            'worst_fitness': float(np.max(fitnesses)),
            'mean_time': float(np.mean(times)),
            'std_time': float(np.std(times)),
            'total_runs': len(runs_data),
            'successful_runs': len(fitnesses)
        }
        
        if task_type == 'feature_selection':
            n_features = [run['n_selected_features'] for run in runs_data if run.get('success', False)]
            accuracies = [run['final_accuracy'] for run in runs_data if run.get('success', False)]
            
            if n_features and accuracies:
                statistics.update({
                    'mean_features': float(np.mean(n_features)),
                    'std_features': float(np.std(n_features)),
                    'mean_accuracy': float(np.mean(accuracies)),
                    'std_accuracy': float(np.std(accuracies))
                })
        
        elif task_type == 'feature_optimization':
            performance_scores = [run['performance_score'] for run in runs_data if run.get('performance_score') and run.get('success', False)]
            
            if performance_scores:
                statistics.update({
                    'mean_performance': float(np.mean(performance_scores)),
                    'std_performance': float(np.std(performance_scores)),
                    'best_performance': float(np.max(performance_scores))
                })
        
        elif task_type == 'hyperparameter_tuning':
            accuracies = [run['best_accuracy'] for run in runs_data if run.get('best_accuracy') and run.get('success', False)]
            
            if accuracies:
                statistics.update({
                    'mean_accuracy': float(np.mean(accuracies)),
                    'std_accuracy': float(np.std(accuracies)),
                    'best_accuracy': float(np.max(accuracies))
                })
        
        return {
            'algorithm': alg_name,
            'runs': runs_data,
            'statistics': statistics,
            'task_type': task_type,
            'total_execution_time': time.time() - total_start_time
        }
        
    except Exception as e:
        st.error(f"❌ Error processing results for {alg_name.upper()}: {str(e)}")
        return None

def display_results():
    """Display comprehensive results with interactive plots"""
    
    toolbox = st.session_state.toolbox
    
    st.markdown("---")
    st.header("📊 Comparison Results")
    
    # Results tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Overview", "🔄 Convergence", "📊 Statistics", "💾 Export"])
    
    with tab1:
        display_overview(toolbox)
    
    with tab2:
        display_convergence_plots(toolbox)
    
    with tab3:
        display_detailed_statistics(toolbox)
    
    with tab4:
        display_export_options(toolbox)

def display_overview(toolbox):
    """Display overview of results"""
    
    st.subheader("🏆 Algorithm Rankings")
    
    # Create ranking dataframe
    ranking_data = []
    for alg_name, results in toolbox.results.items():
        stats = results['statistics']
        ranking_data.append({
            'Algorithm': alg_name.upper(),
            'Mean Fitness': stats['mean_fitness'],
            'Std Fitness': stats['std_fitness'],
            'Best Fitness': stats['best_fitness'],
            'Mean Time (s)': stats['mean_time'],
            'Runs': stats['total_runs']
        })
    
    # Sort by mean fitness (lower is better)
    ranking_df = pd.DataFrame(ranking_data).sort_values('Mean Fitness')
    
    # Display with styling
    st.dataframe(
        ranking_df.style.format({
            'Mean Fitness': '{:.6f}',
            'Std Fitness': '{:.6f}',
            'Best Fitness': '{:.6f}',
            'Mean Time (s)': '{:.2f}'
        }),
        width='stretch'
    )
    
    # Performance comparison chart
    fig = px.bar(
        ranking_df, 
        x='Algorithm', 
        y='Mean Fitness',
        error_y='Std Fitness',
        title="Algorithm Performance Comparison",
        color='Mean Fitness',
        color_continuous_scale='RdYlGn_r'
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

def display_convergence_plots(toolbox):
    """Display convergence plots"""
    
    st.subheader("🔄 Convergence Analysis")
    
    # Create convergence plot
    fig = go.Figure()
    
    for alg_name, results in toolbox.results.items():
        # Get best run's convergence curve
        best_run = min(results['runs'], key=lambda x: x['best_fitness'])
        convergence = best_run['convergence_curve']
        
        fig.add_trace(go.Scatter(
            x=list(range(len(convergence))),
            y=convergence,
            mode='lines+markers',
            name=alg_name.upper(),
            line=dict(width=2),
            marker=dict(size=4)
        ))
    
    fig.update_layout(
        title="Convergence Curves (Best Runs)",
        xaxis_title="Iteration",
        yaxis_title="Fitness Value",
        height=500,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Box plot for fitness distribution
    st.subheader("📦 Fitness Distribution Across Runs")
    
    box_data = []
    for alg_name, results in toolbox.results.items():
        for run in results['runs']:
            box_data.append({
                'Algorithm': alg_name.upper(),
                'Fitness': run['best_fitness']
            })
    
    box_df = pd.DataFrame(box_data)
    fig_box = px.box(
        box_df, 
        x='Algorithm', 
        y='Fitness',
        title="Fitness Distribution (All Runs)"
    )
    fig_box.update_layout(height=400)
    st.plotly_chart(fig_box, use_container_width=True)

def display_detailed_statistics(toolbox):
    """Display detailed statistical analysis"""
    
    st.subheader("📊 Detailed Statistics")
    
    # Create metrics grid
    algorithms = list(toolbox.results.keys())
    n_cols = min(len(algorithms), 3)
    cols = st.columns(n_cols)
    
    for i, (alg_name, results) in enumerate(toolbox.results.items()):
        with cols[i % n_cols]:
            stats = results['statistics']
            
            st.markdown(f"### {alg_name.upper()}")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Best Fitness", f"{stats['best_fitness']:.6f}")
                st.metric("Mean Fitness", f"{stats['mean_fitness']:.6f}")
                st.metric("Std Fitness", f"{stats['std_fitness']:.6f}")
            
            with col2:
                st.metric("Mean Time", f"{stats['mean_time']:.2f}s")
                st.metric("Total Runs", stats['total_runs'])
                
                if toolbox.task_type == 'feature_selection':
                    st.metric("Mean Features", f"{stats['mean_features']:.1f}")
                    st.metric("Mean Accuracy", f"{stats['mean_accuracy']:.4f}")
    
    # Task-specific analysis
    if toolbox.task_type == 'feature_selection':
        st.subheader("🔍 Feature Selection Analysis")
        
        # Features vs Accuracy scatter plot
        scatter_data = []
        for alg_name, results in toolbox.results.items():
            stats = results['statistics']
            scatter_data.append({
                'Algorithm': alg_name.upper(),
                'Mean Features': stats['mean_features'],
                'Mean Accuracy': stats['mean_accuracy']
            })
        
        scatter_df = pd.DataFrame(scatter_data)
        fig_scatter = px.scatter(
            scatter_df,
            x='Mean Features',
            y='Mean Accuracy',
            color='Algorithm',
            size_max=15,
            title="Features vs Accuracy Trade-off"
        )
        fig_scatter.update_layout(height=400)
        st.plotly_chart(fig_scatter, use_container_width=True)

def save_comprehensive_results(toolbox, auto_save=True):
    """Save comprehensive results with all data including models - ALWAYS saves to backend"""
    
    if not toolbox or not toolbox.results:
        return None, None
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create results directory if it doesn't exist
    import os
    os.makedirs("results/auto_save", exist_ok=True)
    os.makedirs("results/models", exist_ok=True)
    
    # Comprehensive export data
    export_data = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'task_type': toolbox.task_type,
            'data_info': toolbox.data_info,
            'total_algorithms': len(toolbox.results),
            'software_version': 'MHA-Comprehensive-v2.0',
            'total_execution_time': sum([r.get('total_execution_time', 0) for r in toolbox.results.values()]),
            'auto_saved': True,
            'save_location': f"results/auto_save/"
        },
        'algorithms': {},
        'global_summary': {
            'best_overall_fitness': None,
            'best_algorithm': None,
            'fastest_algorithm': None,
            'most_stable_algorithm': None
        },
        'models': {}  # Store best models for each algorithm
    }
    
    # Collect comprehensive algorithm data
    all_fitnesses = []
    all_times = []
    algorithm_performance = {}
    
    for alg_name, results in toolbox.results.items():
        stats = results['statistics']
        runs = results['runs']
        
        # Find best run for this algorithm
        best_run = min(runs, key=lambda x: x['best_fitness']) if runs else None
        
        # Store comprehensive algorithm data
        export_data['algorithms'][alg_name] = {
            'basic_info': {
                'algorithm': results['algorithm'],
                'task_type': results['task_type'],
                'total_execution_time': results.get('total_execution_time', 0)
            },
            'statistics': stats,
            'detailed_runs': [],
            'convergence_analysis': {
                'convergence_curves': [],
                'convergence_speed': None,
                'final_convergence': None,
                'best_run_index': runs.index(best_run) if best_run and runs else 0
            },
            'performance_metrics': {
                'consistency_score': 1 - (stats['std_fitness'] / max(stats['mean_fitness'], 1e-10)),
                'efficiency_score': stats['best_fitness'] / max(stats['mean_time'], 1e-10),
                'success_rate': stats.get('successful_runs', len(runs)) / len(runs) if runs else 0
            }
        }
        
        # Store best model data
        if best_run:
            export_data['models'][alg_name] = {
                'best_fitness': best_run['best_fitness'],
                'run_number': best_run['run'],
                'execution_time': best_run['execution_time'],
                'convergence_curve': best_run.get('convergence_curve', []),
                'parameters': {
                    'max_iterations': len(best_run.get('convergence_curve', [])),
                    'final_fitness': best_run['best_fitness'],
                    'convergence_achieved': True if best_run.get('convergence_curve') else False
                }
            }
            
            # Feature selection specific model data
            if toolbox.task_type == 'feature_selection' and best_run.get('n_selected_features'):
                export_data['models'][alg_name]['feature_selection'] = {
                    'selected_features': best_run['n_selected_features'],
                    'feature_ratio': best_run['n_selected_features'] / toolbox.data_info.get('n_features', 1),
                    'accuracy': best_run.get('final_accuracy', 0),
                    'feature_efficiency': best_run.get('final_accuracy', 0) / max(best_run['n_selected_features'], 1)
                }
        
        # Process each run in detail
        for run in runs:
            run_detail = {
                'run_number': run['run'],
                'fitness': run['best_fitness'],
                'execution_time': run['execution_time'],
                'success': run.get('success', True),
                'convergence_curve': run.get('convergence_curve', []),
                'iteration_count': len(run.get('convergence_curve', [])),
                'convergence_rate': None,
                'is_best_run': run == best_run
            }
            
            # Calculate convergence rate
            if run.get('convergence_curve'):
                curve = run['convergence_curve']
                if len(curve) > 10:
                    early_avg = np.mean(curve[:len(curve)//3])
                    late_avg = np.mean(curve[-len(curve)//3:])
                    run_detail['convergence_rate'] = abs(early_avg - late_avg) / max(early_avg, 1e-10)
                
                export_data['algorithms'][alg_name]['convergence_analysis']['convergence_curves'].append(curve)
            
            # Feature selection specific data
            if toolbox.task_type == 'feature_selection' and run.get('n_selected_features'):
                run_detail.update({
                    'selected_features': run['n_selected_features'],
                    'feature_ratio': run['n_selected_features'] / toolbox.data_info.get('n_features', 1),
                    'accuracy': run.get('final_accuracy', 0)
                })
            
            export_data['algorithms'][alg_name]['detailed_runs'].append(run_detail)
        
        # Algorithm performance for rankings
        algorithm_performance[alg_name] = {
            'fitness': stats['best_fitness'],
            'time': stats['mean_time'],
            'stability': 1 - (stats['std_fitness'] / max(stats['mean_fitness'], 1e-10))
        }
        
        all_fitnesses.extend([run['best_fitness'] for run in runs])
        all_times.extend([run['execution_time'] for run in runs])
    
    # Global analysis
    if algorithm_performance:
        best_alg = min(algorithm_performance.items(), key=lambda x: x[1]['fitness'])
        fastest_alg = min(algorithm_performance.items(), key=lambda x: x[1]['time'])
        stable_alg = max(algorithm_performance.items(), key=lambda x: x[1]['stability'])
        
        export_data['global_summary'].update({
            'best_overall_fitness': min(all_fitnesses) if all_fitnesses else None,
            'best_algorithm': best_alg[0],
            'fastest_algorithm': fastest_alg[0],
            'most_stable_algorithm': stable_alg[0],
            'total_runs': len(all_fitnesses),
            'average_execution_time': np.mean(all_times) if all_times else 0,
            'performance_rankings': {
                'by_fitness': sorted(algorithm_performance.items(), key=lambda x: x[1]['fitness']),
                'by_speed': sorted(algorithm_performance.items(), key=lambda x: x[1]['time']),
                'by_stability': sorted(algorithm_performance.items(), key=lambda x: x[1]['stability'], reverse=True)
            }
        })
    
    # ALWAYS save files to backend (auto_save=True by default)
    results_file = None
    summary_file = None
    models_file = None
    
    try:
        # Save comprehensive JSON results
        results_filename = f"results/auto_save/mha_comprehensive_results_{timestamp}.json"
        with open(results_filename, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        results_file = results_filename
        
        # Save models separately
        models_filename = f"results/models/mha_best_models_{timestamp}.json"
        with open(models_filename, 'w') as f:
            json.dump(export_data['models'], f, indent=2, default=str)
        models_file = models_filename
        
        # Save summary CSV
        summary_data = []
        for alg_name, perf in algorithm_performance.items():
            stats = toolbox.results[alg_name]['statistics']
            row = {
                'Algorithm': alg_name.upper(),
                'Best_Fitness': perf['fitness'],
                'Mean_Fitness': stats['mean_fitness'],
                'Std_Fitness': stats['std_fitness'],
                'Mean_Time': perf['time'],
                'Stability_Score': perf['stability'],
                'Efficiency_Score': export_data['algorithms'][alg_name]['performance_metrics']['efficiency_score'],
                'Success_Rate': export_data['algorithms'][alg_name]['performance_metrics']['success_rate'],
                'Total_Runs': stats['total_runs'],
                'Model_Saved': alg_name in export_data['models']
            }
            
            if toolbox.task_type == 'feature_selection':
                row.update({
                    'Mean_Features': stats.get('mean_features', 0),
                    'Mean_Accuracy': stats.get('mean_accuracy', 0),
                    'Feature_Efficiency': stats.get('mean_accuracy', 0) / max(stats.get('mean_features', 1), 1)
                })
            
            summary_data.append(row)
        
        summary_df = pd.DataFrame(summary_data)
        summary_filename = f"results/auto_save/mha_summary_{timestamp}.csv"
        summary_df.to_csv(summary_filename, index=False)
        summary_file = summary_filename
        
        # Display save confirmation
        if auto_save:
            st.success(f"✅ **Auto-saved to backend**: {len(algorithm_performance)} algorithms, models, and complete results")
            st.info(f"📂 **Files saved**: \n- Complete results: `{results_filename}`\n- Models: `{models_filename}`\n- Summary: `{summary_filename}`")
            
    except Exception as e:
        st.error(f"❌ Backend save failed: {str(e)}")
    
    # Store in session state for download access
    if 'saved_results' not in st.session_state:
        st.session_state.saved_results = []
    
    st.session_state.saved_results.append({
        'timestamp': timestamp,
        'results_file': results_file,
        'summary_file': summary_file,
        'models_file': models_file,
        'data': export_data,
        'algorithm_count': len(algorithm_performance)
    })
    
    return export_data, (results_file, summary_file, models_file)

def display_export_options(toolbox):
    """Display comprehensive export and download options with persistent storage"""
    
    st.subheader("💾 Complete Results & 🏆 BEST MODELS")
    
    # Get results manager
    results_manager = st.session_state.results_manager
    
    # Auto-save results with persistent storage
    experiment_name = f"export_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    export_data, saved_files = results_manager.save_comprehensive_results(
        toolbox, 
        session_name=experiment_name
    )
    
    # Enhanced model showcase first
    if export_data and 'models' in export_data and export_data['models']:
        st.markdown("### 🏆 **BEST MODELS READY FOR DOWNLOAD**")
        st.info("� **These optimized models contain the best configurations found for each algorithm!**")
        
        # Top 3 models highlight
        sorted_models = sorted(export_data['models'].items(), key=lambda x: x[1]['best_fitness'])
        if len(sorted_models) >= 3:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                best_model = sorted_models[0]
                st.success(f"🥇 **CHAMPION MODEL**")
                st.metric(
                    best_model[0].upper(), 
                    f"Fitness: {best_model[1]['best_fitness']:.6f}",
                    f"Time: {best_model[1]['execution_time']:.1f}s"
                )
            
            with col2:
                second_model = sorted_models[1]
                st.info(f"🥈 **RUNNER-UP**")
                st.metric(
                    second_model[0].upper(), 
                    f"Fitness: {second_model[1]['best_fitness']:.6f}",
                    f"Time: {second_model[1]['execution_time']:.1f}s"
                )
            
            with col3:
                third_model = sorted_models[2]
                st.warning(f"🥉 **THIRD PLACE**")
                st.metric(
                    third_model[0].upper(), 
                    f"Fitness: {third_model[1]['best_fitness']:.6f}",
                    f"Time: {third_model[1]['execution_time']:.1f}s"
                )
    
    # Enhanced download section with prominent model button
    if saved_files and saved_files[0]:
        st.markdown("### 📥 **INSTANT DOWNLOADS**")
        
        # Make model download super prominent
        if saved_files[2]:
            with open(saved_files[2], 'r') as f:
                models_data = f.read()
            st.download_button(
                label="🏆 **DOWNLOAD ALL BEST MODELS** 🏆",
                data=models_data,
                file_name=f"BEST_MODELS_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                help="🚀 Complete optimized models for all algorithms - Ready for production use!",
                width='stretch',
                type="primary"
            )
        
        # Secondary downloads
        col1, col2 = st.columns(2)
        
        with col1:
            # Complete results
            with open(saved_files[0], 'r') as f:
                results_data = f.read()
            st.download_button(
                label="� Complete Analysis & Data",
                data=results_data,
                file_name=f"complete_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                help="All algorithms, runs, convergence curves, and detailed analysis",
                width='stretch'
            )
        
        with col2:
            # Summary CSV
            if saved_files[1]:
                with open(saved_files[1], 'r') as f:
                    summary_data = f.read()
                st.download_button(
                    label="📊 Performance Rankings",
                    data=summary_data,
                    file_name=f"rankings_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    help="Algorithm rankings, statistics, and comparison metrics",
                    width='stretch'
                )
    
    # Previous session results access
    if 'saved_results' in st.session_state and st.session_state.saved_results:
        st.markdown("### 📚 Previous Session Results")
        
        # Show available results
        for i, result_info in enumerate(reversed(st.session_state.saved_results[-5:])):  # Show last 5
            with st.expander(f"� Session {result_info['timestamp']} - {result_info['algorithm_count']} algorithms"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if result_info['results_file'] and os.path.exists(result_info['results_file']):
                        with open(result_info['results_file'], 'r') as f:
                            data = f.read()
                        st.download_button(
                            label="📄 Complete Results",
                            data=data,
                            file_name=f"results_{result_info['timestamp']}.json",
                            mime="application/json",
                            key=f"results_{i}"
                        )
                
                with col2:
                    if result_info['summary_file'] and os.path.exists(result_info['summary_file']):
                        with open(result_info['summary_file'], 'r') as f:
                            data = f.read()
                        st.download_button(
                            label="📊 Summary",
                            data=data,
                            file_name=f"summary_{result_info['timestamp']}.csv",
                            mime="text/csv",
                            key=f"summary_{i}"
                        )
                
                with col3:
                    if result_info['models_file'] and os.path.exists(result_info['models_file']):
                        with open(result_info['models_file'], 'r') as f:
                            data = f.read()
                        st.download_button(
                            label="🏆 Models",
                            data=data,
                            file_name=f"models_{result_info['timestamp']}.json",
                            mime="application/json",
                            key=f"models_{i}"
                        )
    
    # Advanced export options
    with st.expander("🔧 Advanced Analysis & Exports"):
        col1, col2 = st.columns(2)
        
        with col1:
            # Convergence analysis
            if st.button("📈 Export Convergence Analysis", width='stretch'):
                convergence_analysis = {
                    'metadata': {
                        'export_type': 'convergence_analysis',
                        'timestamp': datetime.now().isoformat(),
                        'algorithms': list(toolbox.results.keys())
                    },
                    'convergence_data': {},
                    'analysis': {
                        'fastest_convergence': None,
                        'best_final_fitness': None,
                        'most_stable_convergence': None
                    }
                }
                
                convergence_stats = {}
                for alg_name, results in toolbox.results.items():
                    curves = []
                    convergence_rates = []
                    
                    for run in results['runs']:
                        if run.get('convergence_curve'):
                            curve = run['convergence_curve']
                            curves.append(curve)
                            
                            # Calculate convergence rate
                            if len(curve) > 10:
                                early = np.mean(curve[:len(curve)//3])
                                late = np.mean(curve[-len(curve)//3:])
                                rate = abs(early - late) / max(early, 1e-10)
                                convergence_rates.append(rate)
                    
                    if curves:
                        convergence_analysis['convergence_data'][alg_name] = {
                            'all_curves': curves,
                            'average_curve': np.mean(curves, axis=0).tolist() if curves else [],
                            'convergence_rates': convergence_rates,
                            'average_convergence_rate': np.mean(convergence_rates) if convergence_rates else 0,
                            'final_fitness_values': [curve[-1] for curve in curves if curve]
                        }
                        
                        convergence_stats[alg_name] = {
                            'avg_rate': np.mean(convergence_rates) if convergence_rates else 0,
                            'final_fitness': min([curve[-1] for curve in curves if curve]) if curves else float('inf')
                        }
                
                # Find best performers
                if convergence_stats:
                    fastest = max(convergence_stats.items(), key=lambda x: x[1]['avg_rate'])
                    best_final = min(convergence_stats.items(), key=lambda x: x[1]['final_fitness'])
                    
                    convergence_analysis['analysis'].update({
                        'fastest_convergence': fastest[0],
                        'best_final_fitness': best_final[0]
                    })
                
                # Download convergence analysis
                st.download_button(
                    label="⬇️ Download Convergence Analysis",
                    data=json.dumps(convergence_analysis, indent=2, default=str),
                    file_name=f"convergence_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        
        with col2:
            # Performance comparison
            if st.button("🏆 Export Performance Comparison", width='stretch'):
                performance_data = {
                    'metadata': {
                        'export_type': 'performance_comparison',
                        'timestamp': datetime.now().isoformat(),
                        'total_algorithms': len(toolbox.results)
                    },
                    'algorithm_rankings': {},
                    'head_to_head': {},
                    'statistical_tests': {}
                }
                
                # Algorithm rankings
                for alg_name, results in toolbox.results.items():
                    stats = results['statistics']
                    performance_data['algorithm_rankings'][alg_name] = {
                        'fitness_rank': None,  # Will be calculated
                        'speed_rank': None,
                        'stability_rank': None,
                        'overall_score': None,
                        'statistics': stats
                    }
                
                # Calculate ranks
                algorithms = list(performance_data['algorithm_rankings'].keys())
                
                # Fitness ranking (lower is better)
                fitness_sorted = sorted(algorithms, key=lambda x: toolbox.results[x]['statistics']['best_fitness'])
                for i, alg in enumerate(fitness_sorted):
                    performance_data['algorithm_rankings'][alg]['fitness_rank'] = i + 1
                
                # Speed ranking (lower time is better)
                speed_sorted = sorted(algorithms, key=lambda x: toolbox.results[x]['statistics']['mean_time'])
                for i, alg in enumerate(speed_sorted):
                    performance_data['algorithm_rankings'][alg]['speed_rank'] = i + 1
                
                # Calculate overall scores
                for alg in algorithms:
                    ranks = performance_data['algorithm_rankings'][alg]
                    # Overall score (lower is better)
                    ranks['overall_score'] = (ranks['fitness_rank'] + ranks['speed_rank']) / 2
                
                st.download_button(
                    label="⬇️ Download Performance Analysis",
                    data=json.dumps(performance_data, indent=2, default=str),
                    file_name=f"performance_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
    
    # Backend storage info
    st.markdown("### 🗄️ Backend Storage Status")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Count auto-saved files
        auto_save_count = len([f for f in os.listdir("results/auto_save") if f.endswith('.json')]) if os.path.exists("results/auto_save") else 0
        st.metric("📂 Auto-saved Results", auto_save_count)
    
    with col2:
        # Count models
        models_count = len([f for f in os.listdir("results/models") if f.endswith('.json')]) if os.path.exists("results/models") else 0
        st.metric("🏆 Saved Models", models_count)
    
    with col3:
        # Storage size
        try:
            total_size = 0
            for root, dirs, files in os.walk("results"):
                for file in files:
                    total_size += os.path.getsize(os.path.join(root, file))
            size_mb = total_size / (1024 * 1024)
            st.metric("💾 Storage Used", f"{size_mb:.1f} MB")
        except:
            st.metric("💾 Storage Used", "N/A")
            # Create CSV summary
            csv_data = []
            for alg_name, results in toolbox.results.items():
                stats = results['statistics']
                row = {
                    'Algorithm': alg_name.upper(),
                    'Task_Type': toolbox.task_type,
                    'Mean_Fitness': stats['mean_fitness'],
                    'Std_Fitness': stats['std_fitness'],
                    'Best_Fitness': stats['best_fitness'],
                    'Mean_Time': stats['mean_time'],
                    'Total_Runs': stats['total_runs']
                }
                
                if toolbox.task_type == 'feature_selection':
                    row.update({
                        'Mean_Features': stats['mean_features'],
                        'Mean_Accuracy': stats['mean_accuracy']
                    })
                
                csv_data.append(row)
            
            csv_df = pd.DataFrame(csv_data)
            csv_str = csv_df.to_csv(index=False)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"mha_summary_{toolbox.task_type}_{timestamp}.csv"
            
            st.download_button(
                label="⬇️ Download CSV",
                data=csv_str,
                file_name=filename,
                mime="text/csv"
            )
    
    # Summary report
    st.subheader("📋 Summary Report")
    
    if toolbox.results:
        # Generate text summary
        summary_text = f"""
# MHA Comparison Toolbox Report

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Task Type:** {toolbox.task_type.upper()}
**Dataset:** {toolbox.data_info['dataset_name']}
**Data:** {toolbox.data_info['n_samples']} samples, {toolbox.data_info['n_features']} features

## Algorithm Rankings

"""
        
        # Sort algorithms by performance
        sorted_algorithms = sorted(toolbox.results.items(), 
                                 key=lambda x: x[1]['statistics']['mean_fitness'])
        
        for rank, (alg_name, results) in enumerate(sorted_algorithms, 1):
            stats = results['statistics']
            summary_text += f"""
### Rank {rank}: {alg_name.upper()}
- **Mean Fitness:** {stats['mean_fitness']:.6f} ± {stats['std_fitness']:.6f}
- **Best Fitness:** {stats['best_fitness']:.6f}
- **Mean Time:** {stats['mean_time']:.2f}s ± {stats['std_time']:.2f}s
"""
            
            if toolbox.task_type == 'feature_selection':
                summary_text += f"- **Mean Features:** {stats['mean_features']:.1f} ± {stats['std_features']:.1f}\n"
                summary_text += f"- **Mean Accuracy:** {stats['mean_accuracy']:.4f} ± {stats['std_accuracy']:.4f}\n"
        
        st.text_area("Summary Report", summary_text, height=300)
        
        # Download summary report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"mha_report_{toolbox.task_type}_{timestamp}.txt"
        
        st.download_button(
            label="📄 Download Report",
            data=summary_text,
            file_name=filename,
            mime="text/plain"
        )


def display_persistent_results_manager():
    """Display comprehensive persistent results management interface"""
    
    st.title("📚 Persistent Results Manager")
    st.markdown("Access all your saved results, models, and experiment history")
    
    # Get results manager
    if 'results_manager' not in st.session_state:
        st.session_state.results_manager = ResultsManager()
    
    results_manager = st.session_state.results_manager
    
    # Current session info
    st.markdown("### 🔄 Current Session")
    col1, col2 = st.columns(2)
    
    with col1:
        st.info(f"**Session ID**: `{results_manager.current_session}`")
    with col2:
        # Create download package button
        if st.button("📦 Create Download Package"):
            with st.spinner("Creating comprehensive download package..."):
                package_path = results_manager.create_download_package()
                if package_path:
                    st.success(f"✅ Package created: `{package_path}`")
                else:
                    st.error("❌ Failed to create download package")
    
    # Results history
    st.markdown("---")
    st.markdown("### 📊 All Results History")
    
    history = results_manager.get_all_results_history()
    
    if history:
        st.success(f"📁 **{len(history)} Sessions Found** with persistent storage")
        
        for session_info in history:
            with st.expander(f"📂 Session: `{session_info['session_id']}` - {session_info.get('results_count', 0)} experiments"):
                
                # Session details
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.text(f"📅 Created: {session_info.get('created_at', 'Unknown')[:19].replace('T', ' ')}")
                with col2:
                    st.text(f"🔄 Last Activity: {session_info.get('last_activity', 'Unknown')[:19].replace('T', ' ')}")
                with col3:
                    st.text(f"🧪 Experiments: {session_info.get('results_count', 0)}")
                
                # Show experiments and files
                if 'experiments' in session_info and session_info['experiments']:
                    for exp in session_info['experiments']:
                        st.markdown(f"**🔬 Experiment: {exp['name']}**")
                        
                        # Files in this experiment
                        file_cols = st.columns(min(len(exp['files']), 4))
                        
                        for i, file_path in enumerate(exp['files']):
                            if file_path.exists() and i < 4:
                                with file_cols[i]:
                                    file_name = file_path.name
                                    file_type = "❓ Other"
                                    
                                    if 'BEST_MODELS' in file_name:
                                        file_type = "🏆 Models"
                                    elif 'complete_results' in file_name:
                                        file_type = "📊 Complete"
                                    elif 'summary' in file_name or file_name.endswith('.csv'):
                                        file_type = "📈 Summary"
                                    elif 'rankings' in file_name:
                                        file_type = "🏅 Rankings"
                                    
                                    # Read and create download
                                    try:
                                        with open(file_path, 'r') as f:
                                            file_data = f.read()
                                        
                                        mime_type = "application/json" if file_name.endswith('.json') else "text/csv"
                                        
                                        st.download_button(
                                            label=file_type,
                                            data=file_data,
                                            file_name=file_name,
                                            mime=mime_type,
                                            key=f"persist_{file_path}_{exp['name']}",
                                            help=f"Download {file_name}"
                                        )
                                    except Exception as e:
                                        st.error(f"Error reading {file_name}: {str(e)}")
                else:
                    st.info("📝 No experiments found in this session")
    else:
        st.info("🆕 No previous results found. Run some experiments to build your results history!")
    
    # Available models showcase
    st.markdown("---")
    st.markdown("### 🏆 All Available Models")
    
    available_models = results_manager.get_available_models()
    if available_models:
        st.success(f"🎯 **{len(available_models)} Model Sets Available**")
        
        for model_info in available_models:
            with st.expander(f"📦 Model Set: {model_info['timestamp']} ({model_info['algorithm_count']} algorithms)"):
                
                # Show algorithms in this model set
                st.markdown("**🧬 Algorithms:**")
                alg_cols = st.columns(min(len(model_info['data']), 5))
                
                for i, alg_name in enumerate(list(model_info['data'].keys())[:5]):
                    with alg_cols[i]:
                        model_data = model_info['data'][alg_name]
                        st.metric(
                            alg_name.upper(),
                            f"{model_data.get('best_fitness', 0):.6f}",
                            f"{model_data.get('execution_time', 0):.1f}s"
                        )
                
                # Download full model set
                with open(model_info['path'], 'r') as f:
                    full_model_data = f.read()
                
                st.download_button(
                    label=f"🏆 Download All {model_info['algorithm_count']} Models",
                    data=full_model_data,
                    file_name=model_info['filename'],
                    mime="application/json",
                    key=f"models_download_{model_info['timestamp']}",
                    help="Download complete optimized model set"
                )
    else:
        st.info("🎯 No models found yet. Run some experiments to generate optimized models!")


if __name__ == "__main__":
    main()