"""
MHA Toolbox - Complete Web Interface
=====================================
Professional, intuitive interface for meta-heuristic optimization
with multi-user support and comprehensive documentation.

Launch:
    python mha_ui_complete.py
    or
    streamlit run mha_ui_complete.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from pathlib import Path
import time
import uuid
import platform
import hashlib

# Import MHA Toolbox
from mha_toolbox import MHAToolbox
from mha_toolbox.user_profile_optimized import (
    create_session_profile,
    create_profile,
    save_profile, 
    load_profile,
    list_profiles,
    cleanup_expired_sessions
)
from mha_toolbox.algorithm_categories import ALGORITHM_CATEGORIES

# Page configuration
st.set_page_config(
    page_title="MHA Toolbox - Optimization Suite",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional design
st.markdown("""
<style>
    /* Main styling */
    .main-header {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    
    /* Metric cards - Purple gradient with WHITE text */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        padding: 2rem;
        border-radius: 15px;
        color: white !important;
        text-align: center;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
    }
    
    .metric-card * {
        color: white !important;
    }
    
    .metric-number {
        font-size: 3rem;
        font-weight: 800;
        margin: 0;
        color: white !important;
    }
    
    .metric-label {
        font-size: 1.1rem;
        opacity: 0.9;
        margin-top: 0.5rem;
        color: white !important;
    }
    
    /* Info boxes - DARK text for visibility */
    .info-box {
        background: #e3f2fd;
        border-left: 5px solid #2196f3;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
        color: #0c5460 !important;
    }
    
    .info-box h1, .info-box h2, .info-box h3, .info-box h4, .info-box h5, .info-box h6,
    .info-box p, .info-box span, .info-box li {
        color: #0c5460 !important;
    }
    
    .success-box {
        background: #e8f5e9;
        border-left: 5px solid #4caf50;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
        color: #155724 !important;
    }
    
    .success-box h1, .success-box h2, .success-box h3, .success-box h4, .success-box h5, .success-box h6,
    .success-box p, .success-box span, .success-box li {
        color: #155724 !important;
    }
    
    .warning-box {
        background: #fff3e0;
        border-left: 5px solid #ff9800;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
        color: #856404 !important;
    }
    
    .warning-box h1, .warning-box h2, .warning-box h3, .warning-box h4, .warning-box h5, .warning-box h6,
    .warning-box p, .warning-box span, .warning-box li {
        color: #856404 !important;
    }
    
    .disclaimer-box {
        background: #ffebee;
        border: 2px solid #f44336;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 2rem 0;
        color: #721c24 !important;
    }
    
    .disclaimer-box h1, .disclaimer-box h2, .disclaimer-box h3, .disclaimer-box h4, .disclaimer-box h5, .disclaimer-box h6,
    .disclaimer-box p, .disclaimer-box span, .disclaimer-box li, .disclaimer-box ul {
        color: #721c24 !important;
    }
    
    /* Algorithm card - WHITE background with DARK text */
    .algo-card {
        background: white !important;
        border: 2px solid #e0e0e0;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        transition: all 0.3s ease;
        color: #1e1e1e !important;
    }
    
    .algo-card:hover {
        border-color: #667eea;
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.2);
    }
    
    .algo-card * {
        color: #1e1e1e !important;
    }
    
    /* Buttons */
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3.5em;
        font-weight: 600;
        font-size: 1.05rem;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    
    /* User profile badge - WHITE text on purple */
    .user-badge {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        padding: 0.5rem 1.5rem;
        border-radius: 25px;
        display: inline-block;
        font-weight: 600;
        margin: 0.5rem 0;
    }
    
    /* Step indicator - DARK text on light background */
    .step-indicator {
        background: #f5f5f5;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
        color: #1e1e1e !important;
    }
    
    .step-indicator h1, .step-indicator h2, .step-indicator h3, .step-indicator h4,
    .step-indicator h5, .step-indicator h6, .step-indicator p, .step-indicator span {
        color: #1e1e1e !important;
    }
    
    /* CRITICAL: Main content area - force dark text on white/light sections */
    .main p, .main li, .main span, .main div:not(.metric-card):not(.user-badge) {
        color: #1e1e1e !important;
    }
    
    .main h1, .main h2, .main h3, .main h4, .main h5, .main h6 {
        color: #1e1e1e !important;
    }
    
    /* Override for ALL elements in main except special cards */
    .main .stMarkdown:not(:has(.metric-card)):not(:has(.user-badge)) * {
        color: #1e1e1e !important;
    }
    
    /* Column content - dark text */
    div[data-testid="column"]:not(:has(.metric-card)) {
        color: #1e1e1e !important;
    }
    
    div[data-testid="column"]:not(:has(.metric-card)) * {
        color: #1e1e1e !important;
    }
    
    /* Block container */
    .block-container:not(:has(.metric-card)):not(:has(.user-badge)) {
        color: #1e1e1e !important;
    }
    
    /* Streamlit expanders */
    div[data-testid="stExpander"] {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        color: #1e1e1e !important;
    }
    
    div[data-testid="stExpander"] * {
        color: #1e1e1e !important;
    }
    
    /* Form elements */
    input, textarea, select {
        background-color: #ffffff !important;
        color: #1e1e1e !important;
        border: 1px solid #ced4da !important;
    }
    
    /* CRITICAL: Labels and text must be visible */
    label, .stNumberInput label, .stSlider label, .stSelectbox label {
        color: #ffffff !important;
        font-weight: 600 !important;
    }
    
    /* Number input labels */
    div[data-testid="stNumberInput"] label {
        color: #ffffff !important;
    }
    
    /* Slider label */
    div[data-testid="stSlider"] label {
        color: #ffffff !important;
    }
    
    /* Help text */
    .stTextInput small, .stNumberInput small, .stSlider small {
        color: #cccccc !important;
    }
    
    /* Tables */
    .main table {
        color: #1e1e1e !important;
    }
    
    .main table th, .main table td {
        color: #1e1e1e !important;
    }
    
    /* Sidebar - keep dark */
    section[data-testid="stSidebar"] {
        background-color: #262730;
    }
    
    /* Code blocks */
    code {
        background-color: #f8f9fa;
        color: #d63384;
        padding: 0.2rem 0.4rem;
        border-radius: 4px;
    }
    
    pre {
        background-color: #f8f9fa;
        color: #1e1e1e;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    """Initialize all session state variables"""
    
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        st.session_state.current_user = None
        st.session_state.user_profile = None
        st.session_state.session_id = str(uuid.uuid4())
        
        # Initialize toolbox with ALL 130+ algorithms
        with st.spinner('🔍 Discovering all algorithms...'):
            st.session_state.toolbox = MHAToolbox(verbose=False)
            total_algos = len(st.session_state.toolbox.list_algorithms())
            hybrid_algos = len(st.session_state.toolbox.registry.list_hybrid_algorithms())
            st.session_state.total_algorithms = total_algos
            st.session_state.hybrid_algorithms = hybrid_algos
            st.session_state.categorized_algorithms = st.session_state.toolbox.list_algorithms_by_category()
        
        st.session_state.current_data = None
        st.session_state.optimization_results = None
        st.session_state.current_page = "🏠 Home"
        st.session_state.show_disclaimer = True
        st.session_state.disclaimer_accepted = False
        st.session_state.user_authenticated = False
        st.session_state.user_password = None
        
        # Clean up expired sessions on startup
        cleanup_expired_sessions()

init_session_state()


def show_user_switcher():
    """Show user profile switcher in sidebar with authentication"""
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 👤 User Profile")
    
    # Current user display
    if st.session_state.user_authenticated and st.session_state.current_user:
        current_user = st.session_state.current_user
        st.sidebar.markdown(f'<div class="user-badge">👤 {current_user}</div>', unsafe_allow_html=True)
        
        # Logout button
        if st.sidebar.button("🚪 Logout", key="logout_btn"):
            # Clear all authentication-related session state
            st.session_state.user_authenticated = False
            st.session_state.current_user = None
            st.session_state.user_profile = None
            st.session_state.user_password = None
            if 'auth_cache' in st.session_state:
                st.session_state.auth_cache.clear()
            # Clear any cached keys
            if 'login_password' in st.session_state:
                del st.session_state['login_password']
            if 'login_select' in st.session_state:
                del st.session_state['login_select']
            st.success("✅ Logged out successfully")
            st.rerun()
    else:
        st.sidebar.info("👤 Please login to continue")
    
    # Login/Switch User
    with st.sidebar.expander("� Login / Switch User", expanded=not st.session_state.user_authenticated):
        # Get all profiles
        profiles = list_profiles()
        profile_names = [p['username'] for p in profiles] if profiles else []
        
        # Tabs for login and new user
        tab1, tab2 = st.tabs(["🔑 Login", "🆕 New User"])
        
        with tab1:
            if profile_names:
                selected_user = st.selectbox("Select User", [""] + profile_names, key="login_select")
                password = st.text_input("Password", type="password", key="login_password")
                
                if st.button("🔓 Login", key="login_btn", width='stretch'):
                    if selected_user and password:
                        # Authenticate user
                        if authenticate_user(selected_user, password):
                            st.session_state.current_user = selected_user
                            st.session_state.user_authenticated = True
                            st.session_state.session_id = str(uuid.uuid4())
                            st.session_state.user_password = password
                            
                            # Load user profile (creates session profile with persistent preferences)
                            st.session_state.user_profile = create_session_profile(
                                selected_user,
                                session_id=st.session_state.session_id,
                                system_id=platform.node()
                            )
                            st.session_state.user_profile.update_preference('last_system', platform.node())
                            st.session_state.user_profile.total_sessions += 1
                            save_profile(st.session_state.user_profile)
                            
                            st.success(f"✅ Logged in as {selected_user}")
                            st.info(f"📊 Profile loaded: {st.session_state.user_profile.total_experiments} experiments completed")
                            st.rerun()
                        else:
                            st.error("❌ Invalid password! Please check your password and try again.")
                            st.info("💡 Tip: Passwords are case-sensitive. Make sure Caps Lock is off.")
                    else:
                        st.warning("⚠️ Please select a user and enter password")
            else:
                st.info("No users yet. Create a new user!")
        
        with tab2:
            new_username = st.text_input("Username", key="new_user_name")
            new_password = st.text_input("Password", type="password", key="new_user_password")
            confirm_password = st.text_input("Confirm Password", type="password", key="new_user_confirm")
            
            if st.button("✅ Create User", key="create_user_btn", width='stretch'):
                if new_username and new_password and confirm_password:
                    if new_password == confirm_password:
                        if new_username not in profile_names:
                            # Hash password first
                            import hashlib
                            hashed_pw = hashlib.sha256(new_password.encode()).hexdigest()
                            
                            # STEP 1: Create persistent profile FIRST (so password is saved permanently)
                            persistent_profile = create_profile(new_username, platform.node(), session_id=None)
                            persistent_profile.update_preference('password_hash', hashed_pw)
                            persistent_profile.update_preference('created_system', platform.node())
                            persistent_profile.update_preference('created_at', datetime.now().isoformat())
                            persistent_profile.total_experiments = 0
                            persistent_profile.total_sessions = 1
                            save_profile(persistent_profile)
                            
                            # STEP 2: Create session profile (will load password from persistent profile)
                            st.session_state.current_user = new_username
                            st.session_state.user_authenticated = True
                            st.session_state.session_id = str(uuid.uuid4())
                            st.session_state.user_password = new_password
                            
                            st.session_state.user_profile = create_session_profile(
                                new_username,
                                session_id=st.session_state.session_id,
                                system_id=platform.node()
                            )
                            
                            # Session profile should now have the password from persistent profile
                            # But ensure it's set in case of any loading issues
                            if 'password_hash' not in st.session_state.user_profile.preferences:
                                st.session_state.user_profile.update_preference('password_hash', hashed_pw)
                            
                            st.session_state.user_profile.update_preference('last_system', platform.node())
                            save_profile(st.session_state.user_profile)
                            
                            st.success(f"✅ User created and saved: {new_username}")
                            st.info("💾 Your profile has been saved permanently. You can now login anytime!")
                            st.rerun()
                        else:
                            st.error("❌ Username already exists!")
                    else:
                        st.error("❌ Passwords don't match!")
                else:
                    st.warning("⚠️ Please fill all fields")
    
    # User stats (only if authenticated)
    if st.session_state.user_authenticated and st.session_state.user_profile:
        st.sidebar.markdown("---")
        st.sidebar.markdown("**📊 Session Info:**")
        
        profile = st.session_state.user_profile
        current_system = platform.node()
        last_system = profile.preferences.get('last_system', 'Unknown')
        
        # System switch detection
        if last_system != current_system:
            st.sidebar.warning(f"⚠️ System changed!\nFrom: {last_system}\nTo: {current_system}")
        
        st.sidebar.info(f"""
        **Experiments:** {profile.total_experiments}
        **Sessions:** {profile.total_sessions}
        **System:** `{current_system}`
        **Session:** `{st.session_state.session_id[:8]}...`
        """)


def authenticate_user(username, password):
    """Authenticate user with password - FIXED VERSION"""
    try:
        import hashlib
        import platform
        
        # Clear any cached data to prevent stale state
        if 'auth_cache' not in st.session_state:
            st.session_state.auth_cache = {}
        
        # Try loading persistent profile first (without session_id)
        profile = None
        
        # Try from current system
        try:
            profile = load_profile(username, system_id=platform.node(), session_id=None)
        except:
            pass
        
        if not profile:
            # Try loading from any system (for cross-system compatibility)
            all_profiles = list_profiles()
            matching = [p for p in all_profiles if p['username'] == username]
            if matching:
                # Load using the stored system_id
                try:
                    profile = load_profile(username, 
                                         system_id=matching[0].get('system_id', platform.node()), 
                                         session_id=None)
                except:
                    pass
        
        if profile:
            stored_hash = profile.preferences.get('password_hash', '')
            
            if not stored_hash:
                # No password set - this shouldn't happen for valid users
                st.warning(f"⚠️ Profile found but no password set for '{username}'. Please create a new account.")
                return False
            
            # Hash the input password
            input_hash = hashlib.sha256(password.encode()).hexdigest()
            
            # Compare hashes
            is_valid = (stored_hash == input_hash)
            
            if is_valid:
                # Cache successful authentication
                st.session_state.auth_cache[username] = True
            
            return is_valid
        else:
            st.error(f"❌ User '{username}' not found. Please check username or create a new account.")
        
        return False
        
    except Exception as e:
        st.error(f"Authentication error: {e}")
        import traceback
        traceback.print_exc()
        return False


def convert_to_json_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    import numpy as np
    
    if isinstance(obj, (np.int32, np.int64, np.int_)):
        return int(obj)
    elif isinstance(obj, (np.float32, np.float64, np.float_)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    else:
        return obj


def save_to_user_history(results, total_time, algorithms, n_runs, task_type):
    """Save optimization results to user's history"""
    try:
        if not st.session_state.user_profile:
            return
        
        from datetime import datetime
        import json
        
        # Create history entry with JSON-safe values
        history_entry = {
            'timestamp': datetime.now().isoformat(),
            'algorithms': algorithms,
            'n_runs': int(n_runs),
            'task_type': task_type,
            'total_time': float(total_time),
            'dataset': st.session_state.current_data.get('name', 'Unknown') if st.session_state.current_data else 'Custom',
            'best_algorithm': min(results.items(), key=lambda x: x[1].get('best_fitness', 1.0))[0] if results else None,
            'best_fitness': float(min(results.values(), key=lambda x: x.get('best_fitness', 1.0)).get('best_fitness', 1.0)) if results else 1.0,
            'algorithms_tested': int(len(algorithms)),
            'results_summary': {
                algo: {
                    'best_fitness': float(res.get('best_fitness', 1.0)),
                    'mean_fitness': float(res.get('mean_fitness', 1.0)),
                    'execution_time': float(res.get('execution_time', 0)),
                    'n_features_selected': int(res.get('n_features_selected', 0))
                }
                for algo, res in results.items() if 'error' not in res
            }
        }
        
        # Convert all values to JSON-serializable types
        history_entry = convert_to_json_serializable(history_entry)
        
        # Save to profile's history
        if 'optimization_history' not in st.session_state.user_profile.preferences:
            st.session_state.user_profile.preferences['optimization_history'] = []
        
        # Limit history to last 100 entries
        history = st.session_state.user_profile.preferences['optimization_history']
        history.append(history_entry)
        if len(history) > 100:
            history = history[-100:]  # Keep only last 100
        
        st.session_state.user_profile.preferences['optimization_history'] = history
        
        # Update profile statistics
        st.session_state.user_profile.increment_experiments()
        st.session_state.user_profile.update_preference('last_optimization', datetime.now().isoformat())
        st.session_state.user_profile.update_preference('total_algorithms_tested', 
            st.session_state.user_profile.preferences.get('total_algorithms_tested', 0) + len(algorithms))
        
        # Save profile
        save_profile(st.session_state.user_profile)
        
    except Exception as e:
        print(f"Error saving to user history: {e}")
        import traceback
        traceback.print_exc()


def show_disclaimer():
    """Show disclaimer modal on first visit - must be accepted to continue"""
    if not st.session_state.disclaimer_accepted:
        # Full screen overlay for disclaimer
        st.markdown('<h1 class="main-header">🧬 MHA Toolbox</h1>', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">Meta-Heuristic Algorithm Optimization Suite</p>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="disclaimer-box">
            <h2 style="color: #f44336; margin-top: 0;">⚠️ IMPORTANT DISCLAIMER & USER GUIDE</h2>
            <p style="font-size: 1.15rem; line-height: 1.8; font-weight: 600;">
                <strong>⚠️ PLEASE READ THIS CAREFULLY BEFORE PROCEEDING</strong>
            </p>
            <hr>
            
            <h3>🚀 Quick Start Guide - How to Use This System</h3>
            <div style="background: #e3f2fd; padding: 1.5rem; border-radius: 8px; margin-bottom: 1.5rem;">
                <ol style="font-size: 1.05rem; line-height: 2.0; margin: 0;">
                    <li><strong>Create Account:</strong> Register with a unique username and password (first-time users)</li>
                    <li><strong>Login:</strong> Sign in with your credentials to access your personal workspace</li>
                    <li><strong>Upload Dataset:</strong> Go to "🏠 Home" → Upload CSV/Excel file with your data</li>
                    <li><strong>Select Target:</strong> Choose target column and features for optimization</li>
                    <li><strong>Choose Algorithm:</strong> Pick from 130+ algorithms (PSO, GA, GWO, WOA, or hybrids like AMSHA)</li>
                    <li><strong>Configure Parameters:</strong> Set population size (20-100), iterations (50-500), threshold (0.0-1.0)</li>
                    <li><strong>Run Optimization:</strong> Click "Run Optimization" and wait for results</li>
                    <li><strong>View Results:</strong> See fitness values, selected features, and visualizations</li>
                    <li><strong>Export Data:</strong> Download results in CSV, Excel, JSON, or NPZ format</li>
                    <li><strong>Compare Algorithms:</strong> Use "🔬 Compare" tab to run multiple algorithms</li>
                </ol>
            </div>
            
            <h3>💡 Key Features Available</h3>
            <ul style="font-size: 1.05rem; line-height: 1.8;">
                <li><strong>130+ Algorithms:</strong> Swarm, Evolutionary, Bio-inspired, Physics-based, and 26 Hybrid combinations</li>
                <li><strong>Feature Selection:</strong> Automatically select best features for ML models</li>
                <li><strong>Real-time Threshold:</strong> Interactive slider (0.0-1.0) to control feature selection sensitivity</li>
                <li><strong>15+ Visualizations:</strong> Convergence plots, box plots, heatmaps, feature importance charts</li>
                <li><strong>Session Management:</strong> Save/load your work, compare runs, track history</li>
                <li><strong>Export Options:</strong> Multiple formats for integration with other tools</li>
            </ul>
            
            <h3>📋 Terms of Use</h3>
            <ul style="font-size: 1.05rem; line-height: 1.8;">
                <li><strong>AS-IS Software:</strong> Provided without warranty of any kind, express or implied</li>
                <li><strong>No Liability:</strong> Authors not liable for damages, data loss, or incorrect results</li>
                <li><strong>Results Validation:</strong> Always validate results with domain expertise before use</li>
                <li><strong>Research Purpose:</strong> Designed for research, education, and experimentation</li>
                <li><strong>Production Use:</strong> Conduct thorough testing before using in production</li>
                <li><strong>Data Privacy:</strong> All data stored locally in <code>persistent_state/</code> directory</li>
            </ul>
            
            <h3>👥 Multi-User Environment</h3>
            <ul style="font-size: 1.05rem; line-height: 1.8;">
                <li><strong>User Authentication:</strong> Create unique account - each user has isolated workspace</li>
                <li><strong>Session Isolation:</strong> Your data, results, and settings are private</li>
                <li><strong>System Tracking:</strong> Login attempts and system changes tracked for security</li>
                <li><strong>Profile Protection:</strong> Cannot access or modify other users' data</li>
                <li><strong>Auto Cleanup:</strong> Inactive sessions cleaned after 24 hours</li>
            </ul>
            
            <h3>🔒 Security & Best Practices</h3>
            <ul style="font-size: 1.05rem; line-height: 1.8;">
                <li><strong>Strong Password:</strong> Use combination of letters, numbers, symbols (min 6 characters)</li>
                <li><strong>No Sharing:</strong> Keep your credentials private and secure</li>
                <li><strong>Local Storage:</strong> Passwords hashed with bcrypt (industry standard)</li>
                <li><strong>Data Backup:</strong> Regularly export important results for backup</li>
                <li><strong>File Size:</strong> Recommended max 10MB for optimal performance</li>
                <li><strong>Browser Cache:</strong> Clear if experiencing login issues</li>
            </ul>
            
            <h3>⚠️ Important Reminders</h3>
            <ul style="font-size: 1.05rem; line-height: 1.8;">
                <li><strong>Algorithm Selection:</strong> Different algorithms work better for different problems</li>
                <li><strong>Computation Time:</strong> Larger datasets and more iterations take longer</li>
                <li><strong>Threshold Impact:</strong> Lower threshold = more features, higher threshold = fewer features</li>
                <li><strong>Result Interpretation:</strong> Lower fitness value = better optimization (for minimization)</li>
                <li><strong>Multiple Runs:</strong> Run same algorithm multiple times for statistical reliability</li>
            </ul>
            
            <p style="margin-top: 2rem; font-size: 1.1rem; font-weight: 600; color: #f44336;">
                ⚠️ By accepting, you acknowledge: (1) You understand how to use the system, (2) You accept all terms and conditions, 
                (3) You will validate results before making decisions, (4) You understand this is research/educational software.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Acceptance checkbox
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            accept_check = st.checkbox("✅ I have read and accept all terms and conditions", key="accept_disclaimer_check")
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            if st.button("✅ I Accept - Continue to Application", 
                        type="primary", 
                        width='stretch',
                        disabled=not accept_check):
                st.session_state.disclaimer_accepted = True
                st.session_state.show_disclaimer = False
                st.success("✅ Disclaimer accepted! Redirecting...")
                time.sleep(1)
                st.rerun()
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            if st.button("📖 Learn More About System", width='stretch'):
                st.session_state.disclaimer_accepted = True
                st.session_state.show_disclaimer = False
                st.session_state.current_page = "📖 About"
                st.rerun()
        
        return False
    return True


def main():
    """Main application"""
    
    # Show disclaimer first (blocks everything)
    if not show_disclaimer():
        return
    
    # Require authentication after disclaimer
    if not st.session_state.user_authenticated:
        st.markdown('<h1 class="main-header">🧬 MHA Toolbox</h1>', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">Meta-Heuristic Algorithm Optimization Suite</p>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
            <h3 style="margin-top:0;">🔐 Authentication Required</h3>
            <p style="font-size: 1.1rem;">
                Please login or create a new user account in the sidebar to continue.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Show user switcher in sidebar
        with st.sidebar:
            st.markdown("## 🔐 Access Control")
            show_user_switcher()
        
        return
    
    # Header
    st.markdown('<h1 class="main-header">🧬 MHA Toolbox</h1>', unsafe_allow_html=True)
    st.markdown(f'<p class="sub-header">Meta-Heuristic Algorithm Optimization Suite - {st.session_state.total_algorithms} Algorithms</p>', unsafe_allow_html=True)
    
    # Sidebar navigation
    with st.sidebar:
        st.markdown("## 📋 Navigation")
        
        pages = [
            "🏠 Home",
            "🚀 New Optimization",
            "� History",
            "�📊 Results",
            "📖 About",
            "⚙️ Settings"
        ]
        
        st.session_state.current_page = st.radio(
            "Go to",
            pages,
            index=pages.index(st.session_state.current_page) if st.session_state.current_page in pages else 0,
            label_visibility="collapsed"
        )
        
        # User switcher
        show_user_switcher()
        
        # Quick info
        st.markdown("---")
        st.markdown("### 📚 Quick Info")
        st.info(f"""
        **{st.session_state.total_algorithms} Algorithms**
        
        - Swarm Intelligence
        - Evolutionary
        - Physics-based
        - Hybrid Algorithms ⭐
        
        **Features:**
        - Feature Selection
        - Real-time Tracking
        - Multi-user Support
        - Export Results
        """)
    
    # Route to pages
    if st.session_state.current_page == "🏠 Home":
        show_home()
    elif st.session_state.current_page == "🚀 New Optimization":
        show_optimization()
    elif st.session_state.current_page == "� History":
        show_history()
    elif st.session_state.current_page == "�📊 Results":
        show_results()
    elif st.session_state.current_page == "📖 About":
        show_about()
    elif st.session_state.current_page == "⚙️ Settings":
        show_settings()


def show_home():
    """Home page with overview"""
    st.markdown("## 🏠 Welcome!")
    
    # Get dynamic algorithm count from session state
    total_algos = st.session_state.get('total_algorithms', 130)
    hybrid_algos = st.session_state.get('hybrid_algorithms', 22)
    
    # Metric cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-number">{total_algos}</div>
            <div class="metric-label">Total Algorithms</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-number">{hybrid_algos}</div>
            <div class="metric-label">Hybrid Algorithms</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-number">Secure</div>
            <div class="metric-label">Multi-User</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        # Get category count
        categories = st.session_state.get('categorized_algorithms', {})
        cat_count = len([c for c in categories.values() if c])
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-number">{cat_count}</div>
            <div class="metric-label">Categories</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # System Flow
    st.markdown("### 🔄 System Flow")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="step-indicator">
            <h4>📤 Step 1: Upload Data</h4>
            <p>Upload CSV, use sample datasets, or generate random data</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="step-indicator">
            <h4>🎯 Step 2: Select Algorithm</h4>
            <p>Choose from 40+ algorithms across multiple categories</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="step-indicator">
            <h4>⚙️ Step 3: Configure</h4>
            <p>Set iterations, population size, and features to select</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="step-indicator">
            <h4>🚀 Step 4: Run & Export</h4>
            <p>View real-time results and export to CSV/Excel</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Quick actions
    st.markdown("### ⚡ Quick Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🚀 Start New Optimization", type="primary", width='stretch'):
            st.session_state.current_page = "🚀 New Optimization"
            st.rerun()
    
    with col2:
        if st.button("📊 View Results", width='stretch'):
            st.session_state.current_page = "📊 Results"
            st.rerun()
    
    with col3:
        if st.button("📖 Learn More", width='stretch'):
            st.session_state.current_page = "📖 About"
            st.rerun()
    
    st.markdown("---")
    
    # Comprehensive User Guide with TABS instead of expanders
    st.markdown("### 📚 Complete User Guide & System Flow")
    
    guide_tabs = st.tabs([
        "🎯 Understanding MHA",
        "🔄 Workflow",
        "📊 Visualizations",
        "⚙️ Algorithms",
        "💡 Tips",
        "🔧 Troubleshooting"
    ])
    
    # TAB 1: Understanding MHA Output
    with guide_tabs[0]:
        st.markdown("""
        ### 🔍 What is the MHA Output?
        
        **Meta-Heuristic Algorithms (MHA)** optimize problems by finding the best solution. The output consists of:
        
        #### 1️⃣ Best Fitness Score (Lower is Better)
        - **Range**: 0.0 to 1.0
        - **0.0 = Perfect** (100% accuracy, no errors)
        - **1.0 = Worst** (0% accuracy, all errors)
        - **Typical good results**: 0.1 to 0.3 (70-90% accuracy)
        
        #### 2️⃣ Best Solution (Search Agent Position)
        - Each dimension represents a **feature** in your dataset
        - **Position value**: 0.0 (least favored) to 1.0 (most important)
        - **Example**: `[0.85, 0.23, 0.91, 0.12, ...]` means:
          - Feature 1: Highly important (0.85) ✅
          - Feature 2: Less important (0.23) ⚠️
          - Feature 3: Very important (0.91) ✅
          - Feature 4: Not important (0.12) ❌
        
        #### 3️⃣ Interactive Feature Selection with Threshold
        
        **How the Threshold Works:**
        
        | Threshold | Meaning | Use Case |
        |-----------|---------|----------|
        | **0.5** | Standard selection | Default - significantly important features |
        | **0.75** | High selectivity | Only most critical features |
        | **0.25** | Low selectivity | Include sometimes-useful features |
        | **0.0** | All features | Complete ranking for analysis |
        
        **Color Coding:**
        - 🟢 **Green bars** = Selected (≥ threshold)
        - ⚪ **Gray bars** = Not selected (< threshold)
        
        **Example Interpretation:**
        ```
        Feature_1: 0.92 ✅ Selected at all thresholds
        Feature_2: 0.67 ✅ Selected at 0.5, not at 0.75
        Feature_3: 0.48 ❌ Not selected at 0.5
        Feature_4: 0.89 ✅ Selected at all thresholds
        ```
        
        #### 4️⃣ Multiple Runs & Statistics
        - **3 Runs Per Algorithm** (default): Ensures reliability
        - **Statistics Calculated**:
          - **Best Fitness**: Lowest (best) fitness achieved
          - **Mean Fitness**: Average across all runs
          - **Standard Deviation**: Consistency measure (lower = more reliable)
          - **Execution Time**: Time taken for optimization
        
        #### 5️⃣ Saving & Downloading Results
        
        **Automatically Saved:**
        - All results: `persistent_state/results/`
        - Sessions: `persistent_state/sessions/`
        - User profiles: `persistent_state/users/`
        
        **Export Options:**
        - **CSV**: Best for data analysis (Excel, Python, R)
        - **Excel**: Formatted with multiple sheets
        - **JSON**: For programmatic access
        - **NPZ**: For NumPy/scientific computing
        """)
    
    # TAB 2: Step-by-Step Workflow
    with guide_tabs[1]:
        st.markdown("""
        ### 📋 Complete System Workflow
        
        #### 🔐 Step 1: Authentication
        1. Read and accept disclaimer (first time only)
        2. Login with username and password
        3. Or create new account
        4. Your data is isolated from other users
        
        #### 📤 Step 2: Upload Data
        
        **Option A: Upload CSV**
        - CSV must have feature columns + 1 target column
        - Target column should be last column
        - Example format: `feature1,feature2,...,target`
        
        **Option B: Use Sample Dataset**
        - **Iris**: 4 features, 150 samples, 3 classes
        - **Wine**: 13 features, 178 samples, 3 classes
        - **Breast Cancer**: 30 features, 569 samples, 2 classes
        
        **Option C: Generate Random Data**
        - Configure features, samples, classes
        - Useful for testing and experimentation
        
        #### 🤖 Step 3: Algorithm Selection
        
        **Automatic Recommendation** (Recommended):
        - System analyzes your dataset
        - Recommends top 10 algorithms with confidence scores (5-10)
        - **Auto-selection**: Top 3 are pre-selected ✅
        
        **Manual Selection**:
        - Browse by category (Swarm, Evolutionary, Physics, etc.)
        - Select up to 10 algorithms
        - Mix different algorithm types
        
        #### ⚙️ Step 4: Configure Optimization
        
        **Task Type**:
        - **Feature Selection**: Binary selection (select/reject) - Most common
        - **Feature Optimization**: Weighted feature importance
        - **Hyperparameter Tuning**: Optimize model parameters
        
        **Parameters**:
        
        | Parameter | Quick | Standard | Deep |
        |-----------|-------|----------|------|
        | **Iterations** | 50-100 | 100-200 | 300-500 |
        | **Population** | 20-30 | 30-50 | 50-100 |
        | **Runs** | 1 | 3 | 5-10 |
        
        #### 🚀 Step 5: Run Optimization
        - Click "Start Optimization" button
        - Watch real-time progress bar
        - See fitness values for each algorithm
        - Wait for completion message
        
        #### 📊 Step 6: Analyze Results (5 Tabs)
        
        1. **� Summary**: Overall comparison table & rankings
        2. **🎯 Feature Analysis**: Interactive threshold slider
        3. **⚖️ Comparative**: 9 different comparison charts
        4. **� Convergence**: Iteration-by-iteration curves
        5. **💾 Export**: Download in multiple formats
        
        #### � Step 7: Export & Use
        - Download summary CSV
        - Download selected features
        - Import into your ML pipeline
        - Test in actual models
        """)
    
    # TAB 3: Visualization Guide
    with guide_tabs[2]:
        st.markdown("""
        ### 📊 Understanding All Visualizations
        
        #### 1️⃣ Summary Table
        - **Columns**: Algorithm | Best Fitness | Mean | SD | Time | Features
        - **Sorting**: By best fitness (ascending)
        - **Highlights**: Best performer marked
        
        #### 2️⃣ Feature Importance Bar Plot
        - **X-axis**: Feature names (Feature_1, Feature_2, ...)
        - **Y-axis**: Position value (0.0 to 1.0)
        - **Colors**: 🟢 Green (≥threshold) | ⚪ Gray (<threshold)
        - **Interactive**: Adjust threshold with slider (0.0 - 1.0)
        - **Red line**: Current threshold
        
        #### 3️⃣ Box Plot (Distribution Comparison)
        - **Box**: 25th to 75th percentile (middle 50%)
        - **Line in box**: Median fitness
        - **Whiskers**: Min and max values
        - **Points**: Individual run data
        - **Use**: Compare consistency between algorithms
        
        #### 4️⃣ Mean Comparison Bar Chart
        - **Height**: Mean fitness across runs
        - **Error bars**: Standard deviation
        - **Lower is better** (closer to 0)
        - **Use**: Quick average performance comparison
        
        #### 5️⃣ Accuracy Comparison
        - **Formula**: (1 - fitness) × 100%
        - **Bar chart** with percentages
        - **Higher is better**
        - **Use**: More intuitive than fitness
        
        #### 6️⃣ Standard Deviation Chart
        - **Lower SD**: More consistent/reliable
        - **Higher SD**: Results vary between runs
        - **Use**: Choose reliable algorithms for production
        
        #### 7️⃣ Execution Time Comparison
        - **Shows**: Time taken by each algorithm
        - **With error bars**: Time variability
        - **Use**: Speed vs accuracy trade-off
        
        #### 8️⃣ Multi-Dimensional Scatter
        - **X-axis**: Execution time
        - **Y-axis**: Fitness
        - **Bubble size**: Features selected
        - **Ideal**: Bottom-left corner (fast + accurate)
        
        #### 9️⃣ Overall Ranking Table
        - **Composite score**: 60% fitness + 20% consistency + 20% speed
        - **Medals**: 🥇🥈🥉 for top 3
        - **Complete stats**: All metrics in one view
        """)
    
    # TAB 4: Algorithm Categories
    with guide_tabs[3]:
        st.markdown("""
        ### ⚙️ Algorithm Categories Explained
        
        #### 🐝 Swarm Intelligence
        - **Inspiration**: Collective behavior of groups (bees, birds, fish)
        - **Examples**: PSO (particles), ABC (bees), ACO (ants), GWO (wolves)
        - **Good for**: Continuous optimization, feature selection, pattern recognition
        - **Characteristics**: Fast convergence, good exploration, simple implementation
        - **Best use**: When you need quick results with good quality
        
        #### 🧬 Evolutionary Algorithms
        - **Inspiration**: Natural evolution (selection, crossover, mutation)
        - **Examples**: GA (genetic), DE (differential evolution), ES (evolution strategy)
        - **Good for**: Complex problems, multi-objective optimization, constraint handling
        - **Characteristics**: Robust, versatile, handles non-linear problems
        - **Best use**: Complex optimization with multiple constraints
        
        #### ⚡ Physics-Based
        - **Inspiration**: Physical phenomena (gravity, black holes, multiverse)
        - **Examples**: GWO (gray wolves), WOA (whales), MVO (multiverse), SA (annealing)
        - **Good for**: Global optimization, avoiding local optima, large search spaces
        - **Characteristics**: Strong exploration, diverse solutions, balance exploitation
        - **Best use**: When stuck in local optima with other algorithms
        
        #### 🦅 Bio-Inspired
        - **Inspiration**: Biological behaviors and processes
        - **Examples**: BA (bats), CS (cuckoo search), FA (fireflies), BFO (bacteria)
        - **Good for**: Pattern recognition, feature selection, clustering
        - **Characteristics**: Nature-inspired, intuitive, effective for specific problems
        - **Best use**: Pattern matching and classification tasks
        
        #### 🎯 Hybrid Algorithms
        - **Inspiration**: Combining strengths of multiple algorithms
        - **Examples**: PSO-GA, ABC-DE, GWO-PSO, Custom combinations
        - **Good for**: Difficult problems requiring both exploration and exploitation
        - **Characteristics**: Balanced, robust, leverages multiple strategies
        - **Best use**: When single algorithms fail or need better performance
        
        ---
        
        ### 🤖 Algorithm Recommendation System
        
        **How It Analyzes Your Dataset:**
        
        1. **Dimensionality**: Number of features vs samples ratio
        2. **Data Size**: Total samples and computational feasibility
        3. **Problem Type**: Classification, regression, or feature selection
        4. **Complexity**: Linear vs non-linear patterns
        5. **Noise Level**: Data quality assessment
        
        **Matching Process:**
        
        | Step | Action | Output |
        |------|--------|--------|
        | 1 | Analyze dataset characteristics | Profile |
        | 2 | Match with algorithm strengths | Candidates |
        | 3 | Calculate confidence scores | Ranking |
        | 4 | Select top 10 recommendations | List |
        | 5 | Auto-select top 3 | Pre-selected |
        
        **Confidence Score Interpretation:**
        
        - **9.0-10.0**: 🟢 Excellent match - Highly recommended
        - **7.0-8.9**: 🟡 Good match - Reliable choice
        - **5.0-6.9**: 🟠 Acceptable - May work adequately
        - **<5.0**: 🔴 Not recommended - Hidden from list
        
        **What Increases Confidence:**
        - Algorithm specializes in your problem type
        - Dataset size matches algorithm requirements
        - Feature count in algorithm's sweet spot
        - Good track record with similar datasets
        - Balanced exploration-exploitation for your complexity
        """)
    
    # TAB 5: Tips & Best Practices
    with guide_tabs[4]:
        st.markdown("""
        ### 💡 Optimization Tips & Best Practices
        
        #### ✅ For Best Results
        
        | Practice | Why It Matters | How To Do It |
        |----------|----------------|--------------|
        | **Trust recommendations** | System analyzes your data scientifically | Use auto-selected algorithms first |
        | **Multiple runs** | Reduces luck factor, ensures reliability | Set runs to 3-5 |
        | **Sufficient iterations** | Allows algorithms to converge | Start with 100-200 |
        | **Validate results** | Confirms real-world effectiveness | Test in actual ML model |
        | **Compare algorithms** | Different algorithms find different patterns | Run 3-5 different types |
        
        #### ⚡ For Faster Results
        
        **Quick Testing Mode:**
        - Iterations: 50-100
        - Population: 20-30
        - Runs: 1
        - Algorithms: 2-3
        - **Expected time**: 1-5 minutes
        
        **When to Use**: Initial exploration, parameter testing, dataset validation
        
        #### 🎯 For Feature Selection
        
        **Best Practices:**
        
        1. **Set Realistic Target**: 30-50% of original features typically works
        2. **Start with 0.5 Threshold**: Standard selection criterion
        3. **Check Multiple Algorithms**: Different perspectives on importance
        4. **Validate in ML Model**: Selected features must improve model
        5. **Iterate Based on Results**: Adjust threshold if needed
        
        **Common Scenarios:**
        
        | Scenario | Strategy |
        |----------|----------|
        | **Too many features** | Increase threshold to 0.6-0.75 |
        | **Too few features** | Decrease threshold to 0.3-0.4 |
        | **Need diversity** | Run multiple algorithms |
        | **Unclear importance** | Use threshold slider to explore |
        
        #### 🔬 For Research & Publications
        
        **Rigorous Testing:**
        - **Runs**: 10+ for statistical significance
        - **Iterations**: 300-500 for thorough search
        - **Algorithms**: 5-10 different types
        - **Documentation**: Record all parameters and results
        - **Statistical Tests**: Use mean ± SD, confidence intervals
        
        **What to Report:**
        - Dataset characteristics
        - Algorithm parameters
        - Number of runs
        - Mean ± Standard deviation
        - Best, worst, median results
        - Convergence curves
        - Feature selection details
        
        #### ⚠️ Common Mistakes to Avoid
        
        | Mistake | Why It's Bad | Solution |
        |---------|--------------|----------|
        | **Too few iterations** | Algorithm doesn't converge | Use 100+ iterations |
        | **Single run only** | Results not reliable | Use 3+ runs |
        | **Ignoring convergence** | May need more iterations | Check convergence tab |
        | **Not validating** | Selected features may not help | Test in actual model |
        | **Comparing unfairly** | Biased conclusions | Use mean of multiple runs |
        | **Wrong task type** | Poor results | Use feature selection for most ML tasks |
        | **Too many features** | Overfitting risk | Target 30-50% reduction |
        
        #### 🎓 Pro Tips
        
        1. **Start Small**: Test with sample dataset first
        2. **Use Progress**: Monitor real-time fitness values
        3. **Check Convergence**: If not converged, increase iterations
        4. **SD Matters**: Low SD = reliable algorithm
        5. **Time vs Quality**: Fast isn't always better
        6. **Download Everything**: Save results before closing
        7. **Track Changes**: Note what works for your data type
        8. **Read Rankings**: Overall ranking considers multiple factors
        """)
    
    # TAB 6: Troubleshooting
    with guide_tabs[5]:
        st.markdown("""
        ### 🔧 Troubleshooting Guide
        
        #### ❌ Problem: All fitness values are 1.0
        
        **Symptoms**:
        - Every algorithm returns fitness = 1.0
        - No improvement over iterations
        - Convergence curve is flat
        
        **Possible Causes & Solutions**:
        
        | Cause | Check | Solution |
        |-------|-------|----------|
        | **Too few samples** | Dataset size | Minimum 30 samples needed |
        | **Wrong target** | Target column | Verify correct target variable |
        | **Invalid data** | NaN, infinity | Clean dataset before upload |
        | **Mismatched task** | Task type | Try different task type |
        | **Features too many** | Target features | Reduce target feature count |
        
        ---
        
        #### ❌ Problem: ValueError - Array length mismatch
        
        **Status**: ✅ AUTO-FIXED in current version
        
        **If still occurs**:
        - Refresh the page (Ctrl+R / Cmd+R)
        - Re-upload dataset
        - Check CSV format (consistent columns)
        - Report bug if persists
        
        ---
        
        #### ❌ Problem: No convergence in results
        
        **Symptoms**:
        - Fitness not improving
        - Convergence curve still dropping at end
        - High variation between runs
        
        **Solutions**:
        
        1. **Increase iterations** to 200-300
        2. **Increase population** to 50-100
        3. **Try different algorithm** (physics-based often better)
        4. **Check dataset quality** (remove outliers)
        5. **Adjust task parameters** (reduce target features)
        
        ---
        
        #### ❌ Problem: Authentication failed
        
        **Symptoms**:
        - "Invalid password" message
        - "User not found" error
        - Cannot login to existing account
        
        **Troubleshooting Steps**:
        
        1. ✅ **Check Caps Lock** (most common!)
        2. ✅ **Verify username spelling** (case-sensitive)
        3. ✅ **Check password** (re-type carefully)
        4. ✅ **Look at error message** (specific details)
        5. ✅ **Check folder exists**: `persistent_state/users/`
        6. ✅ **Try creating new account** if old one lost
        
        ---
        
        #### ❌ Problem: Results not saving
        
        **Symptoms**:
        - Results disappear after session
        - Cannot find exported files
        - Export buttons don't work
        
        **Solutions**:
        
        1. **Check folder permissions**:
           - Navigate to `persistent_state/` folder
           - Ensure write permissions
           - Check disk space available
        
        2. **Run as administrator** (Windows):
           - Right-click command prompt
           - Select "Run as administrator"
           - Start application again
        
        3. **Check download location**:
           - Browser's download folder
           - Check browser download settings
           - Try different export format
        
        ---
        
        #### ❌ Problem: Slow performance
        
        **Symptoms**:
        - Taking too long to complete
        - System freezing
        - High CPU usage
        
        **Quick Fixes**:
        
        | Parameter | Current | Reduce To | Speedup |
        |-----------|---------|-----------|---------|
        | Iterations | 200+ | 50-100 | 2-4x faster |
        | Population | 50+ | 20-30 | 2x faster |
        | Algorithms | 5+ | 2-3 | 2-3x faster |
        | Runs | 5+ | 1-3 | 2-5x faster |
        
        **Expected Times** (typical dataset):
        - Quick mode (1 algo, 1 run, 50 iter): 30s - 2min
        - Standard mode (3 algos, 3 runs, 100 iter): 3-10min
        - Deep mode (5 algos, 5 runs, 300 iter): 20-60min
        
        ---
        
        #### ❌ Problem: Windows multiprocessing errors
        
        **Status**: ✅ FIXED in current version (n_jobs=1)
        
        **If still occurs**:
        - Update scikit-learn: `pip install --upgrade scikit-learn`
        - Update joblib: `pip install --upgrade joblib`
        - Restart application
        - Check Python version (3.8+ recommended)
        
        ---
        
        #### 💬 Getting More Help
        
        **Before asking for help, collect this info**:
        1. **Error message** (exact text or screenshot)
        2. **Dataset info** (rows, columns, type)
        3. **Parameters used** (iterations, population, etc.)
        4. **Steps to reproduce** (what did you click?)
        5. **System info** (Windows/Mac/Linux, Python version)
        
        **Where to get help**:
        - 📖 Check this guide first (most issues covered)
        - 🔍 Search error message online
        - 📝 Check GitHub issues
        - 📧 Contact maintainer with details above
        """)


def show_optimization():
    """Optimization workflow page"""
    st.markdown("## 🚀 New Optimization")
    
    # Step 1: Data Upload
    st.markdown('<div class="step-indicator"><h3>📤 Step 1: Upload Data</h3></div>', unsafe_allow_html=True)
    
    data_source = st.radio(
        "Choose data source",
        ["📁 Upload CSV", "📦 Sample Dataset", "🎲 Generate Random Data"],
        horizontal=True
    )
    
    X, y = None, None
    
    if data_source == "📁 Upload CSV":
        uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Loaded {df.shape[0]} samples with {df.shape[1]} features")
            st.dataframe(df.head(10), width='stretch')
            
            target_col = st.selectbox("Select target column", df.columns.tolist())
            if target_col:
                X = df.drop(columns=[target_col]).values
                y = df[target_col].values
                st.session_state.current_data = {'X': X, 'y': y, 'df': df, 'target_col': target_col}
    
    elif data_source == "📦 Sample Dataset":
        dataset_name = st.selectbox("Select dataset", ["Iris", "Wine", "Breast Cancer", "Digits"])
        
        if st.button("📥 Load Dataset", type="primary"):
            from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_digits
            
            if dataset_name == "Iris":
                data = load_iris()
            elif dataset_name == "Wine":
                data = load_wine()
            elif dataset_name == "Digits":
                data = load_digits()
            else:
                data = load_breast_cancer()
            
            X, y = data.data, data.target
            df = pd.DataFrame(X, columns=[f"Feature_{i}" for i in range(X.shape[1])])
            df['Target'] = y
            
            st.success(f"✅ Loaded {dataset_name}: {X.shape[0]} samples, {X.shape[1]} features")
            st.dataframe(df.head(10), width='stretch')
            st.session_state.current_data = {'X': X, 'y': y, 'df': df, 'target_col': 'Target'}
    
    elif data_source == "🎲 Generate Random Data":
        col1, col2, col3 = st.columns(3)
        with col1:
            n_samples = st.number_input("Samples", 100, 10000, 1000)
        with col2:
            n_features = st.number_input("Features", 5, 100, 20)
        with col3:
            n_classes = st.number_input("Classes", 2, 10, 2)
        
        if st.button("🎲 Generate Dataset", type="primary"):
            from sklearn.datasets import make_classification
            X, y = make_classification(
                n_samples=n_samples, 
                n_features=n_features, 
                n_classes=n_classes, 
                n_informative=int(n_features*0.7),
                n_redundant=int(n_features*0.2),
                random_state=42
            )
            df = pd.DataFrame(X, columns=[f"Feature_{i}" for i in range(n_features)])
            df['Target'] = y
            
            st.success(f"✅ Generated {n_samples} samples with {n_features} features")
            st.dataframe(df.head(10), width='stretch')
            st.session_state.current_data = {'X': X, 'y': y, 'df': df, 'target_col': 'Target'}
    
    # Continue with rest if data loaded
    if st.session_state.current_data is not None:
        X = st.session_state.current_data['X']
        y = st.session_state.current_data['y']
        
        # Add AI-powered algorithm recommendations
        st.markdown("---")
        st.markdown('<div class="step-indicator"><h3>🤖 AI-Powered Algorithm Recommendations</h3></div>', unsafe_allow_html=True)
        
        # Import and use algorithm recommender
        from mha_toolbox.algorithm_recommender import AlgorithmRecommender
        
        recommender = AlgorithmRecommender()
        
        # Get dataset characteristics
        characteristics = recommender.analyze_dataset(X, y)
        
        # Display dataset analysis
        with st.expander("📊 Dataset Analysis", expanded=True):
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Samples", characteristics['n_samples'])
                st.metric("Features", characteristics['n_features'])
            
            with col2:
                st.metric("Dimensionality", characteristics['dimensionality'].upper())
                st.metric("Sample Size", characteristics['sample_size'].replace('_', ' ').title())
            
            with col3:
                st.metric("Data Type", characteristics['data_type'].replace('_', ' ').title())
                st.metric("Complexity", characteristics['complexity'].title())
            
            with col4:
                st.metric("Has Noise", "Yes" if characteristics['has_noise'] else "No")
                # Check for class_balance key (only available if y is provided)
                if 'class_balance' in characteristics:
                    balance_status = characteristics['class_balance'].replace('_', ' ').title()
                    st.metric("Class Balance", balance_status)
                else:
                    st.metric("Task Type", characteristics.get('task_type', 'Unknown').title())
        
        # Get top 10 recommendations with improved scoring
        recommendations = recommender.recommend_algorithms(X, y, top_k=10)
        
        # Display recommendations in compact format with auto-selection
        with st.expander("🎯 Top 10 Recommended Algorithms", expanded=True):
            st.info("Based on your dataset characteristics, these algorithms are most suitable. Top 3 are **auto-selected** for you!")
            
            # Auto-select top 3 algorithms if not already done
            if 'auto_selected_done' not in st.session_state:
                top_3_algos = [algo_name for algo_name, _, _ in recommendations[:3]]
                st.session_state.quick_select_algos = top_3_algos
                st.session_state.auto_selected_done = True
                st.success(f"✅ Auto-selected top 3: {', '.join([a.upper() for a in top_3_algos])}")
            
            # Create a compact table with select buttons
            for idx, (algo_name, confidence, reason) in enumerate(recommendations, 1):
                col1, col2, col3 = st.columns([0.5, 3, 1])
                
                # Check if already selected
                is_selected = 'quick_select_algos' in st.session_state and algo_name in st.session_state.quick_select_algos
                
                with col1:
                    st.markdown(f"**#{idx}**")
                
                with col2:
                    # Improved confidence display (now properly scaled to 8-10 range)
                    display_confidence = 5.0 + (confidence / 2.0)  # Scale to better range
                    st.markdown(f"**{algo_name.upper()}** (Confidence: {display_confidence:.1f}/10) {'✅' if is_selected else ''}")
                    st.caption(reason)
                
                with col3:
                    if is_selected:
                        if st.button("❌ Remove", key=f"quick_select_{idx}", width='stretch'):
                            st.session_state.quick_select_algos.remove(algo_name)
                            st.rerun()
                    else:
                        if st.button("✅ Add", key=f"quick_select_{idx}", width='stretch'):
                            if 'quick_select_algos' not in st.session_state:
                                st.session_state.quick_select_algos = []
                            st.session_state.quick_select_algos.append(algo_name)
                            st.rerun()
                
                if idx < len(recommendations):
                    st.markdown('<hr style="margin: 0.5rem 0; border: 0; border-top: 1px solid #e0e0e0;">', unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown('<div class="step-indicator"><h3>🎯 Step 2: Select Algorithm(s)</h3></div>', unsafe_allow_html=True)
        
        # Algorithm selection with auto-selected algorithms
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Show auto-selected algorithms
            auto_selected = st.session_state.get('quick_select_algos', [])
            
            if auto_selected:
                st.success(f"✨ **Auto-Selected Algorithms ({len(auto_selected)}):** {', '.join([a.upper() for a in auto_selected])}")
                st.info("💡 These were automatically selected based on your dataset. You can modify the selection below or add more algorithms.")
            
            # Get categories from ALGORITHM_CATEGORIES
            categories = list(ALGORITHM_CATEGORIES.keys())
            category = st.selectbox("Algorithm Category", categories, 
                                   help="Choose from different algorithm families including Hybrid Algorithms")
            
            # Get algorithms for selected category
            category_data = ALGORITHM_CATEGORIES[category]
            if isinstance(category_data, dict) and 'algorithms' in category_data:
                algorithms = category_data['algorithms']
                description = category_data.get('description', '')
                difficulty = category_data.get('difficulty', 'intermediate')
                
                # Show category info
                st.info(f"**{description}**\n\nDifficulty: {difficulty.upper()}")
            else:
                algorithms = category_data if isinstance(category_data, list) else []
            
            # Pre-select auto-selected algorithms if they're in this category
            default_selection = [algo for algo in auto_selected if algo in algorithms]
            if not default_selection and algorithms:
                default_selection = [algorithms[0]]
            
            selected_algos = st.multiselect(
                f"Select algorithms from {category}",
                algorithms,
                default=default_selection,
                help="Select one or more algorithms for comparison. Top recommended algorithms are pre-selected."
            )
            
            # Merge auto-selected with manually selected
            all_selected = list(set(auto_selected + selected_algos))
        
        with col2:
            st.markdown("**Selected:**")
            if all_selected:
                for algo in all_selected:
                    st.markdown(f"""
                    <div class="algo-card">
                        ✓ {algo.upper()}
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("Select algorithms")
        
        if all_selected:
            st.markdown("---")
            st.markdown('<div class="step-indicator"><h3>⚙️ Step 3: Configure Parameters</h3></div>', unsafe_allow_html=True)
            
            # Task type selection
            st.markdown("#### 🎯 Optimization Task")
            task_type = st.radio(
                "Select optimization task:",
                ["feature_selection", "feature_optimization", "hyperparameter_tuning"],
                format_func=lambda x: {
                    "feature_selection": "🎯 Feature Selection (Select best features)",
                    "feature_optimization": "⚡ Feature Optimization (Optimize feature weights)",
                    "hyperparameter_tuning": "⚙️ Hyperparameter Tuning (Tune model parameters)"
                }[x],
                help="Choose the type of optimization task"
            )
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                n_iterations = st.number_input("Iterations", 10, 1000, 100, 10,
                                             help="Number of optimization iterations")
            with col2:
                population_size = st.number_input("Population Size", 10, 200, 30, 5,
                                                help="Number of solutions in population")
            with col3:
                if task_type == "feature_selection":
                    n_features_to_select = st.slider(
                        "Features to Select", 
                        1, X.shape[1], 
                        int(X.shape[1] * 0.5),
                        help="Target number of features to select"
                    )
                else:
                    n_features_to_select = X.shape[1]
                    st.metric("Features", X.shape[1], help="All features will be used")
            
            # Advanced options
            with st.expander("🔧 Advanced Options"):
                n_runs = st.slider("Number of Runs", 1, 10, 3, help="Multiple runs for statistical analysis")
                enable_tracking = st.checkbox("Enable detailed tracking", value=True, 
                                            help="Track iteration-by-iteration progress")
                save_results = st.checkbox("Auto-save results", value=True,
                                         help="Automatically save results to history")
            
            st.markdown("---")
            st.markdown('<div class="step-indicator"><h3>🚀 Step 4: Run Optimization</h3></div>', unsafe_allow_html=True)
            
            if st.button("▶️ Start Optimization", type="primary", width='stretch'):
                run_optimization(
                    all_selected, 
                    n_iterations, 
                    population_size, 
                    n_features_to_select,
                    task_type=task_type,
                    n_runs=n_runs if 'n_runs' in locals() else 3,
                    enable_tracking=enable_tracking if 'enable_tracking' in locals() else True
                )


def run_optimization(algorithms, n_iterations, population_size, n_features, 
                    task_type='feature_selection', n_runs=3, enable_tracking=True):
    """Run optimization with comprehensive tracking like web interface"""
    
    # Disable joblib parallel processing to avoid Windows CreateProcess errors
    import os
    os.environ['LOKY_MAX_CPU_COUNT'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'
    
    X = st.session_state.current_data['X']
    y = st.session_state.current_data['y']
    
    results = {}
    
    # Progress tracking
    progress_container = st.container()
    with progress_container:
        st.markdown("### 🔄 Optimization Progress")
        st.info(f"🎯 Task: **{task_type.replace('_', ' ').title()}** | Algorithms: {len(algorithms)} | Runs per algorithm: {n_runs}")
        progress_bar = st.progress(0)
        status_text = st.empty()
        time_text = st.empty()
        metrics_placeholder = st.empty()
    
    start_time = time.time()
    total_algorithms = len(algorithms)
    
    for algo_idx, algo in enumerate(algorithms):
        algo_start = time.time()
        status_text.markdown(f"**Running:** `{algo.upper()}` ({algo_idx+1}/{total_algorithms})")
        
        # Create results container for this algorithm
        algo_runs = []
        
        try:
            # Run multiple times for statistical significance
            for run in range(n_runs):
                run_start = time.time()
                status_text.markdown(f"**Running:** `{algo.upper()}` - Run {run+1}/{n_runs} ({algo_idx+1}/{total_algorithms})")
                
                # Import and initialize MHA toolbox
                from mha_toolbox import MHAToolbox
                toolbox = MHAToolbox()
                
                # Create objective function based on task type
                if task_type == 'feature_selection':
                    # Binary feature selection - select best n_features
                    from sklearn.model_selection import cross_val_score
                    from sklearn.neighbors import KNeighborsClassifier
                    
                    def objective_function(solution):
                        try:
                            # Ensure solution length matches features
                            if len(solution) != X.shape[1]:
                                # Truncate or pad solution to match X
                                if len(solution) > X.shape[1]:
                                    solution = solution[:X.shape[1]]
                                else:
                                    solution = np.pad(solution, (0, X.shape[1] - len(solution)), constant_values=0.5)
                            
                            # Convert continuous values to binary selection
                            selected = solution >= 0.5
                            if np.sum(selected) == 0:
                                return 1.0  # Worst fitness if no features selected
                            
                            # Use only selected features
                            X_selected = X[:, selected]
                            
                            # Quick evaluation with KNN (n_jobs=1 to avoid Windows multiprocessing issues)
                            knn = KNeighborsClassifier(n_neighbors=min(3, len(y)//2))
                            scores = cross_val_score(knn, X_selected, y, cv=min(3, len(y)//10), scoring='accuracy', n_jobs=1)
                            
                            # Return error rate (1 - accuracy) as we minimize
                            fitness = 1.0 - scores.mean()
                            
                            # Add penalty for selecting too many/few features
                            n_selected = np.sum(selected)
                            penalty = abs(n_selected - n_features) / X.shape[1] * 0.1
                            return fitness + penalty
                        except Exception as e:
                            print(f"Error in objective function: {e}")
                            return 1.0
                    
                elif task_type == 'feature_optimization':
                    # Optimize feature weights (continuous values)
                    from sklearn.model_selection import cross_val_score
                    from sklearn.neighbors import KNeighborsClassifier
                    
                    def objective_function(solution):
                        # Use solution as feature weights
                        try:
                            # Ensure solution length matches features
                            if len(solution) != X.shape[1]:
                                if len(solution) > X.shape[1]:
                                    solution = solution[:X.shape[1]]
                                else:
                                    solution = np.pad(solution, (0, X.shape[1] - len(solution)), constant_values=0.5)
                            
                            X_weighted = X * solution
                            knn = KNeighborsClassifier(n_neighbors=min(3, len(y)//2))
                            scores = cross_val_score(knn, X_weighted, y, cv=min(3, len(y)//10), scoring='accuracy', n_jobs=1)
                            return 1.0 - scores.mean()
                        except Exception as e:
                            print(f"Error in feature_optimization: {e}")
                            return 1.0
                
                else:  # hyperparameter_tuning
                    # Simple hyperparameter optimization
                    from sklearn.model_selection import cross_val_score
                    from sklearn.neighbors import KNeighborsClassifier
                    
                    def objective_function(solution):
                        try:
                            # Map solution to hyperparameters
                            n_neighbors = max(1, int(solution[0] * 20))  # 1-20
                            # Use first few features only
                            X_subset = X[:, :min(10, X.shape[1])]
                            knn = KNeighborsClassifier(n_neighbors=n_neighbors)
                            scores = cross_val_score(knn, X_subset, y, cv=min(3, len(y)//10), scoring='accuracy', n_jobs=1)
                            return 1.0 - scores.mean()
                        except Exception as e:
                            print(f"Error in hyperparameter_tuning: {e}")
                            return 1.0
                
                # Run optimization
                try:
                    # Disable joblib parallel processing to avoid Windows CreateProcess errors
                    import os
                    os.environ['LOKY_MAX_CPU_COUNT'] = '1'  # Force joblib to use single core
                    
                    # MHAToolbox.optimize() expects: algorithm_name, then either (X, y) or (objective_function)
                    # We'll use objective_function approach since we need custom fitness calculation
                    result = toolbox.optimize(
                        algo,  # algorithm_name as first positional argument
                        objective_function=objective_function,
                        dimensions=X.shape[1] if task_type != 'hyperparameter_tuning' else 3,
                        lower_bound=0.0,
                        upper_bound=1.0,
                        population_size=population_size,
                        max_iterations=n_iterations,
                        verbose=False
                    )
                    
                    # Extract results
                    best_fitness = float(result.best_fitness_) if hasattr(result, 'best_fitness_') else 1.0
                    best_solution = result.best_solution_ if hasattr(result, 'best_solution_') else np.zeros(X.shape[1])
                    convergence = result.global_fitness_ if hasattr(result, 'global_fitness_') else [best_fitness]
                    
                    run_result = {
                        'run': run + 1,
                        'best_fitness': best_fitness,
                        'convergence_curve': convergence,
                        'execution_time': time.time() - run_start,
                        'best_solution': best_solution,
                        'success': True
                    }
                    
                    algo_runs.append(run_result)
                    
                    # Update metrics display
                    with metrics_placeholder.container():
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric(f"{algo.upper()} - Run {run+1}", 
                                    "Best Fitness", f"{best_fitness:.6f}")
                        with col2:
                            st.metric("Time", f"{run_result['execution_time']:.2f}s")
                        with col3:
                            accuracy = (1.0 - best_fitness) * 100
                            st.metric("Accuracy", f"{accuracy:.2f}%")
                
                except Exception as e:
                    st.warning(f"⚠️ Run {run+1} encountered an error: {str(e)}")
                    algo_runs.append({
                        'run': run + 1,
                        'best_fitness': 1.0,
                        'convergence_curve': [1.0],
                        'execution_time': time.time() - run_start,
                        'best_solution': np.zeros(X.shape[1]),
                        'success': False,
                        'error': str(e)
                    })
                
                # Update progress
                total_progress = (algo_idx * n_runs + run + 1) / (total_algorithms * n_runs)
                progress_bar.progress(total_progress)
            
            # Calculate statistics across runs
            if algo_runs:
                successful_runs = [r for r in algo_runs if r['success']]
                if successful_runs:
                    fitness_values = [r['best_fitness'] for r in successful_runs]
                    time_values = [r['execution_time'] for r in successful_runs]
                    
                    # Find best run
                    best_run = min(successful_runs, key=lambda x: x['best_fitness'])
                    
                    results[algo] = {
                        'best_fitness': best_run['best_fitness'],
                        'mean_fitness': np.mean(fitness_values),
                        'std_fitness': np.std(fitness_values),
                        'best_accuracy': (1.0 - best_run['best_fitness']) * 100,
                        'mean_accuracy': (1.0 - np.mean(fitness_values)) * 100,
                        'convergence_curve': best_run['convergence_curve'],
                        'execution_time': best_run['execution_time'],
                        'mean_time': np.mean(time_values),
                        'std_time': np.std(time_values),
                        'best_solution': best_run['best_solution'],
                        'n_features_selected': np.sum(best_run['best_solution'] >= 0.5) if task_type == 'feature_selection' else X.shape[1],
                        'runs': algo_runs,
                        'n_runs': len(successful_runs),
                        'task_type': task_type
                    }
                    
                    # Track algorithm usage in profile
                    if st.session_state.user_profile:
                        st.session_state.user_profile.track_algorithm_usage(
                            algorithm=algo,
                            runtime=best_run['execution_time'],
                            accuracy=results[algo]['best_accuracy'] / 100.0
                        )
            
        except Exception as e:
            st.error(f"❌ Error with {algo}: {str(e)}")
            results[algo] = {
                'best_fitness': 1.0,
                'mean_fitness': 1.0,
                'std_fitness': 0.0,
                'best_accuracy': 0.0,
                'mean_accuracy': 0.0,
                'convergence_curve': [1.0],
                'execution_time': time.time() - algo_start,
                'mean_time': time.time() - algo_start,
                'std_time': 0.0,
                'best_solution': np.zeros(X.shape[1]),
                'n_features_selected': 0,
                'runs': [],
                'n_runs': 0,
                'task_type': task_type,
                'error': str(e)
            }
    
    total_time = time.time() - start_time
    st.session_state.optimization_results = results
    
    # Save to user history if logged in
    if st.session_state.user_authenticated and st.session_state.user_profile:
        save_to_user_history(results, total_time, algorithms, n_runs, task_type)
    
    progress_bar.empty()
    status_text.empty()
    time_text.empty()
    metrics_placeholder.empty()
    
    st.markdown(f"""
    <div class="success-box">
        <h3 style="margin-top:0;">✅ Optimization Complete!</h3>
        <p><strong>Total Time:</strong> {total_time:.2f} seconds</p>
        <p><strong>Algorithms Tested:</strong> {len(algorithms)}</p>
        <p><strong>Runs per Algorithm:</strong> {n_runs}</p>
        <p><strong>Task Type:</strong> {task_type.replace('_', ' ').title()}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Update user profile
    if st.session_state.user_profile:
        st.session_state.user_profile.increment_experiments()
        save_profile(st.session_state.user_profile)
    
    show_results_inline(results)


def show_results_inline(results):
    """Display comprehensive results with all visualizations"""
    st.markdown("---")
    st.markdown("### 📊 Optimization Results")
    
    # Check if results are empty
    if not results:
        st.warning("⚠️ No results to display. The optimization may have encountered errors.")
        return
    
    # Import visualization functions
    from mha_toolbox.professional_visualizer import plot_feature_threshold, plot_comparison_box_with_stats
    
    # Calculate statistics for each algorithm
    processed_results = {}
    for algo, result in results.items():
        # Skip if result has error or no data
        if 'error' in result and not result.get('runs'):
            continue
            
        # Get fitness (lower is better for optimization)
        fitness = result.get('best_fitness', 1.0)
        execution_time = result.get('execution_time', 0)
        convergence = result.get('convergence_curve', [])
        selected_features = result.get('best_solution', [])
        n_features = result.get('n_features_selected', 0)
        
        # Calculate statistics
        processed_results[algo] = {
            'best_fitness': fitness,
            'mean_fitness': result.get('mean_fitness', fitness),
            'std_fitness': result.get('std_fitness', 0.0),
            'execution_time': execution_time,
            'mean_time': result.get('mean_time', execution_time),
            'std_time': result.get('std_time', 0.0),
            'convergence_curve': convergence,
            'selected_features': selected_features,
            'n_features_selected': n_features,
            'best_solution': selected_features,
            'runs': result.get('runs', [])  # ADD THIS LINE - Include runs data for box plots
        }
    
    # Store processed results in session state to persist across reruns (for threshold slider)
    st.session_state.processed_results = processed_results
    
    # Check if we have any valid results
    if not processed_results:
        st.error("❌ All algorithms failed. Please check your configuration and try again.")
        return
    
    # Summary metrics with icons
    col1, col2, col3, col4 = st.columns(4)
    
    best_alg = min(processed_results.items(), key=lambda x: x[1]['best_fitness'])
    fastest_alg = min(processed_results.items(), key=lambda x: x[1]['execution_time'])
    avg_time = np.mean([r['execution_time'] for r in processed_results.values()])
    
    with col1:
        st.metric("🧬 Algorithms Tested", len(results))
    
    with col2:
        st.metric("🏆 Best Performer", 
                 best_alg[0].upper(),
                 f"Fitness: {best_alg[1]['best_fitness']:.6f}")
    
    with col3:
        st.metric("⚡ Fastest", 
                 fastest_alg[0].upper(),
                 f"{fastest_alg[1]['execution_time']:.2f}s")
    
    with col4:
        st.metric("⏱️ Avg Time", f"{avg_time:.2f}s")
    
    # Create tabs for different analyses
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Summary", 
        "🎯 Feature Analysis", 
        "⚖️ Comparative Analysis", 
        "🔄 Convergence Analysis", 
        "💾 Export"
    ])
    
    with tab1:
        show_results_summary(processed_results)
    
    with tab2:
        # Use processed_results from session state if available (for threshold slider persistence)
        results_to_use = st.session_state.get('processed_results', processed_results)
        show_feature_analysis(results_to_use)
    
    with tab3:
        show_comparative_analysis(processed_results)
    
    with tab4:
        show_convergence_analysis(processed_results)
    
    with tab5:
        show_export_options(processed_results)


def show_results_summary(results):
    """Show summary table and main performance chart"""
    st.markdown("### 📊 Performance Summary")
    
    # Create detailed summary table
    summary_data = []
    for alg_name, result in results.items():
        summary_data.append({
            'Algorithm': alg_name.upper(),
            'Best Fitness': result['best_fitness'],
            'Mean Fitness': result['mean_fitness'],
            'Std Dev': result['std_fitness'],
            'Time (s)': result['execution_time'],
            'Features Selected': result['n_features_selected']
        })
    
    df = pd.DataFrame(summary_data)
    
    # Display table with formatting
    st.dataframe(df.style.format({
        'Best Fitness': '{:.6f}',
        'Mean Fitness': '{:.6f}',
        'Std Dev': '{:.6f}',
        'Time (s)': '{:.2f}'
    }), width='stretch')
    
    # Performance bar chart with dynamic coloring
    st.markdown("### 📈 Performance Comparison: Fitness Values")
    
    fig_bar = px.bar(
        df,
        x='Algorithm',
        y='Best Fitness',
        color='Best Fitness',
        color_continuous_scale='RdYlGn_r',  # Green for low (good), Red for high (bad)
        title="Algorithm Performance: Best Fitness (Lower is Better)",
        text_auto='.6f'
    )
    fig_bar.update_layout(height=500)
    st.plotly_chart(fig_bar, use_container_width=True)


def show_feature_analysis(results):
    """Show feature selection analysis with interactive threshold"""
    st.markdown("### 🎯 Feature Selection Analysis")
    
    # Safety check: Ensure we have required data
    if not results:
        st.warning("⚠️ No results available for feature analysis.")
        return
    
    if 'current_data' not in st.session_state or st.session_state.current_data is None:
        st.warning("⚠️ Dataset not available. Please run optimization again.")
        return
    
    st.info("""
    **💡 Understanding Feature Importance:**
    
    The MHA search agents explore feature space with positions from 0.0 (least important) to 1.0 (most important).
    Use the **threshold slider** below to interactively control which features to select:
    
    - **Threshold = 0.5** (Default): Standard selection - shows significantly important features
    - **Threshold > 0.5** (e.g., 0.75): High selectivity - shows only the most critical features
    - **Threshold < 0.5** (e.g., 0.25): Low selectivity - includes features that contributed sometimes
    - **Threshold = 0.0**: Shows all features with any contribution
    
    **🎨 Enhanced Color Coding:**
    - 🟢 **Dark Green**: Very important features (≥ threshold + 0.2)
    - 🟢 **Green**: Important features (≥ threshold)
    - 🟠 **Orange**: Borderline features (threshold - 0.1 to threshold)
    - 🟠 **Dark Orange**: Moderately low (threshold - 0.2 to threshold - 0.1)
    - ⚪ **Gray**: Not important features (< threshold - 0.2)
    """)
    
    # Get best algorithm's solution
    try:
        best_algo = min(results.items(), key=lambda x: x[1]['best_fitness'])
        best_solution = best_algo[1].get('best_solution', [])
    except Exception as e:
        st.error(f"Error processing results: {e}")
        return
    
    if len(best_solution) > 0:
        st.markdown(f"#### 🏆 Features from Best Algorithm: **{best_algo[0].upper()}**")
        st.markdown(f"**Best Fitness**: {best_algo[1]['best_fitness']:.6f} | **Accuracy**: {(1 - best_algo[1]['best_fitness']) * 100:.2f}%")
        
        # Get feature names from dataset
        X = st.session_state.current_data['X']
        n_features = X.shape[1]
        feature_names = [f"Feature_{i+1}" for i in range(n_features)]
        
        # Ensure best_solution matches feature count
        if len(best_solution) != n_features:
            if len(best_solution) > n_features:
                best_solution = best_solution[:n_features]
            else:
                best_solution = list(best_solution) + [0.5] * (n_features - len(best_solution))
        
        # Interactive threshold slider
        st.markdown("#### 🎚️ Interactive Threshold Selection")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Use session state to preserve threshold value
            if 'feature_threshold' not in st.session_state:
                st.session_state.feature_threshold = 0.5
            
            threshold = st.slider(
                "Feature Selection Threshold",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.feature_threshold,
                step=0.05,
                key="feature_threshold_slider",
                help="Adjust to control feature selection sensitivity. Higher = more selective.",
                on_change=lambda: setattr(st.session_state, 'feature_threshold', st.session_state.feature_threshold_slider)
            )
            
            # Update session state
            st.session_state.feature_threshold = threshold
        
        with col2:
            # Show selection count at current threshold
            selected_count = sum(1 for val in best_solution if val >= threshold)
            st.metric("Features Selected", f"{selected_count}/{n_features}")
        
        # Create dynamic bar chart based on threshold with enhanced color coding
        import plotly.graph_objects as go
        
        # Enhanced color coding: gradient based on distance from threshold
        def get_feature_color(value, threshold):
            """Return color based on feature importance relative to threshold"""
            if value >= threshold + 0.2:
                return '#27AE60'  # Dark green: Very important (far above threshold)
            elif value >= threshold:
                return '#2ECC71'  # Green: Important (above threshold)
            elif value >= threshold - 0.1:
                return '#F39C12'  # Orange: Borderline (just below threshold)
            elif value >= threshold - 0.2:
                return '#E67E22'  # Dark orange: Moderately low
            else:
                return '#95A5A6'  # Gray: Not important (well below threshold)
        
        colors = [get_feature_color(val, threshold) for val in best_solution]
        
        fig_features = go.Figure()
        fig_features.add_trace(go.Bar(
            x=feature_names,
            y=best_solution,
            marker_color=colors,
            text=[f'{val:.3f}' for val in best_solution],
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Importance: %{y:.3f}<extra></extra>',
            showlegend=False
        ))
        
        # Add threshold line
        fig_features.add_hline(
            y=threshold,
            line_dash="dash",
            line_color="red",
            line_width=2,
            annotation_text=f"Threshold = {threshold:.2f}",
            annotation_position="top right"
        )
        
        # Add y-axis limits
        fig_features.update_layout(
            title=f"Feature Importance Bar Plot (Threshold: {threshold:.2f})",
            xaxis_title="Features",
            yaxis_title="Position Value (0 = least favored, 1 = most important)",
            yaxis_range=[0, 1.1],  # Fixed range from 0 to 1
            height=600,
            showlegend=False,
            hovermode='x unified'
        )
        fig_features.update_xaxes(tickangle=-45)
        
        st.plotly_chart(fig_features, use_container_width=True)
        
        # Add color legend
        st.markdown("""
        **🎨 Color Legend:**
        - 🟢 **Dark Green**: Very important (≥ threshold + 0.2)
        - 🟢 **Green**: Important (≥ threshold)
        - 🟠 **Orange**: Borderline (threshold - 0.1 to threshold)
        - 🟠 **Dark Orange**: Moderately low (threshold - 0.2 to threshold - 0.1)
        - ⚪ **Gray**: Not important (< threshold - 0.2)
        """)
        
        # Show dynamic statistics based on threshold
        st.markdown(f"#### 📊 Selection Statistics (Threshold: {threshold:.2f})")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Total Features", n_features)
        
        with col2:
            selected_at_threshold = sum(1 for val in best_solution if val >= threshold)
            st.metric(f"Selected (≥{threshold:.2f})", selected_at_threshold)
        
        with col3:
            percentage = (selected_at_threshold / n_features) * 100
            st.metric("Selection %", f"{percentage:.1f}%")
        
        with col4:
            avg_selected = np.mean([v for v in best_solution if v >= threshold]) if selected_at_threshold > 0 else 0
            st.metric("Avg (Selected)", f"{avg_selected:.3f}")
        
        with col5:
            avg_all = np.mean(best_solution)
            st.metric("Avg (All)", f"{avg_all:.3f}")
        
        # Comparison at different thresholds
        st.markdown("#### 📊 Threshold Comparison Table")
        
        threshold_data = []
        for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
            count = sum(1 for val in best_solution if val >= t)
            percentage = (count / n_features) * 100
            threshold_data.append({
                'Threshold': f"{t:.2f}",
                'Features Selected': count,
                'Percentage': f"{percentage:.1f}%",
                'Reduction': f"{n_features - count} features removed"
            })
        
        thresh_df = pd.DataFrame(threshold_data)
        st.dataframe(thresh_df, width='stretch', hide_index=True)
        
        # Feature ranking table
        st.markdown("#### 🏆 Feature Ranking (All Features)")
        
        feature_ranking = sorted(
            [(i+1, feature_names[i], best_solution[i], '✅' if best_solution[i] >= threshold else '❌') 
             for i in range(len(best_solution))],
            key=lambda x: x[2],
            reverse=True
        )
        
        ranking_df = pd.DataFrame(
            feature_ranking,
            columns=['Rank', 'Feature', 'Position Value', f'Selected @{threshold:.2f}']
        )
        
        # Show top 20 features
        st.markdown(f"**Top 20 Features:**")
        st.dataframe(
            ranking_df.head(20).style.format({'Position Value': '{:.4f}'}),
            width='stretch',
            hide_index=True
        )
        
        # Expandable section for all features
        with st.expander("🔍 View All Features (Complete Ranking)"):
            st.dataframe(
                ranking_df.style.format({'Position Value': '{:.4f}'}),
                width='stretch',
                hide_index=True,
                height=400
            )
        
        # Download selected features
        st.markdown("#### 💾 Download Selected Features")
        
        selected_features = [
            {'Feature': feature_names[i], 'Position_Value': best_solution[i]}
            for i in range(len(best_solution))
            if best_solution[i] >= threshold
        ]
        
        if selected_features:
            selected_df = pd.DataFrame(selected_features)
            csv_data = selected_df.to_csv(index=False)
            
            st.download_button(
                label=f"📥 Download {len(selected_features)} Selected Features (CSV)",
                data=csv_data,
                file_name=f"selected_features_threshold_{threshold:.2f}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                width='stretch'
            )
        else:
            st.warning(f"⚠️ No features selected at threshold {threshold:.2f}. Lower the threshold to select features.")
        
    else:
        st.warning("⚠️ No feature selection data available for visualization.")



def show_comparative_analysis(results):
    """Show comprehensive comparative analysis with box plots and statistical comparisons"""
    st.markdown("### ⚖️ Comprehensive Comparative Analysis")
    
    # Prepare data for box plots (need individual run data)
    box_plot_data = []
    summary_data = []
    
    for alg_name, result in results.items():
        # Get all runs for box plot
        if 'runs' in result and result['runs']:
            for run in result['runs']:
                # Check if run has fitness data (success might not be explicitly set)
                if 'best_fitness' in run and run['best_fitness'] is not None:
                    box_plot_data.append({
                        'Algorithm': alg_name.upper(),
                        'Fitness': run['best_fitness'],
                        'Time': run.get('execution_time', 0)
                    })
        
        # Summary statistics
        summary_data.append({
            'Algorithm': alg_name.upper(),
            'Best Fitness': result['best_fitness'],
            'Mean Fitness': result['mean_fitness'],
            'Std Dev Fitness': result['std_fitness'],
            'Mean Time': result.get('mean_time', result['execution_time']),
            'Std Dev Time': result.get('std_time', 0),
            'Best Accuracy %': (1 - result['best_fitness']) * 100,
            'Mean Accuracy %': (1 - result['mean_fitness']) * 100,
            'Features Selected': result['n_features_selected']
        })
    
    df_summary = pd.DataFrame(summary_data)
    df_box = pd.DataFrame(box_plot_data)
    
    # 1. BOX PLOT: Fitness Distribution Across Runs
    st.markdown("#### 📦 Box Plot: Fitness Distribution (Multiple Runs)")
    st.info("""
    **How to read this box plot:**
    - **Box**: Contains middle 50% of results (25th to 75th percentile)
    - **Line in box**: Median fitness
    - **Whiskers**: Min and max values
    - **Narrower box**: More consistent algorithm (reliable)
    - **Lower position**: Better performance (lower fitness is better)
    """)
    
    # Check if we have multiple runs per algorithm (at least 2 data points per algorithm)
    if len(df_box) > 0:
        # Count runs per algorithm
        runs_per_algo = df_box.groupby('Algorithm').size()
        if runs_per_algo.min() >= 2:  # At least 2 runs per algorithm
            fig_box = px.box(
                df_box,
                x='Algorithm',
                y='Fitness',
                title="Fitness Distribution Across Multiple Runs (Lower is Better)",
                color='Algorithm',
                points='all'  # Show all data points
            )
            fig_box.update_layout(
                height=500,
                showlegend=False,
                yaxis_title="Fitness Value (0 = perfect, 1 = worst)"
            )
            st.plotly_chart(fig_box, use_container_width=True)
            
            # Show summary stats
            st.markdown("**📊 Statistical Summary per Algorithm:**")
            stats_data = []
            for alg in df_box['Algorithm'].unique():
                alg_data = df_box[df_box['Algorithm'] == alg]['Fitness']
                stats_data.append({
                    'Algorithm': alg,
                    'Runs': len(alg_data),
                    'Min': f"{alg_data.min():.6f}",
                    'Max': f"{alg_data.max():.6f}",
                    'Median': f"{alg_data.median():.6f}",
                    'Mean': f"{alg_data.mean():.6f}",
                    'Std Dev': f"{alg_data.std():.6f}"
                })
            st.dataframe(pd.DataFrame(stats_data), use_container_width=True)
        else:
            st.warning(f"⚠️ Box plot requires multiple runs per algorithm. Current runs: {runs_per_algo.to_dict()}")
    else:
        st.warning("⚠️ No run data available for box plot. Please ensure 'Runs per Algorithm' is set to 2 or more.")
    
    # 2. MEAN FITNESS COMPARISON with Error Bars (SD)
    st.markdown("#### � Mean Fitness with Standard Deviation")
    
    fig_mean = go.Figure()
    fig_mean.add_trace(go.Bar(
        x=df_summary['Algorithm'],
        y=df_summary['Mean Fitness'],
        error_y=dict(
            type='data',
            array=df_summary['Std Dev Fitness'],
            visible=True
        ),
        marker_color='lightcoral',
        text=[f"{val:.4f}" for val in df_summary['Mean Fitness']],
        textposition='outside',
        name='Mean Fitness'
    ))
    
    fig_mean.update_layout(
        title="Mean Fitness Comparison with Standard Deviation (Lower is Better)",
        xaxis_title="Algorithm",
        yaxis_title="Mean Fitness ± SD",
        height=500,
        showlegend=False
    )
    fig_mean.update_xaxes(tickangle=-45)
    st.plotly_chart(fig_mean, use_container_width=True)
    
    # 3. ACCURACY COMPARISON (inverse of fitness)
    st.markdown("#### 🎯 Accuracy Comparison (Higher is Better)")
    
    fig_acc = go.Figure()
    fig_acc.add_trace(go.Bar(
        x=df_summary['Algorithm'],
        y=df_summary['Mean Accuracy %'],
        marker_color='lightgreen',
        text=[f"{val:.2f}%" for val in df_summary['Mean Accuracy %']],
        textposition='outside',
        name='Accuracy'
    ))
    
    fig_acc.update_layout(
        title="Mean Accuracy Comparison (1 - Fitness) × 100%",
        xaxis_title="Algorithm",
        yaxis_title="Accuracy (%)",
        height=500,
        yaxis_range=[0, 105],
        showlegend=False
    )
    fig_acc.update_xaxes(tickangle=-45)
    st.plotly_chart(fig_acc, use_container_width=True)
    
    # 4. STANDARD DEVIATION COMPARISON (Reliability)
    st.markdown("#### 📐 Standard Deviation Comparison (Consistency)")
    st.info("**Lower SD = More consistent/reliable algorithm** across multiple runs")
    
    fig_std = go.Figure()
    fig_std.add_trace(go.Bar(
        x=df_summary['Algorithm'],
        y=df_summary['Std Dev Fitness'],
        marker_color='lightskyblue',
        text=[f"{val:.4f}" for val in df_summary['Std Dev Fitness']],
        textposition='outside',
        name='Std Dev'
    ))
    
    fig_std.update_layout(
        title="Standard Deviation of Fitness (Lower = More Reliable)",
        xaxis_title="Algorithm",
        yaxis_title="Standard Deviation",
        height=500,
        showlegend=False
    )
    fig_std.update_xaxes(tickangle=-45)
    st.plotly_chart(fig_std, use_container_width=True)
    
    # 5. EXECUTION TIME COMPARISON with SD
    st.markdown("#### ⏱️ Execution Time Comparison")
    
    fig_time = go.Figure()
    fig_time.add_trace(go.Bar(
        x=df_summary['Algorithm'],
        y=df_summary['Mean Time'],
        error_y=dict(
            type='data',
            array=df_summary['Std Dev Time'],
            visible=True
        ),
        marker_color='plum',
        text=[f"{val:.2f}s" for val in df_summary['Mean Time']],
        textposition='outside',
        name='Time'
    ))
    
    fig_time.update_layout(
        title="Mean Execution Time with Standard Deviation (Lower is Better)",
        xaxis_title="Algorithm",
        yaxis_title="Time (seconds) ± SD",
        height=500,
        showlegend=False
    )
    fig_time.update_xaxes(tickangle=-45)
    st.plotly_chart(fig_time, use_container_width=True)
    
    # 6. FEATURE SELECTION COMPARISON
    if 'Features Selected' in df_summary.columns:
        st.markdown("#### 🎯 Features Selected Comparison")
        
        X = st.session_state.current_data['X']
        target_features = X.shape[1] // 2  # Assume target is 50%
        
        fig_features = go.Figure()
        fig_features.add_trace(go.Bar(
            x=df_summary['Algorithm'],
            y=df_summary['Features Selected'],
            marker_color='lightcyan',
            text=df_summary['Features Selected'],
            textposition='outside',
            name='Features'
        ))
        
        # Add target line
        fig_features.add_hline(
            y=target_features,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Target: {target_features} features"
        )
        
        fig_features.update_layout(
            title=f"Number of Features Selected (Target: ~{target_features})",
            xaxis_title="Algorithm",
            yaxis_title="Number of Features",
            height=500,
            showlegend=False
        )
        fig_features.update_xaxes(tickangle=-45)
        st.plotly_chart(fig_features, use_container_width=True)
    
    # 7. MULTI-DIMENSIONAL COMPARISON (Scatter plot)
    st.markdown("#### 🔄 Multi-Dimensional Comparison")
    st.info("**Ideal position**: Bottom-left corner (low fitness, fast time)")
    
    fig_scatter = px.scatter(
        df_summary,
        x='Mean Time',
        y='Mean Fitness',
        size='Features Selected',
        color='Algorithm',
        hover_data=['Mean Accuracy %', 'Std Dev Fitness'],
        title="Performance vs Speed (Bubble size = Features Selected)",
        text='Algorithm'
    )
    fig_scatter.update_traces(textposition='top center')
    fig_scatter.update_layout(
        height=600,
        xaxis_title="Mean Execution Time (seconds) - Lower is Better",
        yaxis_title="Mean Fitness - Lower is Better"
    )
    st.plotly_chart(fig_scatter, use_container_width=True)
    
    # 8. RANKING TABLE
    st.markdown("#### 🏆 Overall Ranking")
    
    # Calculate composite score (lower is better)
    df_summary['Composite Score'] = (
        df_summary['Mean Fitness'] * 0.6 +  # 60% weight on fitness
        (df_summary['Std Dev Fitness'] / df_summary['Std Dev Fitness'].max()) * 0.2 +  # 20% on consistency
        (df_summary['Mean Time'] / df_summary['Mean Time'].max()) * 0.2  # 20% on speed
    )
    
    df_ranked = df_summary.sort_values('Composite Score').reset_index(drop=True)
    df_ranked.insert(0, 'Rank', range(1, len(df_ranked) + 1))
    
    # Add medals
    def add_medal(rank):
        if rank == 1:
            return "🥇"
        elif rank == 2:
            return "🥈"
        elif rank == 3:
            return "🥉"
        else:
            return ""
    
    df_ranked['Medal'] = df_ranked['Rank'].apply(add_medal)
    
    display_cols = ['Rank', 'Medal', 'Algorithm', 'Best Fitness', 'Mean Fitness', 
                    'Std Dev Fitness', 'Mean Accuracy %', 'Mean Time', 'Features Selected']
    
    st.dataframe(
        df_ranked[display_cols].style.format({
            'Best Fitness': '{:.6f}',
            'Mean Fitness': '{:.6f}',
            'Std Dev Fitness': '{:.6f}',
            'Mean Accuracy %': '{:.2f}%',
            'Mean Time': '{:.2f}s'
        }),
        width='stretch',
        hide_index=True
    )
    
    # 9. STATISTICAL SUMMARY
    st.markdown("#### 📊 Statistical Summary")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**🏆 Best Performers:**")
        best_fitness_algo = df_summary.loc[df_summary['Best Fitness'].idxmin(), 'Algorithm']
        best_acc_algo = df_summary.loc[df_summary['Mean Accuracy %'].idxmax(), 'Algorithm']
        st.write(f"- **Best Fitness**: {best_fitness_algo}")
        st.write(f"- **Best Accuracy**: {best_acc_algo}")
    
    with col2:
        st.markdown("**⚡ Efficiency:**")
        fastest_algo = df_summary.loc[df_summary['Mean Time'].idxmin(), 'Algorithm']
        most_consistent = df_summary.loc[df_summary['Std Dev Fitness'].idxmin(), 'Algorithm']
        st.write(f"- **Fastest**: {fastest_algo}")
        st.write(f"- **Most Consistent**: {most_consistent}")
    
    with col3:
        st.markdown("**📊 Statistics:**")
        st.write(f"- **Avg Fitness**: {df_summary['Mean Fitness'].mean():.4f}")
        st.write(f"- **Avg Accuracy**: {df_summary['Mean Accuracy %'].mean():.2f}%")
        st.write(f"- **Avg Time**: {df_summary['Mean Time'].mean():.2f}s")


def show_convergence_analysis(results):
    """Show detailed convergence analysis"""
    st.markdown("### 🔄 Convergence Analysis")
    
    # Convergence curves
    fig = go.Figure()
    
    for alg_name, result in results.items():
        convergence = result.get('convergence_curve', [])
        if len(convergence) > 0:
            fig.add_trace(go.Scatter(
                x=list(range(1, len(convergence) + 1)),
                y=convergence,
                mode='lines+markers',
                name=alg_name.upper(),
                line=dict(width=2),
                marker=dict(size=4),
                hovertemplate='<b>' + alg_name.upper() + '</b><br>Iteration: %{x}<br>Fitness: %{y:.6f}<extra></extra>'
            ))
    
    fig.update_layout(
        title="Convergence Curves: Fitness vs. Iterations",
        xaxis_title="Iteration",
        yaxis_title="Fitness Value (Lower is Better)",
        height=600,
        hovermode='x unified',
        template='plotly_white'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Convergence statistics
    st.markdown("#### 📊 Convergence Statistics")
    
    convergence_stats = []
    for alg_name, result in results.items():
        convergence = result.get('convergence_curve', [])
        if len(convergence) > 1:
            initial_fitness = convergence[0]
            final_fitness = convergence[-1]
            improvement = initial_fitness - final_fitness
            improvement_pct = (improvement / initial_fitness * 100) if initial_fitness != 0 else 0
            
            # Find iteration where 90% of improvement happened
            target_fitness = initial_fitness - (0.9 * improvement)
            convergence_90 = next((i for i, f in enumerate(convergence) if f <= target_fitness), len(convergence))
            
            convergence_stats.append({
                'Algorithm': alg_name.upper(),
                'Initial Fitness': initial_fitness,
                'Final Fitness': final_fitness,
                'Improvement': improvement,
                'Improvement %': improvement_pct,
                '90% Conv. Iter': convergence_90
            })
    
    if convergence_stats:
        conv_df = pd.DataFrame(convergence_stats)
        st.dataframe(conv_df.style.format({
            'Initial Fitness': '{:.6f}',
            'Final Fitness': '{:.6f}',
            'Improvement': '{:.6f}',
            'Improvement %': '{:.2f}%',
            '90% Conv. Iter': '{:.0f}'
        }), width='stretch')
    
    # Convergence speed comparison
    st.markdown("#### ⚡ Convergence Speed Comparison")
    st.info("""
    **90% Convergence Iteration** shows how quickly each algorithm reaches 90% of its total improvement.
    Lower values indicate faster convergence.
    """)


def show_export_options(results):
    """Show export and download options"""
    st.markdown("### 💾 Export Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # CSV export
        summary_data = []
        for alg_name, result in results.items():
            summary_data.append({
                'Algorithm': alg_name.upper(),
                'Best_Fitness': result['best_fitness'],
                'Mean_Fitness': result['mean_fitness'],
                'Std_Fitness': result['std_fitness'],
                'Execution_Time': result['execution_time'],
                'Features_Selected': result['n_features_selected']
            })
        
        df_export = pd.DataFrame(summary_data)
        csv = df_export.to_csv(index=False)
        
        st.download_button(
            label="📥 Download Summary (CSV)",
            data=csv,
            file_name=f"mha_results_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        # JSON export with all details
        import json
        json_data = json.dumps(results, indent=2, default=str)
        
        st.download_button(
            label="📥 Download Full Results (JSON)",
            data=json_data,
            file_name=f"mha_results_full_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True
        )
    
    st.markdown("---")
    
    # Additional export options
    st.markdown("#### 📊 Additional Export Options")
    
    col3, col4 = st.columns(2)
    
    with col3:
        # Export convergence data
        convergence_data = []
        for alg_name, result in results.items():
            conv = result.get('convergence_curve', [])
            for iter_num, fitness in enumerate(conv, 1):
                convergence_data.append({
                    'Algorithm': alg_name.upper(),
                    'Iteration': iter_num,
                    'Fitness': fitness
                })
        
        if convergence_data:
            conv_df = pd.DataFrame(convergence_data)
            conv_csv = conv_df.to_csv(index=False)
            
            st.download_button(
                label="📈 Download Convergence Data (CSV)",
                data=conv_csv,
                file_name=f"mha_convergence_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    with col4:
        # Export feature selection data
        best_algo = min(results.items(), key=lambda x: x[1]['best_fitness'])
        best_solution = best_algo[1].get('best_solution', [])
        
        if len(best_solution) > 0:
            X = st.session_state.current_data['X']
            n_features = X.shape[1]
            feature_names = [f"Feature_{i+1}" for i in range(n_features)]
            
            # Ensure best_solution matches feature count
            if len(best_solution) != n_features:
                if len(best_solution) > n_features:
                    best_solution = best_solution[:n_features]
                else:
                    best_solution = list(best_solution) + [0.5] * (n_features - len(best_solution))
            
            feature_data = pd.DataFrame({
                'Feature': feature_names,
                'Position_Value': best_solution,
                'Selected_0.5': ['Yes' if v >= 0.5 else 'No' for v in best_solution],
                'Selected_0.75': ['Yes' if v >= 0.75 else 'No' for v in best_solution]
            })
            
            feature_csv = feature_data.to_csv(index=False)
            
            st.download_button(
                label="🎯 Download Feature Selection (CSV)",
                data=feature_csv,
                file_name=f"mha_features_{best_algo[0]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    st.success("✅ All export options are available above. Choose the format you need!")
    
    # Quick action button
    if st.button("📊 View in Results History Page", type="primary", use_container_width=True):
        st.session_state.current_page = "📊 Results"
        st.rerun()


def show_results():
    """Results history page"""
    st.markdown("## 📊 Results History")
    
    if st.session_state.optimization_results:
        st.info("💡 **Tip:** Results from your most recent optimization are shown below with comprehensive analysis.")
        show_results_inline(st.session_state.optimization_results)
    else:
        st.markdown("""
        <div class="info-box">
            <h3 style="margin-top:0;">📊 No Results Yet</h3>
            <p>Run an optimization to see comprehensive results and analysis here.</p>
            <p><strong>You'll get:</strong></p>
            <ul>
                <li>📈 Performance summary with statistics</li>
                <li>🎯 Interactive feature selection analysis with importance ranking</li>
                <li>⚖️ Comparative analysis charts</li>
                <li>🔄 Convergence curves and speed metrics</li>
                <li>💾 Multiple export formats (CSV, JSON)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🚀 Start New Optimization", type="primary", use_container_width=True):
            st.session_state.current_page = "🚀 New Optimization"
            st.rerun()


def show_about():
    """About page with system information using tabs"""
    st.markdown("## 📖 About MHA Toolbox")
    
    st.markdown("""
    <div class="info-box">
        <h3 style="margin-top:0;">🧬 What is MHA Toolbox?</h3>
        <p style="font-size: 1.05rem; line-height: 1.8;">
            MHA Toolbox is a comprehensive Python library and web interface for 
            <strong>Meta-Heuristic Algorithm Optimization</strong>. It provides 98+ 
            state-of-the-art algorithms for solving complex optimization problems 
            including feature selection, classification, and function optimization.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown(f"""
    **📊 Current Session:**
    - Algorithms Available: {st.session_state.total_algorithms}
    - User: {st.session_state.current_user if st.session_state.user_authenticated else 'Guest'}
    """)


def show_history():
    """User's optimization history page"""
    st.markdown("## 📜 Optimization History")
    
    # Check if user is logged in
    if not st.session_state.user_authenticated or not st.session_state.user_profile:
        st.warning("⚠️ Please log in to view your optimization history.")
        st.markdown("""
        <div class="info-box">
            <h3 style="margin-top:0;">🔐 Login Required</h3>
            <p>The optimization history feature requires a user account.</p>
            <p><strong>Benefits of creating an account:</strong></p>
            <ul>
                <li>📜 Track all your optimization runs</li>
                <li>📊 View performance statistics and trends</li>
                <li>🎯 Compare different datasets and algorithms</li>
                <li>💾 Persistent storage of results</li>
                <li>⚙️ Personalized settings and preferences</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Get history from user profile
    history = st.session_state.user_profile.preferences.get('optimization_history', [])
    
    if not history:
        st.info("📭 No optimization history yet. Run your first optimization to start tracking!")
        st.markdown("""
        <div class="info-box">
            <h3 style="margin-top:0;">🚀 Start Optimizing!</h3>
            <p>Your optimization runs will be automatically saved here.</p>
            <p><strong>What gets tracked:</strong></p>
            <ul>
                <li>⏱️ Timestamp and execution time</li>
                <li>🧬 Algorithms tested</li>
                <li>📊 Best fitness achieved</li>
                <li>📁 Dataset information</li>
                <li>🎯 Task type (feature selection, optimization, etc.)</li>
                <li>📈 Complete results summary</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🚀 Start New Optimization", type="primary", use_container_width=True):
            st.session_state.current_page = "🚀 New Optimization"
            st.rerun()
        return
    
    # Display statistics
    st.markdown("### 📊 Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    total_runs = len(history)
    unique_algorithms = len(set(algo for entry in history for algo in entry.get('algorithms', [])))
    unique_datasets = len(set(entry.get('dataset', 'Unknown') for entry in history))
    avg_time = sum(entry.get('total_time', 0) for entry in history) / total_runs if total_runs > 0 else 0
    
    with col1:
        st.metric("Total Runs", total_runs)
    with col2:
        st.metric("Algorithms Tested", unique_algorithms)
    with col3:
        st.metric("Datasets Used", unique_datasets)
    with col4:
        st.metric("Avg Time (s)", f"{avg_time:.2f}")
    
    st.markdown("---")
    
    # Filters
    st.markdown("### 🔍 Filters")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Dataset filter
        all_datasets = sorted(set(entry.get('dataset', 'Unknown') for entry in history))
        selected_dataset = st.selectbox(
            "Dataset",
            ["All"] + all_datasets,
            key="history_dataset_filter"
        )
    
    with col2:
        # Task type filter
        all_task_types = sorted(set(entry.get('task_type', 'Unknown') for entry in history))
        selected_task = st.selectbox(
            "Task Type",
            ["All"] + all_task_types,
            key="history_task_filter"
        )
    
    with col3:
        # Sort by
        sort_by = st.selectbox(
            "Sort By",
            ["Newest First", "Oldest First", "Best Fitness", "Execution Time"],
            key="history_sort"
        )
    
    # Filter history
    filtered_history = history.copy()
    
    if selected_dataset != "All":
        filtered_history = [h for h in filtered_history if h.get('dataset') == selected_dataset]
    
    if selected_task != "All":
        filtered_history = [h for h in filtered_history if h.get('task_type') == selected_task]
    
    # Sort history
    if sort_by == "Newest First":
        filtered_history = sorted(filtered_history, key=lambda x: x.get('timestamp', ''), reverse=True)
    elif sort_by == "Oldest First":
        filtered_history = sorted(filtered_history, key=lambda x: x.get('timestamp', ''))
    elif sort_by == "Best Fitness":
        filtered_history = sorted(filtered_history, key=lambda x: x.get('best_fitness', float('inf')))
    elif sort_by == "Execution Time":
        filtered_history = sorted(filtered_history, key=lambda x: x.get('total_time', 0))
    
    st.markdown(f"### 📋 History ({len(filtered_history)} entries)")
    
    if not filtered_history:
        st.info("No entries match the selected filters.")
        return
    
    # Display history entries
    for idx, entry in enumerate(filtered_history):
        timestamp = entry.get('timestamp', 'Unknown')
        try:
            from datetime import datetime
            dt = datetime.fromisoformat(timestamp)
            formatted_time = dt.strftime("%Y-%m-%d %H:%M:%S")
        except:
            formatted_time = timestamp
        
        algorithms = entry.get('algorithms', [])
        dataset = entry.get('dataset', 'Unknown')
        task_type = entry.get('task_type', 'Unknown')
        best_algo = entry.get('best_algorithm', 'N/A')
        best_fitness = entry.get('best_fitness', 'N/A')
        total_time = entry.get('total_time', 0)
        n_runs = entry.get('n_runs', 1)
        
        with st.expander(f"🔬 Run #{len(filtered_history) - idx}: {formatted_time} - {dataset}"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                **📊 Overview**
                - **Dataset:** {dataset}
                - **Task Type:** {task_type}
                - **Algorithms:** {len(algorithms)}
                - **Runs per Algorithm:** {n_runs}
                - **Total Time:** {total_time:.2f}s
                """)
            
            with col2:
                st.markdown(f"""
                **🏆 Best Result**
                - **Algorithm:** {best_algo}
                - **Fitness:** {best_fitness if isinstance(best_fitness, str) else f'{best_fitness:.6e}'}
                """)
            
            st.markdown("**🧬 Algorithms Tested:**")
            st.write(", ".join(algorithms))
            
            # Results summary
            results_summary = entry.get('results_summary', {})
            if results_summary:
                st.markdown("**📈 Detailed Results:**")
                
                # Create a dataframe for easy viewing
                import pandas as pd
                results_data = []
                for algo, res in results_summary.items():
                    results_data.append({
                        'Algorithm': algo,
                        'Best Fitness': res.get('best_fitness', 'N/A'),
                        'Mean Fitness': res.get('mean_fitness', 'N/A'),
                        'Execution Time (s)': res.get('execution_time', 'N/A'),
                        'Features Selected': res.get('n_features_selected', 'N/A')
                    })
                
                if results_data:
                    df = pd.DataFrame(results_data)
                    st.dataframe(df, use_container_width=True)


def show_settings():
    # Create tabs for organized content
    about_tabs = st.tabs([
        "🔄 System Flow",
        "🎯 Algorithm Categories",
        "👥 Multi-User",
        "💻 Usage Modes",
        "🛠️ Technical Details",
        "📜 License"
    ])
    
    # TAB 1: System Flow
    with about_tabs[0]:
        st.markdown("""
        ### 🔄 System Architecture
        
        ```
        ┌─────────────────────────────────────────────────────────────┐
        │                    MHA TOOLBOX SYSTEM                        │
        │              130+ Algorithms | Multi-User Support            │
        └─────────────────────────────────────────────────────────────┘
                                      │
                      ┌───────────────┴───────────────┐
                      │                               │
                ┌─────▼─────┐                  ┌─────▼─────┐
                │  Web UI   │                  │  Library  │
                │ (Streamlit)│                 │ (Python)  │
                └─────┬─────┘                  └─────┬─────┘
                      │                               │
                      └───────────────┬───────────────┘
                                      │
                           ┌──────────▼──────────┐
                           │   MHA Core Engine   │
                           │  - 130+ Algorithms  │
                           │  - Feature Select   │
                           │  - Classification   │
                           │  - Auto-Recommend   │
                           └──────────┬──────────┘
                                      │
                      ┌───────────────┼───────────────┐
                      │               │               │
                ┌─────▼─────┐  ┌─────▼─────┐  ┌─────▼─────┐
                │   Data    │  │  Results  │  │   User    │
                │  Storage  │  │  Manager  │  │  Profiles │
                └───────────┘  └───────────┘  └───────────┘
        ```
        
        ### 📊 Data Flow
        
        1. **Upload/Load Data** → CSV, Sample, or Generated dataset
        2. **Algorithm Selection** → Auto-recommend or manual selection
        3. **Configuration** → Set parameters (iterations, population, etc.)
        4. **Optimization** → Run algorithms with progress tracking
        5. **Results Analysis** → 5 tabs with comprehensive visualizations
        6. **Export** → Download in CSV, Excel, JSON, or NPZ format
        
        ### 🔐 Security & Isolation
        
        - **Session-based**: Each user gets unique session ID
        - **Data isolation**: User data stored separately
        - **Password protection**: SHA-256 hashed passwords
        - **Auto cleanup**: Expired sessions removed after 24 hours
        """)
    
    # TAB 2: Algorithm Categories
    with about_tabs[1]:
        st.markdown("### 🎯 Algorithm Categories")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            #### 🐝 Swarm Intelligence (PSO)
            - Particle Swarm Optimization (PSO)
            - Grey Wolf Optimizer (GWO)
            - Whale Optimization Algorithm (WOA)
            - Ant Colony Optimization (ACO)
            - Artificial Bee Colony (ABC)
            - Firefly Algorithm (FA)
            - Bat Algorithm (BA)
            - Cuckoo Search (CS)
            - Moth-Flame Optimization (MFO)
            - Dragonfly Algorithm (DA)
            
            #### 🧬 Evolutionary (GA, DE)
            - Genetic Algorithm (GA)
            - Differential Evolution (DE)
            - Evolution Strategy (ES)
            - Genetic Programming (GP)
            - Biogeography-Based Optimizer (BBO)
            """)
        
        with col2:
            st.markdown("""
            #### ⚡ Physics-Based (SA, GSA)
            - Simulated Annealing (SA)
            - Gravitational Search Algorithm (GSA)
            - Multi-Verse Optimizer (MVO)
            - Black Hole Algorithm (BH)
            - Thermal Exchange Optimization (TEO)
            - Atom Search Optimization (ASO)
            
            #### 🎯 Hybrid Algorithms (22+)
            - PSO-GA Hybrid
            - GWO-DE Hybrid
            - ABC-DE Hybrid
            - Custom Combinations
            - Multi-Strategy Optimization
            """)
        
        st.markdown("---")
        st.markdown("""
        ### 📈 Algorithm Performance Characteristics
        
        | Category | Speed | Exploration | Exploitation | Best For |
        |----------|-------|-------------|--------------|----------|
        | **Swarm** | ⚡⚡⚡ Fast | 🔍🔍🔍 High | 🎯🎯 Medium | Feature selection, Quick optimization |
        | **Evolutionary** | ⚡⚡ Medium | 🔍🔍 Medium | 🎯🎯🎯 High | Complex problems, Constraints |
        | **Physics** | ⚡ Slow | 🔍🔍🔍 High | 🎯🎯🎯 High | Global optimum, Large search space |
        | **Hybrid** | ⚡⚡ Medium | 🔍🔍🔍 High | 🎯🎯🎯 High | Difficult problems, Best overall |
        """)
    
    # TAB 3: Multi-User Support
    with about_tabs[2]:
        st.markdown("""
        ### 👥 Multi-User System Architecture
        
        #### 🔄 How It Works
        
        **Session Management:**
        1. **User Login**: Username + password authentication
        2. **Session Creation**: Unique UUID assigned
        3. **Data Isolation**: Separate workspace per user
        4. **Profile Loading**: User preferences and history restored
        5. **Auto-Save**: Results saved to user's directory
        6. **Auto-Cleanup**: Expired sessions removed daily
        
        **Storage Structure:**
        ```
        persistent_state/
        ├── users/
        │   ├── alice/
        │   │   ├── profile.json      (persistent settings)
        │   │   └── algorithm_stats.json
        │   └── bob/
        │       ├── profile.json
        │       └── algorithm_stats.json
        ├── sessions/
        │   ├── session_abc123.json   (active session)
        │   └── session_def456.json
        ├── results/
        │   ├── alice_20250106_143022.csv
        │   └── bob_20250106_150315.npz
        └── datasets/
            ├── uploaded_data_001.csv
            └── uploaded_data_002.csv
        ```
        
        #### 🔐 Security Features
        
        | Feature | Implementation | Benefit |
        |---------|----------------|---------|
        | **Password Hashing** | SHA-256 | Secure storage |
        | **Session IDs** | UUID v4 | Unique identification |
        | **Data Isolation** | Separate folders | Privacy protection |
        | **System Tracking** | Platform detection | Security auditing |
        | **Auto-Cleanup** | 24-hour expiry | Disk space management |
        
        #### 👤 User Features
        
        **Profile Tracking:**
        - Total experiments run
        - Total sessions
        - Favorite algorithms
        - Algorithm usage statistics
        - Best results achieved
        - Time spent
        
        **Automatic Saved:**
        - All optimization results
        - Dataset uploads
        - Configuration preferences
        - Export history
        """)
    
    # TAB 4: Usage Modes
    with about_tabs[3]:
        st.markdown("### 💻 Usage Modes")
        
        usage_tabs = st.tabs(["🌐 Web Interface", "📚 Python Library", "⌨️ CLI Mode"])
        
        with usage_tabs[0]:
            st.markdown("""
            ### 🌐 Web Interface (Current)
            
            **Launch Methods:**
            
            ```bash
            # Method 1: Direct Python execution
            python mha_ui_complete.py
            
            # Method 2: Streamlit command
            streamlit run mha_ui_complete.py
            
            # Method 3: Custom port
            streamlit run mha_ui_complete.py --server.port 8502
            
            # Method 4: Module execution
            python -m mha_toolbox ui
            ```
            
            **Features:**
            - ✅ Multi-user authentication
            - ✅ Interactive data upload
            - ✅ 130+ algorithm selection
            - ✅ Real-time progress tracking
            - ✅ 5-tab result analysis
            - ✅ Interactive threshold slider
            - ✅ 9 comparison visualizations
            - ✅ Multiple export formats
            - ✅ Comprehensive user guide
            
            **Recommended For:**
            - Beginners
            - Visual learners
            - Interactive exploration
            - Non-programmers
            - Quick testing
            """)
        
        with usage_tabs[1]:
            st.markdown("""
            ### 📚 Python Library Usage
            
            **Installation:**
            
            ```bash
            pip install mha-toolbox
            # OR
            pip install -e .  # From source
            ```
            
            **Basic Usage:**
            
            ```python
            from mha_toolbox import MHAToolbox
            import pandas as pd
            import numpy as np
            
            # Load data
            df = pd.read_csv('dataset.csv')
            X = df.drop(columns=['target']).values
            y = df['target'].values
            
            # Initialize toolbox
            toolbox = MHAToolbox()
            
            # Quick optimization
            result = toolbox.optimize(
                'PSO',  # Algorithm name
                objective_function=None,  # Auto-create
                dimensions=X.shape[1],
                lower_bound=0.0,
                upper_bound=1.0,
                population_size=30,
                max_iterations=100
            )
            
            # Access results
            print(f"Best fitness: {result.best_fitness_:.6f}")
            print(f"Best solution: {result.best_solution_}")
            print(f"Convergence: {result.global_fitness_}")
            
            # Get selected features (threshold 0.5)
            selected = result.best_solution_ >= 0.5
            X_selected = X[:, selected]
            print(f"Selected {np.sum(selected)} features")
            ```
            
            **Advanced Usage:**
            
            ```python
            # Custom objective function
            def custom_objective(solution):
                selected = solution >= 0.5
                if np.sum(selected) == 0:
                    return 1.0
                
                X_sel = X[:, selected]
                # Your custom evaluation here
                score = evaluate_model(X_sel, y)
                return 1.0 - score  # Minimize
            
            # Use custom objective
            result = toolbox.optimize(
                'GWO',
                objective_function=custom_objective,
                dimensions=X.shape[1],
                lower_bound=0.0,
                upper_bound=1.0
            )
            ```
            
            **Recommended For:**
            - Python developers
            - Integration into pipelines
            - Automated workflows
            - Custom objectives
            - Batch processing
            """)
        
        with usage_tabs[2]:
            st.markdown("""
            ### ⌨️ Command Line Interface
            
            **Available Commands:**
            
            ```bash
            # Show help
            python -m mha_toolbox --help
            
            # List all algorithms
            python -m mha_toolbox list
            
            # List by category
            python -m mha_toolbox list --category swarm
            
            # Optimize with specific algorithm
            python -m mha_toolbox optimize --algorithm PSO \\
                --dataset data.csv \\
                --iterations 100 \\
                --population 30 \\
                --output results.csv
            
            # Run benchmark
            python -m mha_toolbox benchmark \\
                --algorithms PSO,GWO,WOA \\
                --dataset data.csv \\
                --runs 10
            
            # Interactive mode
            python -m mha_toolbox interactive
            ```
            
            **Recommended For:**
            - Shell scripting
            - Automation
            - CI/CD pipelines
            - Batch experiments
            - Server environments
            """)
    
    # TAB 5: Technical Details
    with about_tabs[4]:
        st.markdown("""
        ### 🛠️ Technical Details
        
        #### 📦 Dependencies
        
        **Core:**
        - Python 3.8+
        - NumPy ≥ 1.20.0
        - Pandas ≥ 1.3.0
        - scikit-learn ≥ 1.0.0
        
        **Web Interface:**
        - Streamlit ≥ 1.25.0
        - Plotly ≥ 5.14.0
        
        **Optional:**
        - matplotlib ≥ 3.5.0
        - seaborn ≥ 0.12.0
        
        #### ⚙️ System Requirements
        
        | Component | Minimum | Recommended |
        |-----------|---------|-------------|
        | **Python** | 3.8 | 3.10+ |
        | **RAM** | 4 GB | 8 GB+ |
        | **CPU** | 2 cores | 4+ cores |
        | **Disk** | 500 MB | 2 GB+ |
        | **OS** | Windows 10, macOS 10.14, Ubuntu 18.04 | Latest versions |
        
        #### 🔧 Configuration
        
        **Default Settings:**
        - Iterations: 100
        - Population: 30
        - Runs: 3
        - Threshold: 0.5
        - Cross-validation folds: 3
        - N-jobs: 1 (Windows compatibility)
        
        **Storage Locations:**
        - User data: `persistent_state/users/`
        - Sessions: `persistent_state/sessions/`
        - Results: `persistent_state/results/`
        - Temp: `persistent_state/temp_results/`
        
        #### 🚀 Performance
        
        **Typical Processing Times:**
        
        | Dataset Size | Algorithms | Iterations | Time |
        |--------------|------------|------------|------|
        | Small (100×10) | 3 | 100 | 30s - 2min |
        | Medium (1000×30) | 3 | 100 | 3-10min |
        | Large (10000×50) | 3 | 100 | 15-30min |
        
        **Optimization Tips:**
        - Use fewer iterations for testing (50-100)
        - Reduce population for speed (20-30)
        - Single run for quick checks
        - Parallel processing disabled for Windows compatibility
        """)
    
    # TAB 6: License
    with about_tabs[5]:
        st.markdown("""
        ### 📜 License & Credits
        
        #### License
        
        This software is provided under the **MIT License**.
        
        ```
        MIT License
        
        Copyright (c) 2025 MHA Toolbox Contributors
        
        Permission is hereby granted, free of charge, to any person obtaining a copy
        of this software and associated documentation files (the "Software"), to deal
        in the Software without restriction, including without limitation the rights
        to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
        copies of the Software, and to permit persons to whom the Software is
        furnished to do so, subject to the following conditions:
        
        The above copyright notice and this permission notice shall be included in all
        copies or substantial portions of the Software.
        
        THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
        IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
        FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
        ```
        
        #### 🙏 Credits & Acknowledgments
        
        **Core Libraries:**
        - NumPy - Numerical computing
        - Pandas - Data manipulation
        - scikit-learn - Machine learning
        - Streamlit - Web interface
        - Plotly - Interactive visualizations
        
        **Algorithm Inspirations:**
        - Research papers and publications
        - Open-source implementations
        - Academic contributions
        
        #### 📚 References
        
        Key research papers that inspired algorithm implementations available in documentation.
        
        #### 🤝 Contributing
        
        Contributions welcome! Please check GitHub repository for contribution guidelines.
        
        #### 📧 Contact
        
        For issues, suggestions, or questions, please use the GitHub issue tracker.
        """)
    
    # Technical Details
    st.markdown("### 🔧 Technical Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Dependencies:**
        - NumPy, Pandas, SciPy
        - Scikit-learn
        - Plotly, Matplotlib
        - Streamlit (for web UI)
        - OpenPyXL (for Excel)
        """)
    
    with col2:
        st.markdown("""
        **System Requirements:**
        - Python 3.8+
        - 4GB RAM minimum
        - 1GB disk space
        - Web browser (for UI)
        """)
    
    # License and Credits
    st.markdown("### 📜 License & Credits")
    
    st.markdown("""
    <div class="info-box">
        <p><strong>License:</strong> MIT License</p>
        <p><strong>Version:</strong> 3.0.0</p>
        <p><strong>Repository:</strong> <a href="https://github.com/Achyut103040/MHA-Algorithm">GitHub</a></p>
        <p><strong>Documentation:</strong> See README.md for complete documentation</p>
    </div>
    """, unsafe_allow_html=True)


def show_settings():
    """Advanced settings page with comprehensive profile information"""
    st.markdown("## ⚙️ Settings & Profile")
    
    if not st.session_state.user_profile:
        st.warning("⚠️ No user profile loaded")
        return
    
    profile = st.session_state.user_profile
    
    # Tabs for different settings sections
    tab1, tab2, tab3, tab4 = st.tabs(["👤 Profile", "🎨 Preferences", "📊 Statistics", "🔧 System"])
    
    with tab1:
        st.markdown("### 👤 User Profile Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Account Details:**")
            st.text_input("Username", value=profile.username, disabled=True)
            st.text_input("User ID", value=profile.user_id, disabled=True)
            st.text_input("Created", value=profile.created_at[:19], disabled=True)
            st.text_input("Last Active", value=profile.last_active[:19], disabled=True)
        
        with col2:
            st.markdown("**Session Information:**")
            st.text_input("Session ID", value=st.session_state.session_id[:16] + "...", disabled=True)
            st.text_input("Current System", value=platform.node(), disabled=True)
            created_system = profile.preferences.get('created_system', 'Unknown')
            st.text_input("Created On System", value=created_system, disabled=True)
            
            # System change warning
            last_system = profile.preferences.get('last_system', platform.node())
            if last_system != platform.node():
                st.warning(f"⚠️ System changed from {last_system}")
        
        # Update system tracking
        if last_system != platform.node():
            profile.track_system_change(platform.node())
            save_profile(profile)
        
        st.markdown("---")
        st.markdown("**🔐 Security:**")
        
        with st.expander("Change Password"):
            current_pw = st.text_input("Current Password", type="password", key="current_pw")
            new_pw = st.text_input("New Password", type="password", key="new_pw")
            confirm_pw = st.text_input("Confirm New Password", type="password", key="confirm_pw")
            
            if st.button("🔒 Update Password"):
                if current_pw and new_pw and confirm_pw:
                    if authenticate_user(profile.username, current_pw):
                        if new_pw == confirm_pw:
                            import hashlib
                            new_hash = hashlib.sha256(new_pw.encode()).hexdigest()
                            profile.update_preference('password_hash', new_hash)
                            save_profile(profile)
                            st.success("✅ Password updated successfully!")
                        else:
                            st.error("❌ New passwords don't match!")
                    else:
                        st.error("❌ Current password is incorrect!")
    
    with tab2:
        st.markdown("### 🎨 User Preferences")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Optimization Defaults:**")
            default_iterations = st.number_input(
                "Default Iterations", 
                10, 1000, 
                profile.preferences.get('default_iterations', 100), 
                10
            )
            default_population = st.number_input(
                "Default Population", 
                10, 200, 
                profile.preferences.get('default_population', 30), 
                5
            )
            
            st.markdown("**Export Settings:**")
            auto_export = st.checkbox(
                "Auto-export results", 
                value=profile.preferences.get('auto_export', False)
            )
            export_format = st.selectbox(
                "Export Format", 
                ["csv", "json", "excel"],
                index=["csv", "json", "excel"].index(profile.preferences.get('export_format', 'csv'))
            )
        
        with col2:
            st.markdown("**Algorithm Preferences:**")
            
            # Favorite algorithms
            all_algos = st.session_state.toolbox.list_algorithms()
            current_favorites = profile.preferences.get('favorite_algorithms', [])
            
            favorite_algos = st.multiselect(
                "Favorite Algorithms",
                all_algos,
                default=current_favorites,
                help="Quick access to your favorite algorithms"
            )
            
            st.markdown("**Notifications:**")
            enable_notifications = st.checkbox(
                "Enable Notifications",
                value=profile.preferences.get('enable_notifications', False)
            )
            notification_email = st.text_input(
                "Notification Email",
                value=profile.preferences.get('notification_email', '') or ''
            )
        
        if st.button("💾 Save Preferences", type="primary"):
            profile.update_preferences({
                'default_iterations': default_iterations,
                'default_population': default_population,
                'auto_export': auto_export,
                'export_format': export_format,
                'favorite_algorithms': favorite_algos,
                'enable_notifications': enable_notifications,
                'notification_email': notification_email
            })
            save_profile(profile)
            st.success("✅ Preferences saved successfully!")
    
    with tab3:
        st.markdown("### 📊 Usage Statistics")
        
        # Get statistics summary
        stats = profile.get_statistics_summary()
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Experiments", stats['total_experiments'])
            st.metric("Total Sessions", stats['total_sessions'])
        
        with col2:
            st.metric("Runtime (hours)", f"{stats['total_runtime_hours']:.2f}")
            st.metric("Datasets Processed", stats['datasets_processed'])
        
        with col3:
            st.metric("Algorithms Used", stats['unique_algorithms_used'])
            st.metric("Best Accuracy", f"{stats['best_accuracy']:.4f}")
        
        with col4:
            st.metric("Systems Used", stats['systems_used'])
            st.metric("Most Used", stats['most_used_algorithm'])
        
        # Algorithm usage chart
        st.markdown("---")
        st.markdown("**📈 Algorithm Usage History:**")
        
        algo_stats = profile.preferences.get('algorithms_used', {})
        if algo_stats:
            # Create DataFrame
            df_algos = pd.DataFrame([
                {
                    'Algorithm': algo,
                    'Count': data['count'],
                    'Avg Accuracy': data['avg_accuracy'],
                    'Total Runtime (s)': data['total_runtime']
                }
                for algo, data in algo_stats.items()
            ]).sort_values('Count', ascending=False)
            
            # Bar chart
            fig = px.bar(df_algos.head(10), x='Algorithm', y='Count',
                        title="Top 10 Most Used Algorithms",
                        color='Avg Accuracy',
                        color_continuous_scale='Viridis')
            st.plotly_chart(fig, width='stretch')
            
            # Full table
            with st.expander("📋 View Full Statistics"):
                st.dataframe(df_algos, width='stretch')
        else:
            st.info("No algorithm usage data yet. Run some optimizations!")
        
        # Recent algorithms
        st.markdown("---")
        st.markdown("**🕐 Recent Algorithm Usage:**")
        last_10 = profile.preferences.get('last_10_algorithms', [])
        if last_10:
            cols = st.columns(min(len(last_10), 5))
            for idx, algo in enumerate(last_10[:5]):
                with cols[idx]:
                    st.markdown(f"""
                    <div class="algo-card">
                        {idx+1}. {algo.upper()}
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("No recent algorithm usage")
    
    with tab4:
        st.markdown("### � System Maintenance")
        
        # System information
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📊 System Statistics:**")
            
            profiles = list_profiles()
            st.metric("Total Users", len(profiles))
            
            results_dir = Path("results")
            if results_dir.exists():
                result_files = list(results_dir.glob("*.json"))
                st.metric("Stored Results", len(result_files))
            else:
                st.metric("Stored Results", 0)
            
            sessions_dir = Path("persistent_state/sessions")
            if sessions_dir.exists():
                session_files = list(sessions_dir.glob("*.json"))
                st.metric("Active Sessions", len(session_files))
            else:
                st.metric("Active Sessions", 0)
        
        with col2:
            st.markdown("**🔧 Maintenance Actions:**")
            
            if st.button("🧹 Cleanup Expired Sessions", width='stretch'):
                cleaned = cleanup_expired_sessions()
                st.success(f"✅ Cleaned up {cleaned} expired sessions")
            
            if st.button("📊 Export Profile Data", width='stretch'):
                import json
                profile_data = profile.to_dict()
                json_str = json.dumps(profile_data, indent=2, default=str)
                st.download_button(
                    "📥 Download Profile JSON",
                    json_str,
                    f"profile_{profile.username}_{datetime.now().strftime('%Y%m%d')}.json",
                    "application/json"
                )
            
            if st.button("🔄 Reset Statistics", width='stretch'):
                if st.checkbox("Confirm reset (this cannot be undone)"):
                    profile.preferences['algorithms_used'] = {}
                    profile.preferences['total_runtime_seconds'] = 0
                    profile.preferences['datasets_processed'] = 0
                    profile.preferences['last_10_algorithms'] = []
                    save_profile(profile)
                    st.success("✅ Statistics reset!")
        
        # System history
        st.markdown("---")
        st.markdown("**🖥️ System Access History:**")
        
        system_history = profile.preferences.get('system_history', [])
        if system_history:
            df_systems = pd.DataFrame(system_history[-10:])  # Last 10
            st.dataframe(df_systems, width='stretch')
        else:
            st.info("No system history yet")


if __name__ == "__main__":
    main()
