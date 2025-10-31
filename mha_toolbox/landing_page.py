"""
Landing Page for MHA Toolbox
============================

Professional welcome and introduction page for the MHA Comparison Toolbox
"""

import streamlit as st
from datetime import datetime


class LandingPage:
    """Landing page component for MHA Toolbox"""
    
    def __init__(self):
        self.version = "2.0.0"
        self.last_updated = "October 2025"
    
    def display_landing_page(self):
        """Display the complete landing page"""
        
        # Hero Section
        self._display_hero_section()
        
        # Feature Highlights
        self._display_features()
        
        # Quick Stats
        self._display_quick_stats()
        
        # Get Started Section
        self._display_get_started()
    
    def _display_hero_section(self):
        """Display hero section with welcome message"""
        
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 3rem 2rem;
            border-radius: 20px;
            margin-bottom: 2rem;
            text-align: center;
            color: white;
        ">
            <h1 style="font-size: 3rem; margin-bottom: 1rem; font-weight: bold;">
                🧬 MHA Toolbox Pro
            </h1>
            <h2 style="font-size: 1.5rem; margin-bottom: 1rem; opacity: 0.9;">
                Professional Metaheuristic Algorithm Comparison Platform
            </h2>
            <p style="font-size: 1.2rem; margin-bottom: 2rem; opacity: 0.8;">
                Compare, analyze, and optimize with 25+ advanced algorithms
            </p>
            <div style="display: flex; justify-content: center; gap: 2rem; flex-wrap: wrap;">
                <div style="background: rgba(255,255,255,0.2); padding: 1rem; border-radius: 10px;">
                    <strong>⚡ Real-time Results</strong>
                </div>
                <div style="background: rgba(255,255,255,0.2); padding: 1rem; border-radius: 10px;">
                    <strong>📊 Advanced Visualization</strong>
                </div>
                <div style="background: rgba(255,255,255,0.2); padding: 1rem; border-radius: 10px;">
                    <strong>🔄 Session Management</strong>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    def _display_features(self):
        """Display key features"""
        
        st.markdown("## 🚀 **Key Features**")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            ### 🧬 **Algorithm Library**
            - **25+ Algorithms** including PSO, GA, SMA, WOA
            - **Hybrid Algorithms** for enhanced performance
            - **Custom Parameters** for fine-tuning
            - **Real-time Tracking** of convergence
            """)
        
        with col2:
            st.markdown("""
            ### 📊 **Advanced Analytics**
            - **Convergence Curves** with interactive plots
            - **Statistical Analysis** and comparisons
            - **Performance Metrics** and rankings
            - **Export Capabilities** to CSV/Excel
            """)
        
        with col3:
            st.markdown("""
            ### 🔄 **Session Management**
            - **Persistent Sessions** across browser restarts
            - **Algorithm Updates** with automatic comparison
            - **Result Preservation** and history tracking
            - **Progress Monitoring** with live updates
            """)
    
    def _display_quick_stats(self):
        """Display quick statistics"""
        
        st.markdown("---")
        st.markdown("## 📈 **Platform Statistics**")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="🧬 **Algorithms Available**",
                value="25+",
                delta="Including Hybrids"
            )
        
        with col2:
            st.metric(
                label="📊 **Dataset Types**",
                value="6",
                delta="Classification & Regression"
            )
        
        with col3:
            st.metric(
                label="⚡ **Real-time Updates**",
                value="10sec",
                delta="Progress Refresh Rate"
            )
        
        with col4:
            st.metric(
                label="📈 **Visualization Types**",
                value="15+",
                delta="Charts & Plots"
            )
    
    def _display_get_started(self):
        """Display get started section"""
        
        st.markdown("---")
        st.markdown("## 🎯 **Get Started in 3 Steps**")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div style="
                background: #f8f9fa;
                padding: 2rem;
                border-radius: 15px;
                text-align: center;
                border-left: 5px solid #28a745;
            ">
                <h3>1️⃣ Select Dataset</h3>
                <p>Choose from 6 built-in datasets or upload your own data for analysis</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style="
                background: #f8f9fa;
                padding: 2rem;
                border-radius: 15px;
                text-align: center;
                border-left: 5px solid #007bff;
            ">
                <h3>2️⃣ Configure Algorithms</h3>
                <p>Select algorithms and customize parameters for optimal performance</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div style="
                background: #f8f9fa;
                padding: 2rem;
                border-radius: 15px;
                text-align: center;
                border-left: 5px solid #ffc107;
            ">
                <h3>3️⃣ Analyze Results</h3>
                <p>View real-time progress and comprehensive analysis with exports</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Call to action
        st.markdown("---")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown("""
            <div style="text-align: center; margin: 2rem 0;">
                <h3>🚀 Ready to start optimizing?</h3>
                <p style="font-size: 1.1rem; color: #666;">
                    Begin your journey with advanced metaheuristic algorithms
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("🎯 **Start Your First Experiment**", type="primary", use_container_width=True):
                st.session_state.current_page = "experiments"
                st.rerun()
    
    def display_welcome_back(self, session_summary):
        """Display welcome back message for returning users"""
        
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
            border-radius: 15px;
            margin-bottom: 2rem;
            color: white;
        ">
            <h2>👋 Welcome Back!</h2>
            <p>Your session is still active with great results waiting for you.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Quick session overview
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("📊 **Active Dataset**", session_summary.get('dataset', 'None'))
        
        with col2:
            st.metric("🧬 **Algorithms Run**", session_summary.get('total_algorithms', 0))
        
        with col3:
            if session_summary.get('best_algorithm'):
                st.metric("🏆 **Best Algorithm**", session_summary['best_algorithm'])
        
        # Quick actions
        st.markdown("### ⚡ **Quick Actions**")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("📊 **View Results**", type="primary"):
                st.session_state.current_page = "results"
                st.rerun()
        
        with col2:
            if st.button("➕ **Add Algorithm**"):
                st.session_state.current_page = "experiments"
                st.session_state.show_algorithm_selector = True
                st.rerun()
        
        with col3:
            if st.button("📈 **View Charts**"):
                st.session_state.current_page = "visualization"
                st.rerun()
        
        with col4:
            if st.button("💾 **Export Data**"):
                st.session_state.current_page = "export"
                st.rerun()
    
    def display_algorithm_showcase(self):
        """Display algorithm showcase section"""
        
        st.markdown("---")
        st.markdown("## 🧬 **Algorithm Showcase**")
        
        # Algorithm categories
        tab1, tab2, tab3, tab4 = st.tabs([
            "🐦 **Bio-Inspired**", 
            "🧠 **Evolutionary**", 
            "🌊 **Physics-Based**", 
            "🔧 **Hybrid**"
        ])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                **🐝 Swarm Intelligence:**
                - Particle Swarm Optimization (PSO)
                - Ant Colony Optimization (ACO)
                - Artificial Bee Colony (ABC)
                - Whale Optimization Algorithm (WOA)
                """)
            
            with col2:
                st.markdown("""
                **🦅 Animal Behavior:**
                - Slime Mould Algorithm (SMA)
                - Grey Wolf Optimizer (GWO)
                - Bat Algorithm (BA)
                - Firefly Algorithm (FA)
                """)
        
        with tab2:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                **🧬 Genetic Algorithms:**
                - Genetic Algorithm (GA)
                - Differential Evolution (DE)
                - Evolution Strategy (ES)
                """)
            
            with col2:
                st.markdown("""
                **🔄 Selection Methods:**
                - Tournament Selection
                - Roulette Wheel Selection
                - Rank-based Selection
                """)
        
        with tab3:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                **⚡ Physics Inspired:**
                - Simulated Annealing (SA)
                - Gravitational Search (GSA)
                - Electromagnetism-like (EM)
                """)
            
            with col2:
                st.markdown("""
                **🌡️ Thermodynamics:**
                - Heat Transfer Search
                - Thermal Exchange Optimization
                """)
        
        with tab4:
            st.markdown("""
            ### 🔧 **Hybrid Algorithms** (Coming Soon!)
            
            Combining the best of multiple algorithms:
            - **PSO-GA Hybrid**: Combines PSO exploration with GA exploitation
            - **WOA-SMA Hybrid**: Merges whale optimization with slime mould intelligence
            - **DE-PSO Hybrid**: Integrates differential evolution with particle swarm
            - **Custom Hybrids**: Create your own algorithm combinations
            
            *Hybrid algorithms are currently in development and will be available soon!*
            """)
    
    def display_tips_and_tricks(self):
        """Display tips and best practices"""
        
        st.markdown("---")
        st.markdown("## 💡 **Tips & Best Practices**")
        
        with st.expander("🎯 **Getting the Best Results**", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                **Parameter Tuning:**
                - Start with default parameters
                - Increase population size for complex problems
                - Use more iterations for better convergence
                - Experiment with different runs
                """)
            
            with col2:
                st.markdown("""
                **Algorithm Selection:**
                - PSO: Good for continuous optimization
                - GA: Excellent for discrete problems
                - SMA: Great for exploration
                - WOA: Balanced exploration/exploitation
                """)
        
        with st.expander("📊 **Understanding Results**", expanded=False):
            st.markdown("""
            **Interpreting Convergence Curves:**
            - **Steep Drop**: Fast initial improvement
            - **Plateau**: Algorithm has converged
            - **Oscillation**: Algorithm still exploring
            - **Multiple Drops**: Algorithm found better regions
            
            **Performance Metrics:**
            - **Best Fitness**: The optimal solution found
            - **Mean Fitness**: Average performance across runs
            - **Standard Deviation**: Consistency of results
            - **Execution Time**: Algorithm efficiency
            """)
        
        with st.expander("🔄 **Session Management**", expanded=False):
            st.markdown("""
            **Session Tips:**
            - Sessions automatically save your progress
            - Change datasets only when starting new experiments
            - Algorithm updates replace previous results if better
            - Export your data regularly for backup
            
            **Performance Optimization:**
            - Close unused browser tabs for better performance
            - Use moderate population sizes (20-50) for faster results
            - Monitor progress with real-time updates
            """)
    
    def display_footer(self):
        """Display footer information"""
        
        st.markdown("---")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            **📦 Version**: {self.version}  
            **📅 Updated**: {self.last_updated}  
            **🔧 Status**: Production Ready
            """)
        
        with col2:
            st.markdown("""
            **📚 Resources**:  
            - [Algorithm Documentation](#)  
            - [Best Practices Guide](#)  
            - [Video Tutorials](#)
            """)
        
        with col3:
            st.markdown("""
            **🤝 Support**:  
            - Report Issues  
            - Feature Requests  
            - Community Forum
            """)
        
        st.markdown("""
        <div style="text-align: center; padding: 2rem; color: #666;">
            <p>🧬 <strong>MHA Toolbox Pro</strong> - Empowering optimization research with cutting-edge algorithms</p>
        </div>
        """, unsafe_allow_html=True)