"""
Create Sample CSV Session for Testing
====================================

Creates a sample session to test the CSV dashboard functionality.
"""

import sys
from pathlib import Path
import json
from datetime import datetime
import numpy as np

# Add the parent directory to sys.path
sys.path.append(str(Path(__file__).parent))

from mha_toolbox.csv_session_manager import ComprehensiveCSVManager

def create_sample_session():
    """Create a sample session for testing"""
    
    print("Creating sample CSV session...")
    
    # Initialize CSV manager
    csv_manager = ComprehensiveCSVManager()
    
    # Create a test session
    session_id = csv_manager.initialize_session("Sample_Test_Session", "Breast Cancer")
    print(f"Created session: {session_id}")
    
    # Sample algorithms with different performance levels
    algorithms_data = [
        {
            'name': 'PSO',
            'best_fitness': 0.05,
            'execution_time': 25.5,
            'iterations': 50
        },
        {
            'name': 'GA', 
            'best_fitness': 0.08,
            'execution_time': 32.1,
            'iterations': 50
        },
        {
            'name': 'SMA',
            'best_fitness': 0.12,
            'execution_time': 28.7,
            'iterations': 50
        }
    ]
    
    # Add algorithms to session
    for i, alg_data in enumerate(algorithms_data):
        algorithm_results = {
            'best_fitness': alg_data['best_fitness'],
            'total_iterations': alg_data['iterations'],
            'execution_time': alg_data['execution_time'],
            'statistics': {
                'mean_fitness': alg_data['best_fitness'] + 0.02,
                'std_fitness': 0.01,
                'total_runs': 3
            }
        }
        
        # Create sample convergence data
        np.random.seed(i)
        convergence_curve = []
        current_fitness = 1.0
        
        for j in range(alg_data['iterations']):
            # Simulate convergence with diminishing improvement
            improvement = np.random.exponential(0.02) * np.exp(-j/15)
            current_fitness = max(alg_data['best_fitness'], current_fitness - improvement)
            convergence_curve.append(current_fitness)
        
        # Add to session
        success = csv_manager.add_algorithm_to_session(
            alg_data['name'], 
            algorithm_results, 
            convergence_curve
        )
        
        if success:
            print(f"✅ Added {alg_data['name']} to session")
        else:
            print(f"❌ Failed to add {alg_data['name']} to session")
    
    # Export session summary
    summary_file = csv_manager.export_session_summary_csv()
    print(f"📊 Session summary exported to: {summary_file}")
    
    print(f"\n🎉 Sample session created successfully!")
    print(f"📁 Session directory: {csv_manager.session_dir}")
    print(f"📋 Session contains {len(algorithms_data)} algorithms")
    
    return csv_manager

if __name__ == "__main__":
    create_sample_session()