import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import json
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class MHAComparisonToolbox:
    """
    Complete MHA Comparison Toolbox
    
    Main class that handles algorithm comparison across different tasks:
    - Feature Selection (default)
    - Feature Optimization
    - Hyperparameter Tuning
    """
    
    def __init__(self):
        self.results = {}
        self.algorithms = []
        self.task_type = 'feature_selection'  # default
        self.data_info = {}
        self.plots = {}
        
    def set_task_type(self, task_type):
        """
        Set the type of optimization task
        
        Parameters:
        -----------
        task_type : str
            'feature_selection' (default), 'feature_optimization', 'hyperparameter_tuning'
        """
        valid_tasks = ['feature_selection', 'feature_optimization', 'hyperparameter_tuning']
        if task_type not in valid_tasks:
            raise ValueError(f"Task type must be one of {valid_tasks}")
        self.task_type = task_type
        print(f"✅ Task type set to: {task_type}")
    
    def load_data(self, X, y, dataset_name="custom_dataset"):
        """Load and prepare data for comparison"""
        self.X = X
        self.y = y
        self.dataset_name = dataset_name
        
        # Store data information
        self.data_info = {
            'n_samples': X.shape[0],
            'n_features': X.shape[1],
            'n_classes': len(np.unique(y)) if len(np.unique(y)) < 20 else 'regression',
            'dataset_name': dataset_name,
            'task_type': self.task_type
        }
        
        print(f"📊 Data loaded: {self.data_info['n_samples']} samples, {self.data_info['n_features']} features")
        print(f"🎯 Task: {self.task_type}")
        
    def add_algorithm(self, algorithm_name, algorithm_params=None):
        """Add algorithm to comparison list"""
        if algorithm_params is None:
            algorithm_params = {}
            
        self.algorithms.append({
            'name': algorithm_name,
            'params': algorithm_params
        })
        print(f"➕ Added algorithm: {algorithm_name}")
    
    def run_comparison(self, max_iterations=50, population_size=30, n_runs=3):
        """
        Run comparison across all added algorithms
        
        Parameters:
        -----------
        max_iterations : int
            Maximum iterations for each algorithm
        population_size : int  
            Population size for each algorithm
        n_runs : int
            Number of independent runs for statistical reliability
        """
        
        if not self.algorithms:
            raise ValueError("No algorithms added. Use add_algorithm() first.")
            
        if not hasattr(self, 'X'):
            raise ValueError("No data loaded. Use load_data() first.")
            
        print(f"\n🚀 Starting comparison of {len(self.algorithms)} algorithms")
        print(f"📋 Task: {self.task_type}")
        print(f"🔄 Runs per algorithm: {n_runs}")
        print("="*60)
        
        # Import our toolbox
        import sys
        sys.path.append('.')
        import mha_toolbox as mha
        
        # Run each algorithm
        for i, alg_config in enumerate(self.algorithms):
            alg_name = alg_config['name']
            alg_params = alg_config['params']
            
            print(f"\n🧬 [{i+1}/{len(self.algorithms)}] Running {alg_name.upper()}...")
            
            # Combine parameters
            params = {
                'max_iterations': max_iterations,
                'population_size': population_size,
                'verbose': False,
                **alg_params
            }
            
            # Run multiple times for statistical reliability
            runs_data = []
            for run in range(n_runs):
                print(f"   Run {run+1}/{n_runs}...", end=" ")
                
                start_time = time.time()
                
                try:
                    if self.task_type == 'feature_selection':
                        result = mha.optimize(alg_name, self.X, self.y, **params)
                        
                        # Extract standardized results
                        run_result = {
                            'run': run + 1,
                            'best_fitness': result.best_fitness_,
                            'n_selected_features': result.n_selected_features_,
                            'selected_features': result.best_solution_binary_,
                            'convergence_curve': result.global_fitness_,
                            'execution_time': time.time() - start_time,
                            'final_accuracy': 1 - result.best_fitness_  # Convert fitness to accuracy
                        }
                        
                    elif self.task_type == 'hyperparameter_tuning':
                        # For hyperparameter tuning, we optimize ML model parameters
                        result = self._run_hyperparameter_tuning(alg_name, params)
                        run_result = result
                        
                    elif self.task_type == 'feature_optimization':
                        # For feature optimization, we optimize feature weights
                        result = self._run_feature_optimization(alg_name, params)
                        run_result = result
                    
                    runs_data.append(run_result)
                    print(f"✅ Fitness: {run_result['best_fitness']:.4f}")
                    
                except Exception as e:
                    print(f"❌ Failed: {str(e)}")
                    continue
            
            # Store algorithm results
            if runs_data:
                self.results[alg_name] = {
                    'algorithm': alg_name,
                    'parameters': params,
                    'runs': runs_data,
                    'statistics': self._calculate_statistics(runs_data),
                    'task_type': self.task_type
                }
                
                print(f"   📊 Average fitness: {self.results[alg_name]['statistics']['mean_fitness']:.4f} ± {self.results[alg_name]['statistics']['std_fitness']:.4f}")
            
        print(f"\n✅ Comparison completed! Results for {len(self.results)} algorithms.")
        
    def _calculate_statistics(self, runs_data):
        """Calculate statistics across multiple runs"""
        fitnesses = [run['best_fitness'] for run in runs_data]
        times = [run['execution_time'] for run in runs_data]
        
        stats = {
            'mean_fitness': np.mean(fitnesses),
            'std_fitness': np.std(fitnesses),
            'best_fitness': np.min(fitnesses),
            'worst_fitness': np.max(fitnesses),
            'mean_time': np.mean(times),
            'std_time': np.std(times),
            'total_runs': len(runs_data)
        }
        
        # Add task-specific statistics
        if self.task_type == 'feature_selection':
            n_features = [run['n_selected_features'] for run in runs_data]
            accuracies = [run['final_accuracy'] for run in runs_data]
            stats.update({
                'mean_features': np.mean(n_features),
                'std_features': np.std(n_features),
                'mean_accuracy': np.mean(accuracies),
                'std_accuracy': np.std(accuracies)
            })
            
        return stats
