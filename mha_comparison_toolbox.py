"""
MHA Comparison Toolbox - Complete System
========================================

A comprehensive metaheuristic algorithm comparison system with:
- Multiple algorithm comparison
- Feature Selection (default)
- Feature Optimization  
- Hyperparameter Tuning
- Unified plotting and analysis
- Standardized output format
- Web interface capability

Author: MHA Development Team
Version: 2.0.0
"""

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
    
    def _run_hyperparameter_tuning(self, alg_name, params):
        """Run hyperparameter tuning task"""
        # Simplified hyperparameter tuning simulation
        # In practice, this would optimize ML model hyperparameters
        
        def hyperparameter_objective(x):
            # Simulate tuning Random Forest parameters
            n_estimators = max(10, int(x[0] * 100))
            max_depth = max(3, int(x[1] * 20))
            
            rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
            scores = cross_val_score(rf, self.X, self.y, cv=3)
            return 1 - np.mean(scores)  # Minimize error
        
        import mha_toolbox as mha
        result = mha.optimize(alg_name, objective_function=hyperparameter_objective, dimensions=2, bounds=(0, 1), **params)
        
        best_params = result.best_solution_
        n_estimators = max(10, int(best_params[0] * 100))
        max_depth = max(3, int(best_params[1] * 20))
        
        return {
            'run': 1,
            'best_fitness': result.best_fitness_,
            'optimized_params': {'n_estimators': n_estimators, 'max_depth': max_depth},
            'convergence_curve': result.global_fitness_,
            'execution_time': result.execution_time_,
            'final_accuracy': 1 - result.best_fitness_
        }
    
    def _run_feature_optimization(self, alg_name, params):
        """Run feature optimization task"""
        # Feature optimization optimizes feature weights/transformations
        
        def feature_weight_objective(weights):
            # Apply weights to features
            X_weighted = self.X * weights
            
            # Train classifier and get cross-validation score
            rf = RandomForestClassifier(n_estimators=50, random_state=42)
            scores = cross_val_score(rf, X_weighted, self.y, cv=3)
            return 1 - np.mean(scores)  # Minimize error
        
        import mha_toolbox as mha
        result = mha.optimize(alg_name, objective_function=feature_weight_objective, 
                            dimensions=self.X.shape[1], bounds=(0, 1), **params)
        
        return {
            'run': 1,
            'best_fitness': result.best_fitness_,
            'optimized_weights': result.best_solution_,
            'convergence_curve': result.global_fitness_,
            'execution_time': result.execution_time_,
            'final_accuracy': 1 - result.best_fitness_
        }
    
    def generate_comparison_plots(self, save_plots=True):
        """Generate comprehensive comparison plots"""
        if not self.results:
            print("❌ No results to plot. Run comparison first.")
            return
            
        print("\n📊 Generating comparison plots...")
        
        # Set up plot style
        plt.style.use('seaborn-v0_8')
        fig = plt.figure(figsize=(20, 12))
        
        # 1. Convergence Comparison Plot
        ax1 = plt.subplot(2, 3, 1)
        self._plot_convergence_comparison(ax1)
        
        # 2. Box Plot of Fitness Values
        ax2 = plt.subplot(2, 3, 2)
        self._plot_fitness_boxplot(ax2)
        
        # 3. Algorithm Performance Bar Chart
        ax3 = plt.subplot(2, 3, 3)
        self._plot_performance_bars(ax3)
        
        # 4. Execution Time Comparison
        ax4 = plt.subplot(2, 3, 4)
        self._plot_time_comparison(ax4)
        
        # 5. Task-specific plot
        ax5 = plt.subplot(2, 3, 5)
        if self.task_type == 'feature_selection':
            self._plot_feature_selection_analysis(ax5)
        elif self.task_type == 'hyperparameter_tuning':
            self._plot_hyperparameter_analysis(ax5)
        else:
            self._plot_feature_optimization_analysis(ax5)
            
        # 6. Statistical Summary
        ax6 = plt.subplot(2, 3, 6)
        self._plot_statistical_summary(ax6)
        
        plt.tight_layout()
        
        if save_plots:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"comparison_results_{self.task_type}_{timestamp}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"📁 Plots saved as: {filename}")
            
        plt.show()
        
    def _plot_convergence_comparison(self, ax):
        """Plot convergence curves for all algorithms"""
        ax.set_title('🔄 Convergence Comparison', fontsize=14, fontweight='bold')
        
        for alg_name, results in self.results.items():
            # Get the best run's convergence curve
            best_run = min(results['runs'], key=lambda x: x['best_fitness'])
            convergence = best_run['convergence_curve']
            
            ax.plot(convergence, label=alg_name.upper(), linewidth=2, marker='o', markersize=3)
            
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Fitness Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    def _plot_fitness_boxplot(self, ax):
        """Plot box plot of fitness values across runs"""
        ax.set_title('📦 Fitness Distribution', fontsize=14, fontweight='bold')
        
        data = []
        labels = []
        
        for alg_name, results in self.results.items():
            fitnesses = [run['best_fitness'] for run in results['runs']]
            data.append(fitnesses)
            labels.append(alg_name.upper())
            
        ax.boxplot(data, labels=labels)
        ax.set_ylabel('Best Fitness')
        ax.tick_params(axis='x', rotation=45)
        
    def _plot_performance_bars(self, ax):
        """Plot performance comparison bars"""
        ax.set_title('🏆 Algorithm Performance', fontsize=14, fontweight='bold')
        
        algorithms = list(self.results.keys())
        mean_fitness = [self.results[alg]['statistics']['mean_fitness'] for alg in algorithms]
        std_fitness = [self.results[alg]['statistics']['std_fitness'] for alg in algorithms]
        
        bars = ax.bar([alg.upper() for alg in algorithms], mean_fitness, 
                     yerr=std_fitness, capsize=5, alpha=0.8)
        
        # Color bars based on performance
        colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(bars)))
        for bar, color in zip(bars, colors):
            bar.set_color(color)
            
        ax.set_ylabel('Mean Fitness ± Std')
        ax.tick_params(axis='x', rotation=45)
        
    def _plot_time_comparison(self, ax):
        """Plot execution time comparison"""
        ax.set_title('⏱️ Execution Time', fontsize=14, fontweight='bold')
        
        algorithms = list(self.results.keys())
        mean_times = [self.results[alg]['statistics']['mean_time'] for alg in algorithms]
        
        bars = ax.bar([alg.upper() for alg in algorithms], mean_times, alpha=0.8, color='lightblue')
        
        ax.set_ylabel('Mean Time (seconds)')
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, time_val in zip(bars, mean_times):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{time_val:.2f}s', ha='center', va='bottom', fontsize=10)
    
    def _plot_feature_selection_analysis(self, ax):
        """Plot feature selection specific analysis"""
        ax.set_title('🔍 Feature Selection Analysis', fontsize=14, fontweight='bold')
        
        algorithms = list(self.results.keys())
        mean_features = [self.results[alg]['statistics']['mean_features'] for alg in algorithms]
        mean_accuracy = [self.results[alg]['statistics']['mean_accuracy'] for alg in algorithms]
        
        # Scatter plot: features vs accuracy
        colors = plt.cm.viridis(np.linspace(0, 1, len(algorithms)))
        for i, alg in enumerate(algorithms):
            ax.scatter(mean_features[i], mean_accuracy[i], 
                      s=100, c=[colors[i]], label=alg.upper(), alpha=0.8)
        
        ax.set_xlabel('Mean Selected Features')
        ax.set_ylabel('Mean Accuracy')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    def _plot_hyperparameter_analysis(self, ax):
        """Plot hyperparameter tuning analysis"""
        ax.set_title('⚙️ Hyperparameter Optimization', fontsize=14, fontweight='bold')
        
        algorithms = list(self.results.keys())
        accuracies = [self.results[alg]['statistics'].get('mean_accuracy', 0) for alg in algorithms]
        
        bars = ax.bar([alg.upper() for alg in algorithms], accuracies, alpha=0.8, color='orange')
        ax.set_ylabel('Optimized Model Accuracy')
        ax.tick_params(axis='x', rotation=45)
        
    def _plot_feature_optimization_analysis(self, ax):
        """Plot feature optimization analysis"""
        ax.set_title('🎯 Feature Optimization', fontsize=14, fontweight='bold')
        
        algorithms = list(self.results.keys())
        accuracies = [self.results[alg]['statistics'].get('mean_accuracy', 0) for alg in algorithms]
        
        bars = ax.bar([alg.upper() for alg in algorithms], accuracies, alpha=0.8, color='green')
        ax.set_ylabel('Optimized Feature Accuracy')
        ax.tick_params(axis='x', rotation=45)
        
    def _plot_statistical_summary(self, ax):
        """Plot statistical significance summary"""
        ax.set_title('📈 Statistical Summary', fontsize=14, fontweight='bold')
        
        # Create a summary table
        algorithms = list(self.results.keys())
        summary_data = []
        
        for alg in algorithms:
            stats = self.results[alg]['statistics']
            summary_data.append([
                alg.upper(),
                f"{stats['mean_fitness']:.4f}",
                f"{stats['std_fitness']:.4f}",
                f"{stats['mean_time']:.2f}s"
            ])
        
        # Create table
        table = ax.table(cellText=summary_data,
                        colLabels=['Algorithm', 'Mean Fitness', 'Std Fitness', 'Mean Time'],
                        cellLoc='center',
                        loc='center',
                        bbox=[0, 0, 1, 1])
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        
        # Style the table
        for i in range(len(algorithms) + 1):
            for j in range(4):
                cell = table[(i, j)]
                if i == 0:  # Header
                    cell.set_facecolor('#4CAF50')
                    cell.set_text_props(weight='bold', color='white')
                else:
                    cell.set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
        
        ax.axis('off')
    
    def save_results(self, filename=None):
        """Save complete results to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"mha_comparison_results_{self.task_type}_{timestamp}.json"
        
        # Prepare data for JSON serialization
        save_data = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'task_type': self.task_type,
                'data_info': self.data_info,
                'total_algorithms': len(self.results)
            },
            'results': {}
        }
        
        # Convert numpy arrays to lists for JSON compatibility
        for alg_name, results in self.results.items():
            save_data['results'][alg_name] = {
                'algorithm': results['algorithm'],
                'parameters': results['parameters'],
                'statistics': results['statistics'],
                'task_type': results['task_type'],
                'runs_summary': {
                    'total_runs': len(results['runs']),
                    'best_fitness': min(run['best_fitness'] for run in results['runs']),
                    'mean_fitness': np.mean([run['best_fitness'] for run in results['runs']]),
                    'convergence_curves': [run['convergence_curve'] for run in results['runs']]
                }
            }
        
        with open(filename, 'w') as f:
            json.dump(save_data, f, indent=2, default=str)
        
        print(f"💾 Results saved to: {filename}")
        return filename
    
    def print_summary(self):
        """Print comprehensive summary of results"""
        if not self.results:
            print("❌ No results to summarize. Run comparison first.")
            return
            
        print("\n" + "="*80)
        print("📊 MHA COMPARISON TOOLBOX - SUMMARY REPORT")
        print("="*80)
        
        print(f"🎯 Task Type: {self.task_type.upper()}")
        print(f"📋 Dataset: {self.data_info['dataset_name']}")
        print(f"📊 Data: {self.data_info['n_samples']} samples, {self.data_info['n_features']} features")
        print(f"🧬 Algorithms Compared: {len(self.results)}")
        
        print(f"\n{'='*80}")
        print("🏆 ALGORITHM RANKINGS")
        print("="*80)
        
        # Sort algorithms by performance
        sorted_algorithms = sorted(self.results.items(), 
                                 key=lambda x: x[1]['statistics']['mean_fitness'])
        
        for rank, (alg_name, results) in enumerate(sorted_algorithms, 1):
            stats = results['statistics']
            print(f"\n🥇 Rank {rank}: {alg_name.upper()}")
            print(f"   📈 Mean Fitness: {stats['mean_fitness']:.6f} ± {stats['std_fitness']:.6f}")
            print(f"   🏃 Best Fitness: {stats['best_fitness']:.6f}")
            print(f"   ⏱️  Mean Time: {stats['mean_time']:.2f}s ± {stats['std_time']:.2f}s")
            
            if self.task_type == 'feature_selection':
                print(f"   🔍 Mean Features: {stats['mean_features']:.1f} ± {stats['std_features']:.1f}")
                print(f"   🎯 Mean Accuracy: {stats['mean_accuracy']:.4f} ± {stats['std_accuracy']:.4f}")
        
        print(f"\n{'='*80}")
        print("💡 RECOMMENDATIONS")
        print("="*80)
        
        best_alg = sorted_algorithms[0][0]
        fastest_alg = min(self.results.items(), key=lambda x: x[1]['statistics']['mean_time'])[0]
        most_stable = min(self.results.items(), key=lambda x: x[1]['statistics']['std_fitness'])[0]
        
        print(f"🏆 Best Performance: {best_alg.upper()}")
        print(f"⚡ Fastest Algorithm: {fastest_alg.upper()}")
        print(f"📊 Most Stable: {most_stable.upper()}")
        
        if self.task_type == 'feature_selection':
            min_features = min(self.results.items(), 
                             key=lambda x: x[1]['statistics']['mean_features'])[0]
            print(f"🔍 Most Selective: {min_features.upper()}")
        
        print(f"\n{'='*80}")


# Example usage function
def demo_comparison_toolbox():
    """Demo function showing how to use the comparison toolbox"""
    
    print("🧬 MHA COMPARISON TOOLBOX - DEMO")
    print("="*50)
    
    # Load sample data
    from sklearn.datasets import load_breast_cancer
    X, y = load_breast_cancer(return_X_y=True)
    
    # Initialize toolbox
    toolbox = MHAComparisonToolbox()
    
    # Set task type (user choice)
    task_choice = input("\nSelect task type:\n1. Feature Selection (default)\n2. Feature Optimization\n3. Hyperparameter Tuning\nChoice (1-3): ").strip()
    
    if task_choice == '2':
        toolbox.set_task_type('feature_optimization')
    elif task_choice == '3':
        toolbox.set_task_type('hyperparameter_tuning')
    else:
        toolbox.set_task_type('feature_selection')
    
    # Load data
    toolbox.load_data(X, y, "Breast Cancer Dataset")
    
    # Add algorithms for comparison
    algorithm_choice = input("\nSelect algorithms to compare:\n1. PSO vs GWO (quick demo)\n2. PSO vs GWO vs SCA (full demo)\n3. Custom selection\nChoice (1-3): ").strip()
    
    if algorithm_choice == '2':
        toolbox.add_algorithm('pso')
        toolbox.add_algorithm('gwo')
        toolbox.add_algorithm('sca')
    elif algorithm_choice == '3':
        while True:
            alg = input("Enter algorithm name (or 'done' to finish): ").strip().lower()
            if alg == 'done' or not alg:
                break
            toolbox.add_algorithm(alg)
    else:
        # Default: PSO vs GWO
        toolbox.add_algorithm('pso')
        toolbox.add_algorithm('gwo')
    
    # Run comparison
    print(f"\n🚀 Starting comparison...")
    toolbox.run_comparison(max_iterations=30, population_size=20, n_runs=3)
    
    # Generate plots
    toolbox.generate_comparison_plots()
    
    # Save results
    toolbox.save_results()
    
    # Print summary
    toolbox.print_summary()
    
    return toolbox


if __name__ == "__main__":
    # Run the demo
    toolbox = demo_comparison_toolbox()