"""
Advanced Visualization Demo for MHA Toolbox
===========================================

This script demonstrates all the advanced visualization and statistical analysis features
of the MHA Toolbox, including:
- Convergence analysis with multiple plots
- Statistical distribution analysis
- Exploration-exploitation behavior
- Algorithm comparison
- Search trajectory visualization
- Performance heatmaps and statistical tests

Run this script to see all visualization capabilities in action.
"""

import numpy as np
import sys
import os

# Add the parent directory to the path so we can import mha_toolbox
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import mha_toolbox as mha
from mha_toolbox.utils.visualizations import AdvancedVisualizer
from sklearn.datasets import load_breast_cancer, load_iris, make_classification
import matplotlib.pyplot as plt

def create_test_function():
    """Create a test optimization function (Sphere function)."""
    def sphere_function(x):
        return np.sum(x**2)
    return sphere_function

def demo_single_algorithm_visualization():
    """Demonstrate visualization for a single algorithm."""
    print("\n" + "="*70)
    print("🎯 SINGLE ALGORITHM VISUALIZATION DEMO")
    print("="*70)
    
    # Generate synthetic data for feature selection
    X, y = make_classification(n_samples=200, n_features=20, n_informative=10, 
                              n_redundant=5, n_clusters_per_class=1, random_state=42)
    
    print("Running Aquila Optimizer (AO) for feature selection...")
    
    # Run AO algorithm
    result = mha.ao(X, y, population_size=20, max_iterations=50, verbose=True)
    
    print("\n📊 Basic Convergence Plot:")
    result.plot_convergence()
    
    print("\n📈 Advanced Convergence Analysis:")
    result.plot_advanced(plot_type='convergence')
    
    print("\n🔍 Exploration-Exploitation Analysis:")
    result.plot_advanced(plot_type='exploration')
    
    print("\n📊 Statistical Analysis:")
    result.plot_advanced(plot_type='statistical')
    
    print("\n🌟 Search Trajectory (2D):")
    result.plot_advanced(plot_type='trajectory')
    
    print("\n📋 Comprehensive Statistics:")
    stats = result.get_statistics()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key:<25}: {value:.6f}")
        else:
            print(f"  {key:<25}: {value}")
    
    return result

def demo_algorithm_comparison():
    """Demonstrate comparison between multiple algorithms."""
    print("\n" + "="*70)
    print("⚔️  ALGORITHM COMPARISON DEMO")
    print("="*70)
    
    # Create a simple optimization problem
    objective_func = create_test_function()
    
    print("Running multiple algorithms for comparison...")
    
    # Run PSO
    print("\n🔵 Running PSO...")
    result_pso = mha.pso(objective_function=objective_func, dimensions=10, 
                        population_size=20, max_iterations=50, verbose=False)
    
    # Run SCA 
    print("🟠 Running SCA...")
    result_sca = mha.sca(objective_function=objective_func, dimensions=10,
                        population_size=20, max_iterations=50, verbose=False)
    
    # Run AO
    print("🟢 Running AO...")
    result_ao = mha.ao(objective_function=objective_func, dimensions=10,
                      population_size=20, max_iterations=50, verbose=False)
    
    print("\n📊 Creating Comparison Visualizations...")
    
    # Create comparison visualizer
    visualizer = AdvancedVisualizer([result_pso, result_sca, result_ao])
    
    print("\n1️⃣ Convergence Comparison:")
    visualizer.convergence_plot()
    
    print("\n2️⃣ Statistical Box Plot Comparison:")
    visualizer.box_plot()
    
    print("\n3️⃣ Exploration-Exploitation Comparison:")
    visualizer.exploration_exploitation_plot()
    
    print("\n4️⃣ Comprehensive Statistical Analysis:")
    visualizer.statistical_analysis_plot()
    
    print("\n5️⃣ Search Trajectory Comparison:")
    visualizer.search_trajectory_plot(dimensions=2)
    
    # Direct comparison between two algorithms
    print("\n🆚 Direct PSO vs AO Comparison:")
    comparison = result_pso.compare_with(result_ao)
    
    print("\nComparison Results:")
    for key, value in comparison.items():
        print(f"  {key:<20}: {value}")
    
    return [result_pso, result_sca, result_ao]

def demo_feature_selection_analysis():
    """Demonstrate feature selection specific visualizations."""
    print("\n" + "="*70)
    print("🎯 FEATURE SELECTION ANALYSIS DEMO")
    print("="*70)
    
    # Load real datasets
    print("Loading breast cancer dataset...")
    X, y = load_breast_cancer(return_X_y=True)
    
    print(f"Dataset info: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Run multiple algorithms on feature selection
    print("\n🔴 Running AO for feature selection...")
    result_ao = mha.ao(X, y, population_size=15, max_iterations=30, verbose=False)
    
    print("🔵 Running PSO for feature selection...")
    result_pso = mha.pso(X, y, population_size=15, max_iterations=30, verbose=False)
    
    print("🟡 Running SCA for feature selection...")
    result_sca = mha.sca(X, y, population_size=15, max_iterations=30, verbose=False)
    
    # Analyze results
    print("\n📊 Feature Selection Results:")
    for result in [result_ao, result_pso, result_sca]:
        print(f"\n{result.algorithm_name}:")
        print(f"  Selected Features: {result.n_selected_features}/{len(result.best_solution)}")
        print(f"  Best Fitness (Error Rate): {result.best_fitness:.4f}")
        print(f"  Execution Time: {result.execution_time:.3f}s")
    
    # Create comprehensive comparison
    print("\n📈 Creating Feature Selection Comparison Visualizations...")
    
    visualizer = AdvancedVisualizer([result_ao, result_pso, result_sca])
    
    print("\n1️⃣ Convergence Analysis:")
    visualizer.convergence_plot(show_stats=True)
    
    print("\n2️⃣ Algorithm Performance Comparison:")
    visualizer.box_plot()
    
    print("\n3️⃣ Statistical Analysis:")
    visualizer.statistical_analysis_plot()
    
    return [result_ao, result_pso, result_sca]

def demo_function_optimization_analysis():
    """Demonstrate function optimization with various benchmark functions."""
    print("\n" + "="*70)
    print("🎯 FUNCTION OPTIMIZATION ANALYSIS DEMO")
    print("="*70)
    
    # Define different test functions
    def sphere(x):
        return np.sum(x**2)
    
    def rosenbrock(x):
        return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)
    
    def rastrigin(x):
        A = 10
        n = len(x)
        return A * n + np.sum(x**2 - A * np.cos(2 * np.pi * x))
    
    functions = {
        'Sphere': sphere,
        'Rosenbrock': rosenbrock,
        'Rastrigin': rastrigin
    }
    
    results = {}
    
    for func_name, func in functions.items():
        print(f"\n🎯 Optimizing {func_name} function...")
        
        # Run AO on the function
        result = mha.ao(objective_function=func, dimensions=10, 
                       population_size=20, max_iterations=40, verbose=False)
        
        results[func_name] = result
        
        print(f"  Best fitness: {result.best_fitness:.6f}")
        print(f"  Execution time: {result.execution_time:.3f}s")
    
    # Create comprehensive analysis
    print("\n📊 Function Optimization Analysis:")
    
    # Individual analysis for each function
    for func_name, result in results.items():
        print(f"\n📈 {func_name} Function Analysis:")
        result.plot_convergence(title=f'{func_name} Function - AO Convergence')
        
        # Get detailed statistics
        stats = result.get_statistics()
        print(f"  Improvement ratio: {stats['improvement_ratio']:.4f}")
        print(f"  Convergence rate: {stats['convergence_rate']:.6f}")
        print(f"  Early convergence: {stats.get('early_convergence', 'N/A')}")
    
    # Compare all function optimizations
    print("\n⚖️ Function Comparison:")
    all_results = list(results.values())
    visualizer = AdvancedVisualizer(all_results)
    visualizer.convergence_plot()
    visualizer.box_plot()
    
    return results

def demo_advanced_statistical_analysis():
    """Demonstrate advanced statistical analysis features."""
    print("\n" + "="*70)
    print("📊 ADVANCED STATISTICAL ANALYSIS DEMO")
    print("="*70)
    
    # Run multiple independent runs of the same algorithm for statistical analysis
    print("Running multiple independent runs for statistical significance...")
    
    objective_func = create_test_function()
    results = []
    
    for run in range(5):
        print(f"  Run {run + 1}/5...")
        result = mha.ao(objective_function=objective_func, dimensions=10,
                       population_size=15, max_iterations=30, verbose=False)
        results.append(result)
    
    print("\n📈 Statistical Analysis of Multiple Runs:")
    
    # Extract final fitness values
    final_fitness = [result.best_fitness for result in results]
    execution_times = [result.execution_time for result in results]
    
    print(f"  Mean fitness: {np.mean(final_fitness):.6f} ± {np.std(final_fitness):.6f}")
    print(f"  Best fitness: {np.min(final_fitness):.6f}")
    print(f"  Worst fitness: {np.max(final_fitness):.6f}")
    print(f"  Mean execution time: {np.mean(execution_times):.3f}s ± {np.std(execution_times):.3f}s")
    
    # Create statistical visualization
    visualizer = AdvancedVisualizer(results)
    
    print("\n📊 Multi-run Statistical Analysis:")
    visualizer.convergence_plot(show_stats=True)
    visualizer.statistical_analysis_plot()
    
    return results

def main():
    """Main demo function."""
    print("🚀 MHA TOOLBOX ADVANCED VISUALIZATION DEMO")
    print("=" * 70)
    print("This demo showcases all advanced visualization and analysis features.")
    print("Each section will display different types of plots and statistics.")
    print("\nPress Enter to continue through each demo section...")
    input()
    
    try:
        # Demo 1: Single algorithm analysis
        result1 = demo_single_algorithm_visualization()
        input("\nPress Enter to continue to algorithm comparison...")
        
        # Demo 2: Algorithm comparison
        results2 = demo_algorithm_comparison()
        input("\nPress Enter to continue to feature selection analysis...")
        
        # Demo 3: Feature selection analysis
        results3 = demo_feature_selection_analysis()
        input("\nPress Enter to continue to function optimization analysis...")
        
        # Demo 4: Function optimization analysis
        results4 = demo_function_optimization_analysis()
        input("\nPress Enter to continue to statistical analysis...")
        
        # Demo 5: Advanced statistical analysis
        results5 = demo_advanced_statistical_analysis()
        
        print("\n" + "="*70)
        print("🎉 DEMO COMPLETED SUCCESSFULLY!")
        print("="*70)
        print("All visualization features have been demonstrated.")
        print("Check the 'results' folder for automatically saved results.")
        print("You can now use these visualization methods in your own projects!")
        
    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        print("Please check that all dependencies are installed:")
        print("  - numpy, matplotlib, seaborn, scikit-learn, pandas, scipy")

if __name__ == "__main__":
    main()