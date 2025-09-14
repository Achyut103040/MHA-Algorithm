"""
🎯 MHA TOOLBOX - Simple Algorithm Selection & Showcase Guide
==========================================================

Quick guide showing you how to:
1. Select any algorithm
2. Run it with different parameters 
3. Create beautiful visualizations
4. Compare multiple algorithms

All in a simple, easy-to-understand format!
"""

import numpy as np
import sys
import os

# Add the parent directory to the path so we can import mha_toolbox
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import mha_toolbox as mha
from sklearn.datasets import load_breast_cancer, load_wine
import matplotlib.pyplot as plt

def show_available_algorithms():
    """Show all available algorithms."""
    print("🎯 AVAILABLE ALGORITHMS:")
    print("="*50)
    
    # Get all available algorithms
    algorithms = mha.list_algorithms()
    for i, algo in enumerate(algorithms, 1):
        print(f"   {i:2d}. {algo}")
    
    print(f"\n✅ Total: {len(algorithms)} algorithms ready to use!")
    return algorithms

def basic_algorithm_usage():
    """Show the simplest ways to use algorithms."""
    print("\n🚀 BASIC USAGE:")
    print("="*50)
    
    # Load sample data
    X, y = load_breast_cancer(return_X_y=True)
    print(f"📊 Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    
    print("\n💻 Simple algorithm calls:")
    
    # Method 1: Just algorithm name with data
    print("\n   1. Simplest usage:")
    print("      result = mha.ao(X, y)")
    result1 = mha.ao(X, y, population_size=15, max_iterations=20, verbose=False)
    print(f"      ✅ AO: Fitness {result1.best_fitness:.4f}, Features {result1.n_selected_features}/{X.shape[1]}")
    
    # Method 2: With basic parameters
    print("\n   2. With parameters:")
    print("      result = mha.pso(X, y, population_size=25, max_iterations=30)")
    result2 = mha.pso(X, y, population_size=25, max_iterations=30, verbose=False)
    print(f"      ✅ PSO: Fitness {result2.best_fitness:.4f}, Features {result2.n_selected_features}/{X.shape[1]}")
    
    # Method 3: Algorithm-specific parameters
    print("\n   3. Algorithm-specific parameters:")
    print("      result = mha.pso(X, y, c1=2.0, c2=1.5, w=0.9)")
    result3 = mha.pso(X, y, c1=2.0, c2=1.5, w=0.9, population_size=20, max_iterations=25, verbose=False)
    print(f"      ✅ Custom PSO: Fitness {result3.best_fitness:.4f}, Features {result3.n_selected_features}/{X.shape[1]}")
    
    return [result1, result2, result3]

def algorithm_showcase_individual():
    """Show how to showcase a single algorithm."""
    print("\n🎨 INDIVIDUAL ALGORITHM SHOWCASE:")
    print("="*50)
    
    # Load data
    X, y = load_wine(return_X_y=True)
    print(f"🍷 Dataset: Wine Classification ({X.shape[0]} samples, {X.shape[1]} features)")
    
    # Select and run algorithm
    print("\n🔥 Running Aquila Optimizer...")
    result = mha.ao(X, y, population_size=20, max_iterations=30, verbose=False)
    
    # Show results
    print(f"\n📊 RESULTS:")
    print(f"   🎯 Best Fitness: {result.best_fitness:.6f}")
    print(f"   🎯 Selected Features: {result.n_selected_features}/{X.shape[1]} ({result.n_selected_features/X.shape[1]*100:.1f}%)")
    print(f"   🎯 Execution Time: {result.execution_time:.2f} seconds")
    
    # Get detailed statistics
    stats = result.get_statistics()
    print(f"\n📈 STATISTICS:")
    print(f"   📊 Convergence Rate: {stats['convergence_rate']:.6f}")
    print(f"   📊 Improvement Ratio: {stats['improvement_ratio']:.4f}")
    
    # Create visualizations
    print(f"\n🎨 CREATING VISUALIZATIONS:")
    
    # Basic convergence plot
    print("   📈 1. Basic convergence plot...")
    result.plot_convergence(title="AO Convergence on Wine Dataset")
    plt.close()
    
    # Advanced plots
    print("   📊 2. Advanced convergence analysis...")
    result.plot_advanced('convergence')
    plt.close()
    
    print("   ✅ Individual showcase complete!")
    return result

def algorithm_comparison():
    """Show how to compare multiple algorithms."""
    print("\n⚔️ ALGORITHM COMPARISON:")
    print("="*50)
    
    # Load data
    X, y = load_breast_cancer(return_X_y=True)
    print(f"🩺 Dataset: Breast Cancer ({X.shape[0]} samples, {X.shape[1]} features)")
    
    # Select algorithms to compare
    algorithms = [
        ('AO', mha.ao),
        ('PSO', mha.pso),
        ('GWO', mha.gwo),
        ('WOA', mha.woa)
    ]
    
    print(f"\n🔥 Comparing {len(algorithms)} algorithms:")
    
    results = []
    comparison_data = []
    
    # Run all algorithms
    for name, algo_func in algorithms:
        print(f"   Running {name}...")
        result = algo_func(X, y, population_size=20, max_iterations=25, verbose=False)
        results.append(result)
        
        comparison_data.append({
            'Algorithm': name,
            'Fitness': result.best_fitness,
            'Features': result.n_selected_features,
            'Time': result.execution_time
        })
        
        print(f"   ✅ {name}: Fitness {result.best_fitness:.4f}, Features {result.n_selected_features}, Time {result.execution_time:.2f}s")
    
    # Show comparison table
    print(f"\n📊 COMPARISON TABLE:")
    print("-" * 60)
    print(f"{'Algorithm':<10} | {'Fitness':<8} | {'Features':<8} | {'Time':<8}")
    print("-" * 60)
    for data in comparison_data:
        print(f"{data['Algorithm']:<10} | {data['Fitness']:<8.4f} | {data['Features']:>3}/{X.shape[1]:<3} | {data['Time']:<8.2f}")
    
    # Create comparison visualization
    print(f"\n🎨 Creating comparison visualization...")
    from mha_toolbox.utils.visualizations import AdvancedVisualizer
    
    try:
        visualizer = AdvancedVisualizer(results)
        visualizer.convergence_plot()
        plt.close()
        print("   ✅ Comparison visualization created!")
    except Exception as e:
        print(f"   ⚠️ Visualization skipped due to: {e}")
    
    return results

def user_friendly_selection():
    """Show a user-friendly algorithm selection interface."""
    print("\n🎯 USER-FRIENDLY ALGORITHM SELECTION:")
    print("="*50)
    
    # Create algorithm menu
    algorithm_menu = {
        '1': ('Aquila Optimizer (AO)', mha.ao),
        '2': ('Particle Swarm (PSO)', mha.pso),
        '3': ('Grey Wolf (GWO)', mha.gwo),
        '4': ('Whale Optimization (WOA)', mha.woa),
        '5': ('Sine Cosine (SCA)', mha.sca)
    }
    
    print("📋 Algorithm Menu:")
    for key, (name, _) in algorithm_menu.items():
        print(f"   {key}. {name}")
    
    # Simulate user selection
    print(f"\n💻 Example selection interface:")
    print("   user_choice = input('Select algorithm (1-5): ')")
    print("   selected_name, selected_func = algorithm_menu[user_choice]")
    print("   result = selected_func(X, y)")
    
    # Demonstrate with automatic selection
    user_choice = '1'  # Simulate user choosing option 1
    selected_name, selected_func = algorithm_menu[user_choice]
    
    X, y = load_wine(return_X_y=True)
    result = selected_func(X, y, population_size=15, max_iterations=20, verbose=False)
    
    print(f"\n✅ User selected: {selected_name}")
    print(f"   📊 Result: Fitness {result.best_fitness:.4f}, Features {result.n_selected_features}/{X.shape[1]}")
    
    return result

def showcase_with_different_data():
    """Show how to use algorithms with different types of data."""
    print("\n📊 ALGORITHMS WITH DIFFERENT DATA:")
    print("="*50)
    
    # Different datasets
    datasets = [
        ('Breast Cancer', load_breast_cancer),
        ('Wine Classification', load_wine),
    ]
    
    # Test algorithm on different data
    for dataset_name, dataset_loader in datasets:
        print(f"\n🎯 Testing on {dataset_name}:")
        
        X, y = dataset_loader(return_X_y=True)
        print(f"   📏 Data: {X.shape[0]} samples, {X.shape[1]} features, {len(np.unique(y))} classes")
        
        # Run algorithm
        result = mha.ao(X, y, population_size=15, max_iterations=20, verbose=False)
        print(f"   ✅ AO Result: Fitness {result.best_fitness:.4f}, Features {result.n_selected_features}/{X.shape[1]}")

def quick_tips():
    """Show quick tips and best practices."""
    print("\n💡 QUICK TIPS & BEST PRACTICES:")
    print("="*50)
    
    print("🔥 Performance Tips:")
    print("   • Start with population_size=20-50 for most problems")
    print("   • Use max_iterations=30-100 depending on problem complexity")
    print("   • Larger populations = better exploration but slower")
    print("   • More iterations = better convergence but takes longer")
    
    print("\n🎨 Visualization Tips:")
    print("   • Use result.plot_convergence() for quick plots")
    print("   • Use result.plot_advanced('all') for complete analysis")
    print("   • All plots are automatically saved to results/ folder")
    
    print("\n📊 Parameter Tips:")
    print("   • PSO: Adjust c1, c2 (cognitive/social factors), w (inertia)")
    print("   • GA: Adjust crossover_rate, mutation_rate")
    print("   • All algorithms: population_size, max_iterations always work")
    
    print("\n🚀 Usage Patterns:")
    print("   • Quick test: mha.algorithm_name(X, y)")
    print("   • Custom params: mha.algorithm_name(X, y, param1=value1)")
    print("   • Function optimization: mha.algorithm_name(objective_function=func, dimensions=n)")

def main():
    """Main function that runs the complete simple showcase."""
    print("🎯 MHA TOOLBOX - SIMPLE SHOWCASE GUIDE")
    print("="*60)
    print("Learn how to select and use algorithms in minutes!")
    
    try:
        # 1. Show available algorithms
        algorithms = show_available_algorithms()
        
        # 2. Basic usage examples
        basic_results = basic_algorithm_usage()
        
        # 3. Individual algorithm showcase
        individual_result = algorithm_showcase_individual()
        
        # 4. Algorithm comparison
        comparison_results = algorithm_comparison()
        
        # 5. User-friendly selection
        user_result = user_friendly_selection()
        
        # 6. Different data types
        showcase_with_different_data()
        
        # 7. Quick tips
        quick_tips()
        
        # Final summary
        print("\n" + "="*60)
        print("🎉 SIMPLE SHOWCASE COMPLETE!")
        print("="*60)
        print("✅ You now know how to:")
        print("   🎯 Select any algorithm: mha.algorithm_name()")
        print("   🔧 Use custom parameters: algorithm(X, y, param=value)")
        print("   🎨 Create visualizations: result.plot_convergence()")
        print("   ⚔️ Compare algorithms: run multiple, compare results")
        print("   📊 Work with any dataset: just pass X, y")
        
        print(f"\n🚀 QUICK START:")
        print(f"   import mha_toolbox as mha")
        print(f"   result = mha.ao(X, y)          # Run Aquila Optimizer")
        print(f"   result.plot_convergence()     # Visualize results")
        print(f"   print(result.best_fitness)    # Get best score")
        
    except Exception as e:
        print(f"\n❌ Error in showcase: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()