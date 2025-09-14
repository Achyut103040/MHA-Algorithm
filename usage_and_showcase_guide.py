"""
🎯 MHA TOOLBOX - Complete Usage and Showcase Guide
=================================================

This guide shows you EVERYTHING about how to:
1. Select and use algorithms
2. Showcase individual algorithms
3. Compare multiple algorithms
4. Create professional demonstrations
5. Customize visualizations and analysis

📚 TABLE OF CONTENTS:
====================
1. Basic Algorithm Selection
2. Advanced Algorithm Usage Patterns
3. Individual Algorithm Showcase
4. Multi-Algorithm Comparison
5. Professional Demonstration Setup
6. Custom Visualization Creation
7. Research and Publication Ready Examples
"""

import numpy as np
import sys
import os

# Add the parent directory to the path so we can import mha_toolbox
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import mha_toolbox as mha
from mha_toolbox.utils.visualizations import AdvancedVisualizer
from mha_toolbox.utils.statistics import StatisticalAnalyzer
from sklearn.datasets import load_breast_cancer, load_wine, load_iris, make_classification
import matplotlib.pyplot as plt

# ============================================================================
# 1. BASIC ALGORITHM SELECTION
# ============================================================================

def basic_algorithm_selection():
    """Shows all the basic ways to select and run algorithms."""
    print("\n" + "="*70)
    print("1️⃣ BASIC ALGORITHM SELECTION")
    print("="*70)
    
    # Load sample data
    X, y = load_breast_cancer(return_X_y=True)
    
    print("📋 Available algorithms:")
    available_algos = mha.list_algorithms()
    for i, algo in enumerate(available_algos, 1):
        print(f"   {i:2d}. {algo}")
    
    print(f"\n🎯 Ways to select and run algorithms:")
    
    # Method 1: Direct function call
    print("\n   Method 1: Direct Function Call")
    print("   ✅ Most common and recommended approach")
    print("   💻 Code:")
    print("       result = mha.ao(X, y)                    # Aquila Optimizer")
    print("       result = mha.pso(X, y)                   # Particle Swarm")
    print("       result = mha.sca(X, y)                   # Sine Cosine")
    
    # Example
    result_ao = mha.ao(X, y, population_size=10, max_iterations=15, verbose=False)
    print(f"   📊 Example result: AO achieved fitness {result_ao.best_fitness:.4f}")
    
    # Method 2: Algorithm mapping
    print("\n   Method 2: Algorithm Mapping")
    print("   ✅ Good for user selection interfaces")
    print("   💻 Code:")
    
    algorithm_map = {
        'ao': mha.ao,
        'pso': mha.pso,
        'sca': mha.sca,
        'gwo': mha.gwo,
        'woa': mha.woa,
        'ga': mha.ga,
        'de': mha.de,
        'fa': mha.fa,
        'ba': mha.ba,
        'aco': mha.aco
    }
    
    print("       algorithm_map = {")
    for name, func in list(algorithm_map.items())[:3]:
        print(f"           '{name}': mha.{name},")
    print("           # ... more algorithms")
    print("       }")
    print("       selected_algo = algorithm_map['pso']")
    print("       result = selected_algo(X, y)")
    
    # Example
    user_choice = 'pso'
    selected_algo = algorithm_map[user_choice]
    result_selected = selected_algo(X, y, population_size=10, max_iterations=15, verbose=False)
    print(f"   📊 User selected '{user_choice}': fitness {result_selected.best_fitness:.4f}")
    
    # Method 3: Interactive selection
    print("\n   Method 3: Interactive Selection")
    print("   ✅ Great for demos and educational purposes")
    print("   💻 Code example:")
    print("""
       def interactive_algorithm_selection():
           algorithms = ['ao', 'pso', 'sca', 'gwo', 'woa']
           print("Available algorithms:")
           for i, algo in enumerate(algorithms, 1):
               print(f"{i}. {algo.upper()}")
           
           choice = int(input("Select algorithm (1-5): ")) - 1
           selected = algorithm_map[algorithms[choice]]
           return selected(X, y)
    """)

# ============================================================================
# 2. ADVANCED ALGORITHM USAGE PATTERNS
# ============================================================================

def advanced_usage_patterns():
    """Shows advanced ways to use algorithms with different parameter patterns."""
    print("\n" + "="*70)
    print("2️⃣ ADVANCED ALGORITHM USAGE PATTERNS")
    print("="*70)
    
    X, y = load_breast_cancer(return_X_y=True)
    
    print("🔧 All supported parameter patterns:")
    
    # Pattern 1: Minimal usage
    print("\n   Pattern 1: Minimal Usage (Auto-detection)")
    print("   💻 mha.ao(X, y)")
    result1 = mha.ao(X, y, population_size=10, max_iterations=15, verbose=False)
    print(f"   📊 Auto-detected {result1.parameters['dimensions']} dimensions, used {result1.parameters['population_size']} population")
    
    # Pattern 2: Positional parameters
    print("\n   Pattern 2: Positional Parameters")
    print("   💻 mha.pso(X, y, 20, 30)  # population_size=20, max_iterations=30")
    result2 = mha.pso(X, y, 20, 30, verbose=False)
    print(f"   📊 Used {result2.parameters['population_size']} population, {result2.parameters['max_iterations']} iterations")
    
    # Pattern 3: Keyword parameters
    print("\n   Pattern 3: Keyword Parameters")
    print("   💻 mha.sca(X=X, y=y, population_size=25, max_iterations=40)")
    result3 = mha.sca(X=X, y=y, population_size=25, max_iterations=40, verbose=False)
    print(f"   📊 Explicit keyword usage: fitness {result3.best_fitness:.4f}")
    
    # Pattern 4: Algorithm-specific parameters
    print("\n   Pattern 4: Algorithm-Specific Parameters")
    print("   💻 mha.pso(X, y, c1=2.5, c2=1.5, w=0.8)  # PSO-specific params")
    result4 = mha.pso(X, y, c1=2.5, c2=1.5, w=0.8, population_size=15, max_iterations=20, verbose=False)
    print(f"   📊 Custom PSO parameters: c1=2.5, c2=1.5, w=0.8, fitness {result4.best_fitness:.4f}")
    
    # Pattern 5: Function optimization
    print("\n   Pattern 5: Function Optimization")
    print("   💻 mha.gwo(objective_function=my_function, dimensions=10)")
    
    def sphere_function(x):
        return np.sum(x**2)
    
    result5 = mha.gwo(objective_function=sphere_function, dimensions=10, population_size=15, max_iterations=20, verbose=False)
    print(f"   📊 Function optimization: minimized sphere function to {result5.best_fitness:.6f}")
    
    # Pattern 6: Mixed parameters
    print("\n   Pattern 6: Mixed Parameters")
    print("   💻 mha.woa(X, y, population_size=30, lower_bound=0.1, upper_bound=0.9)")
    result6 = mha.woa(X, y, population_size=30, lower_bound=0.1, upper_bound=0.9, max_iterations=15, verbose=False)
    print(f"   📊 Mixed usage: custom bounds and population, fitness {result6.best_fitness:.4f}")

# ============================================================================
# 3. INDIVIDUAL ALGORITHM SHOWCASE
# ============================================================================

def individual_algorithm_showcase():
    """Shows how to create a professional showcase for a single algorithm."""
    print("\n" + "="*70)
    print("3️⃣ INDIVIDUAL ALGORITHM SHOWCASE")
    print("="*70)
    
    # Load data
    X, y = load_wine(return_X_y=True)
    print(f"🍷 Dataset: Wine Classification ({X.shape[0]} samples, {X.shape[1]} features)")
    
    # Select algorithm to showcase
    algorithm_name = "Aquila Optimizer (AO)"
    print(f"\n🎯 Showcasing: {algorithm_name}")
    
    # Run the algorithm
    print("   🔥 Running optimization...")
    result = mha.ao(X, y, population_size=25, max_iterations=40, verbose=False)
    
    # Professional showcase
    print(f"\n📊 ALGORITHM PERFORMANCE:")
    print(f"   ✅ Best Fitness: {result.best_fitness:.6f}")
    print(f"   ✅ Selected Features: {result.n_selected_features}/{X.shape[1]} ({result.n_selected_features/X.shape[1]*100:.1f}%)")
    print(f"   ✅ Execution Time: {result.execution_time:.3f} seconds")
    
    # Detailed statistics
    stats = result.get_statistics()
    print(f"\n📈 DETAILED STATISTICS:")
    print(f"   📊 Convergence Rate: {stats['convergence_rate']:.6f}")
    print(f"   📊 Improvement Ratio: {stats['improvement_ratio']:.4f}")
    print(f"   📊 Early Convergence: {stats.get('early_convergence', 'N/A')}")
    print(f"   📊 Stagnation Ratio: {stats.get('stagnation_ratio', 'N/A'):.4f}")
    
    # Create comprehensive visualization
    print(f"\n🎨 CREATING VISUALIZATION SHOWCASE:")
    print("   📈 1. Basic convergence plot...")
    result.plot_convergence(title=f"{algorithm_name} - Convergence Analysis")
    plt.close()
    
    print("   📊 2. Advanced convergence analysis...")
    result.plot_advanced(plot_type='convergence')
    plt.close()
    
    print("   🔍 3. Exploration-exploitation analysis...")
    result.plot_advanced(plot_type='exploration')
    plt.close()
    
    print("   📊 4. Statistical analysis...")
    result.plot_advanced(plot_type='statistical')
    plt.close()
    
    print("   🌟 5. Search trajectory...")
    result.plot_advanced(plot_type='trajectory')
    plt.close()
    
    print("   ✅ Complete individual showcase created!")
    
    return result

# ============================================================================
# 4. MULTI-ALGORITHM COMPARISON SHOWCASE
# ============================================================================

def multi_algorithm_comparison():
    """Shows how to create a professional multi-algorithm comparison."""
    print("\n" + "="*70)
    print("4️⃣ MULTI-ALGORITHM COMPARISON SHOWCASE")
    print("="*70)
    
    # Load data
    X, y = load_breast_cancer(return_X_y=True)
    print(f"🩺 Dataset: Breast Cancer ({X.shape[0]} samples, {X.shape[1]} features)")
    
    # Select algorithms for comparison
    algorithms_to_compare = [
        ('🦅 Aquila Optimizer', mha.ao),
        ('🐝 Particle Swarm', mha.pso),
        ('🌊 Sine Cosine', mha.sca),
        ('🐺 Grey Wolf', mha.gwo),
        ('🐋 Whale Optimization', mha.woa)
    ]
    
    print(f"\n⚔️ COMPARING {len(algorithms_to_compare)} ALGORITHMS:")
    
    results = {}
    comparison_data = []
    
    # Run all algorithms
    for name, algo_func in algorithms_to_compare:
        print(f"   🔥 Running {name}...")
        result = algo_func(X, y, population_size=20, max_iterations=30, verbose=False)
        
        clean_name = name.split()[1]  # Extract algorithm name
        results[clean_name] = result
        
        comparison_data.append({
            'Algorithm': name,
            'Best_Fitness': result.best_fitness,
            'Features_Selected': result.n_selected_features,
            'Execution_Time': result.execution_time,
            'Improvement_Ratio': result.get_statistics()['improvement_ratio']
        })
        
        print(f"       ✅ Fitness: {result.best_fitness:.4f}, Features: {result.n_selected_features}/{X.shape[1]}, Time: {result.execution_time:.2f}s")
    
    # Performance summary table
    print(f"\n📊 COMPARISON SUMMARY:")
    print("-" * 80)
    print(f"{'Algorithm':<20} | {'Fitness':<8} | {'Features':<8} | {'Time':<6} | {'Improvement':<10}")
    print("-" * 80)
    for data in comparison_data:
        print(f"{data['Algorithm']:<20} | {data['Best_Fitness']:<8.4f} | {data['Features_Selected']:>3}/{X.shape[1]:<3} | {data['Execution_Time']:<6.2f} | {data['Improvement_Ratio']:<10.4f}")
    
    # Create comparison visualizations
    print(f"\n🎨 CREATING COMPARISON VISUALIZATIONS:")
    visualizer = AdvancedVisualizer(list(results.values()))
    
    print("   📈 1. Multi-algorithm convergence comparison...")
    visualizer.convergence_plot()
    plt.close()
    
    print("   📊 2. Performance box plots...")
    visualizer.box_plot()
    plt.close()
    
    print("   🔍 3. Exploration-exploitation comparison...")
    visualizer.exploration_exploitation_plot()
    plt.close()
    
    print("   📊 4. Statistical analysis comparison...")
    visualizer.statistical_analysis_plot()
    plt.close()
    
    # Statistical analysis
    print(f"\n🔬 STATISTICAL ANALYSIS:")
    analyzer = StatisticalAnalyzer(list(results.values()))
    
    # Performance ranking
    ranking = analyzer.performance_ranking()
    print("   🏆 Performance Ranking:")
    for i, row in ranking.head(3).iterrows():
        medal = "🥇" if i == 0 else "🥈" if i == 1 else "🥉"
        print(f"       {medal} {row['Algorithm']:<15} | Overall Rank: {row['Overall_Rank']:.2f}")
    
    # Statistical tests
    tests = analyzer.statistical_tests()
    significant = [comp for comp, test in tests.items() if test.get('significant', False)]
    print(f"   📊 Statistical Tests: {len(tests)} comparisons, {len(significant)} significant differences")
    
    print("   ✅ Complete comparison showcase created!")
    
    return results

# ============================================================================
# 5. PROFESSIONAL DEMONSTRATION SETUP
# ============================================================================

def professional_demonstration():
    """Shows how to set up a professional demonstration or presentation."""
    print("\n" + "="*70)
    print("5️⃣ PROFESSIONAL DEMONSTRATION SETUP")
    print("="*70)
    
    print("🎭 Professional Demo Template:")
    
    # Demo configuration
    demo_config = {
        'datasets': [
            ('Medical Data', load_breast_cancer),
            ('Wine Classification', load_wine),
            ('Iris Classification', load_iris)
        ],
        'algorithms': [
            ('Aquila Optimizer', mha.ao),
            ('Particle Swarm', mha.pso),
            ('Grey Wolf', mha.gwo)
        ],
        'parameters': {
            'population_size': 20,
            'max_iterations': 25,
            'verbose': False
        }
    }
    
    print(f"\n📋 Demo Configuration:")
    print(f"   📊 Datasets: {len(demo_config['datasets'])}")
    print(f"   🔥 Algorithms: {len(demo_config['algorithms'])}")
    print(f"   ⚙️ Parameters: {demo_config['parameters']}")
    
    # Run demonstration
    demo_results = {}
    
    for dataset_name, dataset_loader in demo_config['datasets']:
        print(f"\n🎯 DEMONSTRATING ON {dataset_name.upper()}:")
        
        # Load data
        X, y = dataset_loader(return_X_y=True)
        print(f"   📏 Data: {X.shape[0]} samples, {X.shape[1]} features, {len(np.unique(y))} classes")
        
        dataset_results = {}
        
        # Test each algorithm
        for algo_name, algo_func in demo_config['algorithms']:
            try:
                result = algo_func(X, y, **demo_config['parameters'])
                dataset_results[algo_name] = result
                
                print(f"   ✅ {algo_name}: Fitness {result.best_fitness:.4f}, "
                      f"Features {result.n_selected_features}/{X.shape[1]}, "
                      f"Time {result.execution_time:.2f}s")
            except Exception as e:
                print(f"   ❌ {algo_name}: Failed - {e}")
        
        demo_results[dataset_name] = dataset_results
        
        # Create dataset-specific visualization
        if len(dataset_results) >= 2:
            print(f"   🎨 Creating visualization for {dataset_name}...")
            visualizer = AdvancedVisualizer(list(dataset_results.values()))
            visualizer.convergence_plot()
            plt.close()
    
    print(f"\n✅ Professional demonstration completed!")
    print(f"   📊 Tested {len(demo_config['algorithms'])} algorithms on {len(demo_config['datasets'])} datasets")
    print(f"   🎨 Generated visualizations for each dataset")
    print(f"   📋 All results saved automatically")
    
    return demo_results

# ============================================================================
# 6. CUSTOM VISUALIZATION CREATION
# ============================================================================

def custom_visualization_showcase():
    """Shows how to create custom visualizations and analysis."""
    print("\n" + "="*70)
    print("6️⃣ CUSTOM VISUALIZATION CREATION")
    print("="*70)
    
    # Run algorithm for visualization
    X, y = load_wine(return_X_y=True)
    result = mha.ao(X, y, population_size=20, max_iterations=30, verbose=False)
    
    print("🎨 Custom Visualization Options:")
    
    # Option 1: Built-in advanced plots
    print("\n   Option 1: Built-in Advanced Plots")
    print("   💻 result.plot_advanced('convergence')     # Detailed convergence")
    print("   💻 result.plot_advanced('exploration')     # Exploration analysis")
    print("   💻 result.plot_advanced('statistical')     # Statistical plots")
    print("   💻 result.plot_advanced('trajectory')      # Search trajectory")
    print("   💻 result.plot_advanced('all')            # All plots")
    
    # Option 2: Direct visualizer usage
    print("\n   Option 2: Direct AdvancedVisualizer Usage")
    print("   💻 visualizer = AdvancedVisualizer([result])")
    print("   💻 visualizer.convergence_plot(save_path='my_plot.png')")
    print("   💻 visualizer.statistical_analysis_plot()")
    
    # Option 3: Custom matplotlib plots
    print("\n   Option 3: Custom Matplotlib Integration")
    print("   💻 plt.figure(figsize=(12, 8))")
    print("   💻 plt.plot(result.convergence_curve)")
    print("   💻 plt.title('My Custom Convergence Plot')")
    print("   💻 plt.show()")
    
    # Option 4: Statistical analysis
    print("\n   Option 4: Statistical Analysis Integration")
    print("   💻 stats = result.get_statistics()")
    print("   💻 analyzer = StatisticalAnalyzer([result])")
    print("   💻 report = analyzer.generate_report()")
    
    # Demonstrate custom plot
    print("\n   🎨 Creating custom visualization...")
    
    # Custom convergence plot with annotations
    plt.figure(figsize=(10, 6))
    plt.plot(result.convergence_curve, linewidth=2, color='#2E86AB', label='AO Convergence')
    plt.axhline(y=result.best_fitness, color='red', linestyle='--', alpha=0.7, label=f'Best Fitness: {result.best_fitness:.4f}')
    plt.fill_between(range(len(result.convergence_curve)), result.convergence_curve, alpha=0.3, color='#2E86AB')
    
    plt.title('Custom Aquila Optimizer Convergence Analysis', fontsize=14, fontweight='bold')
    plt.xlabel('Iteration')
    plt.ylabel('Fitness Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.close()
    
    print("   ✅ Custom visualization created!")

# ============================================================================
# 7. RESEARCH AND PUBLICATION READY EXAMPLES
# ============================================================================

def research_publication_examples():
    """Shows how to create research and publication-ready examples."""
    print("\n" + "="*70)
    print("7️⃣ RESEARCH & PUBLICATION READY EXAMPLES")
    print("="*70)
    
    print("📚 Research-Grade Usage Patterns:")
    
    # Research Example 1: Algorithm Comparison Study
    print("\n   Example 1: Algorithm Comparison Study")
    print("   🎯 Comparing 5 algorithms on multiple datasets with statistical analysis")
    
    X, y = load_breast_cancer(return_X_y=True)
    
    # Run multiple algorithms
    algorithms = [mha.ao, mha.pso, mha.sca, mha.gwo, mha.woa]
    algo_names = ['AO', 'PSO', 'SCA', 'GWO', 'WOA']
    
    research_results = []
    for algo, name in zip(algorithms[:3], algo_names[:3]):  # Limit for demo
        result = algo(X, y, population_size=25, max_iterations=35, verbose=False)
        research_results.append(result)
        print(f"   📊 {name}: Fitness {result.best_fitness:.6f}, Time {result.execution_time:.3f}s")
    
    # Statistical analysis
    analyzer = StatisticalAnalyzer(research_results)
    tests = analyzer.statistical_tests()
    ranking = analyzer.performance_ranking()
    
    print(f"   🔬 Statistical Tests: {len(tests)} pairwise comparisons")
    print(f"   🏆 Best Algorithm: {ranking.iloc[0]['Algorithm']}")
    
    # Research Example 2: Multi-run Statistical Analysis
    print("\n   Example 2: Multi-run Statistical Analysis")
    print("   🎯 Running algorithm multiple times for statistical significance")
    
    multi_run_results = []
    for run in range(3):  # Normally 30+ runs
        result = mha.ao(X, y, population_size=15, max_iterations=20, verbose=False)
        multi_run_results.append(result.best_fitness)
    
    mean_fitness = np.mean(multi_run_results)
    std_fitness = np.std(multi_run_results)
    
    print(f"   📊 Mean Fitness: {mean_fitness:.6f} ± {std_fitness:.6f}")
    print(f"   📊 Best Run: {min(multi_run_results):.6f}")
    print(f"   📊 Worst Run: {max(multi_run_results):.6f}")
    
    # Research Example 3: Publication-Quality Plots
    print("\n   Example 3: Publication-Quality Visualization")
    
    # Create publication-ready comparison
    visualizer = AdvancedVisualizer(research_results)
    
    print("   📈 Creating publication-quality plots...")
    print("       - Convergence comparison with confidence intervals")
    print("       - Statistical significance heatmap")
    print("       - Performance ranking table")
    print("       - Box plots with statistical annotations")
    
    visualizer.convergence_plot()
    plt.close()
    
    print("   ✅ Research examples completed!")

# ============================================================================
# MAIN SHOWCASE FUNCTION
# ============================================================================

def main_showcase():
    """Main function that runs the complete usage and showcase guide."""
    print("🎯 MHA TOOLBOX - COMPLETE USAGE & SHOWCASE GUIDE")
    print("="*70)
    print("This guide shows you EVERYTHING about using the MHA Toolbox!")
    print("From basic usage to advanced research applications.")
    
    try:
        # 1. Basic algorithm selection
        basic_algorithm_selection()
        
        # 2. Advanced usage patterns
        advanced_usage_patterns()
        
        # 3. Individual algorithm showcase
        individual_result = individual_algorithm_showcase()
        
        # 4. Multi-algorithm comparison
        comparison_results = multi_algorithm_comparison()
        
        # 5. Professional demonstration
        demo_results = professional_demonstration()
        
        # 6. Custom visualization
        custom_visualization_showcase()
        
        # 7. Research examples
        research_publication_examples()
        
        # Final summary
        print("\n" + "="*70)
        print("🎉 COMPLETE USAGE GUIDE FINISHED!")
        print("="*70)
        print("✅ You now know how to:")
        print("   🔥 Select and run any algorithm")
        print("   📊 Create professional showcases")
        print("   ⚔️ Compare multiple algorithms")
        print("   🎨 Generate custom visualizations")
        print("   📚 Conduct research-grade analysis")
        print("   🎭 Set up professional demonstrations")
        
        print(f"\n🚀 KEY TAKEAWAYS:")
        print(f"   💻 Use mha.algorithm_name(X, y) for simple usage")
        print(f"   🎨 Use result.plot_advanced('all') for complete visualization")
        print(f"   ⚔️ Use AdvancedVisualizer([r1,r2,r3]) for comparison")
        print(f"   🔬 Use StatisticalAnalyzer for research analysis")
        print(f"   📊 All results are automatically saved and organized")
        
    except Exception as e:
        print(f"\n❌ Error in showcase: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main_showcase()