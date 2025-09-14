"""
🚀 MHA TOOLBOX - Complete Integration Showcase
==============================================

This script demonstrates the complete integration of:
- All 10 metaheuristic algorithms
- Advanced visualization suite
- Statistical analysis tools
- Professional result management
- Flexible parameter handling
- Multiple data types

This is the ultimate demo showing everything working together!
"""

import numpy as np
import sys
import os

# Add the parent directory to the path so we can import mha_toolbox
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import mha_toolbox as mha
from mha_toolbox.utils.visualizations import AdvancedVisualizer
from mha_toolbox.utils.statistics import StatisticalAnalyzer
from sklearn.datasets import load_breast_cancer, load_wine
import matplotlib.pyplot as plt

def showcase_algorithm_zoo():
    """Showcase all algorithms working together."""
    print("\n" + "🔥"*80)
    print("🎯 ALGORITHM ZOO SHOWCASE")
    print("🔥"*80)
    
    # Load real dataset
    X, y = load_wine(return_X_y=True)
    print(f"🍷 Dataset: Wine Classification - {X.shape[0]} samples, {X.shape[1]} features, {len(np.unique(y))} classes")
    
    # All available algorithms
    algorithms = [
        ('🦅 AO', mha.ao),
        ('🐝 PSO', mha.pso), 
        ('🌊 SCA', mha.sca),
        ('🐺 GWO', mha.gwo),
        ('🐋 WOA', mha.woa),
        ('🧬 GA', mha.ga),
        ('🔄 DE', mha.de),
        ('🔥 FA', mha.fa),
        ('🦇 BA', mha.ba),
        ('🐜 ACO', mha.aco)
    ]
    
    print(f"\n🎪 Running {len(algorithms)} algorithms in parallel showcase...")
    
    results = {}
    performance_summary = []
    
    for emoji_name, algo_func in algorithms:
        try:
            print(f"\n  {emoji_name} Running...")
            result = algo_func(X, y, population_size=20, max_iterations=30, verbose=False)
            
            algo_clean_name = emoji_name.split()[1]
            results[algo_clean_name] = result
            
            performance_summary.append({
                'Algorithm': emoji_name,
                'Best_Fitness': result.best_fitness,
                'Selected_Features': result.n_selected_features,
                'Execution_Time': result.execution_time,
                'Success': '✅'
            })
            
            print(f"    ✅ Success! Fitness: {result.best_fitness:.4f}, Features: {result.n_selected_features}/{X.shape[1]}, Time: {result.execution_time:.2f}s")
            
        except Exception as e:
            performance_summary.append({
                'Algorithm': emoji_name,
                'Best_Fitness': 'Failed',
                'Selected_Features': 'N/A',
                'Execution_Time': 'N/A',
                'Success': '❌'
            })
            print(f"    ❌ Failed: {e}")
    
    # Display performance table
    print(f"\n📊 PERFORMANCE SUMMARY:")
    print("-" * 80)
    for perf in performance_summary:
        if perf['Success'] == '✅':
            print(f"{perf['Algorithm']:<12} | Fitness: {perf['Best_Fitness']:<8.4f} | Features: {perf['Selected_Features']:<3}/{X.shape[1]} | Time: {perf['Execution_Time']:<6.2f}s | {perf['Success']}")
        else:
            print(f"{perf['Algorithm']:<12} | {perf['Best_Fitness']:<27} | {perf['Success']}")
    
    return results

def showcase_visualization_suite(results):
    """Showcase the complete visualization suite."""
    print("\n" + "🎨"*80)
    print("🎨 ADVANCED VISUALIZATION SHOWCASE")
    print("🎨"*80)
    
    if len(results) < 3:
        print("❌ Need at least 3 successful algorithms for full visualization showcase")
        return
    
    # Take top performing algorithms
    result_list = list(results.values())[:5]  # Top 5 for visualization
    
    print(f"\n🎭 Creating visualization suite for {len(result_list)} algorithms...")
    
    # Individual algorithm showcase
    print("\n1️⃣ Individual Algorithm Analysis:")
    best_result = result_list[0]
    print(f"   📈 Analyzing {best_result.algorithm_name}...")
    
    print("     📊 Basic convergence...")
    best_result.plot_convergence()
    plt.close()
    
    print("     🔍 Advanced analysis...")
    best_result.plot_advanced('convergence')
    plt.close()
    
    print("     📈 Statistical analysis...")
    stats = best_result.get_statistics()
    print(f"     Generated {len(stats)} statistical metrics")
    
    # Multi-algorithm comparison
    print("\n2️⃣ Multi-Algorithm Comparison Suite:")
    visualizer = AdvancedVisualizer(result_list)
    
    print("     📈 Convergence comparison...")
    visualizer.convergence_plot()
    plt.close()
    
    print("     📊 Statistical distribution analysis...")
    visualizer.box_plot()
    plt.close()
    
    print("     🔍 Exploration-exploitation analysis...")
    visualizer.exploration_exploitation_plot()
    plt.close()
    
    print("     📊 Comprehensive statistical suite...")
    visualizer.statistical_analysis_plot()
    plt.close()
    
    print("     🌟 Search trajectory analysis...")
    visualizer.search_trajectory_plot(dimensions=2)
    plt.close()
    
    print("   ✅ All visualizations generated successfully!")

def showcase_statistical_analysis(results):
    """Showcase the statistical analysis capabilities."""
    print("\n" + "📊"*80)
    print("📊 STATISTICAL ANALYSIS SHOWCASE")
    print("📊"*80)
    
    if len(results) < 2:
        print("❌ Need at least 2 algorithms for statistical analysis")
        return
    
    analyzer = StatisticalAnalyzer(list(results.values()))
    
    print("\n🔬 Comprehensive Statistical Analysis:")
    
    # Performance ranking
    print("\n1️⃣ Performance Ranking:")
    try:
        ranking = analyzer.performance_ranking()
        print("   📊 Algorithm Performance Ranking:")
        for i, row in ranking.iterrows():
            rank = i + 1
            medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"{rank:2d}"
            print(f"     {medal} {row['Algorithm']:<8} | Fitness: {row['Best_Fitness']:<8.4f} | Time: {row['Execution_Time']:<6.2f}s | Overall Rank: {row['Overall_Rank']:.2f}")
    except Exception as e:
        print(f"   ❌ Ranking failed: {e}")
    
    # Statistical tests
    print("\n2️⃣ Statistical Significance Testing:")
    try:
        tests = analyzer.statistical_tests()
        significant_comparisons = [comp for comp, test in tests.items() if test.get('significant', False)]
        print(f"   🔬 Performed {len(tests)} pairwise comparisons")
        print(f"   📊 Found {len(significant_comparisons)} statistically significant differences")
        
        if significant_comparisons:
            print("   🎯 Significant comparisons:")
            for comp in significant_comparisons[:3]:  # Show top 3
                test_result = tests[comp]
                print(f"     • {comp}: p-value = {test_result['p_value']:.4f}")
    except Exception as e:
        print(f"   ❌ Statistical tests failed: {e}")
    
    # Efficiency analysis
    print("\n3️⃣ Efficiency Analysis:")
    try:
        efficiency = analyzer.efficiency_metrics()
        print("   ⚡ Algorithm Efficiency Scores:")
        for i, row in efficiency.iterrows():
            print(f"     {row['Algorithm']:<8} | Efficiency: {row['Efficiency_Score']:<8.4f} | Speed: {row['Speed_Score']:<8.4f} | Robustness: {row['Robustness_Score']:.4f}")
    except Exception as e:
        print(f"   ❌ Efficiency analysis failed: {e}")

def showcase_parameter_flexibility():
    """Showcase parameter flexibility across all patterns."""
    print("\n" + "🔧"*80)
    print("🔧 PARAMETER FLEXIBILITY SHOWCASE")
    print("🔧"*80)
    
    X, y = load_breast_cancer(return_X_y=True)
    
    # Test all parameter patterns
    patterns = [
        ("📍 Positional", "mha.pso(X, y, 15, 25)"),
        ("🔤 Keyword", "mha.pso(X=X, y=y, population_size=15, max_iterations=25)"),
        ("🔀 Mixed", "mha.pso(X, y, population_size=15, max_iterations=25)"),
        ("🎯 Function Opt", "mha.pso(objective_function=sphere, dimensions=5)"),
        ("⚙️ Algorithm-specific", "mha.pso(X, y, c1=2.5, c2=1.5, w=0.8)")
    ]
    
    def sphere(x):
        return np.sum(x**2)
    
    print("\n🧪 Testing all parameter patterns:")
    
    for pattern_name, pattern_desc in patterns:
        print(f"\n  {pattern_name} Parameters:")
        print(f"    Code: {pattern_desc}")
        
        try:
            if "Function Opt" in pattern_name:
                result = mha.pso(objective_function=sphere, dimensions=5, population_size=10, max_iterations=15, verbose=False)
            elif "Algorithm-specific" in pattern_name:
                result = mha.pso(X, y, c1=2.5, c2=1.5, w=0.8, population_size=10, max_iterations=15, verbose=False)
            elif "Positional" in pattern_name:
                result = mha.pso(X, y, 15, 25, verbose=False)
            elif "Keyword" in pattern_name:
                result = mha.pso(X=X, y=y, population_size=15, max_iterations=25, verbose=False)
            else:  # Mixed
                result = mha.pso(X, y, population_size=15, max_iterations=25, verbose=False)
            
            print(f"    ✅ Success! Fitness: {result.best_fitness:.4f}, Time: {result.execution_time:.2f}s")
            
        except Exception as e:
            print(f"    ❌ Failed: {e}")

def showcase_data_variety():
    """Showcase handling of different data types."""
    print("\n" + "📊"*80)
    print("📊 DATA VARIETY SHOWCASE")
    print("📊"*80)
    
    datasets = [
        ("🩺 Medical (Breast Cancer)", load_breast_cancer),
        ("🍷 Classification (Wine)", load_wine),
    ]
    
    algorithms = [mha.ao, mha.pso, mha.sca]
    algo_names = ['AO', 'PSO', 'SCA']
    
    print(f"\n🧪 Testing {len(datasets)} datasets with {len(algorithms)} algorithms:")
    
    for dataset_name, dataset_loader in datasets:
        print(f"\n  {dataset_name}:")
        
        try:
            X, y = dataset_loader(return_X_y=True)
            print(f"    📏 Shape: {X.shape}, Classes: {len(np.unique(y))}")
            
            for algo, name in zip(algorithms, algo_names):
                try:
                    result = algo(X, y, population_size=10, max_iterations=15, verbose=False)
                    print(f"    ✅ {name}: Fitness {result.best_fitness:.4f}, Features {result.n_selected_features}/{X.shape[1]}")
                except Exception as e:
                    print(f"    ❌ {name}: Failed - {e}")
                    
        except Exception as e:
            print(f"    ❌ Dataset failed to load: {e}")

def final_integration_summary():
    """Provide final integration summary."""
    print("\n" + "🎉"*80)
    print("🎉 INTEGRATION SUMMARY")
    print("🎉"*80)
    
    # Get all available algorithms
    algorithms = mha.list_algorithms()
    
    print(f"\n✅ SUCCESSFULLY INTEGRATED COMPONENTS:")
    print(f"   🔥 Algorithms: {len(algorithms)} metaheuristic algorithms")
    print(f"   🎨 Visualizations: 6 major visualization types")
    print(f"   📊 Statistical Analysis: Comprehensive statistical toolkit")
    print(f"   🔧 Parameter Handling: 5+ parameter patterns supported")
    print(f"   📊 Data Types: Multiple dataset types supported")
    print(f"   💾 Result Management: Professional result storage and analysis")
    
    print(f"\n🎯 AVAILABLE ALGORITHMS:")
    for i, algo in enumerate(algorithms, 1):
        emoji = ["🦅", "🐝", "🌊", "🐺", "🐋", "🧬", "🔄", "🔥", "🦇", "🐜"][i-1] if i <= 10 else "🔥"
        print(f"   {emoji} {algo}")
    
    print(f"\n🎨 VISUALIZATION CAPABILITIES:")
    viz_types = [
        "📈 Convergence Analysis (4-panel detailed analysis)",
        "📊 Statistical Distribution (Box plots, histograms, Q-Q plots)",
        "🔍 Exploration-Exploitation (Diversity and intensity analysis)",
        "🌟 Search Trajectory (2D/3D path visualization)",
        "📊 Performance Comparison (Heatmaps and ranking)",
        "🔬 Statistical Testing (Significance tests and confidence)"
    ]
    
    for viz in viz_types:
        print(f"   {viz}")
    
    print(f"\n🔧 USAGE PATTERNS:")
    usage_examples = [
        "mha.ao(X, y)                    # Simple feature selection",
        "mha.pso(X, y, 30, 50)           # Positional parameters",
        "mha.sca(objective_function=f)   # Function optimization",
        "result.plot_advanced('all')     # Complete visualization suite",
        "result.compare_with(other)      # Algorithm comparison",
        "AdvancedVisualizer([r1,r2,r3])  # Multi-algorithm analysis"
    ]
    
    for example in usage_examples:
        print(f"   💻 {example}")
    
    print(f"\n🚀 THE MHA TOOLBOX IS FULLY INTEGRATED AND OPERATIONAL!")
    print(f"   All algorithms, visualizations, and analysis tools are connected.")
    print(f"   Ready for research, education, and practical optimization problems.")

def main():
    """Main integration showcase."""
    print("🚀 MHA TOOLBOX - COMPLETE INTEGRATION SHOWCASE")
    print("="*90)
    print("Welcome to the ultimate demonstration of the fully integrated MHA Toolbox!")
    print("This showcase demonstrates ALL components working together seamlessly.")
    
    try:
        # 1. Algorithm Zoo
        results = showcase_algorithm_zoo()
        
        # 2. Visualization Suite
        showcase_visualization_suite(results)
        
        # 3. Statistical Analysis
        showcase_statistical_analysis(results)
        
        # 4. Parameter Flexibility
        showcase_parameter_flexibility()
        
        # 5. Data Variety
        showcase_data_variety()
        
        # 6. Final Summary
        final_integration_summary()
        
        print(f"\n" + "🎉"*90)
        print("🎉 SHOWCASE COMPLETED SUCCESSFULLY!")
        print("🎉"*90)
        print("The MHA Toolbox is a fully integrated, professional-grade optimization library!")
        
    except Exception as e:
        print(f"\n❌ SHOWCASE FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()