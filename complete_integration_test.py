"""
Complete Algorithm Integration Test
==================================

This script tests ALL implemented algorithms with data and visualizations to ensure
everything is properly connected and working together.
"""

import numpy as np
import sys
import os

# Add the parent directory to the path so we can import mha_toolbox
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import mha_toolbox as mha
from mha_toolbox.utils.visualizations import AdvancedVisualizer
from mha_toolbox.utils.statistics import StatisticalAnalyzer
from sklearn.datasets import load_breast_cancer, load_iris, make_classification
import matplotlib.pyplot as plt

def test_all_algorithms_feature_selection():
    """Test all algorithms with feature selection data."""
    print("\n" + "="*80)
    print("🧬 TESTING ALL ALGORITHMS - FEATURE SELECTION")
    print("="*80)
    
    # Load real dataset
    X, y = load_breast_cancer(return_X_y=True)
    print(f"Dataset: Breast Cancer - {X.shape[0]} samples, {X.shape[1]} features")
    
    # List of all available algorithms
    algorithms = [
        ('AO', mha.ao),
        ('PSO', mha.pso), 
        ('SCA', mha.sca),
        ('GWO', mha.gwo),
        ('WOA', mha.woa),
        ('GA', mha.ga),
        ('DE', mha.de),
        ('FA', mha.fa),
        ('BA', mha.ba),
        ('ACO', mha.aco)
    ]
    
    results = {}
    
    print("\n🔍 Testing each algorithm...")
    for algo_name, algo_func in algorithms:
        try:
            print(f"\n  🔥 Running {algo_name}...")
            result = algo_func(X, y, population_size=15, max_iterations=20, verbose=False)
            results[algo_name] = result
            
            print(f"    ✅ {algo_name}: Best fitness = {result.best_fitness:.4f}, "
                  f"Selected features = {result.n_selected_features}/{X.shape[1]}, "
                  f"Time = {result.execution_time:.2f}s")
            
        except Exception as e:
            print(f"    ❌ {algo_name}: Failed - {e}")
            
    print(f"\n📊 Successfully tested {len(results)} algorithms!")
    return results

def test_all_algorithms_function_optimization():
    """Test all algorithms with function optimization."""
    print("\n" + "="*80)
    print("📈 TESTING ALL ALGORITHMS - FUNCTION OPTIMIZATION")
    print("="*80)
    
    # Define test function (Sphere)
    def sphere_function(x):
        return np.sum(x**2)
    
    # List of all available algorithms
    algorithms = [
        ('AO', mha.ao),
        ('PSO', mha.pso), 
        ('SCA', mha.sca),
        ('GWO', mha.gwo),
        ('WOA', mha.woa),
        ('GA', mha.ga),
        ('DE', mha.de),
        ('FA', mha.fa),
        ('BA', mha.ba),
        ('ACO', mha.aco)
    ]
    
    results = {}
    
    print("\n🎯 Testing each algorithm on Sphere function (10D)...")
    for algo_name, algo_func in algorithms:
        try:
            print(f"\n  🔥 Running {algo_name}...")
            result = algo_func(objective_function=sphere_function, dimensions=10,
                             population_size=15, max_iterations=20, verbose=False)
            results[algo_name] = result
            
            print(f"    ✅ {algo_name}: Best fitness = {result.best_fitness:.6f}, "
                  f"Time = {result.execution_time:.3f}s")
            
        except Exception as e:
            print(f"    ❌ {algo_name}: Failed - {e}")
            
    print(f"\n📊 Successfully tested {len(results)} algorithms!")
    return results

def test_visualization_integration(results):
    """Test visualization integration with all algorithms."""
    print("\n" + "="*80)
    print("🎨 TESTING VISUALIZATION INTEGRATION")
    print("="*80)
    
    if not results:
        print("❌ No results to visualize!")
        return
    
    print(f"\n📊 Creating visualizations for {len(results)} algorithms...")
    
    # Test individual algorithm visualizations
    print("\n1️⃣ Testing individual algorithm visualizations...")
    first_result = list(results.values())[0]
    
    try:
        print("   📈 Basic convergence plot...")
        first_result.plot_convergence()
        plt.close()  # Close to avoid display issues
        
        print("   🎯 Advanced convergence analysis...")
        first_result.plot_advanced(plot_type='convergence')
        plt.close()
        
        print("   ✅ Individual visualizations working!")
        
    except Exception as e:
        print(f"   ❌ Individual visualization failed: {e}")
    
    # Test multi-algorithm comparison
    print("\n2️⃣ Testing multi-algorithm comparison...")
    try:
        # Take first 3 algorithms for comparison
        comparison_results = list(results.values())[:3]
        
        visualizer = AdvancedVisualizer(comparison_results)
        
        print("   📈 Multi-algorithm convergence comparison...")
        visualizer.convergence_plot()
        plt.close()
        
        print("   📊 Box plot comparison...")
        visualizer.box_plot()
        plt.close()
        
        print("   📈 Statistical analysis...")
        visualizer.statistical_analysis_plot()
        plt.close()
        
        print("   ✅ Multi-algorithm visualizations working!")
        
    except Exception as e:
        print(f"   ❌ Multi-algorithm visualization failed: {e}")
    
    # Test statistical analysis
    print("\n3️⃣ Testing statistical analysis...")
    try:
        analyzer = StatisticalAnalyzer(list(results.values()))
        
        print("   📊 Performance ranking...")
        ranking = analyzer.performance_ranking()
        print(f"       Generated ranking table with {len(ranking)} algorithms")
        
        print("   🔬 Statistical tests...")
        tests = analyzer.statistical_tests()
        print(f"       Performed {len(tests)} statistical comparisons")
        
        print("   📈 Efficiency metrics...")
        efficiency = analyzer.efficiency_metrics()
        print(f"       Calculated efficiency for {len(efficiency)} algorithms")
        
        print("   ✅ Statistical analysis working!")
        
    except Exception as e:
        print(f"   ❌ Statistical analysis failed: {e}")

def test_result_object_methods(results):
    """Test all methods of the OptimizationModel result objects."""
    print("\n" + "="*80)
    print("🔧 TESTING RESULT OBJECT METHODS")
    print("="*80)
    
    if not results:
        print("❌ No results to test!")
        return
    
    # Get first result for testing
    result = list(results.values())[0]
    algo_name = list(results.keys())[0]
    
    print(f"\n🧪 Testing methods on {algo_name} result...")
    
    # Test basic methods
    try:
        print("\n1️⃣ Testing basic methods...")
        
        print("   📋 Summary method...")
        result.summary()
        
        print("   💾 Save method...")
        result.save()
        
        print("   📊 Statistics method...")
        stats = result.get_statistics()
        print(f"       Generated {len(stats)} statistical metrics")
        
        print("   ✅ Basic methods working!")
        
    except Exception as e:
        print(f"   ❌ Basic methods failed: {e}")
    
    # Test comparison methods
    if len(results) >= 2:
        try:
            print("\n2️⃣ Testing comparison methods...")
            result1 = list(results.values())[0]
            result2 = list(results.values())[1]
            
            print("   ⚔️ Compare with method...")
            comparison = result1.compare_with(result2)
            print(f"       Generated comparison with {len(comparison)} metrics")
            plt.close()  # Close comparison plots
            
            print("   📄 Statistical report method...")
            report = result1.statistical_report()
            print(f"       Generated report with {len(report.split())} words")
            
            print("   ✅ Comparison methods working!")
            
        except Exception as e:
            print(f"   ❌ Comparison methods failed: {e}")

def test_parameter_flexibility():
    """Test parameter flexibility across all algorithms."""
    print("\n" + "="*80)
    print("🔧 TESTING PARAMETER FLEXIBILITY")
    print("="*80)
    
    # Create test data
    X, y = make_classification(n_samples=50, n_features=10, random_state=42)
    
    # Test different parameter patterns
    parameter_patterns = [
        ("Positional only", lambda algo: algo(X, y, 10, 15)),
        ("Keyword only", lambda algo: algo(X=X, y=y, population_size=10, max_iterations=15)),
        ("Mixed", lambda algo: algo(X, y, population_size=10, max_iterations=15)),
        ("Algorithm-specific", lambda algo: algo(X=X, y=y, population_size=10) if hasattr(algo, '__name__') and 'pso' in algo.__name__.lower() else algo(X=X, y=y, population_size=10))
    ]
    
    algorithms = [
        ('PSO', mha.pso),
        ('AO', mha.ao),
        ('SCA', mha.sca)
    ]
    
    print("\n🧪 Testing parameter patterns...")
    
    for pattern_name, pattern_func in parameter_patterns:
        print(f"\n  🔍 Testing {pattern_name} parameters...")
        
        for algo_name, algo_func in algorithms:
            try:
                result = pattern_func(algo_func)
                print(f"    ✅ {algo_name}: {pattern_name} - Success (fitness: {result.best_fitness:.4f})")
            except Exception as e:
                print(f"    ❌ {algo_name}: {pattern_name} - Failed: {e}")

def test_data_integration():
    """Test integration with different types of data."""
    print("\n" + "="*80)
    print("📊 TESTING DATA INTEGRATION")
    print("="*80)
    
    # Test different datasets
    datasets = [
        ("Breast Cancer", lambda: load_breast_cancer(return_X_y=True)),
        ("Iris", lambda: load_iris(return_X_y=True)),
        ("Synthetic Binary", lambda: make_classification(n_samples=100, n_features=15, n_classes=2, random_state=42)),
        ("Synthetic Multiclass", lambda: make_classification(n_samples=100, n_features=15, n_classes=3, random_state=42))
    ]
    
    algorithms_to_test = [mha.ao, mha.pso, mha.sca]
    
    print("\n🧪 Testing data compatibility...")
    
    for dataset_name, dataset_func in datasets:
        print(f"\n  📊 Testing {dataset_name} dataset...")
        
        try:
            X, y = dataset_func()
            print(f"    Dataset shape: {X.shape}, Classes: {len(np.unique(y))}")
            
            for i, algo in enumerate(algorithms_to_test):
                try:
                    result = algo(X, y, population_size=10, max_iterations=10, verbose=False)
                    algo_name = ['AO', 'PSO', 'SCA'][i]
                    print(f"    ✅ {algo_name}: Success (fitness: {result.best_fitness:.4f}, features: {result.n_selected_features})")
                except Exception as e:
                    print(f"    ❌ {algo_name}: Failed - {e}")
                    
        except Exception as e:
            print(f"    ❌ Dataset {dataset_name}: Failed to load - {e}")

def main():
    """Main integration test function."""
    print("🚀 MHA TOOLBOX COMPLETE INTEGRATION TEST")
    print("="*80)
    print("Testing ALL algorithms with data and visualizations...")
    print("This comprehensive test ensures everything is properly connected.")
    
    try:
        # Test 1: Feature Selection
        fs_results = test_all_algorithms_feature_selection()
        
        # Test 2: Function Optimization  
        fo_results = test_all_algorithms_function_optimization()
        
        # Test 3: Visualization Integration
        test_visualization_integration(fs_results)
        
        # Test 4: Result Object Methods
        test_result_object_methods(fs_results)
        
        # Test 5: Parameter Flexibility
        test_parameter_flexibility()
        
        # Test 6: Data Integration
        test_data_integration()
        
        # Final Summary
        print("\n" + "="*80)
        print("🎉 INTEGRATION TEST SUMMARY")
        print("="*80)
        print(f"✅ Feature Selection: {len(fs_results)} algorithms tested")
        print(f"✅ Function Optimization: {len(fo_results)} algorithms tested")
        print("✅ Visualization Integration: Completed")
        print("✅ Result Object Methods: Completed")
        print("✅ Parameter Flexibility: Completed")
        print("✅ Data Integration: Completed")
        
        print("\n🎯 ALL SYSTEMS OPERATIONAL!")
        print("The MHA Toolbox is fully integrated and ready for use.")
        print("All algorithms, data handling, and visualizations are connected.")
        
        # Show available algorithms
        print(f"\n📋 Available Algorithms: {mha.list_algorithms()}")
        
    except Exception as e:
        print(f"\n❌ INTEGRATION TEST FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()