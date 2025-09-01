"""
Quick algorithm checker for MHA Toolbox
"""

from mha_toolbox import MHAToolbox
from objective_functions.benchmark_functions import BenchmarkFunctions
import numpy as np

def check_single_algorithm():
    """Test single algorithm with different functions"""
    print("🔍 Checking SCA Algorithm")
    print("="*50)
    
    toolbox = MHAToolbox()
    bf = BenchmarkFunctions()
    
    # Test functions
    test_functions = [
        ('Sphere', bf.sphere),
        ('Rastrigin', bf.rastrigin), 
        ('Ackley', bf.ackley),
        ('Rosenbrock', bf.rosenbrock)
    ]
    
    results = []
    
    for func_name, func in test_functions:
        print(f"\n📊 Testing {func_name} function...")
        
        result = toolbox.optimize(
            algorithm_name='SCA',
            objective_function=func,
            pop_size=30,
            max_iter=100,
            dim=10,
            lb=-10,
            ub=10
        )
        
        results.append({
            'function': func_name,
            'best_score': result['best_score'],
            'best_position': result['best_position'][:3]  # Show first 3 values
        })
        
        print(f"✓ Best score: {result['best_score']:.6f}")
        print(f"✓ Best position (first 3): {result['best_position'][:3]}")
    
    return results

def check_algorithm_convergence():
    """Check algorithm convergence behavior"""
    print("\n🎯 Checking Algorithm Convergence")
    print("="*50)
    
    toolbox = MHAToolbox()
    bf = BenchmarkFunctions()
    
    # Test with sphere function (should converge to 0)
    result = toolbox.optimize(
        algorithm_name='SCA',
        objective_function=bf.sphere,
        pop_size=50,
        max_iter=200,
        dim=5,
        lb=-5,
        ub=5
    )
    
    print(f"✓ Final best score: {result['best_score']:.8f}")
    print(f"✓ Best position: {result['best_position']}")
    print(f"✓ Expected: Close to [0, 0, 0, 0, 0]")
    
    # Check if converged reasonably well
    if result['best_score'] < 0.01:
        print("✅ Good convergence!")
    elif result['best_score'] < 0.1:
        print("⚠️ Moderate convergence")
    else:
        print("❌ Poor convergence")
    
    return result

def check_algorithm_parameters():
    """Test algorithm with different parameters"""
    print("\n⚙️ Checking Different Parameters")
    print("="*50)
    
    toolbox = MHAToolbox()
    bf = BenchmarkFunctions()
    
    parameter_sets = [
        {'pop_size': 20, 'max_iter': 50, 'name': 'Small & Fast'},
        {'pop_size': 50, 'max_iter': 100, 'name': 'Medium'},
        {'pop_size': 100, 'max_iter': 200, 'name': 'Large & Thorough'}
    ]
    
    for params in parameter_sets:
        print(f"\n🔧 Testing {params['name']} setup...")
        
        result = toolbox.optimize(
            algorithm_name='SCA',
            objective_function=bf.sphere,
            pop_size=params['pop_size'],
            max_iter=params['max_iter'],
            dim=10,
            lb=-10,
            ub=10
        )
        
        print(f"✓ Pop: {params['pop_size']}, Iter: {params['max_iter']}")
        print(f"✓ Result: {result['best_score']:.6f}")

def check_custom_function():
    """Test with custom objective function"""
    print("\n🎨 Testing Custom Function")
    print("="*50)
    
    def my_function(x):
        """Custom test function: sum of squares + sine"""
        return np.sum(x**2) + np.sum(np.sin(x))
    
    toolbox = MHAToolbox()
    
    result = toolbox.optimize(
        algorithm_name='SCA',
        objective_function=my_function,
        pop_size=30,
        max_iter=100,
        dim=5,
        lb=-3,
        ub=3
    )
    
    print(f"✓ Custom function result: {result['best_score']:.6f}")
    print(f"✓ Best position: {result['best_position']}")

def main():
    """Run all algorithm checks"""
    print("🚀 MHA Algorithm Checker")
    print("="*60)
    
    try:
        # Basic functionality check
        results = check_single_algorithm()
        
        # Convergence check
        convergence_result = check_algorithm_convergence()
        
        # Parameter sensitivity check
        check_algorithm_parameters()
        
        # Custom function check
        check_custom_function()
        
        print("\n" + "="*60)
        print("📋 Summary")
        print("="*60)
        
        for result in results:
            print(f"• {result['function']}: {result['best_score']:.6f}")
        
        print(f"• Convergence: {convergence_result['best_score']:.6f}")
        
        print("\n✅ All checks completed!")
        
    except Exception as e:
        print(f"❌ Error during checking: {e}")

if __name__ == "__main__":
    main()
