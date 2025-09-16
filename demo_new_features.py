#!/usr/bin/env python3
"""
Demo of New MHA Toolbox Features

This script demonstrates:
1. Direct algorithm access (mha.pso, mha.sca, mha.gwo, etc.)
2. Parameter combinations analysis (4! = 24 combinations)
3. Multiple usage patterns

Author: MHA Development Team
"""

import mha_toolbox as mha
import numpy as np

def main():
    print("🚀 MHA Toolbox New Features Demo")
    print("=" * 50)
    
    # 1. Parameter Combinations Analysis
    print("\n📊 1. Parameter Combinations Analysis")
    print("-" * 40)
    mha.parameter_combinations()
    
    # 2. Direct Algorithm Access - Function Optimization
    print("\n🎯 2. Direct Algorithm Access - Function Optimization")
    print("-" * 55)
    
    # Define a simple test function
    def sphere_function(x):
        return sum(x**2)
    
    print("Testing different algorithms with direct access:")
    
    # PSO
    print("• Testing mha.pso()...")
    result_pso = mha.pso(objective_function=sphere_function, dimensions=5, 
                         max_iterations=20, verbose=False)
    print(f"  PSO Result: {result_pso.best_fitness:.6f}")
    
    # SCA  
    print("• Testing mha.sca()...")
    result_sca = mha.sca(objective_function=sphere_function, dimensions=5,
                         max_iterations=20, verbose=False)
    print(f"  SCA Result: {result_sca.best_fitness:.6f}")
    
    # GWO
    print("• Testing mha.gwo()...")
    result_gwo = mha.gwo(objective_function=sphere_function, dimensions=5,
                         max_iterations=20, verbose=False)
    print(f"  GWO Result: {result_gwo.best_fitness:.6f}")
    
    # 3. Direct Algorithm Access - Feature Selection
    print("\n📊 3. Direct Algorithm Access - Feature Selection")
    print("-" * 52)
    
    # Load test data
    X, y = mha.load_data('breast_cancer')
    print(f"Loaded dataset: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Test feature selection with direct access
    print("• Testing mha.pso() for feature selection...")
    result_fs = mha.pso(X, y, max_iterations=10, verbose=False)
    print(f"  Selected {result_fs.n_selected_features} features with fitness {result_fs.best_fitness:.6f}")
    
    # 4. Different Usage Patterns
    print("\n🔄 4. Different Usage Patterns")
    print("-" * 35)
    
    print("Pattern 1: mha.algorithm(X, y)  # Feature selection")
    result1 = mha.pso(X, y, max_iterations=5, verbose=False)
    print(f"  Result: {result1.n_selected_features} features")
    
    print("Pattern 2: mha.algorithm(objective_function, ...)  # Function optimization")
    result2 = mha.gwo(objective_function=lambda x: sum(x**2), dimensions=3, 
                      max_iterations=5, verbose=False)
    print(f"  Result: {result2.best_fitness:.6f}")
    
    print("Pattern 3: mha.algorithm(**kwargs)  # All keyword arguments")
    result3 = mha.sca(objective_function=sphere_function, dimensions=4,
                      max_iterations=5, population_size=20, verbose=False)
    print(f"  Result: {result3.best_fitness:.6f}")
    
    # 5. Parameter Combinations for Specific Algorithm
    print("\n🔍 5. Algorithm-Specific Parameter Analysis")
    print("-" * 45)
    
    print("Analyzing PSO parameters:")
    mha.parameter_combinations('pso')
    
    print("\n✅ Demo completed successfully!")
    print("\nNow you can use:")
    print("• mha.pso(X, y)  # Feature selection")
    print("• mha.sca(objective_function=func, dimensions=10)  # Function optimization")
    print("• mha.gwo(X, y, population_size=50)  # Custom parameters")
    print("• mha.parameter_combinations()  # Analyze combinations")

if __name__ == "__main__":
    main()