#!/usr/bin/env python3
"""
Test script to verify new algorithms work with direct access
"""

from mha_toolbox import MHAToolbox
import numpy as np

# Initialize toolbox
mha = MHAToolbox()

# Test function (Sphere function)
def sphere_function(x):
    return np.sum(x**2)

print("🧪 Testing new algorithms with direct access...")

# Test some of the new algorithms
algorithms_to_test = [
    ('alo', {}),
    ('mrfo', {}),
    ('sma', {}),
    ('tso', {}),
    ('csa', {})
]

for algo_name, params in algorithms_to_test:
    try:
        print(f"\n🔍 Testing {algo_name.upper()}...")
        
        # Run optimization using the toolbox optimize method
        result = mha.optimize(
            algo_name,
            objective_function=sphere_function,
            dimensions=5,
            lower_bound=-5.0,
            upper_bound=5.0,
            population_size=10, 
            max_iterations=5, 
            **params
        )
        
        print(f"✅ {algo_name.upper()} completed successfully!")
        print(f"   Result type: {type(result)}")
        print(f"   Result attributes: {[attr for attr in dir(result) if not attr.startswith('_')]}")
        if hasattr(result, 'best_fitness_'):
            print(f"   Best fitness: {result.best_fitness_:.6f}")
            print(f"   Best solution: {result.best_solution_[:3]}...")  # First 3 values
        else:
            print(f"   No best_fitness_ attribute found")
        
    except Exception as e:
        print(f"❌ {algo_name.upper()} failed: {e}")

print("\n✨ Algorithm testing completed!")