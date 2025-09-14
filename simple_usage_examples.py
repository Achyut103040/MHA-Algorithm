"""
Simple Usage Examples for MHA Toolbox
====================================

This file shows the simplest ways to use the MHA Toolbox with the TensorFlow-style API.
"""

import numpy as np
import mha_toolbox as mha

# Example 1: Feature Selection (most common use case)
print("Example 1: Feature Selection")
print("-" * 30)

# Create sample data
X = np.random.random((100, 8))  # 100 samples, 8 features
y = np.random.choice([0, 1], 100)  # binary classification

# Run Aquila Optimizer for feature selection
result = mha.ao(X=X, y=y, max_iterations=30)
print(f"Selected {result.n_selected_features} out of {X.shape[1]} features")
print(f"Best fitness: {result.best_fitness:.4f}")

# Example 2: Function Optimization
print("\nExample 2: Function Optimization")
print("-" * 35)

# Define a simple test function
def my_function(x):
    return sum(x**2)  # Sphere function

# Optimize the function
result = mha.pso(objective_function=my_function, dimensions=5, max_iterations=50)
print(f"Best solution: {result.best_solution}")
print(f"Best fitness: {result.best_fitness:.6f}")

# Example 3: Compare Multiple Algorithms
print("\nExample 3: Algorithm Comparison")
print("-" * 32)

algorithms = ['ao', 'pso', 'gwo', 'sca']
for algo in algorithms:
    result = getattr(mha, algo)(X=X, y=y, max_iterations=20)
    print(f"{algo.upper():3s}: {result.best_fitness:.4f} ({result.n_selected_features} features)")

print("\nThat's it! Easy as TensorFlow! 🚀")