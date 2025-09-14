"""Test the algorithm-specific API functions."""

import numpy as np
import mha_toolbox as mha

# Create sample data
X = np.random.random((50, 5))  # 50 samples, 5 features
y = np.random.choice([0, 1], 50)  # binary classification labels

print("Testing TensorFlow-style MHA API...")
print("=" * 50)

# Test listing algorithms
print("Available algorithms:")
algorithms = mha.list_algorithms()
print(algorithms)
print()

# Test AO algorithm
print("Testing Aquila Optimizer (AO)...")
try:
    ao_result = mha.ao(X=X, y=y, max_iterations=10, population_size=10)
    print(f"AO Result: {type(ao_result)}")
    print(f"Best fitness: {ao_result.best_fitness}")
    print()
except Exception as e:
    print(f"Error in AO: {e}")
    print()

# Test PSO algorithm  
print("Testing Particle Swarm Optimization (PSO)...")
try:
    pso_result = mha.pso(X=X, y=y, max_iterations=10, population_size=10)
    print(f"PSO Result: {type(pso_result)}")
    print(f"Best fitness: {pso_result.best_fitness}")
    print()
except Exception as e:
    print(f"Error in PSO: {e}")
    print()

# Test with benchmark function
print("Testing with benchmark function...")
try:
    benchmark_result = mha.ao(objective_function=lambda x: sum(x**2), dimensions=5, max_iterations=10, population_size=10)
    print(f"Benchmark Result: {type(benchmark_result)}")
    print(f"Best fitness: {benchmark_result.best_fitness}")
except Exception as e:
    print(f"Error with benchmark: {e}")

print("\nAPI test completed!")