from mha_toolbox import MHAToolbox
from objective_functions.benchmark_functions import BenchmarkFunctions

# Quick interactive test
toolbox = MHAToolbox()
bf = BenchmarkFunctions()

print('Testing SCA on Sphere function...')
result = toolbox.optimize('SCA', bf.sphere, pop_size=20, max_iter=50, dim=3, lb=-5, ub=5)
print(f'Best score: {result["best_score"]:.6f}')
print(f'Best position: {result["best_position"]}')
