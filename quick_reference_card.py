"""
🎯 MHA TOOLBOX - QUICK REFERENCE CARD
====================================

This is your cheat sheet for selecting and using algorithms!
Copy-paste these examples and modify as needed.

📋 AVAILABLE ALGORITHMS:
- mha.ao()    # Aquila Optimizer
- mha.pso()   # Particle Swarm Optimization  
- mha.gwo()   # Grey Wolf Optimizer
- mha.woa()   # Whale Optimization Algorithm
- mha.sca()   # Sine Cosine Algorithm
- mha.ga()    # Genetic Algorithm
- mha.de()    # Differential Evolution
- mha.fa()    # Firefly Algorithm
- mha.ba()    # Bat Algorithm
- mha.aco()   # Ant Colony Optimization

🚀 BASIC USAGE PATTERNS:
========================

1. SIMPLEST USAGE:
------------------
import mha_toolbox as mha
from sklearn.datasets import load_breast_cancer

X, y = load_breast_cancer(return_X_y=True)
result = mha.ao(X, y)                    # Just pass data
print(f"Fitness: {result.best_fitness}")
result.plot_convergence()                # Visualize

2. WITH PARAMETERS:
-------------------
result = mha.pso(X, y, 
                 population_size=30,     # 30 particles
                 max_iterations=50)      # 50 iterations

3. ALGORITHM-SPECIFIC PARAMETERS:
---------------------------------
result = mha.pso(X, y, 
                 c1=2.0,                 # Cognitive factor
                 c2=1.5,                 # Social factor  
                 w=0.9)                  # Inertia weight

result = mha.ga(X, y,
                crossover_rate=0.8,      # GA crossover
                mutation_rate=0.1)       # GA mutation

4. FUNCTION OPTIMIZATION:
-------------------------
def my_function(x):
    return np.sum(x**2)                  # Sphere function

result = mha.ao(objective_function=my_function, 
                dimensions=10)           # 10D optimization

🎨 VISUALIZATION OPTIONS:
========================

1. BASIC PLOTS:
---------------
result.plot_convergence()               # Simple convergence plot
result.plot_convergence(title="My Plot") # With custom title

2. ADVANCED PLOTS:
------------------
result.plot_advanced('convergence')     # Detailed convergence
result.plot_advanced('exploration')     # Exploration analysis
result.plot_advanced('statistical')     # Statistical plots
result.plot_advanced('trajectory')      # Search trajectory
result.plot_advanced('all')            # All plots at once

3. MULTI-ALGORITHM COMPARISON:
------------------------------
from mha_toolbox.utils.visualizations import AdvancedVisualizer

result1 = mha.ao(X, y)
result2 = mha.pso(X, y)
result3 = mha.gwo(X, y)

visualizer = AdvancedVisualizer([result1, result2, result3])
visualizer.convergence_plot()           # Compare convergence
visualizer.box_plot()                   # Performance boxes

⚔️ ALGORITHM COMPARISON:
========================

# Quick comparison template
algorithms = [
    ('AO', mha.ao),
    ('PSO', mha.pso), 
    ('GWO', mha.gwo),
    ('WOA', mha.woa)
]

results = []
for name, algo_func in algorithms:
    result = algo_func(X, y, population_size=25, max_iterations=30)
    results.append(result)
    print(f"{name}: {result.best_fitness:.4f}")

# Visualize comparison
visualizer = AdvancedVisualizer(results)
visualizer.convergence_plot()

🎯 ALGORITHM SELECTION GUIDE:
=============================

WHEN TO USE WHICH ALGORITHM:
-----------------------------
🦅 AO (Aquila Optimizer):        Good all-around performance
🐝 PSO (Particle Swarm):         Fast, well-tested, many parameters
🐺 GWO (Grey Wolf):              Good exploration, social behavior
🐋 WOA (Whale Optimization):     Good for complex landscapes
🌊 SCA (Sine Cosine):            Simple, fast convergence
🧬 GA (Genetic Algorithm):       Classic, good for discrete problems
🔄 DE (Differential Evolution):  Good for continuous optimization
🔥 FA (Firefly Algorithm):       Good for multimodal problems
🦇 BA (Bat Algorithm):           Good for dynamic problems
🐜 ACO (Ant Colony):             Good for path-based problems

PARAMETER RECOMMENDATIONS:
--------------------------
📊 Population Size:     20-50 (start with 30)
🔄 Max Iterations:      30-100 (start with 50)
⚡ For quick tests:     pop=15, iter=20
🎯 For best results:    pop=50, iter=100
🚀 For competitions:    pop=100, iter=200+

📊 RESULT ANALYSIS:
==================

1. BASIC INFORMATION:
---------------------
print(f"Best fitness: {result.best_fitness}")
print(f"Selected features: {result.n_selected_features}")
print(f"Execution time: {result.execution_time:.2f}s")
print(f"Best solution: {result.best_solution}")

2. DETAILED STATISTICS:
-----------------------
stats = result.get_statistics()
print(f"Convergence rate: {stats['convergence_rate']}")
print(f"Improvement ratio: {stats['improvement_ratio']}")

3. ACCESS CONVERGENCE DATA:
---------------------------
print(f"Convergence curve: {result.convergence_curve}")
print(f"Best fitness over time: {result.fitness_history}")

💾 RESULT SAVING:
=================

# Results are automatically saved to:
# results/single_algorithms/result_AlgorithmName_timestamp.json
# results/single_algorithms/result_AlgorithmName_timestamp_convergence.csv

# Access saved results:
result_data = result.to_dict()           # Get as dictionary
result.save_results(custom_path="my_results.json")  # Custom save

🔧 COMMON PATTERNS:
==================

1. PARAMETER SWEEP:
-------------------
pop_sizes = [20, 30, 40, 50]
best_result = None
best_fitness = float('inf')

for pop_size in pop_sizes:
    result = mha.pso(X, y, population_size=pop_size, verbose=False)
    if result.best_fitness < best_fitness:
        best_fitness = result.best_fitness
        best_result = result

2. MULTIPLE RUNS:
-----------------
results = []
for run in range(10):                   # 10 independent runs
    result = mha.ao(X, y, verbose=False)
    results.append(result.best_fitness)

mean_fitness = np.mean(results)
std_fitness = np.std(results)
print(f"Mean ± Std: {mean_fitness:.4f} ± {std_fitness:.4f}")

3. INTERACTIVE SELECTION:
-------------------------
algorithms = {
    '1': ('Aquila Optimizer', mha.ao),
    '2': ('Particle Swarm', mha.pso),
    '3': ('Grey Wolf', mha.gwo)
}

print("Select algorithm:")
for key, (name, _) in algorithms.items():
    print(f"{key}. {name}")

choice = input("Enter choice (1-3): ")
name, algo_func = algorithms[choice]
result = algo_func(X, y)

🚨 TROUBLESHOOTING:
==================

COMMON ERRORS & SOLUTIONS:
---------------------------
❌ "X is required": Pass your data as first argument
✅ result = mha.ao(X, y)

❌ Poor convergence: Increase population or iterations  
✅ result = mha.ao(X, y, population_size=50, max_iterations=100)

❌ Too slow: Reduce population and iterations
✅ result = mha.ao(X, y, population_size=15, max_iterations=20)

❌ Bad results: Try different algorithm or parameters
✅ result = mha.pso(X, y, c1=2.0, c2=1.5)

PERFORMANCE TIPS:
-----------------
⚡ Set verbose=False for faster execution
🎯 Use smaller parameters for testing
📊 Always visualize results to understand behavior
🔧 Try multiple algorithms on your specific problem

🎉 QUICK START CHECKLIST:
=========================
□ Import: import mha_toolbox as mha
□ Load data: X, y = load_your_data()
□ Run algorithm: result = mha.algorithm_name(X, y)
□ Check result: print(result.best_fitness)
□ Visualize: result.plot_convergence()
□ Compare: Try multiple algorithms
□ Optimize: Adjust parameters for better results

🚀 YOU'RE READY TO GO!
======================
Start with: result = mha.ao(X, y)
Then explore the advanced features!
"""

if __name__ == "__main__":
    print("🎯 MHA TOOLBOX - QUICK REFERENCE CARD")
    print("="*50)
    print("This file contains all the essential usage patterns!")
    print("Use it as a reference while working with the MHA Toolbox.")
    print("\n📖 Key points:")
    print("✅ 10 algorithms available")
    print("✅ Simple usage: mha.algorithm_name(X, y)")
    print("✅ Advanced plots: result.plot_advanced('all')")
    print("✅ Auto-save: All results saved automatically")
    print("✅ Flexible: Works with any X, y data")