# MHA Toolbox - Complete User Guide

## Table of Contents
1. [Installation](#installation)
2. [Basic Usage](#basic-usage)
3. [Using Specific Algorithms](#using-specific-algorithms)
4. [Getting Visualizations](#getting-visualizations)
5. [Web Interface](#web-interface)
6. [Advanced Features](#advanced-features)

---

## Installation

### Option 1: Install from PyPI (After Publishing)
```bash
pip install mha-toolbox
```

### Option 2: Install from Local Source
```bash
# Clone the repository
git clone https://github.com/Achyut103040/MHA-Algorithm.git
cd MHA-Algorithm

# Install in editable mode
pip install -e .
```

### Option 3: Install from GitHub
```bash
pip install git+https://github.com/Achyut103040/MHA-Algorithm.git
```

---

## Basic Usage

### Simple Example - Optimize a Function

```python
import numpy as np
from mha_toolbox.algorithms import PSO

# Define your objective function (minimize)
def sphere_function(x):
    return np.sum(x**2)

# Define bounds for each dimension
bounds = np.array([[-10, 10]] * 5)  # 5 dimensions, each in [-10, 10]

# Create and run PSO
pso = PSO(
    objective_func=sphere_function,
    bounds=bounds,
    n_particles=30,
    max_iter=100
)

# Run optimization
best_position, best_fitness = pso.optimize()

print(f"Best solution: {best_position}")
print(f"Best fitness: {best_fitness}")
```

---

## Using Specific Algorithms

### Available Algorithms (104 Total)

```python
from mha_toolbox.algorithms import (
    # Popular algorithms with short aliases
    PSO,   # Particle Swarm Optimization
    GA,    # Genetic Algorithm
    GWO,   # Grey Wolf Optimizer
    WOA,   # Whale Optimization Algorithm
    DE,    # Differential Evolution
    FA,    # Firefly Algorithm
    BA,    # Bat Algorithm
    ACO,   # Ant Colony Optimization
    
    # Full names also work
    ParticleSwarmOptimization,
    GeneticAlgorithm,
    GreyWolfOptimizer,
    # ... and 90+ more!
)
```

### Example: Using Different Algorithms

```python
import numpy as np
from mha_toolbox.algorithms import PSO, GWO, WOA, GA, DE

# Define problem
def rastrigin(x):
    A = 10
    return A * len(x) + np.sum(x**2 - A * np.cos(2 * np.pi * x))

bounds = np.array([[-5.12, 5.12]] * 10)

# Try different algorithms
algorithms = {
    'PSO': PSO(rastrigin, bounds, n_particles=50, max_iter=200),
    'GWO': GWO(rastrigin, bounds, n_wolves=50, max_iter=200),
    'WOA': WOA(rastrigin, bounds, n_whales=50, max_iter=200),
    'GA': GA(rastrigin, bounds, pop_size=50, max_iter=200),
    'DE': DE(rastrigin, bounds, pop_size=50, max_iter=200)
}

results = {}
for name, algo in algorithms.items():
    best_pos, best_fit = algo.optimize()
    results[name] = best_fit
    print(f"{name}: {best_fit:.6f}")

# Find best algorithm
best_algo = min(results, key=results.get)
print(f"\nBest algorithm: {best_algo}")
```

---

## Getting Visualizations

### Method 1: Using Built-in Visualization

```python
import numpy as np
import matplotlib.pyplot as plt
from mha_toolbox.algorithms import PSO

def sphere(x):
    return np.sum(x**2)

bounds = np.array([[-10, 10]] * 2)
pso = PSO(sphere, bounds, n_particles=30, max_iter=100)

# Run optimization and collect history
best_pos, best_fit = pso.optimize()

# Plot convergence curve
if hasattr(pso, 'fitness_history'):
    plt.figure(figsize=(10, 6))
    plt.plot(pso.fitness_history)
    plt.xlabel('Iteration')
    plt.ylabel('Best Fitness')
    plt.title('PSO Convergence Curve')
    plt.grid(True)
    plt.show()
```

### Method 2: Manual Tracking and Visualization

```python
import numpy as np
import matplotlib.pyplot as plt
from mha_toolbox.algorithms import GWO

def rosenbrock(x):
    return sum(100*(x[1:]-x[:-1]**2)**2 + (1-x[:-1])**2)

bounds = np.array([[-5, 5]] * 5)

# Track fitness over iterations
fitness_history = []

class GWOWithTracking(GWO):
    def optimize(self):
        for iteration in range(self.max_iter):
            # Store best fitness each iteration
            fitness_history.append(self.alpha_score)
            # Continue normal optimization
            super().optimize()
        return self.alpha_pos, self.alpha_score

gwo = GWO(rosenbrock, bounds, n_wolves=30, max_iter=100)
best_pos, best_fit = gwo.optimize()

# Create multiple plots
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Convergence curve
axes[0, 0].plot(fitness_history)
axes[0, 0].set_title('Convergence Curve')
axes[0, 0].set_xlabel('Iteration')
axes[0, 0].set_ylabel('Best Fitness')
axes[0, 0].grid(True)

# Log scale convergence
axes[0, 1].semilogy(fitness_history)
axes[0, 1].set_title('Convergence (Log Scale)')
axes[0, 1].set_xlabel('Iteration')
axes[0, 1].set_ylabel('Best Fitness (log)')
axes[0, 1].grid(True)

# Box plot of final solutions
axes[1, 0].bar(range(len(best_pos)), best_pos)
axes[1, 0].set_title('Best Solution Parameters')
axes[1, 0].set_xlabel('Dimension')
axes[1, 0].set_ylabel('Value')

# Summary text
axes[1, 1].axis('off')
summary = f"""
Optimization Results
====================
Algorithm: Grey Wolf Optimizer
Iterations: {len(fitness_history)}
Best Fitness: {best_fit:.6e}
Dimensions: {len(best_pos)}
"""
axes[1, 1].text(0.1, 0.5, summary, fontsize=12, family='monospace')

plt.tight_layout()
plt.show()
```

### Method 3: Compare Multiple Algorithms with Visualization

```python
import numpy as np
import matplotlib.pyplot as plt
from mha_toolbox.algorithms import PSO, GWO, WOA, GA

def ackley(x):
    a, b, c = 20, 0.2, 2 * np.pi
    d = len(x)
    sum1 = np.sum(x**2)
    sum2 = np.sum(np.cos(c * x))
    return -a * np.exp(-b * np.sqrt(sum1/d)) - np.exp(sum2/d) + a + np.e

bounds = np.array([[-5, 5]] * 10)

# Run multiple algorithms and track convergence
algorithms = ['PSO', 'GWO', 'WOA', 'GA']
all_histories = {}

for algo_name in algorithms:
    if algo_name == 'PSO':
        algo = PSO(ackley, bounds, n_particles=40, max_iter=150)
    elif algo_name == 'GWO':
        algo = GWO(ackley, bounds, n_wolves=40, max_iter=150)
    elif algo_name == 'WOA':
        algo = WOA(ackley, bounds, n_whales=40, max_iter=150)
    else:  # GA
        algo = GA(ackley, bounds, pop_size=40, max_iter=150)
    
    best_pos, best_fit = algo.optimize()
    if hasattr(algo, 'fitness_history'):
        all_histories[algo_name] = algo.fitness_history
    print(f"{algo_name} final fitness: {best_fit:.6f}")

# Plot comparison
plt.figure(figsize=(12, 6))
for name, history in all_histories.items():
    plt.plot(history, label=name, linewidth=2)

plt.xlabel('Iteration', fontsize=12)
plt.ylabel('Best Fitness', fontsize=12)
plt.title('Algorithm Comparison on Ackley Function', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

---

## Web Interface

### Launch the Web Interface

The toolbox includes a comprehensive **Streamlit web interface** for easy visualization and experimentation.

```bash
# Method 1: Using the command-line tool
mha-web

# Method 2: Direct streamlit command
streamlit run mha_toolbox_pro_ultimate.py

# Method 3: Using the launch script (Windows)
.\launch.bat
```

### Web Interface Features:

1. **Algorithm Selection**: Choose from 95+ algorithms organized by category
2. **Problem Configuration**: 
   - Select benchmark functions (Sphere, Rastrigin, Rosenbrock, Ackley, etc.)
   - Set dimensions, bounds, population size, iterations
3. **Real-time Visualization**:
   - Convergence curves
   - 2D/3D solution space visualization
   - Performance metrics
4. **Comparison Mode**: Compare multiple algorithms side-by-side
5. **Export Results**: Download results as CSV, JSON, or NPZ files
6. **Session Management**: Save and load optimization sessions

### Using Web Interface Programmatically

```python
import streamlit as st
from mha_toolbox.frontend import run_app

# Launch the web interface
if __name__ == "__main__":
    run_app()
```

---

## Advanced Features

### 1. Hybrid Algorithms

```python
from mha_toolbox.algorithms.hybrid import (
    GWO_PSO_Hybrid,
    WOA_GA_Hybrid,
    ABC_DE_Hybrid,
    SMA_DE_Hybrid,
    FA_GA_Hybrid
)

# Use hybrid algorithm
hybrid = GWO_PSO_Hybrid(
    objective_func=your_function,
    bounds=bounds,
    n_agents=50,
    max_iter=200
)

best_pos, best_fit = hybrid.optimize()
```

### 2. Custom Objective Functions

```python
import numpy as np
from mha_toolbox.algorithms import PSO

# Multi-modal function
def custom_function(x):
    # Your custom optimization problem
    penalty = 0
    if np.any(x < 0):  # Add constraints
        penalty = 1e6
    return np.sum(x**2) + np.sin(x[0]) * np.cos(x[1]) + penalty

bounds = np.array([[-10, 10]] * 5)
pso = PSO(custom_function, bounds, n_particles=50, max_iter=200)
best_pos, best_fit = pso.optimize()
```

### 3. Feature Selection Example

```python
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from mha_toolbox.algorithms import PSO

# Load dataset
data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Define fitness function for feature selection
def feature_selection_fitness(mask):
    # Convert continuous values to binary
    binary_mask = (mask > 0.5).astype(int)
    
    # Need at least one feature
    if np.sum(binary_mask) == 0:
        return 1.0
    
    # Select features
    X_train_selected = X_train[:, binary_mask == 1]
    X_test_selected = X_test[:, binary_mask == 1]
    
    # Train classifier
    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    clf.fit(X_train_selected, y_train)
    
    # Calculate fitness (we minimize, so use 1 - accuracy)
    y_pred = clf.predict(X_test_selected)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Balance accuracy and number of features
    feature_ratio = np.sum(binary_mask) / len(binary_mask)
    fitness = (1 - accuracy) + 0.01 * feature_ratio
    
    return fitness

# Run PSO for feature selection
bounds = np.array([[0, 1]] * X.shape[1])
pso = PSO(feature_selection_fitness, bounds, n_particles=30, max_iter=50)
best_mask, best_fitness = pso.optimize()

# Get selected features
selected_features = (best_mask > 0.5).astype(int)
print(f"Selected {np.sum(selected_features)} out of {len(selected_features)} features")
print(f"Feature indices: {np.where(selected_features)[0]}")
```

### 4. Parallel Optimization

```python
from multiprocessing import Pool
import numpy as np
from mha_toolbox.algorithms import PSO, GWO, WOA

def run_algorithm(algo_config):
    algo_name, objective_func, bounds = algo_config
    
    if algo_name == 'PSO':
        algo = PSO(objective_func, bounds, n_particles=30, max_iter=100)
    elif algo_name == 'GWO':
        algo = GWO(objective_func, bounds, n_wolves=30, max_iter=100)
    else:  # WOA
        algo = WOA(objective_func, bounds, n_whales=30, max_iter=100)
    
    best_pos, best_fit = algo.optimize()
    return algo_name, best_fit

# Define problem
def sphere(x):
    return np.sum(x**2)

bounds = np.array([[-10, 10]] * 5)

# Run algorithms in parallel
configs = [
    ('PSO', sphere, bounds),
    ('GWO', sphere, bounds),
    ('WOA', sphere, bounds)
]

with Pool(3) as pool:
    results = pool.map(run_algorithm, configs)

for name, fitness in results:
    print(f"{name}: {fitness:.6e}")
```

### 5. Save and Load Results

```python
import numpy as np
import json
from mha_toolbox.algorithms import PSO

# Run optimization
def sphere(x):
    return np.sum(x**2)

bounds = np.array([[-10, 10]] * 5)
pso = PSO(sphere, bounds, n_particles=30, max_iter=100)
best_pos, best_fit = pso.optimize()

# Save results
results = {
    'algorithm': 'PSO',
    'best_position': best_pos.tolist(),
    'best_fitness': float(best_fit),
    'bounds': bounds.tolist(),
    'parameters': {
        'n_particles': 30,
        'max_iter': 100
    }
}

with open('optimization_results.json', 'w') as f:
    json.dump(results, f, indent=2)

# Save as numpy file
np.savez('optimization_results.npz',
         best_position=best_pos,
         best_fitness=best_fit,
         bounds=bounds)

# Load results
with open('optimization_results.json', 'r') as f:
    loaded_results = json.load(f)

loaded_data = np.load('optimization_results.npz')
print(f"Loaded fitness: {loaded_data['best_fitness']}")
```

---

## Command-Line Interface

```bash
# Run single algorithm
mha-toolbox --algorithm PSO --function sphere --dimensions 10 --iterations 100

# Compare multiple algorithms
mha-toolbox compare --algorithms PSO GWO WOA --function rastrigin

# Launch web interface
mha-web

# Run demo
mha-demo
```

---

## API Reference

### Algorithm Base Class

All algorithms inherit from `BaseOptimizer` with this interface:

```python
class BaseOptimizer:
    def __init__(self, objective_func, bounds, **kwargs):
        """
        Args:
            objective_func: Function to minimize
            bounds: np.array of shape (n_dims, 2) with [min, max] for each dimension
            **kwargs: Algorithm-specific parameters
        """
        pass
    
    def optimize(self):
        """
        Run optimization
        
        Returns:
            best_position: np.array of best solution found
            best_fitness: float of best fitness value
        """
        pass
```

### Common Parameters

Most algorithms support these parameters:
- `objective_func`: Function to minimize
- `bounds`: Search space boundaries
- `max_iter`: Maximum iterations (default: 100)
- Population size: `n_particles`, `pop_size`, `n_wolves`, etc. (default: 30-50)
- Random seed: `random_state` (for reproducibility)

---

## Troubleshooting

### Import Errors
```python
# Use short aliases if full names have issues
from mha_toolbox.algorithms import PSO, GWO, WOA
# Instead of:
# from mha_toolbox.algorithms import ParticleSwarmOptimization
```

### Memory Issues
```python
# Reduce population size or iterations
pso = PSO(func, bounds, n_particles=20, max_iter=50)  # Instead of 100, 200
```

### Convergence Issues
```python
# Try different algorithms
# Increase population size
# Increase iterations
# Adjust bounds
```

---

## Support and Documentation

- **GitHub**: https://github.com/Achyut103040/MHA-Algorithm
- **Issues**: Report bugs and request features on GitHub Issues
- **Documentation**: See README.md and code docstrings

---

**Happy Optimizing! 🚀**
