# 🧬 MHA Toolbox Pro

**Professional Metaheuristic Algorithm Library for Python**

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Version](https://img.shields.io/badge/version-2.0.0-orange)](https://github.com/Achyut103040/MHA-Algorithm)

A comprehensive, professional-grade library featuring **95+ metaheuristic optimization algorithms** across 7 categories, including 9 powerful hybrid combinations.

---

## ✨ Key Features

- **95+ Algorithms**: Swarm Intelligence, Evolutionary, Bio-Inspired, Physics-Based, Human Behavior, Mathematical, and Hybrid
- **9 Hybrid Algorithms**: Advanced combinations like PSO-GA, GWO-PSO, WOA-GA, SMA-DE, ABC-DE, and more
- **Professional Interface**: Beautiful Streamlit web interface with real-time visualization
- **Easy to Use**: Simple API with intelligent defaults
- **Export Formats**: CSV, JSON, NPZ support
- **Session Management**: Track and manage multiple optimization runs
- **Benchmark Functions**: Built-in test functions (Sphere, Rosenbrock, Rastrigin, Ackley, etc.)
- **Performance Analytics**: Convergence curves, statistical analysis, comparison tools

---

## 📦 Installation

### Option 1: Install from PyPI (Recommended)

```bash
pip install mha-toolbox
```

### Option 2: Install from GitHub

```bash
pip install git+https://github.com/Achyut103040/MHA-Algorithm.git
```

### Option 3: Install from Source

```bash
git clone https://github.com/Achyut103040/MHA-Algorithm.git
cd MHA-Algorithm
pip install -e .
```

---

## 🚀 Quick Start

### Using the Web Interface

```bash
# Launch the web interface
mha-web

# Or use streamlit directly
streamlit run mha_toolbox_pro_ultimate.py

# On Windows, use the batch file
.\launch.bat
```

Then open your browser to `http://localhost:8501`

### Python API - Simple Example

```python
from mha_toolbox.algorithms import PSO
import numpy as np

# Define objective function to minimize
def sphere(x):
    return np.sum(x**2)

# Set search space bounds
bounds = np.array([[-10, 10]] * 5)  # 5 dimensions

# Create and run optimizer
pso = PSO(objective_func=sphere, bounds=bounds, n_particles=30, max_iter=100)
best_position, best_fitness = pso.optimize()

print(f"Best fitness: {best_fitness:.6e}")
print(f"Best position: {best_position}")
```

### Algorithm Comparison

```python
from mha_toolbox.algorithms import PSO, GWO, WOA, GA, DE
import numpy as np

def rastrigin(x):
    A = 10
    return A * len(x) + np.sum(x**2 - A * np.cos(2 * np.pi * x))

bounds = np.array([[-5.12, 5.12]] * 10)

# Compare multiple algorithms
algorithms = {
    'PSO': PSO(rastrigin, bounds, n_particles=50, max_iter=200),
    'GWO': GWO(rastrigin, bounds, n_wolves=50, max_iter=200),
    'WOA': WOA(rastrigin, bounds, n_whales=50, max_iter=200),
    'GA': GA(rastrigin, bounds, pop_size=50, max_iter=200),
    'DE': DE(rastrigin, bounds, pop_size=50, max_iter=200)
}

for name, algo in algorithms.items():
    best_pos, best_fit = algo.optimize()
    print(f"{name}: {best_fit:.6f}")
```

### Using Hybrid Algorithms

```python
from mha_toolbox.algorithms.hybrid import GWO_PSO_Hybrid, WOA_GA_Hybrid

# GWO-PSO Hybrid - combines exploration of GWO with exploitation of PSO
hybrid = GWO_PSO_Hybrid(objective_func=sphere, bounds=bounds, n_agents=40, max_iter=150)
best_pos, best_fit = hybrid.optimize()
print(f"Hybrid result: {best_fit:.6e}")
```

### Run Example Script

```bash
python examples/quick_start.py
```

---

## 📚 Algorithm Categories

### Swarm Intelligence (15 algorithms)
- PSO (Particle Swarm Optimization)
- ABC (Artificial Bee Colony)
- ACO (Ant Colony Optimization)
- WOA (Whale Optimization Algorithm)
- GWO (Grey Wolf Optimizer)
- BA (Bat Algorithm)
- FA (Firefly Algorithm)
- And 8 more...

### Evolutionary (8 algorithms)
- GA (Genetic Algorithm)
- DE (Differential Evolution)
- EO (Equilibrium Optimizer)
- EPO (Emperor Penguin Optimizer)
- And 4 more...

### Bio-Inspired (32 algorithms)
- HHO (Harris Hawks Optimization)
- ALO (Ant Lion Optimizer)
- MPA (Marine Predators Algorithm)
- DA (Dragonfly Algorithm)
- DMOA (Dwarf Mongoose Optimization)
- HBA (Honey Badger Algorithm)
- HGS (Hunger Games Search)
- And 25 more...

### Physics-Based (17 algorithms)
- SA (Simulated Annealing)
- GSA (Gravitational Search Algorithm)
- MVO (Multi-Verse Optimizer)
- ASO (Atom Search Optimization)
- TWO (Tug of War Optimization)
- And 12 more...

### Human Behavior (10 algorithms)
- TLBO (Teaching-Learning-Based Optimization)
- ICA (Imperialist Competitive Algorithm)
- SOS (Symbiotic Organisms Search)
- And 7 more...

### Mathematical (13 algorithms)
- HS (Harmony Search)
- TS (Tabu Search)
- HC (Hill Climbing)
- RUN (RUNge Kutta Optimizer)
- And 9 more...

### Hybrid Algorithms (9 algorithms)
- **PSO-GA**: Particle Swarm + Genetic Algorithm
- **GWO-PSO**: Grey Wolf + Particle Swarm
- **WOA-GA**: Whale + Genetic Algorithm
- **SMA-DE**: Slime Mould + Differential Evolution
- **ABC-DE**: Artificial Bee Colony + Differential Evolution
- **WOA-SMA**: Whale + Slime Mould
- **GA-SA**: Genetic Algorithm + Simulated Annealing
- **DE-PSO**: Differential Evolution + Particle Swarm
- **FA-GA**: Firefly + Genetic Algorithm

---

## 💡 Usage Examples

### Comparing Multiple Algorithms

```python
from mha_toolbox.algorithms import PSO, GWO, WOA, SMA
import numpy as np

# Define benchmark function
def rastrigin(x):
    return 10*len(x) + np.sum(x**2 - 10*np.cos(2*np.pi*x))

algorithms = [
    ('PSO', PSO(30, 100)),
    ('GWO', GWO(30, 100)),
    ('WOA', WOA(30, 100)),
    ('SMA', SMA(30, 100))
]

results = {}
for name, algo in algorithms:
    best_pos, best_fit, conv, _, _ = algo.optimize(
        rastrigin, 
        bounds=(-5.12, 5.12), 
        dimension=10
    )
    results[name] = {'fitness': best_fit, 'convergence': conv}
    print(f"{name}: {best_fit:.6f}")
```

### Export Results

```python
import json
import pandas as pd
import numpy as np

# Save as JSON
with open('results.json', 'w') as f:
    json.dump(results, f, indent=2)

# Save as CSV
df = pd.DataFrame([
    {'algorithm': name, 'best_fitness': data['fitness']}
    for name, data in results.items()
])
df.to_csv('results.csv', index=False)

# Save as NPZ
npz_data = {name: np.array(data['convergence']) 
            for name, data in results.items()}
np.savez('results.npz', **npz_data)
```

---

## 🎨 Web Interface Features

The Streamlit-based web interface provides:

- **Session Management**: Start/stop optimization sessions
- **Algorithm Selection**: Browse and select from 95+ algorithms by category
- **Configuration**: Adjust population size, iterations, bounds
- **Real-time Visualization**: Watch convergence in real-time
- **Results Dashboard**: Compare algorithm performance
- **Export Options**: Download results in multiple formats
- **Professional Design**: Clean, modern interface with gradient themes

---

## 📊 Benchmark Functions

Built-in test functions include:

- **Sphere**: Unimodal, smooth landscape
- **Rosenbrock**: Narrow valley, challenging
- **Rastrigin**: Highly multimodal, many local optima
- **Ackley**: Nearly flat outer region, deep hole at center
- **Griewank**: Product and sum components, multimodal
- **Schwefel**: Deceptive, multimodal

---

## 🔬 Research Applications

Perfect for:

- Function Optimization
- Feature Selection
- Hyperparameter Tuning
- Neural Architecture Search
- Portfolio Optimization
- Scheduling Problems
- Engineering Design Optimization
- Machine Learning Model Selection

---

## 📖 Documentation

For detailed documentation, visit:
- [GitHub Wiki](https://github.com/Achyut103040/MHA-Algorithm/wiki)
- [API Reference](https://github.com/Achyut103040/MHA-Algorithm/docs)
- [Tutorial Notebook](MHA_Toolbox_Tutorial.ipynb)

---

## 📖 Documentation

- **[USER_GUIDE.md](USER_GUIDE.md)** - Complete usage guide with examples
- **[PUBLISHING_GUIDE.md](PUBLISHING_GUIDE.md)** - How to publish and share
- **[IMPORT_FIXES_COMPLETED.md](IMPORT_FIXES_COMPLETED.md)** - Technical details
- **[BUILD_GUIDE.md](BUILD_GUIDE.md)** - Building from source
- **[QUICK_START.md](QUICK_START.md)** - Getting started quickly

### Examples

Run the example script to see the library in action:

```bash
python examples/quick_start.py
```

This will demonstrate:
- Basic optimization with PSO
- Algorithm comparison
- Visualization generation
- Hybrid algorithms

---

## 🎯 Use Cases

### Function Optimization
```python
from mha_toolbox.algorithms import GWO
import numpy as np

def ackley(x):
    a, b, c = 20, 0.2, 2 * np.pi
    d = len(x)
    sum1 = np.sum(x**2)
    sum2 = np.sum(np.cos(c * x))
    return -a * np.exp(-b * np.sqrt(sum1/d)) - np.exp(sum2/d) + a + np.e

bounds = np.array([[-5, 5]] * 10)
gwo = GWO(ackley, bounds, n_wolves=50, max_iter=200)
best_pos, best_fit = gwo.optimize()
```

### Feature Selection
```python
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from mha_toolbox.algorithms import PSO

# Load data
X, y = load_breast_cancer(return_X_y=True)

# Define fitness function
def feature_selection_fitness(mask):
    binary_mask = (mask > 0.5).astype(int)
    if np.sum(binary_mask) == 0:
        return 1.0
    
    X_selected = X[:, binary_mask == 1]
    clf = RandomForestClassifier(n_estimators=10)
    return 1 - cross_val_score(clf, X_selected, y, cv=3).mean()

# Optimize
bounds = np.array([[0, 1]] * X.shape[1])
pso = PSO(feature_selection_fitness, bounds, n_particles=30, max_iter=50)
best_mask, best_fitness = pso.optimize()

selected_features = (best_mask > 0.5).astype(int)
print(f"Selected {np.sum(selected_features)} features")
```

### Hyperparameter Tuning
```python
from sklearn.svm import SVC
from mha_toolbox.algorithms import WOA

def svm_fitness(params):
    C, gamma = params
    clf = SVC(C=C, gamma=gamma)
    return 1 - cross_val_score(clf, X_train, y_train, cv=3).mean()

bounds = np.array([[0.1, 100], [0.001, 10]])
woa = WOA(svm_fitness, bounds, n_whales=20, max_iter=30)
best_params, best_score = woa.optimize()
print(f"Best C={best_params[0]:.3f}, gamma={best_params[1]:.6f}")
```

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 Citation

If you use MHA Toolbox in your research, please cite:

```bibtex
@software{mha_toolbox,
  title = {MHA Toolbox: Professional Metaheuristic Algorithm Library},
  author = {MHA Development Team},
  year = {2025},
  version = {2.0.0},
  url = {https://github.com/Achyut103040/MHA-Algorithm}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

This library builds upon decades of research in metaheuristic optimization. We acknowledge the original authors of each algorithm and the broader research community.

---

## 📧 Contact

- **GitHub**: [Achyut103040](https://github.com/Achyut103040)
- **Issues**: [Bug Tracker](https://github.com/Achyut103040/MHA-Algorithm/issues)
- **Email**: mha.toolbox@gmail.com

---

## 🌟 Star History

If you find this library useful, please consider giving it a star ⭐!

---

**Made with ❤️ by the MHA Development Team**
