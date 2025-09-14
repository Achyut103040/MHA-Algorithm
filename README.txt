# MHA Toolbox - Advanced Metaheuristic Algorithms Library


## 📋 Features

✅ mha.ao(), mha.pso(), mha.sca(), etc.
✅ Strict data requirements with intelligent parameter defaults
✅ Professional result management with automatic saving
✅ Advanced visualization and statistical analysis
✅ Support for feature selection and function optimization
✅ Flexible parameter handling (positional, keyword, mixed)
✅ Comprehensive documentation and examples

## 🚀 Quick Start

1. **Run the visualization demo:**
   ```bash
   python visualization_demo.py
   ```

2. **Basic usage in your code:**
   ```python
   import numpy as np
   import mha_toolbox as mha
   from sklearn.datasets import load_breast_cancer
   
   # Load real dataset
   X, y = load_breast_cancer(return_X_y=True)
   
   # Use any algorithm with simple function calls
   result = mha.ao(X, y)  # Aquila Optimizer
   result = mha.pso(X, y, 30, 50)  # PSO with custom parameters
   
   # View results with advanced visualizations
   result.summary()
   result.plot_convergence()
   result.plot_advanced('all')  # All visualization types
   ```

## 🎨 Advanced Visualization

The library includes comprehensive visualization capabilities:

```python
# Basic convergence plot
result.plot_convergence()

# Advanced visualization suite
result.plot_advanced('convergence')    # Detailed convergence analysis
result.plot_advanced('exploration')    # Exploration-exploitation analysis
result.plot_advanced('statistical')    # Statistical distribution analysis
result.plot_advanced('trajectory')     # Search trajectory visualization
result.plot_advanced('all')           # All visualization types

# Algorithm comparison
from mha_toolbox.utils.visualizations import AdvancedVisualizer

result1 = mha.ao(X, y)
result2 = mha.pso(X, y)
result3 = mha.sca(X, y)

visualizer = AdvancedVisualizer([result1, result2, result3])
visualizer.convergence_plot()
visualizer.box_plot()
visualizer.statistical_analysis_plot()
```

## 📊 Statistical Analysis

Get comprehensive statistics and comparisons:

```python
# Get detailed statistics
stats = result.get_statistics()
report = result.statistical_report()

# Compare algorithms
comparison = result1.compare_with(result2)

# Multi-algorithm statistical analysis
from mha_toolbox.utils.statistics import StatisticalAnalyzer
analyzer = StatisticalAnalyzer([result1, result2, result3])
ranking = analyzer.performance_ranking()
tests = analyzer.statistical_tests()
```

## 📁 Available Algorithms

Call any algorithm as a simple function:
- `mha.ao()` - Aquila Optimizer
- `mha.pso()` - Particle Swarm Optimization
- `mha.gwo()` - Grey Wolf Optimizer
- `mha.sca()` - Sine Cosine Algorithm
- `mha.woa()` - Whale Optimization Algorithm
- `mha.ssa()` - Salp Swarm Algorithm
... and more algorithms being added!

## 🔧 Flexible Parameter Combinations

The API supports ANY parameter combination:

```python
# Feature Selection Examples
mha.ao(X, y)                           # Simple
mha.pso(X=data, y=labels)               # Keyword
mha.gwo(X, y, 30, 50)                  # Positional  
mha.sca(X=data, max_iterations=100)     # Mixed

# Function Optimization Examples
mha.pso(objective_function=sphere_func, dimensions=10)
mha.ao(objective_function=func, 20, 100, 15)  # pop_size, max_iter, dims

# Algorithm-Specific Parameters
mha.pso(X, y, c1=2.0, c2=2.0, w=0.9)   # PSO specific
mha.sca(X, y, a=2.0)                    # SCA specific
mha.ao(X, y, alpha=0.1, delta=0.1)     # AO specific
```

## 🗂️ Automatic Result Management

Results are automatically organized in structured directories:
- `results/single_algorithms/` - Individual algorithm results
- `results/hybrid_algorithms/` - Hybrid algorithm results  
- `results/function_optimization/` - Function optimization results

Each result includes:
- JSON file with complete results and parameters
- CSV file with convergence curve data
- Automatic timestamping and metadata
- Professional result summary and statistics

## 🎯 Visualization Types

The library supports multiple visualization types:

1. **Convergence Analysis** - Multi-panel convergence analysis with log-scale, rate, and moving average
2. **Statistical Distribution** - Box plots, histograms, Q-Q plots, and violin plots
3. **Exploration-Exploitation** - Population diversity, exploitation intensity, and phase analysis
4. **Search Trajectory** - 2D/3D visualization of algorithm search patterns
5. **Performance Comparison** - Heatmaps, statistical tests, and algorithm ranking
6. **Feature Selection Analysis** - Feature importance and selection pattern visualization

## 🔬 Research Features

Perfect for academic and research use:
- Publication-quality plots with customizable styling
- Statistical significance testing (Wilcoxon, t-tests)
- Multi-run analysis and confidence intervals
- Performance benchmarking and ranking
- Detailed convergence and behavior analysis
- Algorithm comparison and visualization

## 📚 Example Scripts

- `main.py` - Basic usage demonstration
- `visualization_demo.py` - Complete visualization showcase
- `simple_usage_examples.py` - Simple API demonstrations
- `test_parameter_flexibility.py` - Parameter handling tests

## 💻 Installation

1. Clone or download this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run demo: `python visualization_demo.py`

## 📈 Professional Results

Every optimization run automatically provides:
- Comprehensive statistical analysis
- Professional visualization suite
- Detailed performance metrics
- Comparison and ranking tools
- Publication-ready plots and reports

This library is designed for both educational use and serious research applications.
- `mha.fa()` - Firefly Algorithm
- `mha.ba()` - Bat Algorithm
- `mha.aco()` - Ant Colony Optimization

## Flexible Parameter Combinations

The API supports ANY parameter combination:

```python
# Feature Selection
mha.ao(X, y)                           # Simple
mha.pso(X=data, y=labels)               # Keyword
mha.gwo(X, y, 30, 50)                  # Positional
mha.sca(X=data, max_iterations=100)     # Mixed

# Function Optimization  
mha.ao(objective_function=func, dimensions=5)
mha.pso(objective_function=func, 25, 100)  # positional params

# Any combination works!
```

## File Structure

**Essential files only:**
- `main.py` - Main demo (run this!)
- `mha_toolbox/` - Core library
- `simple_usage_examples.py` - Simple examples
- `test_parameter_flexibility.py` - Comprehensive tests
- `requirements.txt` - Dependencies
- `results/` - Auto-saved results

## How to Use

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run main demo:**
   ```bash
   python main.py
   ```

3. **Use in your project:**
   ```python
   import mha_toolbox as mha
   result = mha.ao(X=your_data, y=your_labels)
   print(f"Best fitness: {result.best_fitness}")
   print(f"Selected features: {result.n_selected_features}")
   ```

## Key Features

✅ **-like simplicity** - `mha.ao()`, `mha.pso()`, etc.  
✅ **Maximum flexibility** - Any parameter combination works  
✅ **Automatic result saving** - JSON/CSV outputs in `results/`  
✅ **Professional result objects** - Detailed analysis and visualization  
✅ **Both feature selection and function optimization**  
✅ **Intelligent defaults** - Works with minimal parameters  

That's it! Clean, simple, and powerful! 🚀
