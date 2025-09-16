# MHA Toolbox - Professional Metaheuristic Algorithm Library

A professional-grade Python library for metaheuristic optimization algorithms following TensorFlow-style design principles.

## Key Features

- **TensorFlow-style Interface**: Simple, one-line optimization calls
- **Data-First Design**: Input data is always the first positional argument
- **Intelligent Defaults**: All parameters are optional with smart automatic calculation
- **No User Interruption**: Zero prompts - everything works out of the box
- **Standardized Results**: Comprehensive model objects with built-in analysis tools
- **Professional Structure**: Object-oriented design with full documentation

## Quick Start

### Installation

```bash
git clone https://github.com/Achyut103040/MHA-Algorithm.git
cd MHA-Algorithm
pip install -r requirements.txt
```

### Basic Usage (TensorFlow-style)

```python
import mha_toolbox as mha

# Simple function optimization - minimal input required
result = mha.optimize('SCA', 
                     objective_function=lambda x: sum(x**2), 
                     dimensions=10, 
                     verbose=True)

print(f"Best fitness: {result.best_fitness}")
result.plot_convergence()
```

### Feature Selection (Data as First Argument)

```python
import numpy as np
from sklearn.datasets import load_breast_cancer

# Load data
X, y = load_breast_cancer(return_X_y=True)

# Feature selection - X is first argument, everything else optional
result = mha.optimize('SCA', X, y, verbose=True)

# Results
print(f"Selected {sum(result.best_solution_binary)} features")
print(f"Error rate: {result.best_fitness}")
result.summary()  # Complete analysis
```

### Intelligent Parameter Handling

```python
# Only specify what you want to change - everything else uses smart defaults
result = mha.optimize('SCA', X, y, 
                     upper_bound=0.8,  # Only upper bound specified
                     max_iterations=100)  # Lower bound auto-derived

# Or use absolute minimal input
result = mha.optimize('SCA', X, y, verbose=True)  # Everything else automatic
```

## Advanced Usage

### Using the MHAToolbox Class

```python
from mha_toolbox import MHAToolbox

# Create toolbox instance
toolbox = MHAToolbox()

# List available algorithms
print(toolbox.list_algorithms())

# Get optimizer with specific parameters
optimizer = toolbox.get_optimizer('SCA', population_size=50, verbose=True)

# Run optimization
result = optimizer.optimize(X, y)
result.plot_convergence()
```

### Model Object Capabilities

```python
# The result object contains everything you need
result = mha.optimize('SCA', X, y)

# Access individual attributes
print(f"Algorithm: {result.algorithm_name}")
print(f"Best solution: {result.best_solution}")
print(f"Execution time: {result.execution_time}")

# View all parameters used (including auto-calculated ones)
print(result.parameters)

# Built-in analysis methods
result.summary()           # Complete summary
result.plot_convergence()  # Convergence curve

# Access convergence history
for i, fitness in enumerate(result.convergence_curve[:5]):
    print(f"Iteration {i}: {fitness}")
```

## Design Philosophy

This toolbox follows professional software engineering principles:

### 1. User Abstraction
- Users provide minimal input - just data and algorithm name
- All technical parameters are automatically calculated
- No interruptions or prompts for default values

### 2. TensorFlow-style Design
```python
# Like TensorFlow, users import and call functions directly
import mha_toolbox as mha
result = mha.optimize('SCA', X, y)  # Simple and clean
```

### 3. Data-First Approach
```python
# Data is always the first positional argument
result = mha.optimize('SCA', X, y, **optional_params)
```

### 4. Intelligent Defaults
- Bounds automatically derived from data type and values
- Dimensions detected from input data shape
- Iterations scaled based on problem complexity
- Population size optimized for problem type

### 5. Comprehensive Results
- Single model object contains all results and metadata
- Built-in analysis and visualization methods
- All parameters used (including defaults) are preserved

## Available Algorithms

- **SCA (Sine Cosine Algorithm)**: Population-based optimization using sine/cosine functions
- More algorithms coming soon!

## Examples

See the following files for comprehensive examples:

- [`main.py`](main.py): Basic usage demonstration
- [`demo.py`](demo.py): Complete feature showcase  
- [`professional_examples.py`](professional_examples.py): Advanced usage patterns
- [`MHA_Toolbox_Tutorial.ipynb`](MHA_Toolbox_Tutorial.ipynb): Interactive Jupyter tutorial

## Command Line Interface

```bash
# Basic optimization
python mha_cli.py --algorithm SCA --benchmark sphere --dimensions 10

# Feature selection from file
python mha_cli.py --algorithm SCA --dataset data.csv --target-column target

# Custom parameters
python mha_cli.py --algorithm SCA --benchmark rastrigin --pop-size 50 --max-iter 200
```

## Architecture

The toolbox follows a professional object-oriented design:

- **BaseOptimizer**: Abstract base class for all algorithms
- **OptimizationModel**: Standardized result container with analysis methods
- **MHAToolbox**: Central interface for algorithm access and management

## Contributing

To add a new algorithm:

1. Create a class inheriting from `BaseOptimizer`
2. Implement the `_optimize` method
3. Add comprehensive docstrings
4. The toolbox will automatically discover and register your algorithm

## Citation

```bibtex
@software{MHA_Toolbox,
  author = {MHA Toolbox Team},
  title = {MHA Toolbox: Professional Metaheuristic Algorithm Library},
  url = {https://github.com/Achyut103040/MHA-Algorithm},
  year = {2025},
  version = {1.0.0}
}
```

## License

MIT License - see LICENSE file for details.
    max_iter=200,
    dim=10,
    lb=-100,
    ub=100
)

print(f"Best solution: {result['best_position']}")
print(f"Best fitness: {result['best_score']}")
```

## Project Structure

```
MHA-Algorithm/
├── toolbox_algorithms/          # Standardized algorithm implementations
│   └── SCA.py                  # Sine Cosine Algorithm
├── objective_functions/         # Built-in objective functions
│   └── benchmark_functions.py  # Standard benchmark functions
├── utils/                      # Utility functions
│   └── toolbox_utils.py       # Common helper functions
├── original_codes/             # Backup of original implementations
├── mha_toolbox.py             # Main toolbox class
└── setup_project.py           # Setup script
```

## Standardized Interface

All algorithms follow the same signature:

```python
def ALGORITHM_NAME(pop_size, max_iter, lb, ub, dim, obj_func, **kwargs):
    """
    Args:
        pop_size (int): Population size
        max_iter (int): Maximum iterations
        lb (float/list): Lower bounds
        ub (float/list): Upper bounds
        dim (int): Problem dimensions
        obj_func (callable): Objective function
        **kwargs: Algorithm-specific parameters
    
    Returns:
        dict: {
            'best_score': float,
            'best_position': np.ndarray,
            'convergence_curve': np.ndarray,
            'execution_time': float,
            'algorithm_name': str
        }
    """
```

## Built-in Functions

Available benchmark functions:
- `sphere`, `rastrigin`, `rosenbrock`, `ackley`, `griewank`, `schwefel`, `levy`

```python
# Use built-in functions
result = toolbox.optimize('SCA', 'sphere', pop_size=30, max_iter=200, dim=10)
```

## Intelligent Bounds

```python
# Single value bounds (applied to all dimensions)
result = toolbox.optimize(..., lb=-10, ub=10, dim=5)

# Different bounds per dimension
lb_list = [-10, -5, -1, -50, -100]
ub_list = [10, 5, 1, 50, 100]
result = toolbox.optimize(..., lb=lb_list, ub=ub_list, dim=5)
```

## Algorithm Comparison

```python
results = toolbox.compare_algorithms(
    algorithm_names=['SCA'],  # Add more when implemented
    objective_function='sphere',
    runs=5,
    pop_size=30,
    max_iter=200,
    dim=10
)
toolbox.print_comparison_table(results)
```

## Currently Implemented

- **SCA**: Sine Cosine Algorithm ✅

## Adding New Algorithms

1. Copy original algorithm to `original_codes/`
2. Use `SCA.py` as template
3. Follow the standardized signature
4. Use utility functions from `toolbox_utils.py`
5. Test with the framework

## Setup

```bash
python setup_project.py  # Verify installation
python mha_toolbox.py     # Test toolbox
```

This toolbox provides a clean, standardized interface for metaheuristic algorithms, making them easy to use and compare.
