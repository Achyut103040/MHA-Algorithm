# MHA Toolbox - Metaheuristic Algorithm Toolbox

A standardized Python toolbox for metaheuristic optimization algorithms with unified interface.

## Quick Start

```python
from mha_toolbox import MHAToolbox

toolbox = MHAToolbox()

# Define your optimization problem
def my_function(x):
    return sum(x**2)

# Run optimization
result = toolbox.optimize(
    algorithm_name='SCA',
    objective_function=my_function,
    pop_size=30,
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
