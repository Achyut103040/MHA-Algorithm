"""
MHA Toolbox: A TensorFlow-like library for metaheuristic optimization algorithms.
"""

import numpy as np
from .toolbox import MHAToolbox
from .base import OptimizationModel

# Global toolbox instance
_global_toolbox = None

def _get_toolbox():
    """Get or create the global toolbox instance."""
    global _global_toolbox
    if _global_toolbox is None:
        _global_toolbox = MHAToolbox()
    return _global_toolbox

def _run_algorithm(algorithm_name, X=None, y=None, objective_function=None, *args, **kwargs):
    """
    Enhanced helper function to run a specific algorithm with flexible parameter handling.
    
    This function supports all possible parameter combinations:
    - Single parameters: mha.pso(X=data)
    - Multiple parameters: mha.pso(X=data, y=labels, max_iterations=50)
    - Positional arguments: mha.pso(data, labels, 30, 100)  # X, y, pop_size, max_iter
    - Mixed combinations: mha.pso(X=data, max_iterations=50, population_size=30)
    - Any other combinations users might try
    
    Parameters
    ----------
    algorithm_name : str
        Name of the algorithm to run
    X : array-like, optional
        Input data for feature selection problems
    y : array-like, optional  
        Target labels for supervised problems
    objective_function : callable, optional
        Custom objective function for optimization problems
    *args : tuple
        Additional positional arguments (population_size, max_iterations, dimensions, etc.)
    **kwargs : dict
        Additional keyword arguments passed to algorithm constructor
        
    Returns
    -------
    OptimizationModel
        Result object containing optimization results
    """
    toolbox = _get_toolbox()
    
    # Handle flexible positional arguments if provided
    if args:
        # Map common positional argument patterns
        if len(args) >= 1 and 'population_size' not in kwargs:
            kwargs['population_size'] = args[0]
        if len(args) >= 2 and 'max_iterations' not in kwargs:
            kwargs['max_iterations'] = args[1]
        if len(args) >= 3 and 'dimensions' not in kwargs:
            kwargs['dimensions'] = args[2]
        if len(args) >= 4 and 'lower_bound' not in kwargs:
            kwargs['lower_bound'] = args[3]
        if len(args) >= 5 and 'upper_bound' not in kwargs:
            kwargs['upper_bound'] = args[4]
    
    # Create optimizer with all provided parameters
    optimizer = toolbox.get_optimizer(algorithm_name, **kwargs)
    
    # Run optimization with data parameters
    return optimizer.optimize(X=X, y=y, objective_function=objective_function)

def pso(*args, X=None, y=None, objective_function=None, **kwargs):
    """
    Particle Swarm Optimization (PSO) with flexible parameter handling.
    
    Supports multiple calling patterns:
    - pso(X=data, y=labels)
    - pso(data, labels, 30, 100)  # X, y, pop_size, max_iter
    - pso(X=data, max_iterations=50, population_size=30)
    - pso(objective_function=func, dimensions=5)
    """
    # Handle positional arguments for X and y
    if args:
        if len(args) >= 1 and X is None:
            X = args[0]
        if len(args) >= 2 and y is None:
            y = args[1]
        # Pass remaining args to _run_algorithm
        remaining_args = args[2:] if len(args) > 2 else ()
        return _run_algorithm('ParticleSwarmOptimization', X, y, objective_function, *remaining_args, **kwargs)
    
    return _run_algorithm('ParticleSwarmOptimization', X, y, objective_function, **kwargs)

def sca(*args, X=None, y=None, objective_function=None, **kwargs):
    """
    Sine Cosine Algorithm (SCA) with flexible parameter handling.
    
    Supports multiple calling patterns:
    - sca(X=data, y=labels)
    - sca(data, labels, 30, 100)  # X, y, pop_size, max_iter
    - sca(X=data, max_iterations=50, population_size=30)
    - sca(objective_function=func, dimensions=5)
    """
    # Handle positional arguments for X and y
    if args:
        if len(args) >= 1 and X is None:
            X = args[0]
        if len(args) >= 2 and y is None:
            y = args[1]
        # Pass remaining args to _run_algorithm
        remaining_args = args[2:] if len(args) > 2 else ()
        return _run_algorithm('SineCosinAlgorithm', X, y, objective_function, *remaining_args, **kwargs)
    
    return _run_algorithm('SineCosinAlgorithm', X, y, objective_function, **kwargs)

def gwo(*args, X=None, y=None, objective_function=None, **kwargs):
    """
    Grey Wolf Optimizer (GWO) with flexible parameter handling.
    
    Supports multiple calling patterns:
    - gwo(X=data, y=labels)
    - gwo(data, labels, 30, 100)  # X, y, pop_size, max_iter
    - gwo(X=data, max_iterations=50, population_size=30)
    - gwo(objective_function=func, dimensions=5)
    """
    # Handle positional arguments for X and y
    if args:
        if len(args) >= 1 and X is None:
            X = args[0]
        if len(args) >= 2 and y is None:
            y = args[1]
        # Pass remaining args to _run_algorithm
        remaining_args = args[2:] if len(args) > 2 else ()
        return _run_algorithm('GreyWolfOptimizer', X, y, objective_function, *remaining_args, **kwargs)
    
    return _run_algorithm('GreyWolfOptimizer', X, y, objective_function, **kwargs)

def ao(*args, X=None, y=None, objective_function=None, **kwargs):
    """
    Aquila Optimizer (AO) with flexible parameter handling.
    
    Supports multiple calling patterns:
    - ao(X=data, y=labels)
    - ao(data, labels, 30, 100)  # X, y, pop_size, max_iter
    - ao(X=data, max_iterations=50, population_size=30)
    - ao(objective_function=func, dimensions=5)
    """
    # Handle positional arguments for X and y
    if args:
        if len(args) >= 1 and X is None:
            X = args[0]
        if len(args) >= 2 and y is None:
            y = args[1]
        # Pass remaining args to _run_algorithm
        remaining_args = args[2:] if len(args) > 2 else ()
        return _run_algorithm('AquilaOptimizer', X, y, objective_function, *remaining_args, **kwargs)
    
    return _run_algorithm('AquilaOptimizer', X, y, objective_function, **kwargs)

def woa(*args, X=None, y=None, objective_function=None, **kwargs):
    """
    Whale Optimization Algorithm (WOA) with flexible parameter handling.
    
    Supports multiple calling patterns:
    - woa(X=data, y=labels)
    - woa(data, labels, 30, 100)  # X, y, pop_size, max_iter
    - woa(X=data, max_iterations=50, population_size=30)
    - woa(objective_function=func, dimensions=5)
    """
    # Handle positional arguments for X and y
    if args:
        if len(args) >= 1 and X is None:
            X = args[0]
        if len(args) >= 2 and y is None:
            y = args[1]
        # Pass remaining args to _run_algorithm
        remaining_args = args[2:] if len(args) > 2 else ()
        return _run_algorithm('WhaleOptimizationAlgorithm', X, y, objective_function, *remaining_args, **kwargs)
    
    return _run_algorithm('WhaleOptimizationAlgorithm', X, y, objective_function, **kwargs)

def ga(*args, X=None, y=None, objective_function=None, **kwargs):
    """
    Genetic Algorithm (GA) with flexible parameter handling.
    
    Supports multiple calling patterns:
    - ga(X=data, y=labels)
    - ga(data, labels, 30, 100)  # X, y, pop_size, max_iter
    - ga(X=data, max_iterations=50, population_size=30)
    - ga(objective_function=func, dimensions=5)
    """
    # Handle positional arguments for X and y
    if args:
        if len(args) >= 1 and X is None:
            X = args[0]
        if len(args) >= 2 and y is None:
            y = args[1]
        # Pass remaining args to _run_algorithm
        remaining_args = args[2:] if len(args) > 2 else ()
        return _run_algorithm('GeneticAlgorithm', X, y, objective_function, *remaining_args, **kwargs)
    
    return _run_algorithm('GeneticAlgorithm', X, y, objective_function, **kwargs)

def de(*args, X=None, y=None, objective_function=None, **kwargs):
    """
    Differential Evolution (DE) with flexible parameter handling.
    
    Supports multiple calling patterns:
    - de(X=data, y=labels)
    - de(data, labels, 30, 100)  # X, y, pop_size, max_iter
    - de(X=data, max_iterations=50, population_size=30)
    - de(objective_function=func, dimensions=5)
    """
    # Handle positional arguments for X and y
    if args:
        if len(args) >= 1 and X is None:
            X = args[0]
        if len(args) >= 2 and y is None:
            y = args[1]
        # Pass remaining args to _run_algorithm
        remaining_args = args[2:] if len(args) > 2 else ()
        return _run_algorithm('DifferentialEvolution', X, y, objective_function, *remaining_args, **kwargs)
    
    return _run_algorithm('DifferentialEvolution', X, y, objective_function, **kwargs)

def fa(*args, X=None, y=None, objective_function=None, **kwargs):
    """
    Firefly Algorithm (FA) with flexible parameter handling.
    
    Supports multiple calling patterns:
    - fa(X=data, y=labels)
    - fa(data, labels, 30, 100)  # X, y, pop_size, max_iter
    - fa(X=data, max_iterations=50, population_size=30)
    - fa(objective_function=func, dimensions=5)
    """
    # Handle positional arguments for X and y
    if args:
        if len(args) >= 1 and X is None:
            X = args[0]
        if len(args) >= 2 and y is None:
            y = args[1]
        # Pass remaining args to _run_algorithm
        remaining_args = args[2:] if len(args) > 2 else ()
        return _run_algorithm('FireflyAlgorithm', X, y, objective_function, *remaining_args, **kwargs)
    
    return _run_algorithm('FireflyAlgorithm', X, y, objective_function, **kwargs)

def ba(*args, X=None, y=None, objective_function=None, **kwargs):
    """
    Bat Algorithm (BA) with flexible parameter handling.
    
    Supports multiple calling patterns:
    - ba(X=data, y=labels)
    - ba(data, labels, 30, 100)  # X, y, pop_size, max_iter
    - ba(X=data, max_iterations=50, population_size=30)
    - ba(objective_function=func, dimensions=5)
    """
    # Handle positional arguments for X and y
    if args:
        if len(args) >= 1 and X is None:
            X = args[0]
        if len(args) >= 2 and y is None:
            y = args[1]
        # Pass remaining args to _run_algorithm
        remaining_args = args[2:] if len(args) > 2 else ()
        return _run_algorithm('BatAlgorithm', X, y, objective_function, *remaining_args, **kwargs)
    
    return _run_algorithm('BatAlgorithm', X, y, objective_function, **kwargs)

def aco(*args, X=None, y=None, objective_function=None, **kwargs):
    """
    Ant Colony Optimization (ACO) with flexible parameter handling.
    
    Supports multiple calling patterns:
    - aco(X=data, y=labels)
    - aco(data, labels, 30, 100)  # X, y, pop_size, max_iter
    - aco(X=data, max_iterations=50, population_size=30)
    - aco(objective_function=func, dimensions=5)
    """
    # Handle positional arguments for X and y
    if args:
        if len(args) >= 1 and X is None:
            X = args[0]
        if len(args) >= 2 and y is None:
            y = args[1]
        # Pass remaining args to _run_algorithm
        remaining_args = args[2:] if len(args) > 2 else ()
        return _run_algorithm('AntColonyOptimization', X, y, objective_function, *remaining_args, **kwargs)
    
    return _run_algorithm('AntColonyOptimization', X, y, objective_function, **kwargs)

def list_algorithms():
    """List all available algorithms."""
    toolbox = _get_toolbox()
    return toolbox.list_algorithms()

# Legacy imports for backward compatibility
from .toolbox import MHAToolbox
from .base import BaseOptimizer, OptimizationModel
from .hybrid import HybridOptimizer, create_hybrid_optimizer

# Expose algorithm functions
__all__ = [
    'pso', 'sca', 'gwo', 'ao', 'woa', 'ga', 'de', 'fa', 'ba', 'aco',
    'list_algorithms',
    'MHAToolbox', 'BaseOptimizer', 'OptimizationModel', 
    'HybridOptimizer', 'create_hybrid_optimizer'
]