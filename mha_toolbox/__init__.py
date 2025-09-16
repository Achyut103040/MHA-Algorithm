"""
MHA Toolbox: A Professional Meta-Heuristic Algorithms Library

This library provides a simple, intuitive interface for meta-heuristic optimization
algorithms, designed to work like popular ML libraries (TensorFlow, PyTorch, scikit-learn).

🚀 Key Features:
- One-line optimization with intelligent defaults
- Automatic algorithm discovery and registration  
- Support for function optimization and feature selection
- Built-in dataset utilities and visualization
- Comprehensive result analysis and reporting

📖 Quick Start:
--------------
>>> import mha_toolbox as mha
>>> 
>>> # Simple function optimization (just specify algorithm and function)
>>> result = mha.optimize('pso', objective_function=lambda x: sum(x**2), dimensions=10)
>>> print(f"Best fitness: {result.best_fitness}")
>>>
>>> # Feature selection with your data (data first, everything else optional)
>>> result = mha.optimize('gwo', X, y)
>>> print(f"Selected {result.n_selected_features} features")
>>>
>>> # Advanced usage with custom parameters (still simple!)
>>> result = mha.optimize('sca', X, y, population_size=50, max_iterations=200)
>>> result.plot_convergence()
>>> result.summary()

🧬 Available Algorithms:
-----------------------
Swarm Intelligence: pso, gwo, sca, alo, abc, mfo, goa, hho, woa, etc.
Physics-based: sa, sho, efo, etc.  
Bio-inspired: ga, de, evolution, etc.
Human-based: tlo, hsa, etc.

📊 Built-in Datasets:
--------------------
>>> # Load test datasets easily
>>> X, y = mha.load_dataset('breast_cancer')
>>> result = mha.optimize('pso', X, y)

🔬 Compare Algorithms:
---------------------
>>> # Compare multiple algorithms automatically
>>> results = mha.compare(['pso', 'gwo', 'sca'], X, y)
>>> mha.plot_comparison(results)
"""

# Core imports (fix duplicates)
from .toolbox import (
    MHAToolbox, get_optimizer, list_algorithms, get_algorithm_info, 
    run_optimizer, compare_algorithms, get_toolbox
)
from .base import BaseOptimizer, OptimizationModel
from .hybrid import HybridOptimizer, create_hybrid_optimizer

# Utility imports (add missing ones)
from .utils.datasets import load_dataset, list_datasets
from .utils.visualizations import plot_comparison
from .utils.problem_creator import create_problem


def optimize(algorithm_name, X=None, y=None, objective_function=None, **kwargs):
    """
    
    This is the primary function users should use. It provides a simple,
    one-line interface to run any optimization algorithm with intelligent
    parameter handling.
    
    Parameters
    ----------
    algorithm_name : str
        Name of the algorithm to use (e.g., 'SCA', 'SineCosinAlgorithm')
    X : numpy.ndarray, optional
        Input data (features) - prioritized as first positional argument
    y : numpy.ndarray, optional
        Target values (for supervised problems like feature selection)
    objective_function : callable, optional
        Function to optimize (required if X and y are not provided)
    **kwargs : dict
        Additional algorithm parameters (all optional with intelligent defaults)
        
    Returns
    -------
    OptimizationModel
        Model object containing all results, parameters, and analysis methods
        
    Examples
    --------
    >>> # Simple function optimization
    >>> result = optimize('SCA', objective_function=lambda x: sum(x**2), dimensions=10)
    
    >>> # Feature selection (data is first argument)
    >>> result = optimize('SCA', X, y, verbose=True)
    
    >>> # Custom parameters
    >>> result = optimize('SCA', X, y, population_size=50, max_iterations=200)
    """
    return run_optimizer(algorithm_name, X=X, y=y, objective_function=objective_function, **kwargs)

def list_algorithms():
    """
    List all available algorithms.
    
    Returns
    -------
    list
        List of available algorithm names
    """
    # Get categorized algorithms and flatten to a simple list
    from .toolbox import list_algorithms as list_algorithms_internal
    categorized = list_algorithms_internal()
    if isinstance(categorized, dict):
        # Flatten the categorized results
        all_algorithms = []
        for algorithms_list in categorized.values():
            all_algorithms.extend(algorithms_list)
        return sorted(all_algorithms)
    else:
        return categorized

def info(algorithm_name):
    """
    Get information about a specific algorithm.
    
    Parameters
    ----------
    algorithm_name : str
        Name of the algorithm
        
    Returns
    -------
    dict
        Algorithm information including parameters and documentation
    """
    return get_algorithm_info(algorithm_name)

def compare(algorithm_names, X=None, y=None, objective_function=None, **kwargs):
    """
    Compare multiple algorithms on the same problem.
    
    Parameters
    ----------
    algorithm_names : list
        List of algorithm names to compare
    X : numpy.ndarray, optional
        Input data (features)
    y : numpy.ndarray, optional
        Target values (for supervised problems)
    objective_function : callable, optional
        Function to optimize (required if X and y are not provided)
    **kwargs : dict
        Additional parameters passed to all algorithms
        
    Returns
    -------
    dict
        Dictionary mapping algorithm names to their results
    """
    return compare_algorithms(algorithm_names, X=X, y=y, 
                            objective_function=objective_function, **kwargs)

def hybrid(algorithm_names, X=None, y=None, objective_function=None, mode='sequential', **kwargs):
    """
    Run hybrid optimization using multiple algorithms.
    
    This function combines 2-3 algorithms using different strategies to potentially
    achieve better results than individual algorithms.
    
    Parameters
    ----------
    algorithm_names : list
        List of 2-3 algorithm names to combine
    X : numpy.ndarray, optional
        Input data (features)
    y : numpy.ndarray, optional
        Target values (for supervised problems)
    objective_function : callable, optional
        Function to optimize (required if X and y are not provided)
    mode : str, default='sequential'
        Hybridization mode: 'sequential', 'parallel', or 'collaborative'
    **kwargs : dict
        Additional parameters passed to algorithms
        
    Returns
    -------
    OptimizationModel
        Best result from the hybrid optimization
        
    Examples
    --------
    >>> # Sequential hybrid (SCA then PSO)
    >>> result = hybrid(['SCA', 'PSO'], X, y, mode='sequential')
    
    >>> # Parallel hybrid (run simultaneously)
    >>> result = hybrid(['SCA', 'PSO', 'GWO'], X, y, mode='parallel')
    
    >>> # Collaborative hybrid (share information)
    >>> result = hybrid(['SCA', 'PSO'], X, y, mode='collaborative')
    """
    hybrid_optimizer = create_hybrid_optimizer(algorithm_names, mode=mode, **kwargs)
    return hybrid_optimizer.optimize(X=X, y=y, objective_function=objective_function)

# Dataset and utility functions
def load_data(dataset_name):
    """Load built-in datasets for testing and experimentation."""
    return load_dataset(dataset_name)

def available_datasets():
    """List all available built-in datasets."""
    return list_datasets()

def plot_results(results, title="Algorithm Comparison"):
    """Plot comparison of multiple optimization results."""
    plot_comparison(results, title)

def create_custom_problem(objective_function, dimensions, bounds=(-100, 100), 
                         minimize=True, constraints=None):
    """Create a custom optimization problem definition."""
    return create_problem(objective_function, dimensions, bounds, minimize, constraints)

# Quick access functions for beginners
def quick_optimize(algorithm='pso', problem_type='function', **kwargs):
    """
    Quick optimization with minimal setup - perfect for beginners.
    
    Parameters
    ----------
    algorithm : str, default='pso'
        Algorithm to use
    problem_type : str, default='function'
        'function' for function optimization, 'features' for feature selection
    **kwargs
        Problem-specific parameters
        
    Examples
    --------
    >>> # Quick function optimization
    >>> result = quick_optimize()  # Uses default sphere function
    
    >>> # Quick feature selection  
    >>> result = quick_optimize('gwo', 'features', X=data, y=targets)
    """
    if problem_type == 'function':
        # Default sphere function for testing
        objective_func = kwargs.get('objective_function', lambda x: sum(x**2))
        dimensions = kwargs.get('dimensions', 10)
        return optimize(algorithm, objective_function=objective_func, dimensions=dimensions, **kwargs)
    elif problem_type == 'features':
        X = kwargs.pop('X', None)
        y = kwargs.pop('y', None)
        if X is None or y is None:
            raise ValueError("For feature selection, provide X and y data")
        return optimize(algorithm, X, y, **kwargs)
    else:
        raise ValueError("problem_type must be 'function' or 'features'")

def get_best_algorithm(X=None, y=None, objective_function=None, algorithms=None, **kwargs):
    """
    Automatically find the best algorithm for your problem.
    
    This function tests multiple algorithms and returns the best performing one.
    
    Parameters
    ----------
    X, y : array-like, optional
        Data for feature selection problems
    objective_function : callable, optional  
        Function for optimization problems
    algorithms : list, optional
        Algorithms to test (default: ['pso', 'gwo', 'sca', 'woa'])
    **kwargs
        Additional parameters
        
    Returns
    -------
    OptimizationModel
        Best result from all tested algorithms
    """
    if algorithms is None:
        algorithms = ['pso', 'gwo', 'sca', 'woa', 'abc']
    
    print(f"🔍 Testing {len(algorithms)} algorithms to find the best one...")
    results = compare(algorithms, X=X, y=y, objective_function=objective_function, **kwargs)
    
    # Find best result
    best_result = None
    best_fitness = float('inf')
    
    for alg_name, result in results.items():
        if result.best_fitness < best_fitness:
            best_fitness = result.best_fitness
            best_result = result
    
    print(f"🏆 Best algorithm: {best_result.algorithm_name} (fitness: {best_fitness:.6f})")
    return best_result

def parameter_combinations(algorithm_name=None, show_math=True):
    """
    Show parameter combinations and complexity for algorithms.
    
    For algorithms with optional parameters, this shows:
    - Number of possible parameter combinations
    - Mathematical calculation (e.g., 4! = 24 for 4 parameters)
    - Recommended parameter ranges
    
    Parameters
    ----------
    algorithm_name : str, optional
        Specific algorithm to analyze (if None, shows general info)
    show_math : bool, default=True
        Whether to show mathematical calculations
        
    Returns
    -------
    dict
        Parameter combination information
    """
    import math
    
    if algorithm_name:
        try:
            algo_info = get_algorithm_info(algorithm_name)
            params = algo_info.get('parameters', {})
            optional_params = [p for p, info in params.items() 
                             if info.get('required', True) == False]
            
            n_optional = len(optional_params)
            combinations = math.factorial(n_optional) if n_optional <= 10 else "Very Large"
            
            result = {
                'algorithm': algorithm_name,
                'total_parameters': len(params),
                'optional_parameters': n_optional,
                'required_parameters': len(params) - n_optional,
                'parameter_combinations': combinations,
                'mathematical_formula': f"{n_optional}! = {combinations}" if n_optional <= 10 else f"{n_optional}! (factorial)",
                'optional_param_names': optional_params
            }
            
            if show_math and n_optional <= 10:
                print(f"📊 Parameter Analysis for {algorithm_name.upper()}:")
                print(f"   Total parameters: {len(params)}")
                print(f"   Required parameters: {len(params) - n_optional}")
                print(f"   Optional parameters: {n_optional}")
                print(f"   Possible combinations: {n_optional}! = {combinations}")
                if n_optional == 4:
                    print(f"   Yes! With 4 optional parameters: 4! = 4×3×2×1 = 24 combinations")
                print(f"   Optional parameters: {', '.join(optional_params)}")
            
            return result
            
        except Exception as e:
            print(f"❌ Could not analyze {algorithm_name}: {e}")
            return None
    else:
        # General information about parameter combinations
        info = {
            'explanation': 'Parameter combinations in optimization algorithms',
            'factorial_examples': {
                '2 parameters': '2! = 2×1 = 2 combinations',
                '3 parameters': '3! = 3×2×1 = 6 combinations', 
                '4 parameters': '4! = 4×3×2×1 = 24 combinations',
                '5 parameters': '5! = 5×4×3×2×1 = 120 combinations'
            },
            'note': 'These are just ordering combinations. Actual parameter space is much larger!'
        }
        
        if show_math:
            print("🧮 Parameter Combination Mathematics:")
            print("   When you have n optional parameters, there are n! ways to order them")
            print("   Examples:")
            for params, calc in info['factorial_examples'].items():
                print(f"   • {params}: {calc}")
            print("\n   🎯 For your question: YES, 4 optional parameters = 4! = 24 combinations")
            print("   💡 Note: This is just ordering. Real parameter space is much larger!")
        
        return info

# Export main classes and functions
__all__ = [
    # Core classes
    'MHAToolbox', 'BaseOptimizer', 'OptimizationModel',
    'HybridOptimizer', 'create_hybrid_optimizer',
    
    # Main optimization functions
    'optimize', 'list_algorithms', 'info', 'compare', 'hybrid',
    
    # Dataset utilities
    'load_data', 'available_datasets', 
    
    # Utility functions
    'plot_results', 'create_custom_problem', 'parameter_combinations',
    
    # Beginner-friendly functions
    'quick_optimize', 'get_best_algorithm',
    
    # Advanced functions (remove duplicates)
    'get_optimizer', 'get_algorithm_info', 'run_optimizer', 'compare_algorithms'
]

# Version info
__version__ = "1.0.0"
__author__ = "MHA Development Team"

# Convenience aliases
mha = get_toolbox()

# Enhanced AlgorithmAccessor with error handling
class AlgorithmAccessor:
    """Provides direct access to algorithms like mha.pso(), mha.gwo(), mha.sca()"""
    
    def __getattr__(self, algorithm_name):
        # Check if it's a module function first
        builtin_functions = {
            'algorithms': list_algorithms,
            'info': info,
            'compare': compare,
            'hybrid': hybrid,
            'optimize': optimize,
            'load_data': load_data,
            'available_datasets': available_datasets,
            'plot_results': plot_results,
            'parameter_combinations': parameter_combinations,
            'quick_optimize': quick_optimize,
            'get_best_algorithm': get_best_algorithm
        }
        
        if algorithm_name in builtin_functions:
            return builtin_functions[algorithm_name]
        
        # Check if algorithm exists
        try:
            # Test if algorithm is available
            available_algs = list_algorithms()
            if isinstance(available_algs, list):
                available_names = [alg.lower() for alg in available_algs]
            else:
                # Handle dict case
                available_names = []
                for alg_list in available_algs.values():
                    available_names.extend([alg.lower() for alg in alg_list])
            
            if algorithm_name.lower() not in available_names:
                raise AttributeError(f"Algorithm '{algorithm_name}' not found. Available: {available_names[:10]}...")
                
        except Exception as e:
            # If we can't check, proceed anyway and let optimize() handle the error
            pass
        
        def algorithm_runner(*args, **kwargs):
            """Dynamic algorithm runner with proper argument handling"""
            try:
                if len(args) >= 2:
                    return optimize(algorithm_name, args[0], args[1], **kwargs)
                elif len(args) == 1:
                    if callable(args[0]):
                        return optimize(algorithm_name, objective_function=args[0], **kwargs)
                    else:
                        return optimize(algorithm_name, args[0], **kwargs)
                else:
                    return optimize(algorithm_name, **kwargs)
            except Exception as e:
                raise RuntimeError(f"Error running algorithm '{algorithm_name}': {str(e)}")
        
        algorithm_runner.__name__ = algorithm_name
        algorithm_runner.__doc__ = f"""
        Run {algorithm_name.upper()} optimization algorithm.
        
        Examples
        --------
        >>> result = mha.{algorithm_name}(X, y)  # Feature selection
        >>> result = mha.{algorithm_name}(objective_function=func, dimensions=10)  # Function optimization
        """
        return algorithm_runner

# Apply the AlgorithmAccessor to this module
import sys
sys.modules[__name__].__class__ = type(
    'MHAModule', 
    (sys.modules[__name__].__class__, AlgorithmAccessor), 
    {}
)