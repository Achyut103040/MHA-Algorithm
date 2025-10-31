"""
MHA Toolbox: Professional Meta-Heuristic Algorithm Library
Version 2.0.0 - Complete optimization toolbox with 95+ algorithms and 9 hybrid combinations
"""

__version__ = "2.0.0"
__author__ = "MHA Development Team"
__license__ = "MIT"

# Core imports
from .toolbox import MHAToolbox, list_algorithms, run_optimizer
from .base import BaseOptimizer, OptimizationModel
from .advanced_hybrid import AdvancedHybridOptimizer
from .demo_system import MHADemoSystem, run_demo_system
from .robust_optimization import optimize_for_large_dataset

def optimize(algorithm_name, X=None, y=None, objective_function=None, **kwargs):
    """
    Main optimization function - simplified interface with intelligent defaults.
    
    Usage patterns:
    1. Feature Selection: optimize('pso', X, y)
    2. Function Optimization: optimize('pso', objective_function=func, dimensions=10)
    3. With hyperparameter tuning: optimize('pso', X, y, hyperparameter_tuning=True)
    
    Parameters:
    -----------
    algorithm_name : str
        Name of the algorithm (e.g., 'pso', 'gwo', 'sca')
    X : array-like, optional
        Input features/data (required for feature selection)
    y : array-like, optional
        Target values (required for feature selection)
    objective_function : callable, optional
        Function to optimize (required for function optimization)
    **kwargs : optional parameters with intelligent defaults
        - population_size: default 30
        - max_iterations: default 100
        - hyperparameter_tuning: default True (auto-tunes algorithm parameters)
        - feature_selection: default True (when X,y provided)
        - timeout_seconds: default None (no timeout)
        - n_jobs: default 1 (parallel processing cores)
        
    Auto-included features:
    - Hyperparameter optimization using grid search or Bayesian optimization
    - Cross-validation for robust performance estimation
    - Early stopping to prevent overfitting
    - Automatic result visualization and saving
    - Performance metrics calculation
    - Statistical significance testing
    
    Returns:
    --------
    OptimizationModel with:
        - best_parameters: Optimized algorithm hyperparameters
        - best_fitness: Best achieved fitness score
        - selected_features: Selected features (if feature selection)
        - cv_scores: Cross-validation results
        - convergence_history: Training progress
        - performance_metrics: Detailed performance analysis
        - visualization_plots: Generated plots and charts
    """
    
    # Set intelligent defaults
    kwargs.setdefault('population_size', 30)
    kwargs.setdefault('max_iterations', 100)
    kwargs.setdefault('hyperparameter_tuning', True)
    kwargs.setdefault('cross_validation', True)
    kwargs.setdefault('early_stopping', True)
    kwargs.setdefault('save_results', True)
    kwargs.setdefault('verbose', True)
    
    # Enhanced feature selection mode
    if X is not None and y is not None:
        kwargs.setdefault('feature_selection', True)
        kwargs.setdefault('performance_metrics', ['accuracy', 'f1_score', 'precision', 'recall'])
        
    return run_optimizer(algorithm_name, X=X, y=y, objective_function=objective_function, **kwargs)

__all__ = [
    'optimize', 'list_algorithms', 'MHAToolbox', 'AdvancedHybridOptimizer', 
    'run_demo_system', 'optimize_for_large_dataset'
]
