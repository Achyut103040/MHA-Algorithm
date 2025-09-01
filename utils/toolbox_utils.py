import numpy as np
import time
from typing import Union, Callable, Dict, Any

def handle_bounds(lb: Union[float, list], ub: Union[float, list], dim: int):
    """Handle intelligent search bounds - single values or lists per dimension."""
    if isinstance(lb, (int, float)):
        lb = np.full(dim, lb)
    else:
        lb = np.array(lb)
        
    if isinstance(ub, (int, float)):
        ub = np.full(dim, ub)
    else:
        ub = np.array(ub)
        
    return lb, ub

def initialize_population(pop_size: int, dim: int, lb: np.ndarray, ub: np.ndarray):
    """Initialize population within bounds."""
    return np.random.uniform(0, 1, (pop_size, dim)) * (ub - lb) + lb

def boundary_check(positions: np.ndarray, lb: np.ndarray, ub: np.ndarray):
    """Ensure all positions are within bounds."""
    return np.clip(positions, lb, ub)

def evaluate_population(positions: np.ndarray, obj_func: Callable):
    """Evaluate fitness for entire population."""
    fitness = np.zeros(positions.shape[0])
    for i in range(positions.shape[0]):
        fitness[i] = obj_func(positions[i])
    return fitness

def create_result_dict(best_score: float, best_position: np.ndarray, 
                      convergence_curve: np.ndarray, execution_time: float,
                      algorithm_name: str) -> Dict[str, Any]:
    """Create standardized result dictionary."""
    return {
        'best_score': best_score,
        'best_position': best_position.copy(),
        'convergence_curve': convergence_curve.copy(),
        'execution_time': execution_time,
        'algorithm_name': algorithm_name
    }
