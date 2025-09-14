"""
Aquila Optimizer (AO) implementation.

This module implements the Aquila Optimizer algorithm, inspired by the hunting 
behavior of Aquila (golden eagles). The algorithm mimics four different hunting 
strategies for exploration and exploitation phases.

Author: MHA Toolbox
Reference: Abualigah, L., et al. (2021). Aquila Optimizer: A novel meta-heuristic 
           optimization algorithm. Computers & Industrial Engineering, 157, 107250.
"""

import numpy as np
from ..base import BaseOptimizer


class AquilaOptimizer(BaseOptimizer):
    """
    Aquila Optimizer (AO) implementation.
    
    AO is inspired by the hunting behavior of Aquila (golden eagles).
    It mimics four different hunting strategies: expanded exploration,
    narrowed exploration, expanded exploitation, and narrowed exploitation.
    
    Parameters
    ----------
    population_size : int, optional
        Size of the population (number of eagles), default=30
    max_iterations : int, optional
        Maximum number of iterations. If None, will be calculated based on dimensions.
    lower_bound : float or numpy.ndarray, optional
        Lower boundary constraint. If None, will be determined from data.
    upper_bound : float or numpy.ndarray, optional
        Upper boundary constraint. If None, will be determined from data.
    dimensions : int, optional
        Number of dimensions in the search space. If None, will be determined from data.
    verbose : bool, optional
        Whether to display progress information, default=False
    
    Attributes
    ----------
    algorithm_name : str
        Name of the algorithm
    
    References
    ----------
    .. [1] Abualigah, L., Yousri, D., Abd Elaziz, M., Ewees, A. A., Al-qaness, M. A., 
           & Gandomi, A. H. (2021). Aquila Optimizer: A novel meta-heuristic optimization 
           algorithm. Computers & Industrial Engineering, 157, 107250.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize the Aquila Optimizer with algorithm-specific parameters."""
        super().__init__(*args, **kwargs)
        self.algorithm_name = "AquilaOptimizer"
        
        # AO-specific parameters (can be customized via kwargs)
        self.alpha = kwargs.get('alpha', 0.1)
        self.delta = kwargs.get('delta', 0.1)
    
    def _optimize(self, objective_function, **kwargs):
        """
        Implement the Aquila Optimizer algorithm.
        
        The algorithm uses four different hunting strategies:
        1. Expanded exploration (X1) - High altitude soar with vertical stoop
        2. Narrowed exploration (X2) - Contour flight with short glide attack
        3. Expanded exploitation (X3) - Low flight with slow descent attack
        4. Narrowed exploitation (X4) - Walk and grab prey
        
        Parameters
        ----------
        objective_function : callable
            The function to optimize
        **kwargs : dict
            Additional algorithm-specific parameters
            
        Returns
        -------
        tuple
            (best_solution, best_fitness, convergence_curve)
        """
        # Extract parameters
        max_iter = self.max_iterations
        pop_size = self.population_size
        dim = self.dimensions
        lb = self.lower_bound
        ub = self.upper_bound
        
        # Initialize population randomly
        X = np.zeros((pop_size, dim))
        for i in range(pop_size):
            X[i, :] = lb + (ub - lb) * np.random.rand(dim)
        
        # Initialize fitness array
        fitness = np.zeros(pop_size)
        for i in range(pop_size):
            fitness[i] = objective_function(X[i, :])
        
        # Find the best solution
        best_idx = np.argmin(fitness)
        best_x = X[best_idx, :].copy()
        best_f = fitness[best_idx]
        
        # Initialize convergence curve
        convergence_curve = np.zeros(max_iter)
        
        # Main optimization loop
        for t in range(max_iter):
            # Update controlling parameters
            G1 = 2 * np.random.rand() - 1  # Eq. (16) - Random number in [-1, 1]
            G2 = 2 * (1 - t / max_iter)    # Eq. (17) - Linearly decreasing from 2 to 0
            
            # Determine the current phase
            # First 2/3 of iterations: Exploration phase
            # Last 1/3 of iterations: Exploitation phase
            exploration_phase = t <= (2/3) * max_iter
            
            # Update each eagle position
            for i in range(pop_size):
                if exploration_phase:
                    # EXPLORATION PHASE
                    if np.random.rand() < 0.5:
                        # Method 1: Expanded exploration (X1) - Eq. (18)
                        # High altitude soar with vertical stoop
                        r1 = np.random.rand()
                        r2 = np.random.rand()
                        
                        if r1 > 0.5:
                            # First variant
                            X_new = best_x * (1 - r2) + np.random.rand(dim) * ((ub - lb) * r2 + lb)
                        else:
                            # Second variant
                            X_new = best_x * (1 - r2) + np.random.rand(dim) * ((ub - lb) * r2 + lb)
                        
                        X[i, :] = X_new
                    else:
                        # Method 2: Narrowed exploration (X2) - Eq. (19)
                        # Contour flight with short glide attack
                        r3 = np.random.rand()
                        r4 = np.random.rand()
                        
                        X_new = (best_x * np.sin(r3) + 
                                np.random.rand(dim) * np.cos(r4) * (ub - lb))
                        X[i, :] = X_new
                else:
                    # EXPLOITATION PHASE
                    if np.random.rand() < 0.5:
                        # Method 3: Expanded exploitation (X3) - Eq. (20)
                        # Low flight with slow descent attack
                        beta = 2 * np.exp(np.random.rand() * (max_iter - t + 1) / max_iter)
                        
                        X_new = ((best_x - np.mean(X, axis=0)) * self.alpha - 
                                np.random.rand() * ((ub - lb) * beta) + lb)
                        X[i, :] = X_new
                    else:
                        # Method 4: Narrowed exploitation (X4) - Eq. (21)
                        # Walk and grab prey
                        QF = t ** (2 * np.random.rand() - 1)  # Quality function
                        
                        # Levy flight component
                        LF = 0.01 * np.random.rand(dim)
                        
                        X_new = (QF * best_x - 
                                (G1 * X[i, :] * np.random.rand()) - 
                                G2 * LF + np.random.rand() * G1)
                        X[i, :] = X_new
                
                # Apply boundary constraints
                X[i, :] = np.clip(X[i, :], lb, ub)
                
                # Evaluate the new solution
                fitness[i] = objective_function(X[i, :])
                
                # Update the best solution if found
                if fitness[i] < best_f:
                    best_x = X[i, :].copy()
                    best_f = fitness[i]
            
            # Store convergence information
            convergence_curve[t] = best_f
            
            # Display progress if verbose
            if self.verbose and (t + 1) % 10 == 0:
                phase = "Exploration" if exploration_phase else "Exploitation"
                print(f"Iteration {t+1}/{max_iter} ({phase}), Best Fitness: {best_f:.6f}")
        
        return best_x, best_f, convergence_curve