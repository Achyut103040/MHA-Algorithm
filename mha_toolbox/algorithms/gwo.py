"""
Grey Wolf Optimizer (GWO)

Based on: Mirjalili, S., Mirjalili, S. M., & Lewis, A. (2014). 
Grey wolf optimizer. Advances in engineering software, 69, 46-61.
"""

import numpy as np
from ..base import BaseOptimizer


class GreyWolfOptimizer(BaseOptimizer):
    """
    Grey Wolf Optimizer (GWO)
    
    GWO mimics the leadership hierarchy and hunting mechanism of grey wolves.
    The algorithm simulates the social behavior of grey wolves including alpha,
    beta, delta, and omega wolves.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.algorithm_name = "GWO"
    
    def _optimize(self, objective_function, X=None, y=None, **kwargs):
        """GWO optimization implementation with automatic parameter calculation"""
        # Automatically determine problem type and set bounds/dimensions
        if X is not None:
            # Feature selection problem
            self.dimensions = X.shape[1]
            self.lower_bound = np.zeros(self.dimensions)
            self.upper_bound = np.ones(self.dimensions)
        else:
            # Function optimization problem
            if not hasattr(self, 'dimensions') or self.dimensions is None:
                self.dimensions = kwargs.get('dimensions', 10)
            
            # Set bounds if not already set
            if not hasattr(self, 'lower_bound') or self.lower_bound is None:
                lb = kwargs.get('lower_bound', kwargs.get('lb', -10.0))
                self.lower_bound = np.full(self.dimensions, lb) if np.isscalar(lb) else np.array(lb)
            
            if not hasattr(self, 'upper_bound') or self.upper_bound is None:
                ub = kwargs.get('upper_bound', kwargs.get('ub', 10.0))
                self.upper_bound = np.full(self.dimensions, ub) if np.isscalar(ub) else np.array(ub)
        
        # Initialize population
        population = np.random.uniform(
            self.lower_bound, self.upper_bound,
            (self.population_size, self.dimensions)
        )
        
        # Initialize alpha, beta, delta positions
        alpha_pos = np.zeros(self.dimensions)
        beta_pos = np.zeros(self.dimensions)
        delta_pos = np.zeros(self.dimensions)
        
        alpha_score = float('inf')
        beta_score = float('inf')
        delta_score = float('inf')
        
        convergence_curve = []
        
        for iteration in range(self.max_iterations):
            for i in range(len(population)):
                # Return back the search agents that go beyond the boundaries
                population[i] = np.clip(population[i], self.lower_bound, self.upper_bound)
                
                # Calculate objective function for each search agent
                fitness = objective_function(population[i])
                
                # Update Alpha, Beta, and Delta
                if fitness < alpha_score:
                    delta_score = beta_score
                    delta_pos = beta_pos.copy()
                    beta_score = alpha_score
                    beta_pos = alpha_pos.copy()
                    alpha_score = fitness
                    alpha_pos = population[i].copy()
                
                if fitness > alpha_score and fitness < beta_score:
                    delta_score = beta_score
                    delta_pos = beta_pos.copy()
                    beta_score = fitness
                    beta_pos = population[i].copy()
                
                if fitness > alpha_score and fitness > beta_score and fitness < delta_score:
                    delta_score = fitness
                    delta_pos = population[i].copy()
            
            # Calculate convergence curve
            convergence_curve.append(alpha_score)
            
            if self.verbose and (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration + 1}/{self.max_iterations}, Best Fitness: {alpha_score:.6f}")
            
            # Update the position of search agents
            a = 2 - iteration * (2.0 / self.max_iterations)  # a decreases linearly from 2 to 0
            
            for i in range(len(population)):
                for j in range(self.dimensions):
                    r1 = np.random.random()
                    r2 = np.random.random()
                    
                    A1 = 2 * a * r1 - a
                    C1 = 2 * r2
                    
                    D_alpha = abs(C1 * alpha_pos[j] - population[i][j])
                    X1 = alpha_pos[j] - A1 * D_alpha
                    
                    r1 = np.random.random()
                    r2 = np.random.random()
                    
                    A2 = 2 * a * r1 - a
                    C2 = 2 * r2
                    
                    D_beta = abs(C2 * beta_pos[j] - population[i][j])
                    X2 = beta_pos[j] - A2 * D_beta
                    
                    r1 = np.random.random()
                    r2 = np.random.random()
                    
                    A3 = 2 * a * r1 - a
                    C3 = 2 * r2
                    
                    D_delta = abs(C3 * delta_pos[j] - population[i][j])
                    X3 = delta_pos[j] - A3 * D_delta
                    
                    population[i][j] = (X1 + X2 + X3) / 3
        
        return alpha_pos, alpha_score, convergence_curve
