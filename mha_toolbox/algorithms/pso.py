"""
Particle Swarm Optimization (PSO) Algorithm

Based on: Kennedy, J., & Eberhart, R. (1995). Particle swarm optimization.
"""

import numpy as np
from ..base import BaseOptimizer


class ParticleSwarmOptimization(BaseOptimizer):
    """
    Particle Swarm Optimization (PSO) Algorithm
    
    PSO is inspired by the social behavior of bird flocking or fish schooling.
    Each particle represents a potential solution and moves through the search space
    influenced by its own best position and the global best position.
    
    Parameters
    ----------
    w : float, default=0.9
        Inertia weight controlling the influence of previous velocity
    c1 : float, default=2.0
        Acceleration coefficient for personal best (cognitive component)
    c2 : float, default=2.0
        Acceleration coefficient for global best (social component)
    """
    
    def __init__(self, *args, w=0.9, c1=2.0, c2=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.algorithm_name = "PSO"
    
    def _optimize(self, objective_function, X=None, y=None, **kwargs):
        """PSO optimization implementation with automatic parameter calculation"""
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
        
        # Initialize velocities
        velocity = np.random.uniform(-1, 1, (self.population_size, self.dimensions))
        
        # Initialize personal best positions and fitness
        personal_best = population.copy()
        personal_best_fitness = np.array([objective_function(ind) for ind in population])
        
        # Find global best
        global_best_idx = np.argmin(personal_best_fitness)
        global_best = personal_best[global_best_idx].copy()
        global_best_fitness = personal_best_fitness[global_best_idx]
        
        convergence_curve = []
        
        for iteration in range(self.max_iterations):
            for i in range(self.population_size):
                # Update velocity
                r1, r2 = np.random.random(2)
                velocity[i] = (self.w * velocity[i] + 
                             self.c1 * r1 * (personal_best[i] - population[i]) +
                             self.c2 * r2 * (global_best - population[i]))
                
                # Update position
                population[i] += velocity[i]
                
                # Apply bounds
                population[i] = np.clip(population[i], self.lower_bound, self.upper_bound)
                
                # Evaluate fitness
                fitness = objective_function(population[i])
                
                # Update personal best
                if fitness < personal_best_fitness[i]:
                    personal_best[i] = population[i].copy()
                    personal_best_fitness[i] = fitness
                    
                    # Update global best
                    if fitness < global_best_fitness:
                        global_best = population[i].copy()
                        global_best_fitness = fitness
            
            convergence_curve.append(global_best_fitness)
            
            if self.verbose and (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration + 1}/{self.max_iterations}, Best Fitness: {global_best_fitness:.6f}")
        
        return global_best, global_best_fitness, convergence_curve
