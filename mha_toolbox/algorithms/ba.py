"""
Bat Algorithm (BA)

Based on: Yang, X. S. (2010). A new metaheuristic bat-inspired algorithm.
In Nature inspired cooperative strategies for optimization (pp. 65-74).
"""

import numpy as np
from ..base import BaseOptimizer


class BatAlgorithm(BaseOptimizer):
    """
    Bat Algorithm (BA)
    
    BA is inspired by the echolocation behavior of microbats. The algorithm
    uses the idealized behavior of microbats to perform optimization.
    
    Parameters
    ----------
    A : float, default=0.5
        Loudness (constant or decreasing)
    r : float, default=0.5
        Pulse rate (constant or increasing)
    f_min : float, default=0.0
        Minimum frequency
    f_max : float, default=2.0
        Maximum frequency
    """
    
    def __init__(self, A=0.5, r=0.5, f_min=0.0, f_max=2.0, **kwargs):
        super().__init__(**kwargs)
        self.A = A
        self.r = r
        self.f_min = f_min
        self.f_max = f_max
        self.algorithm_name = "BA"
    
    def _optimize(self, objective_function, **kwargs):
        """BA optimization implementation"""
        # Initialize population
        population = np.random.uniform(
            self.lower_bound, self.upper_bound,
            (self.population_size, self.dimensions)
        )
        
        # Initialize velocities
        velocity = np.zeros((self.population_size, self.dimensions))
        
        # Initialize frequency, loudness and pulse rate
        frequency = np.zeros(self.population_size)
        A = np.full(self.population_size, self.A)
        r = np.full(self.population_size, self.r)
        
        # Calculate fitness for all bats
        fitness = np.array([objective_function(ind) for ind in population])
        
        # Find the best solution
        best_idx = np.argmin(fitness)
        best_bat = population[best_idx].copy()
        best_fitness = fitness[best_idx]
        
        convergence_curve = []
        
        for iteration in range(self.max_iterations):
            for i in range(self.population_size):
                # Update frequency
                frequency[i] = self.f_min + (self.f_max - self.f_min) * np.random.random()
                
                # Update velocity
                velocity[i] += (population[i] - best_bat) * frequency[i]
                
                # Update position
                new_position = population[i] + velocity[i]
                
                # Apply local search around the best solution
                if np.random.random() > r[i]:
                    new_position = best_bat + 0.001 * np.random.randn(self.dimensions)
                
                # Apply bounds
                new_position = np.clip(new_position, self.lower_bound, self.upper_bound)
                
                # Evaluate new solution
                new_fitness = objective_function(new_position)
                
                # Accept new solution if it's better and within loudness
                if new_fitness < fitness[i] and np.random.random() < A[i]:
                    population[i] = new_position.copy()
                    fitness[i] = new_fitness
                    
                    # Update loudness and pulse rate
                    A[i] *= 0.9
                    r[i] = self.r * (1 - np.exp(-0.9 * iteration))
                    
                    # Update best solution
                    if new_fitness < best_fitness:
                        best_bat = new_position.copy()
                        best_fitness = new_fitness
            
            convergence_curve.append(best_fitness)
            
            if self.verbose and (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration + 1}/{self.max_iterations}, Best Fitness: {best_fitness:.6f}")
        
        return best_bat, best_fitness, convergence_curve
