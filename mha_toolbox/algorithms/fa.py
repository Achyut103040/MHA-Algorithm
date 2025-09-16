"""
Firefly Algorithm (FA)

Based on: Yang, X. S. (2009). Firefly algorithms for multimodal optimization.
In International symposium on stochastic algorithms (pp. 169-178).
"""

import numpy as np
from ..base import BaseOptimizer


class FireflyAlgorithm(BaseOptimizer):
    """
    Firefly Algorithm (FA)
    
    FA is inspired by the flashing behavior of fireflies. The algorithm
    is based on the assumption that all fireflies are unisex and they
    are attracted to other fireflies regardless of their sex.
    
    Parameters
    ----------
    alpha : float, default=0.2
        Randomization parameter
    beta_0 : float, default=1.0
        Attractiveness at distance r=0
    gamma : float, default=1.0
        Light absorption coefficient
    """
    
    aliases = ["fa", "firefly", "firefly_algorithm"]
    
    def __init__(self, alpha=0.2, beta_0=1.0, gamma=1.0, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.beta_0 = beta_0
        self.gamma = gamma
        self.algorithm_name = "FA"
    
    def _optimize(self, objective_function, **kwargs):
        """FA optimization implementation"""
        # Initialize population
        population = np.random.uniform(
            self.lower_bound, self.upper_bound,
            (self.population_size, self.dimensions)
        )
        
        # Calculate fitness for all fireflies
        fitness = np.array([objective_function(ind) for ind in population])
        
        convergence_curve = []
        
        for iteration in range(self.max_iterations):
            # Update fireflies
            for i in range(self.population_size):
                for j in range(self.population_size):
                    if fitness[j] < fitness[i]:  # j is brighter than i
                        # Calculate distance
                        r = np.linalg.norm(population[i] - population[j])
                        
                        # Calculate attractiveness
                        beta = self.beta_0 * np.exp(-self.gamma * r**2)
                        
                        # Move firefly i towards j
                        population[i] += beta * (population[j] - population[i]) + \
                                       self.alpha * (np.random.random(self.dimensions) - 0.5)
                
                # Apply bounds
                population[i] = np.clip(population[i], self.lower_bound, self.upper_bound)
                
                # Update fitness
                fitness[i] = objective_function(population[i])
            
            # Find best firefly
            best_idx = np.argmin(fitness)
            convergence_curve.append(fitness[best_idx])
            
            if self.verbose and (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration + 1}/{self.max_iterations}, Best Fitness: {fitness[best_idx]:.6f}")
            
            # Reduce alpha (randomization parameter)
            self.alpha *= 0.98
        
        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx], convergence_curve
