"""
Differential Evolution (DE)

Based on: Storn, R., & Price, K. (1997). Differential evolution–a simple and 
efficient heuristic for global optimization over continuous spaces.
"""

import numpy as np
from ..base import BaseOptimizer


class DifferentialEvolution(BaseOptimizer):
    """
    Differential Evolution (DE)
    
    DE is a population-based optimization algorithm that uses vector differences
    for perturbing the vector population. It includes mutation, crossover, and
    selection operations.
    
    Parameters
    ----------
    F : float, default=0.5
        Differential weight (scaling factor)
    CR : float, default=0.7
        Crossover probability
    strategy : str, default='rand/1/bin'
        DE strategy to use
    """
    
    aliases = ["de", "differential", "differential_evolution"]
    
    def __init__(self, F=0.5, CR=0.7, strategy='rand/1/bin', **kwargs):
        super().__init__(**kwargs)
        self.F = F
        self.CR = CR
        self.strategy = strategy
        self.algorithm_name = "DE"
    
    def _optimize(self, objective_function, **kwargs):
        """DE optimization implementation"""
        # Initialize population
        population = np.random.uniform(
            self.lower_bound, self.upper_bound,
            (self.population_size, self.dimensions)
        )
        
        # Evaluate initial population
        fitness = np.array([objective_function(ind) for ind in population])
        
        convergence_curve = []
        
        for iteration in range(self.max_iterations):
            for i in range(self.population_size):
                # Mutation
                if self.strategy == 'rand/1/bin':
                    # Select three random vectors (different from current)
                    candidates = list(range(self.population_size))
                    candidates.remove(i)
                    r1, r2, r3 = np.random.choice(candidates, 3, replace=False)
                    
                    # Create mutant vector
                    mutant = population[r1] + self.F * (population[r2] - population[r3])
                
                elif self.strategy == 'best/1/bin':
                    # Use best individual
                    best_idx = np.argmin(fitness)
                    candidates = list(range(self.population_size))
                    candidates.remove(i)
                    r1, r2 = np.random.choice(candidates, 2, replace=False)
                    
                    mutant = population[best_idx] + self.F * (population[r1] - population[r2])
                
                # Apply bounds to mutant
                mutant = np.clip(mutant, self.lower_bound, self.upper_bound)
                
                # Crossover
                trial = population[i].copy()
                j_rand = np.random.randint(0, self.dimensions)
                
                for j in range(self.dimensions):
                    if np.random.random() < self.CR or j == j_rand:
                        trial[j] = mutant[j]
                
                # Selection
                trial_fitness = objective_function(trial)
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
            
            # Record best fitness
            best_fitness = np.min(fitness)
            convergence_curve.append(best_fitness)
            
            if self.verbose and (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration + 1}/{self.max_iterations}, Best Fitness: {best_fitness:.6f}")
        
        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx], convergence_curve
