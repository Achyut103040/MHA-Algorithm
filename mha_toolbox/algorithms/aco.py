"""
Ant Colony Optimization (ACO)

Based on: Dorigo, M., & Gambardella, L. M. (1997). Ant colony system: 
a cooperative learning approach to the traveling salesman problem.
"""

import numpy as np
from ..base import BaseOptimizer


class AntColonyOptimization(BaseOptimizer):
    """
    Ant Colony Optimization (ACO) for continuous optimization
    
    ACO is inspired by the foraging behavior of ants. This implementation
    adapts ACO for continuous optimization problems using a solution
    construction mechanism based on probability distributions.
    
    Parameters
    ----------
    q : float, default=0.01
        Intensification parameter
    zeta : float, default=1.0
        Deviation-distance ratio parameter
    """
    
    def __init__(self, q=0.01, zeta=1.0, **kwargs):
        super().__init__(**kwargs)
        self.q = q
        self.zeta = zeta
        self.algorithm_name = "ACO"
    
    def _optimize(self, objective_function, **kwargs):
        """ACO optimization implementation"""
        # Initialize solution archive
        archive = []
        archive_size = self.population_size
        
        # Generate initial solutions
        for _ in range(archive_size):
            solution = np.random.uniform(self.lower_bound, self.upper_bound, self.dimensions)
            fitness = objective_function(solution)
            archive.append((solution, fitness))
        
        # Sort archive by fitness
        archive.sort(key=lambda x: x[1])
        
        convergence_curve = []
        
        for iteration in range(self.max_iterations):
            new_solutions = []
            
            for _ in range(self.population_size):
                # Construct solution using probabilistic approach
                solution = self._construct_solution(archive)
                
                # Apply bounds
                solution = np.clip(solution, self.lower_bound, self.upper_bound)
                
                # Evaluate solution
                fitness = objective_function(solution)
                new_solutions.append((solution, fitness))
            
            # Update archive
            archive.extend(new_solutions)
            archive.sort(key=lambda x: x[1])
            archive = archive[:archive_size]  # Keep only best solutions
            
            convergence_curve.append(archive[0][1])
            
            if self.verbose and (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration + 1}/{self.max_iterations}, Best Fitness: {archive[0][1]:.6f}")
        
        best_solution, best_fitness = archive[0]
        return best_solution, best_fitness, convergence_curve
    
    def _construct_solution(self, archive):
        """Construct a solution using the solution archive"""
        solution = np.zeros(self.dimensions)
        
        # Calculate weights for archive solutions
        weights = []
        for i, (_, fitness) in enumerate(archive):
            weight = (1.0 / (len(archive) * self.q * np.sqrt(2 * np.pi))) * \
                    np.exp(-0.5 * ((i) / (self.q * len(archive)))**2)
            weights.append(weight)
        
        weights = np.array(weights)
        weights /= np.sum(weights)  # Normalize weights
        
        for j in range(self.dimensions):
            # Select archive solution based on weights
            selected_idx = np.random.choice(len(archive), p=weights)
            selected_solution = archive[selected_idx][0]
            
            # Calculate standard deviation
            sigma = self.zeta * np.sum([abs(sol[j] - selected_solution[j]) 
                                      for sol, _ in archive]) / (len(archive) - 1)
            
            # Generate component value
            if sigma > 0:
                solution[j] = np.random.normal(selected_solution[j], sigma)
            else:
                solution[j] = selected_solution[j]
        
        return solution
