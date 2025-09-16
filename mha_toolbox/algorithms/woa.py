"""
Whale Optimization Algorithm (WOA)

Based on: Mirjalili, S., & Lewis, A. (2016). The whale optimization algorithm. 
Advances in engineering software, 95, 51-67.
"""

import numpy as np
from ..base import BaseOptimizer


class WhaleOptimizationAlgorithm(BaseOptimizer):
    """
    Whale Optimization Algorithm (WOA)
    
    WOA mimics the social behavior of humpback whales and their bubble-net
    hunting strategy. The algorithm implements encircling prey, bubble-net
    attacking method, and search for prey.
    """
    
    aliases = ["woa", "whale", "whale_optimization"]
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.algorithm_name = "WOA"
    
    def _optimize(self, objective_function, **kwargs):
        """WOA optimization implementation"""
        # Initialize population
        population = np.random.uniform(
            self.lower_bound, self.upper_bound,
            (self.population_size, self.dimensions)
        )
        
        # Initialize the position of leader (best search agent)
        leader_pos = np.zeros(self.dimensions)
        leader_score = float('inf')
        
        convergence_curve = []
        
        for iteration in range(self.max_iterations):
            for i in range(len(population)):
                # Return back the search agents that go beyond the boundaries
                population[i] = np.clip(population[i], self.lower_bound, self.upper_bound)
                
                # Calculate objective function for each search agent
                fitness = objective_function(population[i])
                
                # Update the leader
                if fitness < leader_score:
                    leader_score = fitness
                    leader_pos = population[i].copy()
            
            # Calculate convergence curve
            convergence_curve.append(leader_score)
            
            if self.verbose and (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration + 1}/{self.max_iterations}, Best Fitness: {leader_score:.6f}")
            
            # Update the position of search agents
            a = 2 - iteration * (2.0 / self.max_iterations)  # a decreases linearly from 2 to 0
            
            for i in range(len(population)):
                r1 = np.random.random()
                r2 = np.random.random()
                
                A = 2 * a * r1 - a
                C = 2 * r2
                
                b = 1  # defining the shape of the logarithmic spiral
                l = np.random.uniform(-1, 1)  # random number in [-1,1]
                
                p = np.random.random()  # p in [0,1]
                
                for j in range(self.dimensions):
                    if p < 0.5:
                        if abs(A) >= 1:
                            # Search for prey (exploration phase)
                            rand_leader_index = np.random.randint(0, len(population))
                            X_rand = population[rand_leader_index]
                            D_X_rand = abs(C * X_rand[j] - population[i][j])
                            population[i][j] = X_rand[j] - A * D_X_rand
                        elif abs(A) < 1:
                            # Encircling prey (exploitation phase)
                            D_Leader = abs(C * leader_pos[j] - population[i][j])
                            population[i][j] = leader_pos[j] - A * D_Leader
                    elif p >= 0.5:
                        # Bubble-net attacking method (exploitation phase)
                        distance2Leader = abs(leader_pos[j] - population[i][j])
                        population[i][j] = distance2Leader * np.exp(b * l) * np.cos(l * 2 * np.pi) + leader_pos[j]
        
        return leader_pos, leader_score, convergence_curve
