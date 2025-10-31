"""
ABC-DE Hybrid (Artificial Bee Colony - Differential Evolution)
=============================================================

Hybrid algorithm combining ABC's exploitation with DE's mutation strategy.
"""

import numpy as np
from ...base import BaseOptimizer


class ABC_DE_Hybrid(BaseOptimizer):
    """ABC-DE Hybrid combining bee colony search with differential evolution"""
    
    def __init__(self, population_size=30, max_iterations=100):
        super().__init__(population_size, max_iterations)
        self.algorithm_name = "ABC-DE Hybrid"
        self.aliases = ["abc_de", "abc_de_hybrid", "bee_differential"]
        self.limit = 10  # ABC limit parameter
        self.F = 0.5  # DE scaling factor
        self.CR = 0.9  # DE crossover rate
    
    def _optimize(self, objective_function, bounds, dimension):
        # Initialize population
        population = np.random.uniform(bounds[0], bounds[1], (self.population_size, dimension))
        fitness = np.array([objective_function(ind) for ind in population])
        trial_counter = np.zeros(self.population_size)
        
        best_idx = np.argmin(fitness)
        best_position = population[best_idx].copy()
        best_fitness = fitness[best_idx]
        
        global_fitness = [best_fitness]
        local_fitness = [fitness.copy()]
        local_positions = [population.copy()]
        
        for iteration in range(self.max_iterations):
            # ABC Employed Bee Phase with DE mutation
            for i in range(self.population_size):
                # DE mutation
                candidates = list(range(self.population_size))
                candidates.remove(i)
                a, b, c = np.random.choice(candidates, 3, replace=False)
                mutant = population[a] + self.F * (population[b] - population[c])
                
                # ABC neighborhood search
                phi = np.random.uniform(-1, 1, dimension)
                k = np.random.choice(candidates)
                trial = population[i] + phi * (population[i] - population[k])
                
                # DE crossover
                crossover_mask = np.random.rand(dimension) < self.CR
                trial[crossover_mask] = mutant[crossover_mask]
                trial = np.clip(trial, bounds[0], bounds[1])
                
                trial_fitness = objective_function(trial)
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                    trial_counter[i] = 0
                else:
                    trial_counter[i] += 1
            
            # ABC Onlooker Bee Phase
            probs = (1 / (fitness + 1e-10)) / np.sum(1 / (fitness + 1e-10))
            for i in range(self.population_size):
                selected = np.random.choice(self.population_size, p=probs)
                phi = np.random.uniform(-1, 1, dimension)
                k = np.random.randint(0, self.population_size)
                trial = population[selected] + phi * (population[selected] - population[k])
                trial = np.clip(trial, bounds[0], bounds[1])
                
                trial_fitness = objective_function(trial)
                if trial_fitness < fitness[selected]:
                    population[selected] = trial
                    fitness[selected] = trial_fitness
                    trial_counter[selected] = 0
            
            # Scout Bee Phase
            for i in range(self.population_size):
                if trial_counter[i] > self.limit:
                    population[i] = np.random.uniform(bounds[0], bounds[1], dimension)
                    fitness[i] = objective_function(population[i])
                    trial_counter[i] = 0
            
            current_best_idx = np.argmin(fitness)
            if fitness[current_best_idx] < best_fitness:
                best_position = population[current_best_idx].copy()
                best_fitness = fitness[current_best_idx]
            
            global_fitness.append(best_fitness)
            local_fitness.append(fitness.copy())
            local_positions.append(population.copy())
        
        return best_position, best_fitness, global_fitness, local_fitness, local_positions