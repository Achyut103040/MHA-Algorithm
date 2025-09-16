"""
Genetic Algorithm (GA)

Based on: Holland, J. H. (1992). Genetic algorithms. Scientific American, 267(1), 66-72.
"""

import numpy as np
from ..base import BaseOptimizer


class GeneticAlgorithm(BaseOptimizer):
    """
    Genetic Algorithm (GA)
    
    GA is inspired by the process of natural selection and genetics.
    Solutions evolve through selection, crossover, and mutation operations.
    
    Parameters
    ----------
    crossover_rate : float, default=0.8
        Probability of crossover between two parents
    mutation_rate : float, default=0.1
        Probability of mutation for each gene
    elite_rate : float, default=0.1
        Percentage of best individuals to keep in next generation
    """
    
    aliases = ["ga", "genetic", "genetic_algorithm"]
    
    def __init__(self, crossover_rate=0.8, mutation_rate=0.1, elite_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elite_rate = elite_rate
        self.algorithm_name = "GA"
    
    def _optimize(self, objective_function, X=None, y=None, **kwargs):
        """GA optimization implementation with automatic parameter calculation"""
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
        
        convergence_curve = []
        elite_count = max(1, int(self.elite_rate * self.population_size))
        
        for iteration in range(self.max_iterations):
            # Evaluate fitness
            fitness = np.array([objective_function(ind) for ind in population])
            
            # Sort by fitness (ascending - minimize)
            sorted_indices = np.argsort(fitness)
            population = population[sorted_indices]
            fitness = fitness[sorted_indices]
            
            convergence_curve.append(fitness[0])
            
            if self.verbose and (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration + 1}/{self.max_iterations}, Best Fitness: {fitness[0]:.6f}")
            
            # Create new population
            new_population = np.zeros_like(population)
            
            # Keep elite individuals
            new_population[:elite_count] = population[:elite_count]
            
            # Generate offspring
            for i in range(elite_count, self.population_size):
                # Tournament selection
                parent1 = self._tournament_selection(population, fitness)
                parent2 = self._tournament_selection(population, fitness)
                
                # Crossover
                if np.random.random() < self.crossover_rate:
                    offspring = self._crossover(parent1, parent2)
                else:
                    offspring = parent1.copy()
                
                # Mutation
                offspring = self._mutate(offspring)
                
                new_population[i] = offspring
            
            population = new_population
        
        # Final evaluation
        final_fitness = np.array([objective_function(ind) for ind in population])
        best_idx = np.argmin(final_fitness)
        
        return population[best_idx], final_fitness[best_idx], convergence_curve
    
    def _tournament_selection(self, population, fitness, tournament_size=3):
        """Tournament selection"""
        tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
        tournament_fitness = fitness[tournament_indices]
        winner_idx = tournament_indices[np.argmin(tournament_fitness)]
        return population[winner_idx].copy()
    
    def _crossover(self, parent1, parent2):
        """Single-point crossover"""
        crossover_point = np.random.randint(1, len(parent1))
        offspring = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
        return offspring
    
    def _mutate(self, individual):
        """Gaussian mutation"""
        for i in range(len(individual)):
            if np.random.random() < self.mutation_rate:
                if hasattr(self, 'upper_bound') and hasattr(self, 'lower_bound'):
                    # Handle both scalar and array bounds
                    ub = self.upper_bound[i] if hasattr(self.upper_bound, '__len__') else self.upper_bound
                    lb = self.lower_bound[i] if hasattr(self.lower_bound, '__len__') else self.lower_bound
                    mutation_range = abs(ub - lb)
                else:
                    mutation_range = 2.0
                    
                individual[i] += np.random.normal(0, 0.1 * mutation_range)
                
                if hasattr(self, 'upper_bound') and hasattr(self, 'lower_bound'):
                    ub = self.upper_bound[i] if hasattr(self.upper_bound, '__len__') else self.upper_bound
                    lb = self.lower_bound[i] if hasattr(self.lower_bound, '__len__') else self.lower_bound
                    individual[i] = np.clip(individual[i], lb, ub)
        return individual
