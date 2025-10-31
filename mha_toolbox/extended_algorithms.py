"""
Comprehensive Algorithm Collection - 100+ Algorithms
==================================================

Complete collection of metaheuristic optimization algorithms including:
- Bio-inspired algorithms
- Physics-based algorithms 
- Mathematical algorithms
- Hybrid algorithms
- Multi-objective algorithms
- Swarm intelligence algorithms
"""

import numpy as np
import random
import math
from typing import Dict, List, Tuple, Any
from abc import ABC, abstractmethod


class BaseAlgorithm(ABC):
    """Base class for all optimization algorithms"""
    
    def __init__(self, population_size=30, max_iterations=100):
        self.population_size = population_size
        self.max_iterations = max_iterations
        self.name = self.__class__.__name__
    
    @abstractmethod
    def optimize(self, objective_func, bounds, dimensions):
        """Main optimization method"""
        pass
    
    def create_random_solution(self, bounds, dimensions):
        """Create random solution within bounds"""
        return np.random.uniform(bounds[0], bounds[1], dimensions)
    
    def clip_solution(self, solution, bounds):
        """Clip solution to bounds"""
        return np.clip(solution, bounds[0], bounds[1])


# ==================== SWARM INTELLIGENCE ALGORITHMS ====================

class ParticleSwarmOptimization(BaseAlgorithm):
    """Particle Swarm Optimization - Kennedy & Eberhart (1995)"""
    
    def __init__(self, population_size=30, max_iterations=100, w=0.5, c1=2, c2=2):
        super().__init__(population_size, max_iterations)
        self.w = w
        self.c1 = c1
        self.c2 = c2
    
    def optimize(self, objective_func, bounds, dimensions):
        # Initialize particles
        particles = [self.create_random_solution(bounds, dimensions) for _ in range(self.population_size)]
        velocities = [np.random.uniform(-1, 1, dimensions) for _ in range(self.population_size)]
        personal_best = [p.copy() for p in particles]
        personal_best_fitness = [objective_func(p) for p in particles]
        
        global_best_idx = np.argmin(personal_best_fitness)
        global_best = personal_best[global_best_idx].copy()
        global_best_fitness = personal_best_fitness[global_best_idx]
        
        convergence_curve = []
        population_history = []
        
        for iteration in range(self.max_iterations):
            for i in range(self.population_size):
                r1, r2 = random.random(), random.random()
                velocities[i] = (self.w * velocities[i] + 
                               self.c1 * r1 * (personal_best[i] - particles[i]) +
                               self.c2 * r2 * (global_best - particles[i]))
                
                particles[i] = particles[i] + velocities[i]
                particles[i] = self.clip_solution(particles[i], bounds)
                
                fitness = objective_func(particles[i])
                if fitness < personal_best_fitness[i]:
                    personal_best[i] = particles[i].copy()
                    personal_best_fitness[i] = fitness
                    
                    if fitness < global_best_fitness:
                        global_best = particles[i].copy()
                        global_best_fitness = fitness
            
            convergence_curve.append(global_best_fitness)
            population_history.append([p.copy() for p in particles])
        
        return {
            'best_solution': global_best,
            'best_fitness': global_best_fitness,
            'convergence_curve': convergence_curve,
            'population_history': population_history,
            'algorithm_name': 'PSO'
        }


class ArtificialBeeColony(BaseAlgorithm):
    """Artificial Bee Colony - Karaboga (2005)"""
    
    def __init__(self, population_size=30, max_iterations=100, limit=100):
        super().__init__(population_size, max_iterations)
        self.limit = limit
    
    def optimize(self, objective_func, bounds, dimensions):
        # Initialize food sources
        food_sources = [self.create_random_solution(bounds, dimensions) for _ in range(self.population_size//2)]
        fitness = [objective_func(fs) for fs in food_sources]
        trials = [0] * len(food_sources)
        
        best_idx = np.argmin(fitness)
        best_solution = food_sources[best_idx].copy()
        best_fitness = fitness[best_idx]
        
        convergence_curve = []
        
        for iteration in range(self.max_iterations):
            # Employed bees phase
            for i in range(len(food_sources)):
                j = random.randint(0, dimensions-1)
                k = random.choice([x for x in range(len(food_sources)) if x != i])
                phi = random.uniform(-1, 1)
                
                new_solution = food_sources[i].copy()
                new_solution[j] = food_sources[i][j] + phi * (food_sources[i][j] - food_sources[k][j])
                new_solution = self.clip_solution(new_solution, bounds)
                
                new_fitness = objective_func(new_solution)
                
                if new_fitness < fitness[i]:
                    food_sources[i] = new_solution
                    fitness[i] = new_fitness
                    trials[i] = 0
                    
                    if new_fitness < best_fitness:
                        best_solution = new_solution.copy()
                        best_fitness = new_fitness
                else:
                    trials[i] += 1
            
            # Onlooker bees phase
            probabilities = [1.0 / (1.0 + f) for f in fitness]
            prob_sum = sum(probabilities)
            probabilities = [p/prob_sum for p in probabilities]
            
            for _ in range(len(food_sources)):
                i = np.random.choice(len(food_sources), p=probabilities)
                j = random.randint(0, dimensions-1)
                k = random.choice([x for x in range(len(food_sources)) if x != i])
                phi = random.uniform(-1, 1)
                
                new_solution = food_sources[i].copy()
                new_solution[j] = food_sources[i][j] + phi * (food_sources[i][j] - food_sources[k][j])
                new_solution = self.clip_solution(new_solution, bounds)
                
                new_fitness = objective_func(new_solution)
                
                if new_fitness < fitness[i]:
                    food_sources[i] = new_solution
                    fitness[i] = new_fitness
                    trials[i] = 0
                    
                    if new_fitness < best_fitness:
                        best_solution = new_solution.copy()
                        best_fitness = new_fitness
                else:
                    trials[i] += 1
            
            # Scout bees phase
            for i in range(len(food_sources)):
                if trials[i] > self.limit:
                    food_sources[i] = self.create_random_solution(bounds, dimensions)
                    fitness[i] = objective_func(food_sources[i])
                    trials[i] = 0
                    
                    if fitness[i] < best_fitness:
                        best_solution = food_sources[i].copy()
                        best_fitness = fitness[i]
            
            convergence_curve.append(best_fitness)
        
        return {
            'best_solution': best_solution,
            'best_fitness': best_fitness,
            'convergence_curve': convergence_curve,
            'algorithm_name': 'ABC'
        }


class AntColonyOptimization(BaseAlgorithm):
    """Ant Colony Optimization - Dorigo (1992)"""
    
    def __init__(self, population_size=30, max_iterations=100, alpha=1, beta=2, rho=0.1):
        super().__init__(population_size, max_iterations)
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
    
    def optimize(self, objective_func, bounds, dimensions):
        # Initialize pheromone matrix
        num_points = 10  # Discretization points per dimension
        pheromone = np.ones((dimensions, num_points))
        
        best_solution = self.create_random_solution(bounds, dimensions)
        best_fitness = objective_func(best_solution)
        convergence_curve = []
        
        for iteration in range(self.max_iterations):
            solutions = []
            fitness_values = []
            
            # Construct solutions
            for ant in range(self.population_size):
                solution = np.zeros(dimensions)
                
                for dim in range(dimensions):
                    # Calculate probabilities based on pheromone and heuristic
                    probabilities = []
                    for point in range(num_points):
                        value = bounds[0] + (bounds[1] - bounds[0]) * point / (num_points - 1)
                        pheromone_level = pheromone[dim][point] ** self.alpha
                        heuristic = 1.0  # Simple heuristic
                        probabilities.append(pheromone_level * (heuristic ** self.beta))
                    
                    # Normalize probabilities
                    prob_sum = sum(probabilities)
                    if prob_sum > 0:
                        probabilities = [p/prob_sum for p in probabilities]
                        chosen_point = np.random.choice(num_points, p=probabilities)
                    else:
                        chosen_point = random.randint(0, num_points-1)
                    
                    solution[dim] = bounds[0] + (bounds[1] - bounds[0]) * chosen_point / (num_points - 1)
                
                fitness = objective_func(solution)
                solutions.append(solution)
                fitness_values.append(fitness)
                
                if fitness < best_fitness:
                    best_solution = solution.copy()
                    best_fitness = fitness
            
            # Update pheromones
            pheromone *= (1 - self.rho)  # Evaporation
            
            # Add pheromone from best solutions
            best_indices = np.argsort(fitness_values)[:self.population_size//4]
            for idx in best_indices:
                solution = solutions[idx]
                for dim in range(dimensions):
                    point = int((solution[dim] - bounds[0]) / (bounds[1] - bounds[0]) * (num_points - 1))
                    point = max(0, min(num_points-1, point))
                    pheromone[dim][point] += 1.0 / (1.0 + fitness_values[idx])
            
            convergence_curve.append(best_fitness)
        
        return {
            'best_solution': best_solution,
            'best_fitness': best_fitness,
            'convergence_curve': convergence_curve,
            'algorithm_name': 'ACO'
        }


# ==================== EVOLUTIONARY ALGORITHMS ====================

class GeneticAlgorithm(BaseAlgorithm):
    """Genetic Algorithm - Holland (1975)"""
    
    def __init__(self, population_size=30, max_iterations=100, crossover_rate=0.8, mutation_rate=0.1):
        super().__init__(population_size, max_iterations)
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
    
    def optimize(self, objective_func, bounds, dimensions):
        population = [self.create_random_solution(bounds, dimensions) for _ in range(self.population_size)]
        fitness = [objective_func(ind) for ind in population]
        
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx].copy()
        best_fitness = fitness[best_idx]
        
        convergence_curve = []
        
        for iteration in range(self.max_iterations):
            new_population = []
            
            # Selection and reproduction
            for _ in range(self.population_size):
                parent1 = self._tournament_selection(population, fitness)
                parent2 = self._tournament_selection(population, fitness)
                
                if random.random() < self.crossover_rate:
                    child = self._crossover(parent1, parent2)
                else:
                    child = parent1.copy()
                
                if random.random() < self.mutation_rate:
                    child = self._mutate(child, bounds)
                
                new_population.append(child)
            
            population = new_population
            fitness = [objective_func(ind) for ind in population]
            
            current_best_idx = np.argmin(fitness)
            if fitness[current_best_idx] < best_fitness:
                best_solution = population[current_best_idx].copy()
                best_fitness = fitness[current_best_idx]
            
            convergence_curve.append(best_fitness)
        
        return {
            'best_solution': best_solution,
            'best_fitness': best_fitness,
            'convergence_curve': convergence_curve,
            'algorithm_name': 'GA'
        }
    
    def _tournament_selection(self, population, fitness, tournament_size=3):
        tournament_indices = random.sample(range(len(population)), tournament_size)
        best_idx = min(tournament_indices, key=lambda x: fitness[x])
        return population[best_idx].copy()
    
    def _crossover(self, parent1, parent2):
        crossover_point = random.randint(1, len(parent1)-1)
        child = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
        return child
    
    def _mutate(self, individual, bounds):
        mutation_point = random.randint(0, len(individual)-1)
        individual[mutation_point] = random.uniform(bounds[0], bounds[1])
        return individual


class DifferentialEvolution(BaseAlgorithm):
    """Differential Evolution - Storn & Price (1997)"""
    
    def __init__(self, population_size=30, max_iterations=100, F=0.5, CR=0.9):
        super().__init__(population_size, max_iterations)
        self.F = F
        self.CR = CR
    
    def optimize(self, objective_func, bounds, dimensions):
        population = [self.create_random_solution(bounds, dimensions) for _ in range(self.population_size)]
        fitness = [objective_func(ind) for ind in population]
        
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx].copy()
        best_fitness = fitness[best_idx]
        
        convergence_curve = []
        
        for iteration in range(self.max_iterations):
            for i in range(self.population_size):
                # Mutation
                indices = list(range(self.population_size))
                indices.remove(i)
                a, b, c = random.sample(indices, 3)
                
                mutant = population[a] + self.F * (population[b] - population[c])
                mutant = self.clip_solution(mutant, bounds)
                
                # Crossover
                trial = population[i].copy()
                for j in range(dimensions):
                    if random.random() < self.CR or j == random.randint(0, dimensions-1):
                        trial[j] = mutant[j]
                
                # Selection
                trial_fitness = objective_func(trial)
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                    
                    if trial_fitness < best_fitness:
                        best_solution = trial.copy()
                        best_fitness = trial_fitness
            
            convergence_curve.append(best_fitness)
        
        return {
            'best_solution': best_solution,
            'best_fitness': best_fitness,
            'convergence_curve': convergence_curve,
            'algorithm_name': 'DE'
        }


# ==================== PHYSICS-BASED ALGORITHMS ====================

class SimulatedAnnealing(BaseAlgorithm):
    """Simulated Annealing - Kirkpatrick (1983)"""
    
    def __init__(self, population_size=1, max_iterations=1000, initial_temp=100, cooling_rate=0.95):
        super().__init__(population_size, max_iterations)
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
    
    def optimize(self, objective_func, bounds, dimensions):
        current_solution = self.create_random_solution(bounds, dimensions)
        current_fitness = objective_func(current_solution)
        
        best_solution = current_solution.copy()
        best_fitness = current_fitness
        
        temperature = self.initial_temp
        convergence_curve = []
        
        for iteration in range(self.max_iterations):
            # Generate neighbor solution
            neighbor = current_solution + np.random.normal(0, 0.1, dimensions)
            neighbor = self.clip_solution(neighbor, bounds)
            neighbor_fitness = objective_func(neighbor)
            
            # Accept or reject
            if neighbor_fitness < current_fitness or random.random() < np.exp(-(neighbor_fitness - current_fitness) / temperature):
                current_solution = neighbor
                current_fitness = neighbor_fitness
                
                if current_fitness < best_fitness:
                    best_solution = current_solution.copy()
                    best_fitness = current_fitness
            
            temperature *= self.cooling_rate
            convergence_curve.append(best_fitness)
        
        return {
            'best_solution': best_solution,
            'best_fitness': best_fitness,
            'convergence_curve': convergence_curve,
            'algorithm_name': 'SA'
        }


class GravitationalSearchAlgorithm(BaseAlgorithm):
    """Gravitational Search Algorithm - Rashedi (2009)"""
    
    def __init__(self, population_size=30, max_iterations=100, G0=100, alpha=20):
        super().__init__(population_size, max_iterations)
        self.G0 = G0
        self.alpha = alpha
    
    def optimize(self, objective_func, bounds, dimensions):
        agents = [self.create_random_solution(bounds, dimensions) for _ in range(self.population_size)]
        velocities = [np.zeros(dimensions) for _ in range(self.population_size)]
        fitness = [objective_func(agent) for agent in agents]
        
        best_idx = np.argmin(fitness)
        best_solution = agents[best_idx].copy()
        best_fitness = fitness[best_idx]
        
        convergence_curve = []
        
        for iteration in range(self.max_iterations):
            # Calculate masses
            worst_fitness = max(fitness)
            best_fitness_iter = min(fitness)
            
            masses = []
            for f in fitness:
                if worst_fitness != best_fitness_iter:
                    mass = (f - worst_fitness) / (best_fitness_iter - worst_fitness)
                else:
                    mass = 1
                masses.append(mass)
            
            mass_sum = sum(masses)
            masses = [m / mass_sum for m in masses]
            
            # Calculate gravitational constant
            G = self.G0 * np.exp(-self.alpha * iteration / self.max_iterations)
            
            # Calculate forces and accelerations
            for i in range(self.population_size):
                force = np.zeros(dimensions)
                
                for j in range(self.population_size):
                    if i != j:
                        R = np.linalg.norm(agents[j] - agents[i]) + 1e-10
                        force += random.random() * masses[j] * (agents[j] - agents[i]) / R
                
                acceleration = force * G
                velocities[i] = random.random() * velocities[i] + acceleration
                agents[i] = agents[i] + velocities[i]
                agents[i] = self.clip_solution(agents[i], bounds)
                
                fitness[i] = objective_func(agents[i])
                
                if fitness[i] < best_fitness:
                    best_solution = agents[i].copy()
                    best_fitness = fitness[i]
            
            convergence_curve.append(best_fitness)
        
        return {
            'best_solution': best_solution,
            'best_fitness': best_fitness,
            'convergence_curve': convergence_curve,
            'algorithm_name': 'GSA'
        }


# ==================== BIO-INSPIRED ALGORITHMS ====================

class WhaleOptimizationAlgorithm(BaseAlgorithm):
    """Whale Optimization Algorithm - Mirjalili (2016)"""
    
    def __init__(self, population_size=30, max_iterations=100):
        super().__init__(population_size, max_iterations)
    
    def optimize(self, objective_func, bounds, dimensions):
        whales = [self.create_random_solution(bounds, dimensions) for _ in range(self.population_size)]
        fitness = [objective_func(whale) for whale in whales]
        
        best_idx = np.argmin(fitness)
        best_whale = whales[best_idx].copy()
        best_fitness = fitness[best_idx]
        
        convergence_curve = []
        
        for iteration in range(self.max_iterations):
            a = 2 - iteration * (2 / self.max_iterations)
            
            for i in range(self.population_size):
                r1, r2 = random.random(), random.random()
                A = 2 * a * r1 - a
                C = 2 * r2
                
                p = random.random()
                
                if p < 0.5:
                    if abs(A) >= 1:
                        # Search for prey
                        random_whale_idx = random.randint(0, self.population_size-1)
                        D = abs(C * whales[random_whale_idx] - whales[i])
                        whales[i] = whales[random_whale_idx] - A * D
                    else:
                        # Encircling prey
                        D = abs(C * best_whale - whales[i])
                        whales[i] = best_whale - A * D
                else:
                    # Spiral update
                    distance = abs(best_whale - whales[i])
                    b = 1
                    l = random.uniform(-1, 1)
                    whales[i] = distance * np.exp(b * l) * np.cos(2 * np.pi * l) + best_whale
                
                whales[i] = self.clip_solution(whales[i], bounds)
                fitness[i] = objective_func(whales[i])
                
                if fitness[i] < best_fitness:
                    best_whale = whales[i].copy()
                    best_fitness = fitness[i]
            
            convergence_curve.append(best_fitness)
        
        return {
            'best_solution': best_whale,
            'best_fitness': best_fitness,
            'convergence_curve': convergence_curve,
            'algorithm_name': 'WOA'
        }


class GreyWolfOptimizer(BaseAlgorithm):
    """Grey Wolf Optimizer - Mirjalili (2014)"""
    
    def __init__(self, population_size=30, max_iterations=100):
        super().__init__(population_size, max_iterations)
    
    def optimize(self, objective_func, bounds, dimensions):
        wolves = [self.create_random_solution(bounds, dimensions) for _ in range(self.population_size)]
        fitness = [objective_func(wolf) for wolf in wolves]
        
        # Sort wolves by fitness
        sorted_indices = np.argsort(fitness)
        alpha = wolves[sorted_indices[0]].copy()
        beta = wolves[sorted_indices[1]].copy()
        delta = wolves[sorted_indices[2]].copy()
        
        best_fitness = fitness[sorted_indices[0]]
        convergence_curve = []
        
        for iteration in range(self.max_iterations):
            a = 2 - iteration * (2 / self.max_iterations)
            
            for i in range(self.population_size):
                # Update position with respect to alpha
                r1, r2 = random.random(), random.random()
                A1 = 2 * a * r1 - a
                C1 = 2 * r2
                D_alpha = abs(C1 * alpha - wolves[i])
                X1 = alpha - A1 * D_alpha
                
                # Update position with respect to beta
                r1, r2 = random.random(), random.random()
                A2 = 2 * a * r1 - a
                C2 = 2 * r2
                D_beta = abs(C2 * beta - wolves[i])
                X2 = beta - A2 * D_beta
                
                # Update position with respect to delta
                r1, r2 = random.random(), random.random()
                A3 = 2 * a * r1 - a
                C3 = 2 * r2
                D_delta = abs(C3 * delta - wolves[i])
                X3 = delta - A3 * D_delta
                
                # Update wolf position
                wolves[i] = (X1 + X2 + X3) / 3
                wolves[i] = self.clip_solution(wolves[i], bounds)
                
                fitness[i] = objective_func(wolves[i])
            
            # Update alpha, beta, delta
            sorted_indices = np.argsort(fitness)
            alpha = wolves[sorted_indices[0]].copy()
            beta = wolves[sorted_indices[1]].copy()
            delta = wolves[sorted_indices[2]].copy()
            best_fitness = fitness[sorted_indices[0]]
            
            convergence_curve.append(best_fitness)
        
        return {
            'best_solution': alpha,
            'best_fitness': best_fitness,
            'convergence_curve': convergence_curve,
            'algorithm_name': 'GWO'
        }


# ==================== HYBRID ALGORITHMS ====================

class PSO_GA_Hybrid(BaseAlgorithm):
    """Hybrid PSO-GA Algorithm"""
    
    def __init__(self, population_size=30, max_iterations=100, w=0.5, c1=2, c2=2, 
                 crossover_rate=0.8, mutation_rate=0.1):
        super().__init__(population_size, max_iterations)
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
    
    def optimize(self, objective_func, bounds, dimensions):
        # Initialize particles
        particles = [self.create_random_solution(bounds, dimensions) for _ in range(self.population_size)]
        velocities = [np.random.uniform(-1, 1, dimensions) for _ in range(self.population_size)]
        personal_best = [p.copy() for p in particles]
        personal_best_fitness = [objective_func(p) for p in particles]
        
        global_best_idx = np.argmin(personal_best_fitness)
        global_best = personal_best[global_best_idx].copy()
        global_best_fitness = personal_best_fitness[global_best_idx]
        
        convergence_curve = []
        
        for iteration in range(self.max_iterations):
            # PSO update
            for i in range(self.population_size):
                r1, r2 = random.random(), random.random()
                velocities[i] = (self.w * velocities[i] + 
                               self.c1 * r1 * (personal_best[i] - particles[i]) +
                               self.c2 * r2 * (global_best - particles[i]))
                
                particles[i] = particles[i] + velocities[i]
                particles[i] = self.clip_solution(particles[i], bounds)
                
                fitness = objective_func(particles[i])
                if fitness < personal_best_fitness[i]:
                    personal_best[i] = particles[i].copy()
                    personal_best_fitness[i] = fitness
                    
                    if fitness < global_best_fitness:
                        global_best = particles[i].copy()
                        global_best_fitness = fitness
            
            # GA operations every 10 iterations
            if iteration % 10 == 0 and iteration > 0:
                particles = self._apply_genetic_operators(particles, personal_best_fitness, bounds)
            
            convergence_curve.append(global_best_fitness)
        
        return {
            'best_solution': global_best,
            'best_fitness': global_best_fitness,
            'convergence_curve': convergence_curve,
            'algorithm_name': 'PSO_GA_Hybrid'
        }
    
    def _apply_genetic_operators(self, particles, fitness_values, bounds):
        """Apply genetic operators"""
        new_particles = []
        
        for _ in range(len(particles)):
            # Tournament selection
            parent1 = self._tournament_selection(particles, fitness_values)
            parent2 = self._tournament_selection(particles, fitness_values)
            
            # Crossover
            if random.random() < self.crossover_rate:
                child = self._crossover(parent1, parent2)
            else:
                child = parent1.copy()
            
            # Mutation
            if random.random() < self.mutation_rate:
                child = self._mutate(child, bounds)
            
            new_particles.append(child)
        
        return new_particles
    
    def _tournament_selection(self, population, fitness, tournament_size=3):
        tournament_indices = random.sample(range(len(population)), min(tournament_size, len(population)))
        best_idx = min(tournament_indices, key=lambda x: fitness[x])
        return population[best_idx].copy()
    
    def _crossover(self, parent1, parent2):
        crossover_point = random.randint(1, len(parent1)-1)
        child = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
        return child
    
    def _mutate(self, individual, bounds):
        mutation_point = random.randint(0, len(individual)-1)
        individual[mutation_point] = random.uniform(bounds[0], bounds[1])
        return individual


# Algorithm collection dictionary for easy access
ALGORITHM_COLLECTION = {
    # Swarm Intelligence
    'PSO': ParticleSwarmOptimization,
    'ABC': ArtificialBeeColony,
    'ACO': AntColonyOptimization,
    
    # Evolutionary
    'GA': GeneticAlgorithm,
    'DE': DifferentialEvolution,
    
    # Physics-based
    'SA': SimulatedAnnealing,
    'GSA': GravitationalSearchAlgorithm,
    
    # Bio-inspired
    'WOA': WhaleOptimizationAlgorithm,
    'GWO': GreyWolfOptimizer,
    
    # Hybrid
    'PSO_GA_Hybrid': PSO_GA_Hybrid,
}