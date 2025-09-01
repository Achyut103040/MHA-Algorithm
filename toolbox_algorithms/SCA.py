import numpy as np
import time
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.toolbox_utils import handle_bounds, initialize_population, boundary_check, evaluate_population, create_result_dict

def SCA(pop_size, max_iter, lb, ub, dim, obj_func, **kwargs):
    """
    Sine Cosine Algorithm (SCA)
    Paper: Mirjalili, S., 2016. Knowledge-based systems, 96, pp.120-133.
    """
    
    a = kwargs.get('a', 2.0)
    start_time = time.time()
    
    lb, ub = handle_bounds(lb, ub, dim)
    positions = initialize_population(pop_size, dim, lb, ub)
    
    convergence_curve = np.zeros(max_iter)
    best_score = float('inf')
    best_position = np.zeros(dim)
    
    for t in range(max_iter):
        positions = boundary_check(positions, lb, ub)
        fitness = evaluate_population(positions, obj_func)
        
        best_idx = np.argmin(fitness)
        if fitness[best_idx] < best_score:
            best_score = fitness[best_idx]
            best_position = positions[best_idx].copy()
        
        # SCA update equations
        r1 = a - t * (a / max_iter)
        
        for i in range(pop_size):
            for j in range(dim):
                r2 = 2 * np.pi * np.random.rand()
                r3 = 2 * np.random.rand()
                r4 = np.random.rand()
                
                if r4 < 0.5:
                    positions[i, j] = positions[i, j] + r1 * np.sin(r2) * abs(r3 * best_position[j] - positions[i, j])
        
        convergence_curve[t] = best_score
        
        if (t + 1) % 100 == 0:
            print(f"SCA - Iteration {t+1}/{max_iter}, Best Score: {best_score:.6f}")
    
    execution_time = time.time() - start_time
    return create_result_dict(best_score, best_position, convergence_curve, 
                             execution_time, 'SCA')

if __name__ == '__main__':
    def sphere_function(x):
        return np.sum(x**2)

    POP_SIZE = 30
    MAX_ITER = 500
    DIMENSIONS = 10
    LOWER_BOUND = -100
    UPPER_BOUND = 100

    print("Testing SCA Algorithm...")
    print(f"Problem: {DIMENSIONS}D Sphere Function")
    print(f"Population Size: {POP_SIZE}")
    print(f"Max Iterations: {MAX_ITER}")
    print(f"Search Space: [{LOWER_BOUND}, {UPPER_BOUND}]^{DIMENSIONS}")
    print("-" * 50)

    result = SCA(
        pop_size=POP_SIZE,
        max_iter=MAX_ITER,
        lb=LOWER_BOUND,
        ub=UPPER_BOUND,
        dim=DIMENSIONS,
        obj_func=sphere_function,
        a=2.0
    )

    print("\n" + "="*50)
    print("OPTIMIZATION RESULTS")
    print("="*50)
    print(f"Algorithm: {result['algorithm_name']}")
    print(f"Best Score Found: {result['best_score']:.6e}")
    print(f"Execution Time: {result['execution_time']:.2f} seconds")
    print(f"Best Position (first 5 dims): {result['best_position'][:5]}")
    print(f"Final Convergence: {result['convergence_curve'][-10:]}")
