import sys
import os
import importlib.util
import inspect
from pathlib import Path
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from objective_functions.benchmark_functions import BENCHMARK_FUNCTIONS, FUNCTION_BOUNDS

class MHAToolbox:
    """Metaheuristic Algorithm Toolbox - Unified interface for optimization algorithms"""
    
    def __init__(self):
        self.algorithms = {}
        self.load_algorithms()
        
    def load_algorithms(self):
        """Load all algorithms from toolbox_algorithms folder"""
        toolbox_path = Path(__file__).parent / "toolbox_algorithms"
        
        if not toolbox_path.exists():
            print(f"Warning: toolbox_algorithms folder not found")
            return
        
        print("Loading algorithms...")
        
        for py_file in toolbox_path.glob("*.py"):
            if py_file.name.startswith("__"):
                continue
                
            module_name = py_file.stem
            
            try:
                spec = importlib.util.spec_from_file_location(module_name, py_file)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                if hasattr(module, module_name):
                    func = getattr(module, module_name)
                    if callable(func):
                        self.algorithms[module_name] = func
                        print(f"✓ Loaded algorithm: {module_name}")
                        
            except Exception as e:
                print(f"✗ Failed to load {module_name}: {e}")
        
        print(f"Total algorithms loaded: {len(self.algorithms)}")
    
    def list_algorithms(self):
        """List all available algorithms"""
        return list(self.algorithms.keys())
    
    def list_benchmark_functions(self):
        """List all available benchmark functions"""
        return list(BENCHMARK_FUNCTIONS.keys())
    
    def get_algorithm_info(self, algorithm_name):
        """Get information about a specific algorithm"""
        if algorithm_name not in self.algorithms:
            return f"Algorithm '{algorithm_name}' not found."
        
        func = self.algorithms[algorithm_name]
        
        # Get function signature
        sig = inspect.signature(func)
        
        # Get docstring
        docstring = func.__doc__ or "No documentation available."
        
        return {
            'name': algorithm_name,
            'signature': str(sig),
            'docstring': docstring,
            'parameters': list(sig.parameters.keys())
        }
    
    def optimize(self, algorithm_name, objective_function, pop_size=30, max_iter=500, 
                 lb=-100, ub=100, dim=30, **algorithm_params):
        """Main optimization function"""
        
        if algorithm_name not in self.algorithms:
            available = ", ".join(self.list_algorithms())
            raise ValueError(f"Algorithm '{algorithm_name}' not found. Available: {available}")
        
        if isinstance(objective_function, str):
            if objective_function not in BENCHMARK_FUNCTIONS:
                available = ", ".join(self.list_benchmark_functions())
                raise ValueError(f"Function '{objective_function}' not found. Available: {available}")
            obj_func = BENCHMARK_FUNCTIONS[objective_function]
            
            if isinstance(lb, (int, float)) and isinstance(ub, (int, float)):
                if lb == -100 and ub == 100:
                    recommended_bounds = FUNCTION_BOUNDS[objective_function]
                    lb, ub = recommended_bounds
                    print(f"Using recommended bounds for {objective_function}: [{lb}, {ub}]")
                    
        elif callable(objective_function):
            obj_func = objective_function
        else:
            raise ValueError("objective_function must be string or callable")
        
        algorithm_func = self.algorithms[algorithm_name]
        
        print(f"Running {algorithm_name} optimization...")
        print(f"Parameters: pop_size={pop_size}, max_iter={max_iter}, dim={dim}")
        print(f"Bounds: lb={lb}, ub={ub}")
        if algorithm_params:
            print(f"Algorithm-specific params: {algorithm_params}")
        print("-" * 60)
        
        result = algorithm_func(
            pop_size=pop_size,
            max_iter=max_iter,
            lb=lb,
            ub=ub,
            dim=dim,
            obj_func=obj_func,
            **algorithm_params
        )
        
        print(f"✓ Optimization completed. Best score: {result['best_score']:.6e}")
        return result
    
    def compare_algorithms(self, algorithm_names, objective_function, runs=5, **params):
        """Compare multiple algorithms on the same problem"""
        if not algorithm_names:
            raise ValueError("Must provide at least one algorithm name")
        
        results = []
        
        for alg_name in algorithm_names:
            if alg_name not in self.algorithms:
                print(f"Skipping {alg_name} - not found")
                continue
                
            run_scores = []
            run_times = []
            
            for run in range(runs):
                try:
                    result = self.optimize(
                        algorithm_name=alg_name,
                        objective_function=objective_function,
                        **params
                    )
                    run_scores.append(result['best_score'])
                    run_times.append(result['execution_time'])
                    
                except Exception as e:
                    print(f"Run {run+1} failed: {e}")
                    continue
            
            if run_scores:
                results.append({
                    'algorithm': alg_name,
                    'mean_score': np.mean(run_scores),
                    'std_score': np.std(run_scores),
                    'best_score': np.min(run_scores),
                    'worst_score': np.max(run_scores),
                    'mean_time': np.mean(run_times),
                    'successful_runs': len(run_scores),
                    'total_runs': runs
                })
        return results
    
    def print_comparison_table(self, comparison_results):
        """Print formatted comparison table"""
        if not comparison_results:
            print("No results to display")
            return
        
        print("\n" + "="*100)
        print("ALGORITHM COMPARISON RESULTS")
        print("="*100)
        print(f"{'Algorithm':<15} {'Best Score':<15} {'Mean Score':<15} {'Std Dev':<15} {'Mean Time':<15} {'Success Rate':<15}")
        print("-" * 100)
        
        for result in comparison_results:
            success_rate = result['successful_runs'] / result['total_runs'] * 100
            print(f"{result['algorithm']:<15} "
                  f"{result['best_score']:<15.2e} "
                  f"{result['mean_score']:<15.2e} "
                  f"{result['std_score']:<15.2e} "
                  f"{result['mean_time']:<15.2f} "
                  f"{success_rate:<15.1f}%")

if __name__ == "__main__":
    toolbox = MHAToolbox()
    
    if 'SCA' in toolbox.list_algorithms():
        result = toolbox.optimize(
            algorithm_name='SCA',
            objective_function='sphere',
            pop_size=30,
            max_iter=200,
            dim=10,
            a=2.0
        )
        print(f"Best Score: {result['best_score']:.6e}")
        print(f"Execution Time: {result['execution_time']:.2f}s")
