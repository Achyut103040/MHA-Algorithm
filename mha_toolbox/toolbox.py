import importlib
import os
import inspect
import numpy as np
from mha_toolbox.base import BaseOptimizer

class MHAToolbox:
    """
    Main interface for the Metaheuristic Algorithm Toolbox.
    
    This class provides a centralized access point to all available optimization
    algorithms in the toolbox. Users can easily get any algorithm, see what's 
    available, and run optimizations with minimal configuration.
    
    The toolbox follows the TensorFlow-style design where users import the library
    and call specific functions as needed, rather than writing complete programs
    from scratch for each algorithm.
    
    Examples
    --------
    >>> from mha_toolbox import MHAToolbox
    >>> mha = MHAToolbox()
    >>> # Get a list of all available algorithms
    >>> print(mha.list_algorithms())
    >>> # Run Sine Cosine Algorithm with default parameters
    >>> X = np.random.rand(100, 5)  # Example data
    >>> sca = mha.get_optimizer('SineCosinAlgorithm')
    >>> result = sca.optimize(X, objective_function=lambda x: np.sum(x**2))
    >>> print(f"Best fitness: {result.best_fitness}")
    >>> # Plot convergence curve
    >>> result.plot_convergence()
    """
    
    def __init__(self):
        """Initialize the MHAToolbox and discover available algorithms."""
        self.algorithms = {}
        self._discover_algorithms()
    
    def _discover_algorithms(self):
        """Dynamically discover all algorithm classes in the toolbox."""
        # Manually register known algorithms first
        try:
            from mha_toolbox.algorithms.sca import SineCosinAlgorithm
            self.algorithms['SineCosinAlgorithm'] = SineCosinAlgorithm
            self.algorithms['SCA'] = SineCosinAlgorithm  # Add alias
        except ImportError as e:
            print(f"Warning: Could not import SineCosinAlgorithm: {e}")
            
        try:
            from mha_toolbox.algorithms.pso import ParticleSwarmOptimization
            self.algorithms['ParticleSwarmOptimization'] = ParticleSwarmOptimization
            self.algorithms['PSO'] = ParticleSwarmOptimization  # Add alias
        except ImportError as e:
            print(f"Warning: Could not import ParticleSwarmOptimization: {e}")
            
        try:
            from mha_toolbox.algorithms.gwo import GreyWolfOptimizer
            self.algorithms['GreyWolfOptimizer'] = GreyWolfOptimizer
            self.algorithms['GWO'] = GreyWolfOptimizer  # Add alias
        except ImportError as e:
            print(f"Warning: Could not import GreyWolfOptimizer: {e}")
        
        # Try to discover other algorithms automatically
        try:
            # Get the path to the algorithms directory
            import mha_toolbox.algorithms
            algorithms_dir = os.path.dirname(mha_toolbox.algorithms.__file__)
            
            # Scan for other algorithm modules
            for filename in os.listdir(algorithms_dir):
                if filename.endswith('.py') and not filename.startswith('__') and filename != 'sca.py':
                    module_name = f"mha_toolbox.algorithms.{filename[:-3]}"
                    try:
                        # Import the module
                        module = importlib.import_module(module_name)
                        
                        # Find all classes that inherit from BaseOptimizer
                        for name, obj in inspect.getmembers(module):
                            if (inspect.isclass(obj) and issubclass(obj, BaseOptimizer) 
                                and obj is not BaseOptimizer):
                                self.algorithms[name] = obj
                                
                                # Add common abbreviation as alias
                                abbr = ''.join([c for c in name if c.isupper()])
                                if abbr and abbr != name and abbr not in self.algorithms:
                                    self.algorithms[abbr] = obj
                    except ImportError:
                        # Skip modules that can't be imported
                        pass
        except Exception as e:
            print(f"Warning: Error during algorithm discovery: {e}")
    
    def get_optimizer(self, name, *args, **kwargs):
        """
        Get an instance of the specified optimizer.
        
        Supports flexible argument parsing:
        - get_optimizer('SCA', 15, 100) -> population_size=15, max_iterations=100
        - get_optimizer('SCA', population_size=15, max_iterations=100)
        
        Parameters
        ----------
        name : str
            Name of the optimizer to instantiate
        *args : tuple
            Positional arguments: population_size, max_iterations, dimensions
        **kwargs : dict
            Parameters to pass to the optimizer constructor
            
        Returns
        -------
        BaseOptimizer
            Instance of the requested optimizer
            
        Raises
        ------
        ValueError
            If the requested optimizer is not found
        """
        if name not in self.algorithms:
            available = ', '.join(sorted(self.algorithms.keys()))
            raise ValueError(f"Algorithm '{name}' not found. Available algorithms: {available}")
        
        # Instantiate the optimizer with the provided parameters
        return self.algorithms[name](*args, **kwargs)
    
    def list_algorithms(self):
        """
        List all available optimization algorithms.
        
        Returns
        -------
        list
            Sorted list of algorithm names
        """
        # Return only unique algorithm classes (not aliases)
        unique_algorithms = set(self.algorithms.values())
        return sorted([cls.__name__ for cls in unique_algorithms])
    
    def get_algorithm_info(self, name):
        """
        Get detailed information about an algorithm.
        
        Parameters
        ----------
        name : str
            Name of the algorithm
            
        Returns
        -------
        dict
            Dictionary containing algorithm information
            
        Raises
        ------
        ValueError
            If the requested algorithm is not found
        """
        if name not in self.algorithms:
            available = ', '.join(sorted(self.algorithms.keys()))
            raise ValueError(f"Algorithm '{name}' not found. Available algorithms: {available}")
        
        algorithm_class = self.algorithms[name]
        
        return {
            'name': algorithm_class.__name__,
            'description': algorithm_class.__doc__,
            'parameters': inspect.signature(algorithm_class.__init__)
        }
    
    def run_optimizer(self, algorithm_name, X=None, y=None, objective_function=None, **kwargs):
        """
        Run an optimization algorithm directly with given parameters.
        
        This is a convenience method that gets the optimizer and runs it in one step.
        
        Parameters
        ----------
        algorithm_name : str
            Name of the algorithm to use
        X : numpy.ndarray, optional
            Input data (features) - this is the first positional argument as required
        y : numpy.ndarray, optional
            Target values (for supervised problems like feature selection)
        objective_function : callable, optional
            Function to optimize (required if X and y are not provided)
        **kwargs : dict
            Additional parameters to pass to the optimizer
            
        Returns
        -------
        OptimizationModel
            Model containing all results and parameters
        """
        optimizer = self.get_optimizer(algorithm_name, **kwargs)
        return optimizer.optimize(X=X, y=y, objective_function=objective_function)
    
    def compare_algorithms(self, algorithm_names, X=None, y=None, objective_function=None, 
                          plot_comparison=True, **kwargs):
        """
        Run multiple algorithms and compare their performance.
        
        Parameters
        ----------
        algorithm_names : list
            List of algorithm names to compare
        X : numpy.ndarray, optional
            Input data (features)
        y : numpy.ndarray, optional
            Target values (for supervised problems)
        objective_function : callable, optional
            Function to optimize (required if X and y are not provided)
        plot_comparison : bool, optional
            Whether to plot comparison of convergence curves, default=True
        **kwargs : dict
            Additional parameters to pass to all optimizers
            
        Returns
        -------
        dict
            Dictionary mapping algorithm names to their respective models
        """
        results = {}
        
        print("Running algorithm comparison...")
        print("=" * 50)
        
        for name in algorithm_names:
            print(f"Running {name}...")
            try:
                optimizer = self.get_optimizer(name, **kwargs)
                result = optimizer.optimize(X=X, y=y, objective_function=objective_function)
                results[name] = result
                print(f"{name} - Best Fitness: {result.best_fitness:.6f}")
                print(f"{name} - Execution Time: {result.execution_time:.4f} seconds")
                print("-" * 30)
            except Exception as e:
                print(f"Error running {name}: {str(e)}")
                print("-" * 30)
        
        # Plot comparison if requested
        if plot_comparison and len(results) > 1:
            self._plot_algorithm_comparison(results)
        
        return results
    
    def _plot_algorithm_comparison(self, results):
        """
        Plot comparison of multiple algorithm results.
        
        Parameters
        ----------
        results : dict
            Dictionary mapping algorithm names to their models
        """
        try:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(12, 8))
            
            for name, result in results.items():
                if result.convergence_curve is not None:
                    plt.plot(result.convergence_curve, label=f"{name} (Best: {result.best_fitness:.6f})")
            
            plt.title("Algorithm Performance Comparison")
            plt.xlabel("Iteration")
            plt.ylabel("Best Fitness")
            plt.legend()
            plt.grid(True)
            plt.show()
            
        except ImportError:
            print("Matplotlib not available. Skipping comparison plot.")
        except Exception as e:
            print(f"Error creating comparison plot: {str(e)}")


# Create a singleton instance for easy import
_toolbox_instance = None

def get_toolbox():
    """
    Get the singleton MHAToolbox instance.
    
    Returns
    -------
    MHAToolbox
        The singleton toolbox instance
    """
    global _toolbox_instance
    if _toolbox_instance is None:
        _toolbox_instance = MHAToolbox()
    return _toolbox_instance

# Convenience functions for direct access
def get_optimizer(name, **kwargs):
    """Get an optimizer instance by name"""
    return get_toolbox().get_optimizer(name, **kwargs)

def list_algorithms():
    """List all available algorithms"""
    return get_toolbox().list_algorithms()

def get_algorithm_info(name):
    """Get detailed information about an algorithm"""
    return get_toolbox().get_algorithm_info(name)

def run_optimizer(algorithm_name, X=None, y=None, objective_function=None, **kwargs):
    """Run an optimization algorithm directly"""
    return get_toolbox().run_optimizer(algorithm_name, X=X, y=y, 
                                      objective_function=objective_function, **kwargs)

def compare_algorithms(algorithm_names, X=None, y=None, objective_function=None, **kwargs):
    """Compare multiple algorithms"""
    return get_toolbox().compare_algorithms(algorithm_names, X=X, y=y, 
                                           objective_function=objective_function, **kwargs)
