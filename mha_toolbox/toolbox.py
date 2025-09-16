import importlib
import os
import inspect
import numpy as np
import time
from mha_toolbox.base import BaseOptimizer, OptimizationModel
from mha_toolbox.utils import plot_comparison, create_problem

class MHAToolbox:
    """
    Main interface for the Metaheuristic Algorithm Toolbox.
    
    This class provides a centralized access point to all available optimization
    algorithms in the toolbox. Users can easily get any algorithm, see what's 
    available, and run optimizations with minimal configuration.
    
    The toolbox follows the TensorFlow-style design where users import the library
    and call specific functions as needed, rather than writing complete programs
    from scratch for each algorithm.
    
    🚀 Key Features:
    - Automatic algorithm discovery and registration
    - Intelligent parameter defaults based on problem type
    - Support for both function optimization and feature selection
    - One-line optimization with comprehensive results
    - Algorithm comparison and analysis tools
    
    """
    
    def __init__(self):
        """Initialize the MHAToolbox and discover available algorithms."""
        if hasattr(self, '_initialized'):
            return
        self.algorithms = {}
        self.algorithm_aliases = {}
        self._discover_algorithms()
        self._create_aliases()
        self._initialized = True
    
    def _discover_algorithms(self):
        """Dynamically discover all algorithm classes in the toolbox."""
        
        if hasattr(self, '_discovery_done'):
            return
            
        print("🔍 Discovering available algorithms...")
        
        # Manually register known algorithms first
        try:
            from mha_toolbox.algorithms.sca import SineCosinAlgorithm
            self.algorithms['SineCosinAlgorithm'] = SineCosinAlgorithm
            print("  ✓ SineCosinAlgorithm")
        except ImportError as e:
            print(f"  ⚠ Could not import SineCosinAlgorithm: {e}")
            
        try:
            from mha_toolbox.algorithms.pso import ParticleSwarmOptimization
            self.algorithms['ParticleSwarmOptimization'] = ParticleSwarmOptimization
            print("  ✓ ParticleSwarmOptimization")
        except ImportError as e:
            print(f"  ⚠ Could not import ParticleSwarmOptimization: {e}")
            
        try:
            from mha_toolbox.algorithms.gwo import GreyWolfOptimizer
            self.algorithms['GreyWolfOptimizer'] = GreyWolfOptimizer
            print("  ✓ GreyWolfOptimizer")
        except ImportError as e:
            print(f"  ⚠ Could not import GreyWolfOptimizer: {e}")
        
        # Try to discover other algorithms automatically
        try:
            # Get the path to the algorithms directory
            import mha_toolbox.algorithms
            algorithms_dir = os.path.dirname(mha_toolbox.algorithms.__file__)
            
            # Scan for other algorithm modules
            discovered_count = 0
            for filename in os.listdir(algorithms_dir):
                if filename.endswith('.py') and not filename.startswith('__'):
                    module_name = f"mha_toolbox.algorithms.{filename[:-3]}"
                    try:
                        # Import the module
                        module = importlib.import_module(module_name)
                        
                        # Find all classes that inherit from BaseOptimizer
                        for name, obj in inspect.getmembers(module):
                            if (inspect.isclass(obj) and issubclass(obj, BaseOptimizer) 
                                and obj is not BaseOptimizer and name not in self.algorithms):
                                self.algorithms[name] = obj
                                discovered_count += 1
                                print(f"  ✓ {name}")
                                
                    except ImportError as e:
                        print(f"  ⚠ Could not import {module_name}: {e}")
            
            print(f"📊 Total algorithms discovered: {len(self.algorithms)}")
                        
        except Exception as e:
            print(f"⚠ Error during algorithm discovery: {e}")
        
        self._discovery_done = True
    
    def _create_aliases(self):
        """Create user-friendly aliases for algorithms."""
        
        # Clear existing aliases
        self.algorithm_aliases = {}
        
        # Create dynamic aliases from algorithm class aliases attribute
        for alg_name, alg_class in self.algorithms.items():
            if hasattr(alg_class, 'aliases'):
                for alias in alg_class.aliases:
                    self.algorithm_aliases[alias.lower()] = alg_name
        
        # Common abbreviations and alternative names (fallback)
        alias_map = {
            'pso': 'ParticleSwarmOptimization',
            'particle_swarm': 'ParticleSwarmOptimization',
            'sca': 'SineCosinAlgorithm', 
            'sine_cosine': 'SineCosinAlgorithm',
            'gwo': 'GreyWolfOptimizer',
            'grey_wolf': 'GreyWolfOptimizer',
            'woa': 'WhaleOptimizationAlgorithm',
            'whale': 'WhaleOptimizationAlgorithm',
            'ga': 'GeneticAlgorithm',
            'genetic': 'GeneticAlgorithm',
            'de': 'DifferentialEvolution',
            'differential': 'DifferentialEvolution',
            'abc': 'ArtificialBeeColony',
            'bee': 'ArtificialBeeColony',
            'aco': 'AntColonyOptimization',
            'ant': 'AntColonyOptimization',
            'ba': 'BatAlgorithm',
            'bat': 'BatAlgorithm',
            'fa': 'FireflyAlgorithm',
            'firefly': 'FireflyAlgorithm',
            'ao': 'AquilaOptimizer',
            'aquila': 'AquilaOptimizer'
        }
        
        # Create aliases for existing algorithms (only if not already set)
        for alias, full_name in alias_map.items():
            if full_name in self.algorithms and alias.lower() not in self.algorithm_aliases:
                self.algorithm_aliases[alias.lower()] = full_name
                
        # Create automatic abbreviations
        for alg_name in self.algorithms.keys():
            # Create abbreviation from capital letters
            abbr = ''.join([c for c in alg_name if c.isupper()]).lower()
            if abbr and abbr not in self.algorithm_aliases:
                self.algorithm_aliases[abbr] = alg_name
                
        print(f"📝 Created {len(self.algorithm_aliases)} algorithm aliases")
    
    def _resolve_algorithm_name(self, name):
        """Resolve algorithm name from alias if needed."""
        # Check direct name first
        if name in self.algorithms:
            return name
        # Check aliases
        if name.lower() in self.algorithm_aliases:
            return self.algorithm_aliases[name.lower()]
        # Check case-insensitive match
        for alg_name in self.algorithms.keys():
            if alg_name.lower() == name.lower():
                return alg_name
        return None
    
    def get_optimizer(self, name, **kwargs):
        """
        Get an instance of the specified optimizer with intelligent defaults.
        
        This method automatically sets appropriate default parameters based on
        the problem type and algorithm characteristics.
        
        Parameters
        ----------
        name : str
            Name of the optimizer to instantiate (supports aliases)
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
        
        # Resolve algorithm name
        resolved_name = self._resolve_algorithm_name(name)
        if resolved_name is None:
            available = self._get_available_names()
            raise ValueError(f"Algorithm '{name}' not found. Available: {', '.join(available[:10])}...")
        
        # Get algorithm class
        algorithm_class = self.algorithms[resolved_name]
        
        # Set intelligent defaults if not provided
        defaults = self._get_intelligent_defaults(resolved_name, **kwargs)
        
        # Merge user parameters with defaults
        final_params = {**defaults, **kwargs}
        
        # Filter parameters to only those accepted by the algorithm
        filtered_params = self._filter_parameters(algorithm_class, final_params)
        
        try:
            return algorithm_class(**filtered_params)
        except Exception as e:
            print(f"⚠ Error creating {resolved_name}: {e}")
            print(f"  Available parameters: {list(inspect.signature(algorithm_class.__init__).parameters.keys())}")
            # Try with minimal parameters
            minimal_params = {k: v for k, v in filtered_params.items() 
                            if k in ['population_size', 'max_iterations']}
            return algorithm_class(**minimal_params)
    
    def _get_intelligent_defaults(self, algorithm_name, **kwargs):
        """Get intelligent default parameters based on algorithm and problem."""
        
        # Base defaults
        defaults = {
            'population_size': 30,
            'max_iterations': 100
        }
        
        # Algorithm-specific defaults
        if 'Particle' in algorithm_name:
            defaults.update({
                'c1': 2.0, 'c2': 2.0, 'w': 0.9, 'w_min': 0.4, 'w_max': 0.9
            })
        elif 'SineCosine' in algorithm_name or 'SCA' in algorithm_name:
            defaults.update({
                'a': 2.0, 'r1_min': 0, 'r1_max': 2
            })
        elif 'GreyWolf' in algorithm_name or 'GWO' in algorithm_name:
            defaults.update({
                'a_linearly_decrease': True
            })
        
        # Adjust based on problem hints
        dimensions = kwargs.get('dimensions', 10)
        if dimensions > 50:
            defaults['population_size'] = min(50, dimensions)
            defaults['max_iterations'] = max(200, dimensions * 2)
        elif dimensions > 20:
            defaults['population_size'] = 40
            defaults['max_iterations'] = 150
        
        return defaults
    
    def _filter_parameters(self, algorithm_class, params):
        """Filter parameters to only those accepted by the algorithm."""
        try:
            sig = inspect.signature(algorithm_class.__init__)
            valid_params = set(sig.parameters.keys()) - {'self'}
            
            filtered = {}
            for key, value in params.items():
                if key in valid_params:
                    filtered[key] = value
                    
            return filtered
        except Exception:
            # If inspection fails, return original params
            return params
    
    def _get_available_names(self):
        """Get all available algorithm names including aliases."""
        names = list(self.algorithms.keys())
        names.extend(self.algorithm_aliases.keys())
        return sorted(names)
    
    def list_algorithms(self):
        """
        List all available optimization algorithms with their aliases.
        
        Returns
        -------
        dict
            Dictionary mapping categories to algorithm lists
        """
        
        # Categorize algorithms
        categories = {
            'Swarm Intelligence': [],
            'Physics Based': [],
            'Bio Inspired': [],
            'Human Based': [],
            'Math Based': [],
            'Other': []
        }
        
        for alg_name, alg_class in self.algorithms.items():
            # Simple categorization based on name patterns
            name_lower = alg_name.lower()
            if any(word in name_lower for word in ['swarm', 'particle', 'bee', 'ant', 'whale', 'wolf', 'bat']):
                categories['Swarm Intelligence'].append(alg_name)
            elif any(word in name_lower for word in ['sine', 'gravity', 'physics', 'electromagnetic']):
                categories['Physics Based'].append(alg_name)
            elif any(word in name_lower for word in ['genetic', 'evolution', 'bio', 'dna']):
                categories['Bio Inspired'].append(alg_name)
            elif any(word in name_lower for word in ['human', 'social', 'cultural']):
                categories['Human Based'].append(alg_name)
            elif any(word in name_lower for word in ['math', 'optimization', 'search']):
                categories['Math Based'].append(alg_name)
            else:
                categories['Other'].append(alg_name)
        
        # Remove empty categories and sort
        result = {}
        for category, algorithms in categories.items():
            if algorithms:
                result[category] = sorted(algorithms)
        
        return result
    
    def get_all_algorithm_names(self):
        """Get all algorithm names including aliases."""
        all_names = list(self.algorithms.keys())
        all_names.extend(self.algorithm_aliases.keys())
        return sorted(set(all_names))
    
    def get_algorithm_info(self, name):
        """
        Get detailed information about an algorithm.
        
        Parameters
        ----------
        name : str
            Name of the algorithm (supports aliases)
            
        Returns
        -------
        dict
            Dictionary containing algorithm information
        """
        
        resolved_name = self._resolve_algorithm_name(name)
        if resolved_name is None:
            available = self._get_available_names()
            raise ValueError(f"Algorithm '{name}' not found. Available: {', '.join(available[:10])}...")
        
        algorithm_class = self.algorithms[resolved_name]
        
        # Get parameter information
        try:
            sig = inspect.signature(algorithm_class.__init__)
            params = {}
            for param_name, param in sig.parameters.items():
                if param_name != 'self':
                    params[param_name] = {
                        'default': param.default if param.default != param.empty else 'Required',
                        'annotation': param.annotation if param.annotation != param.empty else 'Any'
                    }
        except Exception:
            params = {'Error': 'Could not extract parameter information'}
        
        return {
            'name': algorithm_class.__name__,
            'description': algorithm_class.__doc__ or 'No description available',
            'parameters': params,
            'aliases': [alias for alias, full_name in self.algorithm_aliases.items() 
                       if full_name == resolved_name]
        }
    
    def optimize(self, algorithm_name, X=None, y=None, objective_function=None, **kwargs):
        """
        Main optimization function - the heart of the library.
        
        This method provides the TensorFlow-style interface that users love.
        It automatically handles problem detection, parameter defaults, and
        result packaging.
        
        Parameters
        ----------
        algorithm_name : str
            Name of the algorithm to use (supports aliases like 'pso', 'gwo')
        X : numpy.ndarray, optional
            Input data (features) - prioritized as first argument
        y : numpy.ndarray, optional
            Target values (for feature selection problems)
        objective_function : callable, optional
            Function to optimize (required if X and y are not provided)
        **kwargs : dict
            Additional parameters (all optional with intelligent defaults)
            
        Returns
        -------
        OptimizationModel
            Comprehensive result object
            
        Examples
        --------
        >>> # Simple function optimization
        >>> result = toolbox.optimize('pso', objective_function=lambda x: sum(x**2), dimensions=10)
        >>> 
        >>> # Feature selection (data first!)
        >>> result = toolbox.optimize('gwo', X, y)
        >>>
        >>> # Custom parameters
        >>> result = toolbox.optimize('sca', X, y, population_size=50, max_iterations=200)
        """
        
        print(f"🚀 Starting optimization with {algorithm_name}...")
        
        # Create problem definition
        if X is not None and y is not None:
            print(f"📊 Detected feature selection problem: {X.shape[0]} samples, {X.shape[1]} features")
            problem = create_problem(X=X, y=y, problem_type='feature_selection')
            kwargs.setdefault('dimensions', X.shape[1])
        elif objective_function is not None:
            dimensions = kwargs.get('dimensions', 10)
            print(f"🎯 Detected function optimization problem: {dimensions} dimensions")
            problem = create_problem(objective_function=objective_function, 
                                   dimensions=dimensions, 
                                   problem_type='function')
        else:
            raise ValueError("Must provide either (X, y) for feature selection or objective_function for optimization")
        
        # Get optimizer with intelligent defaults
        optimizer = self.get_optimizer(algorithm_name, **kwargs)
        
        # Run optimization
        start_time = time.time()
        result = optimizer.optimize(problem)
        execution_time = time.time() - start_time
        
        # Package results
        if hasattr(result, 'execution_time'):
            result.execution_time = execution_time
        
        print(f"✅ Optimization completed in {execution_time:.3f} seconds")
        print(f"🏆 Best fitness: {result.best_fitness:.6f}")
        
        return result
    
    def compare_algorithms(self, algorithm_names, X=None, y=None, objective_function=None, 
                          n_runs=1, plot_results=True, **kwargs):
        """
        Compare multiple algorithms on the same problem.
        
        This method runs multiple algorithms and provides comprehensive comparison
        including statistical analysis and visualization.
        
        Parameters
        ----------
        algorithm_names : list
            List of algorithm names to compare
        X, y : array-like, optional
            Data for feature selection problems
        objective_function : callable, optional
            Function for optimization problems
        n_runs : int, default=1
            Number of independent runs for statistical analysis
        plot_results : bool, default=True
            Whether to plot comparison results
        **kwargs : dict
            Additional parameters passed to all algorithms
            
        Returns
        -------
        dict
            Dictionary mapping algorithm names to their results
        """
        
        print(f"🔬 Comparing {len(algorithm_names)} algorithms...")
        print(f"📊 Running {n_runs} independent runs each...")
        
        results = {}
        
        for alg_name in algorithm_names:
            print(f"\n  Running {alg_name}...")
            
            if n_runs == 1:
                # Single run
                result = self.optimize(alg_name, X=X, y=y, 
                                     objective_function=objective_function, **kwargs)
                results[alg_name] = result
            else:
                # Multiple runs for statistics
                run_results = []
                for run in range(n_runs):
                    print(f"    Run {run+1}/{n_runs}...")
                    result = self.optimize(alg_name, X=X, y=y, 
                                         objective_function=objective_function, **kwargs)
                    run_results.append(result)
                
                # Aggregate results
                best_result = min(run_results, key=lambda r: r.best_fitness)
                best_result.statistics = self._calculate_statistics(run_results)
                results[alg_name] = best_result
        
        print(f"\n🏁 Comparison completed!")
        
        # Plot results if requested
        if plot_results:
            try:
                plot_comparison(results, title=f"Algorithm Comparison ({n_runs} runs)")
            except Exception as e:
                print(f"⚠ Could not plot results: {e}")
        
        return results
    
    def _calculate_statistics(self, results):
        """Calculate statistical measures for multiple runs."""
        fitness_values = [r.best_fitness for r in results]
        execution_times = [getattr(r, 'execution_time', 0) for r in results]
        
        return {
            'fitness_mean': np.mean(fitness_values),
            'fitness_std': np.std(fitness_values),
            'fitness_min': np.min(fitness_values),
            'fitness_max': np.max(fitness_values),
            'time_mean': np.mean(execution_times),
            'time_std': np.std(execution_times),
            'success_rate': sum(1 for f in fitness_values if f < 1e-6) / len(fitness_values)
        }
    
    def quick_start(self, problem_type='function', **kwargs):
        """
        Quick start guide for beginners.
        
        This method provides an interactive way to get started with the library.
        
        Parameters
        ----------
        problem_type : str, default='function'
            Type of problem: 'function', 'feature_selection', or 'demo'
        **kwargs
            Additional parameters
            
        Returns
        -------
        OptimizationModel
            Result from the quick start optimization
        """
        
        print("🌟 Welcome to MHA Toolbox Quick Start!")
        print("=" * 50)
        
        if problem_type == 'demo':
            print("🎯 Running demo with sphere function...")
            result = self.optimize('pso', 
                                 objective_function=lambda x: np.sum(x**2),
                                 dimensions=10,
                                 verbose=True)
            
        elif problem_type == 'function':
            print("🎯 Setting up function optimization...")
            print("   Using sphere function: f(x) = sum(x²)")
            result = self.optimize('pso',
                                 objective_function=lambda x: np.sum(x**2),
                                 dimensions=kwargs.get('dimensions', 10))
            
        elif problem_type == 'feature_selection':
            print("📊 Setting up feature selection...")
            print("   Loading sample dataset...")
            from mha_toolbox.utils import load_dataset
            X, y = load_dataset('breast_cancer')
            result = self.optimize('gwo', X, y)
            
        print("\n🎉 Quick start completed!")
        print("💡 Tip: Try result.plot_convergence() to see the optimization progress")
        print("💡 Tip: Try result.summary() for detailed information")
        
        return result

# Global functions for TensorFlow-style interface
_global_toolbox = None

def get_toolbox():
    """Get or create the global toolbox instance."""
    global _global_toolbox
    if _global_toolbox is None:
        _global_toolbox = MHAToolbox()
    return _global_toolbox

def get_optimizer(name, **kwargs):
    """Global function to get an optimizer."""
    return get_toolbox().get_optimizer(name, **kwargs)

def list_algorithms():
    """Global function to list algorithms.""" 
    return get_toolbox().list_algorithms()

def get_algorithm_info(name):
    """Global function to get algorithm info."""
    return get_toolbox().get_algorithm_info(name)

def run_optimizer(algorithm_name, X=None, y=None, objective_function=None, **kwargs):
    """Global function to run optimization."""
    return get_toolbox().optimize(algorithm_name, X=X, y=y, 
                                   objective_function=objective_function, **kwargs)

def compare_algorithms(algorithm_names, X=None, y=None, objective_function=None, **kwargs):
    """Global function to compare algorithms."""
    return get_toolbox().compare_algorithms(algorithm_names, X=X, y=y,
                                            objective_function=objective_function, **kwargs)
