import numpy as np
import time
import json
import os
from datetime import datetime
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt

class OptimizationModel:
    """
    A standardized object to store and manage optimization results.
    
    This class provides a consistent interface for accessing all information
    related to an optimization run, including the best solution, fitness,
    convergence history, and parameters used. It also includes methods for
    analysis and visualization.
    """
    
    def __init__(self, algorithm_name, best_solution, best_fitness, 
                 convergence_curve, execution_time, parameters,
                 problem_type='unknown', X_data=None, y_data=None):
        
        self.algorithm_name = algorithm_name
        self.best_solution = np.array(best_solution)
        self.best_fitness = best_fitness
        self.convergence_curve = convergence_curve
        self.execution_time = execution_time
        self.parameters = parameters
        self.problem_type = problem_type
        self.timestamp = datetime.now().isoformat()
        
        # For feature selection, store binary solution and selected features
        if self.problem_type == 'feature_selection':
            self.best_solution_binary = (self.best_solution > 0.5).astype(int)
            self.n_selected_features = sum(self.best_solution_binary)
            self.selected_feature_indices = np.where(self.best_solution_binary)[0]
        
        # Store a reference to the data if provided
        self._X_data = X_data
        self._y_data = y_data

    def summary(self):
        """
        Print a comprehensive summary of the optimization results.
        """
        print("\n" + "="*60)
        print(f"📈 Optimization Results Summary: {self.algorithm_name}")
        print("="*60)
        print(f"  - Timestamp: {self.timestamp}")
        print(f"  - Problem Type: {self.problem_type}")
        print(f"  - Execution Time: {self.execution_time:.4f} seconds")
        print(f"  - Best Fitness: {self.best_fitness:.6f}")
        
        if self.problem_type == 'feature_selection':
            print(f"  - Selected Features: {self.n_selected_features} / {len(self.best_solution)}")
            print(f"  - Selected Indices: {self.selected_feature_indices[:10]}...")
        else:
            print(f"  - Best Solution: {self.best_solution[:10]}...")
        
        print("\n" + "-"*60)
        print("⚙️  Parameters Used:")
        print("-"*60)
        for key, value in self.parameters.items():
            print(f"  - {key:<20}: {value}")
        print("="*60)

    def plot_convergence(self, title=None, save_path=None):
        """
        Plot the convergence curve of the optimization.
        
        Parameters
        ----------
        title : str, optional
            Title for the plot
        save_path : str, optional
            Path to save the plot image
        """
        plt.figure(figsize=(10, 6))
        plt.plot(self.convergence_curve, linewidth=2)
        plt.title(title or f'{self.algorithm_name} Convergence Curve')
        plt.xlabel('Iteration')
        plt.ylabel('Best Fitness')
        plt.grid(True, linestyle='--', alpha=0.6)
        
        if save_path:
            plt.savefig(save_path, dpi=300)
            print(f"Convergence plot saved to {save_path}")
        
        plt.show()

    def save(self, filename=None, format='json'):
        """
        Save the optimization results to a file.
        
        Parameters
        ----------
        filename : str, optional
            Filename to save results. If None, a default is generated.
        format : str
            'json' or 'txt'
        """
        if filename is None:
            # Determine the appropriate subdirectory based on problem type and algorithm
            if 'Hybrid' in self.algorithm_name:
                subdir = 'results/hybrid_algorithms'
            elif self.problem_type == 'function_optimization':
                subdir = 'results/function_optimization'
            else:
                subdir = 'results/single_algorithms'
            
            # Create directory if it doesn't exist
            import os
            os.makedirs(subdir, exist_ok=True)
            
            filename = f"{subdir}/result_{self.algorithm_name}_{datetime.now():%Y%m%d_%H%M%S}.{format}"
        
        data_to_save = {
            'algorithm_name': self.algorithm_name,
            'best_fitness': float(self.best_fitness),  # Ensure it's a Python float
            'best_solution': self.best_solution.tolist(),  # Convert numpy array to list
            'convergence_curve': [float(x) for x in self.convergence_curve],  # Ensure all are Python floats
            'execution_time': float(self.execution_time),
            'parameters': self._serialize_parameters(self.parameters),  # Handle numpy arrays in parameters
            'problem_type': self.problem_type,
            'timestamp': self.timestamp
        }
        
        if self.problem_type == 'feature_selection':
            data_to_save['n_selected_features'] = int(self.n_selected_features)
            data_to_save['selected_feature_indices'] = self.selected_feature_indices.tolist()

        try:
            with open(filename, 'w') as f:
                if format == 'json':
                    json.dump(data_to_save, f, indent=4)
                else:
                    f.write(json.dumps(data_to_save, indent=4))
            print(f"Results successfully saved to {filename}")
            
            # Also save convergence curve as CSV
            csv_filename = filename.replace('.json', '_convergence.csv').replace('.txt', '_convergence.csv')
            self._save_convergence_csv(csv_filename)
            
        except Exception as e:
            print(f"Error saving results: {e}")

    def _save_convergence_csv(self, filename):
        """Save convergence curve as CSV file."""
        try:
            import csv
            with open(filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['iteration', 'best_fitness'])
                for i, fitness in enumerate(self.convergence_curve):
                    writer.writerow([i+1, float(fitness)])
            print(f"Convergence curve saved to {filename}")
        except Exception as e:
            print(f"Error saving convergence curve: {e}")

    def _serialize_parameters(self, params):
        """Convert numpy arrays in parameters to lists for JSON serialization."""
        serialized = {}
        for key, value in params.items():
            if isinstance(value, np.ndarray):
                serialized[key] = value.tolist()
            elif isinstance(value, (np.integer, np.floating)):
                serialized[key] = value.item()  # Convert numpy scalar to Python scalar
            else:
                serialized[key] = value
        return serialized

    @classmethod
    def load(cls, filename):
        """
        Load optimization results from a file.
        """
        with open(filename, 'r') as f:
            data = json.load(f)
        
        # Recreate the model object
        model = cls(
            algorithm_name=data['algorithm_name'],
            best_solution=data['best_solution'],
            best_fitness=data['best_fitness'],
            convergence_curve=data['convergence_curve'],
            execution_time=data['execution_time'],
            parameters=data['parameters'],
            problem_type=data.get('problem_type', 'unknown')
        )
        return model

class BaseOptimizer(ABC):
    """
    Abstract base class for all metaheuristic optimizers.
    
    This class provides the core structure and functionality for all algorithms,
    including parameter initialization, data handling, and result formatting.
    
    Supports flexible initialization:
    - PSO(15, 100) -> population_size=15, max_iterations=100
    - PSO(population_size=30, max_iterations=200)
    """
    
    def __init__(self, *args, population_size=30, max_iterations=None, 
                 lower_bound=None, upper_bound=None, dimensions=None, 
                 verbose=True, **kwargs):
        
        # Handle flexible positional arguments: PSO(15, 100)
        if len(args) >= 1:
            population_size = args[0]
        if len(args) >= 2:
            max_iterations = args[1]
        if len(args) >= 3:
            dimensions = args[2]
            
        self.population_size = population_size
        self.max_iterations = max_iterations
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.dimensions = dimensions
        self.verbose = verbose
        self.algorithm_name = self.__class__.__name__
        
        # Store any extra parameters
        self.extra_params = kwargs

    def get_params(self):
        """Get all parameters of the optimizer."""
        params = {
            'population_size': self.population_size,
            'max_iterations': self.max_iterations,
            'lower_bound': self.lower_bound,
            'upper_bound': self.upper_bound,
            'dimensions': self.dimensions,
            'verbose': self.verbose
        }
        params.update(self.extra_params)
        return params

    def _initialize_parameters(self, X=None, y=None, objective_function=None):
        """
        Automatically initialize parameters if they are not provided.
        """
        # Determine problem type
        if y is not None and X is not None:
            self.problem_type = 'feature_selection'
        elif objective_function is not None:
            self.problem_type = 'function_optimization'
        else:
            raise ValueError("Either (X, y) for feature selection or objective_function must be provided.")

        # Auto-detect dimensions
        if self.dimensions is None:
            if self.problem_type == 'feature_selection':
                self.dimensions = X.shape[1]
            elif self.problem_type == 'function_optimization':
                self.dimensions = 10 # Default for function optimization
            if self.verbose:
                print(f"Auto-detected dimensions: {self.dimensions}")

        # Auto-detect bounds
        if self.lower_bound is None or self.upper_bound is None:
            if self.problem_type == 'feature_selection':
                defaults = {'lower_bound': 0.0, 'upper_bound': 1.0}
            else: # function_optimization
                defaults = {'lower_bound': -100.0, 'upper_bound': 100.0}
            
            self.lower_bound = self.lower_bound if self.lower_bound is not None else defaults['lower_bound']
            self.upper_bound = self.upper_bound if self.upper_bound is not None else defaults['upper_bound']
            
            if self.verbose:
                print(f"No bounds specified. Using default bounds for {self.problem_type}: [{self.lower_bound}, {self.upper_bound}]")

        # Auto-detect max_iterations
        if self.max_iterations is None:
            self.max_iterations = max(100, 10 * self.dimensions)
            if self.verbose:
                print(f"Auto-calculated max_iterations: {self.max_iterations}")

        # Format bounds to be arrays
        self._format_bounds()

    def _format_bounds(self):
        """Ensure bounds are numpy arrays of correct dimension."""
        if not isinstance(self.lower_bound, np.ndarray) or len(self.lower_bound) == 1:
            self.lower_bound = np.full(self.dimensions, self.lower_bound)
        if not isinstance(self.upper_bound, np.ndarray) or len(self.upper_bound) == 1:
            self.upper_bound = np.full(self.dimensions, self.upper_bound)

    def _create_objective_function(self, X, y):
        """
        Create a default objective function for feature selection.
        This function calculates classification error.
        """
        from sklearn.model_selection import train_test_split
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.metrics import accuracy_score

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        def objective(solution):
            # Convert solution to binary
            binary_solution = (np.array(solution) > 0.5).astype(int)
            
            # If no features are selected, return worst fitness
            if sum(binary_solution) == 0:
                return 1.0
            
            # Select features
            selected_features = X_train[:, binary_solution == 1]
            selected_features_test = X_test[:, binary_solution == 1]
            
            # Train a simple classifier
            model = KNeighborsClassifier(n_neighbors=3)
            model.fit(selected_features, y_train)
            
            # Calculate error rate
            y_pred = model.predict(selected_features_test)
            error_rate = 1.0 - accuracy_score(y_test, y_pred)
            
            return error_rate
        
        return objective

    def optimize(self, problem=None, X=None, y=None, objective_function=None):
        """
        Run the optimization process.
        
        This method can accept either a problem object or direct parameters.
        
        Parameters
        ----------
        problem : dict, optional
            Problem definition dictionary from create_problem()
        X : array-like, optional
            Feature matrix (alternative to problem)
        y : array-like, optional
            Target values (alternative to problem)
        objective_function : callable, optional
            Objective function (alternative to problem)
        """
        
        # Handle problem object
        if problem is not None:
            objective_function = problem['objective_function']
            X = problem.get('X')
            y = problem.get('y')
            if 'dimensions' in problem:
                self.dimensions = problem['dimensions']
            if 'bounds' in problem:
                bounds = problem['bounds']
                self.lower_bound = [b[0] for b in bounds]
                self.upper_bound = [b[1] for b in bounds]
        
        # Enforce that we have what we need
        if objective_function is None:
            raise ValueError("Objective function is required")
        
        start_time = time.time()
        
        self._initialize_parameters(X, y, objective_function)
        
        # Run the algorithm-specific optimization
        best_solution, best_fitness, convergence_curve = self._optimize(
            objective_function=objective_function, X=X, y=y
        )
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Create and return the standardized model object
        model = self._create_model(
            best_solution, best_fitness, convergence_curve, execution_time, X, y
        )
        
        if self.verbose:
            print(f"\nOptimization finished in {execution_time:.4f} seconds.")
            print(f"Best fitness: {best_fitness:.6f}")
        
        return model

    def _create_model(self, best_solution, best_fitness, convergence_curve, 
                      execution_time, X=None, y=None):
        """Create and return the OptimizationModel object."""
        return OptimizationModel(
            algorithm_name=self.algorithm_name,
            best_solution=best_solution,
            best_fitness=best_fitness,
            convergence_curve=convergence_curve,
            execution_time=execution_time,
            parameters=self.get_params(),
            problem_type=getattr(self, 'problem_type', 'unknown'),
            X_data=X,
            y_data=y
        )

    @abstractmethod
    def _optimize(self, objective_function, **kwargs):
        """
        Core optimization logic for the specific algorithm.
        
        This method must be implemented by all subclasses. It should contain
        the main loop of the algorithm and return the best solution, best
        fitness, and convergence history.
        
        Returns
        -------
        tuple
            (best_solution, best_fitness, convergence_curve)
        """
        pass
