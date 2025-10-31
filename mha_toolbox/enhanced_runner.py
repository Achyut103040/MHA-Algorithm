"""
Enhanced Runner with Live Progress Support
==========================================

This module provides generator-based algorithm execution that yields results
after each algorithm completes, enabling live UI updates.
"""

import numpy as np
import time
from datetime import datetime


def run_comparison_with_live_progress(X, y, dataset_name, task_type, algorithms,
                                      max_iterations, population_size, n_runs, 
                                      timeout_minutes):
    """
    Generator function that yields results for each algorithm as it completes.
    
    This enables live progress updates in the UI by yielding after each algorithm finishes.
    
    Args:
        X: Feature matrix
        y: Target vector
        dataset_name: Name of the dataset
        task_type: Type of optimization task
        algorithms: List of algorithm names to run
        max_iterations: Maximum iterations per algorithm
        population_size: Population size for algorithms
        n_runs: Number of independent runs per algorithm
        timeout_minutes: Timeout in minutes per algorithm
        
    Yields:
        dict: Result dictionary for each completed algorithm containing:
            - algorithm: Algorithm name
            - status: 'running', 'completed', or 'failed'
            - result_data: Full results if completed
            - error: Error message if failed
    """
    
    # Import MHA toolbox
    try:
        import mha_toolbox as mha
    except ImportError:
        yield {
            'algorithm': 'system',
            'status': 'failed',
            'error': 'mha_toolbox not found. Please install it first.'
        }
        return
    
    total_algorithms = len(algorithms)
    
    for i, alg_name in enumerate(algorithms):
        # Yield "running" status
        yield {
            'algorithm': alg_name,
            'status': 'running',
            'progress': i / total_algorithms,
            'iteration': f"{i+1}/{total_algorithms}"
        }
        
        try:
            # Run algorithm with timeout protection
            alg_start_time = time.time()
            
            # Parameters for this algorithm
            params = {
                'max_iterations': max_iterations,
                'population_size': population_size,
                'verbose': False
            }
            
            # Run multiple times for reliability
            runs_data = []
            
            for run in range(n_runs):
                # Check timeout
                if time.time() - alg_start_time > timeout_minutes * 60:
                    break
                
                run_start_time = time.time()
                
                try:
                    if task_type == 'feature_selection':
                        result = mha.optimize(alg_name, X, y, **params)
                        
                        run_result = {
                            'run': run + 1,
                            'best_fitness': float(result.best_fitness_),
                            'n_selected_features': int(result.n_selected_features_),
                            'convergence_curve': [float(x) for x in result.global_fitness_],
                            'execution_time': time.time() - run_start_time,
                            'final_accuracy': float(1 - result.best_fitness_),
                            'success': True,
                            'total_iterations': len(result.global_fitness_)
                        }
                        
                    elif task_type == 'feature_optimization':
                        # Feature optimization objective
                        def feature_objective(weights):
                            try:
                                weights = np.array(weights)
                                weights = np.clip(weights, 0, 1)
                                
                                if len(weights) != X.shape[1]:
                                    if len(weights) < X.shape[1]:
                                        weights = np.tile(weights, (X.shape[1] // len(weights)) + 1)[:X.shape[1]]
                                    else:
                                        weights = weights[:X.shape[1]]
                                
                                X_weighted = X * weights.reshape(1, -1)
                                
                                from sklearn.ensemble import RandomForestClassifier
                                from sklearn.model_selection import cross_val_score
                                
                                model = RandomForestClassifier(n_estimators=10, random_state=42)
                                scores = cross_val_score(model, X_weighted, y, cv=3, scoring='accuracy')
                                return 1.0 - np.mean(scores)
                                
                            except Exception:
                                return 1.0
                        
                        result = mha.optimize(
                            alg_name,
                            objective_function=feature_objective,
                            dimensions=X.shape[1],
                            bounds=[(0, 1)] * X.shape[1],
                            **params
                        )
                        
                        run_result = {
                            'run': run + 1,
                            'best_fitness': float(result.best_fitness_),
                            'convergence_curve': [float(x) for x in result.global_fitness_],
                            'execution_time': time.time() - run_start_time,
                            'optimized_weights': result.best_solution_.tolist(),
                            'performance_score': float(1 - result.best_fitness_),
                            'success': True,
                            'total_iterations': len(result.global_fitness_)
                        }
                        
                    elif task_type == 'hyperparameter_tuning':
                        # Hyperparameter optimization objective
                        def hyperparameter_objective(params_vector):
                            try:
                                params_vector = np.array(params_vector)
                                params_vector = np.clip(params_vector, 0, 1)
                                
                                n_estimators = max(10, int(params_vector[0] * 190 + 10))
                                max_depth = max(3, int(params_vector[1] * 17 + 3)) if params_vector[1] > 0.1 else None
                                min_samples_split = max(2, int(params_vector[2] * 18 + 2))
                                
                                from sklearn.ensemble import RandomForestClassifier
                                from sklearn.model_selection import cross_val_score
                                
                                rf = RandomForestClassifier(
                                    n_estimators=n_estimators,
                                    max_depth=max_depth,
                                    min_samples_split=min_samples_split,
                                    random_state=42
                                )
                                
                                scores = cross_val_score(rf, X, y, cv=3, scoring='accuracy')
                                return 1.0 - np.mean(scores)
                                
                            except Exception:
                                return 1.0
                        
                        result = mha.optimize(
                            alg_name,
                            objective_function=hyperparameter_objective,
                            dimensions=3,
                            bounds=[(0, 1)] * 3,
                            **params
                        )
                        
                        best_params = result.best_solution_
                        best_n_estimators = max(10, int(best_params[0] * 190 + 10))
                        best_max_depth = max(3, int(best_params[1] * 17 + 3)) if best_params[1] > 0.1 else None
                        best_min_samples_split = max(2, int(best_params[2] * 18 + 2))
                        
                        run_result = {
                            'run': run + 1,
                            'best_fitness': float(result.best_fitness_),
                            'convergence_curve': [float(x) for x in result.global_fitness_],
                            'execution_time': time.time() - run_start_time,
                            'best_accuracy': float(1 - result.best_fitness_),
                            'optimized_params': result.best_solution_.tolist(),
                            'best_hyperparameters': {
                                'n_estimators': best_n_estimators,
                                'max_depth': best_max_depth,
                                'min_samples_split': best_min_samples_split
                            },
                            'success': True,
                            'total_iterations': len(result.global_fitness_)
                        }
                    
                    runs_data.append(run_result)
                    
                except Exception as e:
                    continue
            
            # Calculate statistics if we have successful runs
            if runs_data:
                fitnesses = [run['best_fitness'] for run in runs_data if run.get('success', False)]
                times = [run['execution_time'] for run in runs_data if run.get('success', False)]
                
                if fitnesses:
                    statistics = {
                        'mean_fitness': float(np.mean(fitnesses)),
                        'std_fitness': float(np.std(fitnesses)),
                        'best_fitness': float(np.min(fitnesses)),
                        'worst_fitness': float(np.max(fitnesses)),
                        'mean_time': float(np.mean(times)),
                        'std_time': float(np.std(times)),
                        'total_runs': len(runs_data),
                        'successful_runs': len(fitnesses)
                    }
                    
                    # Task-specific statistics
                    if task_type == 'feature_selection':
                        n_features = [run['n_selected_features'] for run in runs_data if run.get('success', False)]
                        accuracies = [run['final_accuracy'] for run in runs_data if run.get('success', False)]
                        
                        if n_features and accuracies:
                            statistics.update({
                                'mean_features': float(np.mean(n_features)),
                                'std_features': float(np.std(n_features)),
                                'mean_accuracy': float(np.mean(accuracies)),
                                'std_accuracy': float(np.std(accuracies))
                            })
                    
                    # Yield completed status with results
                    yield {
                        'algorithm': alg_name,
                        'status': 'completed',
                        'result_data': {
                            'algorithm': alg_name,
                            'runs': runs_data,
                            'statistics': statistics,
                            'task_type': task_type,
                            'total_execution_time': time.time() - alg_start_time
                        },
                        'progress': (i + 1) / total_algorithms
                    }
                else:
                    # All runs failed
                    yield {
                        'algorithm': alg_name,
                        'status': 'failed',
                        'error': 'All runs failed or timed out',
                        'progress': (i + 1) / total_algorithms
                    }
            else:
                # No successful runs
                yield {
                    'algorithm': alg_name,
                    'status': 'failed',
                    'error': 'No successful runs completed',
                    'progress': (i + 1) / total_algorithms
                }
                
        except Exception as e:
            # Algorithm failed completely
            yield {
                'algorithm': alg_name,
                'status': 'failed',
                'error': str(e),
                'progress': (i + 1) / total_algorithms
            }
    
    # Yield final completion
    yield {
        'algorithm': 'all',
        'status': 'completed',
        'progress': 1.0,
        'message': 'All algorithms completed'
    }
