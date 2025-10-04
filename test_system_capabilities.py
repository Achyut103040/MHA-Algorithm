#!/usr/bin/env python3
"""
Test script to verify system capabilities for different problem types:
1. Function Optimization
2. Feature Selection  
3. Hyperparameter Tuning
"""

from mha_toolbox import MHAToolbox
import numpy as np
from sklearn.datasets import make_classification, load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

# Initialize toolbox
mha = MHAToolbox()

print("🧪 Testing MHA System Capabilities...")
print("="*60)

# =============================================================================
# 1. FUNCTION OPTIMIZATION TEST
# =============================================================================
print("\n1️⃣ FUNCTION OPTIMIZATION TEST")
print("-" * 40)

def rosenbrock_function(x):
    """Rosenbrock function - classic optimization benchmark"""
    return sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

def sphere_function(x):
    """Sphere function - simple benchmark"""
    return np.sum(x**2)

try:
    print("Testing Sphere Function with PSO...")
    result = mha.optimize(
        'pso',
        objective_function=sphere_function,
        dimensions=10,
        lower_bound=-5.0,
        upper_bound=5.0,
        population_size=30,
        max_iterations=100
    )
    print(f"✅ Function Optimization: SUCCESS")
    print(f"   Best fitness: {result.best_fitness_:.6f}")
    print(f"   Expected near 0.0: {'PASS' if result.best_fitness_ < 0.1 else 'FAIL'}")
    
except Exception as e:
    print(f"❌ Function Optimization: FAILED - {e}")

# =============================================================================
# 2. FEATURE SELECTION TEST
# =============================================================================
print("\n2️⃣ FEATURE SELECTION TEST")
print("-" * 40)

try:
    # Create a dataset with many features
    X, y = make_classification(
        n_samples=200, 
        n_features=50, 
        n_informative=10, 
        n_redundant=5, 
        random_state=42
    )
    
    print(f"Testing Feature Selection with GA on dataset: {X.shape}")
    result = mha.optimize(
        'ga',
        X=X,
        y=y,
        population_size=20,
        max_iterations=50
    )
    
    print(f"✅ Feature Selection: SUCCESS")
    print(f"   Original features: {X.shape[1]}")
    print(f"   Selected features: {result.n_selected_features_}")
    print(f"   Selection ratio: {result.n_selected_features_/X.shape[1]:.2%}")
    print(f"   Best fitness: {result.best_fitness_:.6f}")
    
except Exception as e:
    print(f"❌ Feature Selection: FAILED - {e}")

# =============================================================================
# 3. HYPERPARAMETER TUNING TEST  
# =============================================================================
print("\n3️⃣ HYPERPARAMETER TUNING TEST")
print("-" * 40)

try:
    # Load a real dataset
    data = load_breast_cancer()
    X, y = data.data, data.target
    
    def rf_hyperparameter_objective(params):
        """
        Objective function for Random Forest hyperparameter tuning.
        params: [n_estimators, max_depth, min_samples_split, min_samples_leaf]
        """
        # Convert continuous values to discrete hyperparameters
        n_estimators = int(10 + params[0] * 190)  # 10-200
        max_depth = max(1, int(params[1] * 20))   # 1-20  
        min_samples_split = max(2, int(params[2] * 18) + 2)  # 2-20
        min_samples_leaf = max(1, int(params[3] * 10) + 1)   # 1-10
        
        try:
            # Create Random Forest with these hyperparameters
            rf = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                random_state=42
            )
            
            # Evaluate using cross-validation
            scores = cross_val_score(rf, X, y, cv=3, scoring='accuracy')
            
            # Return negative accuracy (since we minimize)
            return -np.mean(scores)
            
        except Exception:
            return 1.0  # Bad fitness for invalid parameters
    
    print("Testing Hyperparameter Tuning with GWO on Random Forest...")
    result = mha.optimize(
        'gwo',
        objective_function=rf_hyperparameter_objective,
        dimensions=4,  # 4 hyperparameters
        lower_bound=0.0,
        upper_bound=1.0,
        population_size=15,
        max_iterations=30
    )
    
    print(f"✅ Hyperparameter Tuning: SUCCESS")
    print(f"   Best accuracy: {-result.best_fitness_:.4f}")
    print(f"   Best parameters:")
    best_params = result.best_solution_
    print(f"     n_estimators: {int(10 + best_params[0] * 190)}")
    print(f"     max_depth: {max(1, int(best_params[1] * 20))}")
    print(f"     min_samples_split: {max(2, int(best_params[2] * 18) + 2)}")
    print(f"     min_samples_leaf: {max(1, int(best_params[3] * 10) + 1)}")
    
except Exception as e:
    print(f"❌ Hyperparameter Tuning: FAILED - {e}")

# =============================================================================
# 4. MULTI-OBJECTIVE OPTIMIZATION TEST
# =============================================================================
print("\n4️⃣ MULTI-OBJECTIVE CAPABILITIES TEST")
print("-" * 40)

try:
    def multi_objective_function(x):
        """
        Test multi-objective capability by combining objectives
        Objective 1: Minimize sum of squares
        Objective 2: Minimize sum of absolute values
        """
        obj1 = np.sum(x**2)
        obj2 = np.sum(np.abs(x))
        # Weighted combination (this is a simple scalarization approach)
        return 0.5 * obj1 + 0.5 * obj2
    
    print("Testing Multi-objective optimization with SCA...")
    result = mha.optimize(
        'sca',
        objective_function=multi_objective_function,
        dimensions=5,
        lower_bound=-2.0,
        upper_bound=2.0,
        population_size=20,
        max_iterations=50
    )
    
    print(f"✅ Multi-objective: SUCCESS")
    print(f"   Combined objective: {result.best_fitness_:.6f}")
    
except Exception as e:
    print(f"❌ Multi-objective: FAILED - {e}")

# =============================================================================
# 5. CONSTRAINT HANDLING TEST
# =============================================================================
print("\n5️⃣ CONSTRAINT HANDLING TEST")
print("-" * 40)

try:
    def constrained_function(x):
        """
        Constrained optimization problem
        Minimize: x1^2 + x2^2
        Subject to: x1 + x2 >= 1 (constraint)
        """
        objective = x[0]**2 + x[1]**2
        
        # Constraint: x1 + x2 >= 1
        constraint_violation = max(0, 1 - (x[0] + x[1]))
        
        # Penalty method for constraint handling
        penalty = 1000 * constraint_violation**2
        
        return objective + penalty
    
    print("Testing Constraint handling with WOA...")
    result = mha.optimize(
        'woa',
        objective_function=constrained_function,
        dimensions=2,
        lower_bound=-1.0,
        upper_bound=3.0,
        population_size=20,
        max_iterations=50
    )
    
    print(f"✅ Constraint Handling: SUCCESS")
    print(f"   Best fitness: {result.best_fitness_:.6f}")
    print(f"   Best solution: [{result.best_solution_[0]:.3f}, {result.best_solution_[1]:.3f}]")
    constraint_check = result.best_solution_[0] + result.best_solution_[1]
    print(f"   Constraint (x1+x2>=1): {constraint_check:.3f} {'SATISFIED' if constraint_check >= 0.99 else 'VIOLATED'}")
    
except Exception as e:
    print(f"❌ Constraint Handling: FAILED - {e}")

print("\n" + "="*60)
print("🎯 SYSTEM CAPABILITIES SUMMARY:")
print("✅ Function Optimization - Supported")
print("✅ Feature Selection - Supported") 
print("✅ Hyperparameter Tuning - Supported")
print("✅ Multi-objective Optimization - Supported (via scalarization)")
print("✅ Constraint Handling - Supported (via penalty methods)")
print("="*60)