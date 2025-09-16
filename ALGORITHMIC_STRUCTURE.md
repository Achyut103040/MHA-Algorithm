# MHA Algorithm Toolbox: Algorithmic Structure & Implementation Guide

## 🔄 Complete System Flowchart

```
                           MHA TOOLBOX EXECUTION FLOW
                          ================================

User Interface Layer:
┌─────────────────────────────────────────────────────────────────┐
│  User API Calls:                                                │
│  • mha.optimize('pso', X, y)                                   │
│  • mha.pso(X, y, population_size=50)                           │
│  • mha.compare(['pso', 'gwo'], X, y)                           │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
Parameter Processing Layer:
┌─────────────────────────────────────────────────────────────────┐
│  1. Validate Input Parameters                                   │
│  2. Apply Intelligent Defaults                                  │
│  3. Resolve Algorithm Aliases                                   │
│     • 'pso' → 'ParticleSwarmOptimization'                      │
│     • 'gwo' → 'GreyWolfOptimizer'                              │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
Problem Detection Layer:
┌─────────────────────────────────────────────────────────────────┐
│                Problem Type Analysis                            │
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐        │
│  │ (X, y) Data │    │ Objective   │    │   Custom    │        │
│  │ Provided?   │    │ Function?   │    │  Problem?   │        │
│  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘        │
│         │                  │                  │               │
│         ▼                  ▼                  ▼               │
│  Feature Selection   Function Optimization   User Defined     │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
Algorithm Resolution Layer:
┌─────────────────────────────────────────────────────────────────┐
│  1. Load Algorithm Class                                        │
│  2. Create Algorithm Instance                                   │
│  3. Configure Algorithm Parameters                              │
│     • Core params: population_size, max_iterations             │
│     • Algorithm-specific: c1, c2, w for PSO                   │
│     • Problem-adaptive: bounds, dimensions                     │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
Optimization Execution Layer:
┌─────────────────────────────────────────────────────────────────┐
│                    MAIN OPTIMIZATION LOOP                      │
│                                                                 │
│  ┌─ Initialize Population ────────────────────────────────────┐ │
│  │  • Random positions within bounds                          │ │
│  │  • Initialize velocities (if applicable)                   │ │
│  │  • Set initial algorithm parameters                        │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                              │                                 │
│                              ▼                                 │
│  ┌─ FOR iteration = 1 to max_iterations ─────────────────────┐ │
│  │                                                           │ │
│  │  ┌─ Evaluate Fitness ─────────────────────────────────┐   │ │
│  │  │  • For each particle/individual                    │   │ │
│  │  │  • Apply objective function or fitness calculation │   │ │
│  │  │  • Handle boundary constraints                     │   │ │
│  │  └─────────────────────────────────────────────────────┘   │ │
│  │                              │                             │ │
│  │                              ▼                             │ │
│  │  ┌─ Update Best Solutions ────────────────────────────┐   │ │
│  │  │  • Track global best                               │   │ │
│  │  │  • Update personal bests (if applicable)           │   │ │
│  │  │  • Record convergence data                         │   │ │
│  │  └─────────────────────────────────────────────────────┘   │ │
│  │                              │                             │ │
│  │                              ▼                             │ │
│  │  ┌─ Algorithm-Specific Updates ──────────────────────┐   │ │
│  │  │  PSO: Update velocities and positions             │   │ │
│  │  │  GWO: Update α, β, δ wolves and hunting positions │   │ │
│  │  │  SCA: Update sine/cosine position formula         │   │ │
│  │  │  GA: Selection, crossover, mutation operations    │   │ │
│  │  └─────────────────────────────────────────────────────┘   │ │
│  │                              │                             │ │
│  │                              ▼                             │ │
│  │  ┌─ Convergence Check ────────────────────────────────┐   │ │
│  │  │  • Check if stopping criteria met                 │   │ │
│  │  │  • Early termination if optimal found             │   │ │
│  │  │  • Update progress indicators                      │   │ │
│  │  └─────────────────────────────────────────────────────┘   │ │
│  │                              │                             │ │
│  └──────────────────────────────┼─────────────────────────────┘ │
│                              │                                 │
└──────────────────────────────┼─────────────────────────────────┘
                              │
                              ▼
Results Processing Layer:
┌─────────────────────────────────────────────────────────────────┐
│  1. Package Optimization Results                                │
│     • Best fitness value                                        │
│     • Best solution vector                                      │
│     • Convergence curve                                         │
│     • Execution statistics                                      │
│                                                                 │
│  2. Feature Selection Post-Processing (if applicable)           │
│     • Convert binary solution to feature mask                   │
│     • Calculate feature importance scores                       │
│     • Validate selected feature subset                          │
│                                                                 │
│  3. Create OptimizationModel Instance                          │
│     • Comprehensive result object                               │
│     • Built-in analysis methods                                 │
│     • Visualization capabilities                                │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
Return to User:
┌─────────────────────────────────────────────────────────────────┐
│  OptimizationModel with:                                        │
│  • .best_fitness                                                │
│  • .best_solution                                               │
│  • .plot_convergence()                                          │
│  • .summary()                                                   │
│  • Feature selection specific attributes                        │
└─────────────────────────────────────────────────────────────────┘
```

## 🧬 Algorithm-Specific Implementation Structures

### 1. Particle Swarm Optimization (PSO)
```
INITIALIZATION:
├── Create N particles with random positions and velocities
├── Set cognitive (c1) and social (c2) parameters
├── Initialize inertia weight (w) with linear decrease schedule
└── Set personal and global best positions

MAIN LOOP:
For each iteration:
├── FOR each particle i:
│   ├── Evaluate fitness f(x_i)
│   ├── Update personal best if f(x_i) < f(p_best_i)
│   └── Update global best if f(x_i) < f(g_best)
├── FOR each particle i:
│   ├── Update velocity: v_i = w*v_i + c1*r1*(p_best_i - x_i) + c2*r2*(g_best - x_i)
│   ├── Update position: x_i = x_i + v_i
│   └── Apply boundary constraints
└── Decrease inertia weight: w = w_max - (w_max - w_min) * iter/max_iter
```

### 2. Grey Wolf Optimizer (GWO)
```
INITIALIZATION:
├── Create N wolves with random positions
├── Initialize a = 2 (control parameter)
├── Designate α, β, δ wolves (best three solutions)
└── Set remaining wolves as ω wolves

MAIN LOOP:
For each iteration:
├── FOR each wolf:
│   ├── Evaluate fitness
│   └── Update α, β, δ if better solutions found
├── Calculate a = 2 - 2 * iter/max_iter (linearly decrease)
├── FOR each wolf:
│   ├── Update position based on α, β, δ wolves:
│   │   ├── Calculate D_α, D_β, D_δ (distances to leader wolves)
│   │   ├── Calculate X1, X2, X3 (positions based on leaders)
│   │   └── Average: X(t+1) = (X1 + X2 + X3) / 3
│   └── Apply boundary constraints
└── Update convergence curve
```

### 3. Sine Cosine Algorithm (SCA)
```
INITIALIZATION:
├── Create N solutions with random positions
├── Set control parameter a = 2
├── Initialize best solution
└── Set r1, r2, r3, r4 ranges

MAIN LOOP:
For each iteration:
├── Update control parameter: a = 2 - 2 * iter/max_iter
├── FOR each solution i:
│   ├── Update r1, r2, r3, r4 (random parameters)
│   ├── IF r4 < 0.5:
│   │   └── X_i = X_i + r1 * sin(r2) * |r3 * P_best - X_i|
│   ├── ELSE:
│   │   └── X_i = X_i + r1 * cos(r2) * |r3 * P_best - X_i|
│   └── Apply boundary constraints
├── Evaluate all solutions
└── Update best solution
```

## 📊 Parameter Management System

### Intelligent Defaults Engine
```
def get_intelligent_defaults(algorithm_name, problem_type, **hints):
    base_defaults = {
        'population_size': 30,
        'max_iterations': 100
    }
    
    # Problem-adaptive scaling
    if hints.get('dimensions', 10) > 50:
        base_defaults['population_size'] = min(50, dimensions)
        base_defaults['max_iterations'] = max(200, dimensions * 2)
    
    # Algorithm-specific parameters
    if algorithm_name == 'PSO':
        return {**base_defaults, 'c1': 2.0, 'c2': 2.0, 'w': 0.9}
    elif algorithm_name == 'GWO':
        return {**base_defaults, 'a_linearly_decrease': True}
    elif algorithm_name == 'SCA':
        return {**base_defaults, 'a': 2.0}
    
    return base_defaults
```

### Parameter Validation Pipeline
```
Parameter Flow:
User Input → Alias Resolution → Default Merging → Type Validation → Range Checking → Algorithm Instance
```

## 🎯 Problem Type Detection & Handling

### Feature Selection Problem
```
Input: (X, y) where X is features, y is targets
Process:
├── Problem Type: Binary optimization (0/1 for each feature)
├── Dimensions: X.shape[1] (number of features)
├── Bounds: [0, 1] for each dimension
├── Objective Function: Classification accuracy using selected features
├── Constraints: Minimum number of features (avoid empty selection)
└── Post-processing: Convert continuous [0,1] to binary {0,1}
```

### Function Optimization Problem
```
Input: objective_function + dimensions
Process:
├── Problem Type: Continuous optimization
├── Dimensions: User-specified
├── Bounds: User-specified or default [-100, 100]
├── Objective Function: Direct user function
├── Constraints: User-defined or none
└── Post-processing: Direct result from optimization
```

## 🔧 Error Handling & Robustness

### Exception Hierarchy
```
MHAToolboxError
├── AlgorithmNotFoundError
├── InvalidParameterError
├── ProblemDefinitionError
├── OptimizationError
└── ResultProcessingError
```

### Graceful Degradation
```
Error Recovery Strategy:
1. Parameter Error → Use defaults, warn user
2. Algorithm Error → Suggest alternatives
3. Convergence Issues → Return best available result
4. Memory Issues → Reduce population size automatically
```

## 📈 Performance Optimization

### Computational Efficiency
```
Optimization Strategies:
├── Vectorized Operations: Use NumPy for population-wide calculations
├── Early Termination: Stop when convergence criteria met
├── Adaptive Parameters: Adjust based on problem characteristics
├── Memory Management: Efficient storage of convergence data
└── Parallel Evaluation: Multi-core fitness evaluation (future enhancement)
```

### Scalability Considerations
```
Problem Size Handling:
├── Small (< 10 dimensions): Standard parameters
├── Medium (10-50 dimensions): Increased population
├── Large (50+ dimensions): Adaptive scaling
└── Very Large (100+ dimensions): Special handling recommendations
```

This comprehensive algorithmic structure ensures that the MHA Toolbox operates efficiently, reliably, and provides consistent results across all supported optimization algorithms while maintaining ease of use for all skill levels.