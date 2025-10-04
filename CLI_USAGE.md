# MHA Toolbox CLI Usage

## Quick Start

The MHA Toolbox provides a command-line interface for easy access to metaheuristic algorithms.

### List Available Algorithms

```bash
python -m mha_toolbox list
```

### Get Algorithm Information

```bash
python -m mha_toolbox info pso
python -m mha_toolbox info ant_lion
```

### Run Algorithms

```bash
# Run PSO on breast cancer dataset
python -m mha_toolbox run pso --dataset breast_cancer --population_size 20 --max_iterations 30

# Run Ant Lion Optimizer
python -m mha_toolbox run alo --dataset breast_cancer --population_size 15 --max_iterations 50

# Run Slime Mould Algorithm
python -m mha_toolbox run sma --dataset breast_cancer --population_size 25 --max_iterations 40

# Save the model as well
python -m mha_toolbox run tso --dataset breast_cancer --save_model
```

### Available Algorithms

The toolbox now includes 20+ algorithms:

- **Classic**: PSO, GA, DE, ACO
- **Bio-inspired**: ALO, BA, CSA, COA, MRFO, MSA, PFA, SSA, Spider, TSO, SMA, ANTS
- **Physics-based**: SCA, GWO, WOA, FA

### Available Options

- `--dataset`: breast_cancer (more datasets can be added)
- `--population_size`: Number of agents/particles (default: 30)
- `--max_iterations`: Maximum iterations (default: 100)
- `--dimensions`: Problem dimensions (auto-detected for datasets)
- `--output`: Output directory (default: results)
- `--save_model`: Also save pickled model file

### Output Files

Each run automatically generates:
- `.json`: Complete results and parameters
- `_convergence.csv`: Convergence data
- `_convergence.png`: Convergence plot
- `_local_mean.png`: Local fitness plot
- `_history.json`: Training history (all agents per iteration)
- `_model.pkl`: Pickled model (if --save_model used)

### Example Usage in Python

```python
from mha_toolbox import MHAToolbox

# Initialize toolbox
mha = MHAToolbox()

# Direct algorithm access
pso = mha.pso(population_size=30, max_iterations=100)
result = pso.optimize(X=X, y=y, objective_function=my_function)

# Auto-export results with plots
result.save('my_results.json')  # Generates all outputs automatically

# Load saved model
from mha_toolbox.base import OptimizationModel
model = OptimizationModel.load_model('my_results_model.pkl')
```