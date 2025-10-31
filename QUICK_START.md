# 🚀 Quick Start Guide - MHA Toolbox Pro v2.0.0

## Three Ways to Get Started

### 1. 🌐 Web Interface (Recommended)

**Windows:**
```bash
launch.bat
```

**Mac/Linux:**
```bash
streamlit run mha_toolbox_pro_ultimate.py
```

Then open: http://localhost:8501

---

### 2. 🐍 Python API

**Basic Usage:**
```python
from mha_toolbox.algorithms import PSO
import numpy as np

# Define objective function
def sphere(x):
    return np.sum(x**2)

# Run optimization
optimizer = PSO(population_size=30, max_iterations=100)
best_pos, best_fit, convergence, _, _ = optimizer.optimize(
    sphere, bounds=(-100, 100), dimension=10
)

print(f"Best fitness: {best_fit}")
```

**Using Hybrids:**
```python
from mha_toolbox.algorithms.hybrid import GWO_PSO_Hybrid

optimizer = GWO_PSO_Hybrid(population_size=30, max_iterations=100)
best_pos, best_fit, convergence, _, _ = optimizer.optimize(
    sphere, bounds=(-100, 100), dimension=10
)
```

---

### 3. 📦 Command Line (Coming Soon)

```bash
mha-toolbox --algorithm pso --function sphere --dim 10
```

---

## Available Categories

1. **Swarm Intelligence (15)**: PSO, ABC, ACO, WOA, GWO, BA, FA, SSA, SMA, HHO, etc.
2. **Evolutionary (8)**: GA, DE, EO, EPO, GBO, AO, AOA, CGO
3. **Bio-Inspired (32)**: ALO, BBO, BOA, CA, DA, DMOA, EHO, FPA, GAO, GOA, HBA, HGS, etc.
4. **Physics-Based (17)**: SA, GSA, CSS, ASO, CRO, GBMO, MVO, NRO, TWO, WWO, etc.
5. **Human Behavior (10)**: TLBO, ICA, SOS, LCA, SAR, GSK, LCBO, SSD, THRO, INNOV
6. **Mathematical (13)**: AEO, CEM, GCO, HC, HS, PM, RUN, TS, VNS, SHO, SLO, CSA, MSA
7. **Hybrid (9)**: PSO-GA, GWO-PSO, WOA-GA, SMA-DE, ABC-DE, WOA-SMA, GA-SA, DE-PSO, FA-GA

---

## Benchmark Functions

- **Sphere**: `lambda x: np.sum(x**2)`
- **Rastrigin**: `lambda x: 10*len(x) + np.sum(x**2 - 10*np.cos(2*np.pi*x))`
- **Ackley**: Complex multimodal function
- **Rosenbrock**: Narrow valley optimization
- **Griewank**: Product and sum components
- **Schwefel**: Deceptive multimodal

---

## Export Your Results

**CSV:**
```python
df.to_csv('results.csv', index=False)
```

**JSON:**
```python
import json
with open('results.json', 'w') as f:
    json.dump(results, f, indent=2)
```

**NPZ:**
```python
import numpy as np
np.savez('results.npz', **data)
```

---

## Tips for Best Results

1. **Population Size**: Start with 30, increase for complex problems
2. **Iterations**: 100-500 depending on convergence
3. **Bounds**: Set appropriate for your problem domain
4. **Algorithm Choice**: 
   - Unimodal: PSO, GWO
   - Multimodal: WOA, SMA, HHO
   - Hybrid: Best of both worlds

---

## Need Help?

- 📖 See README.md for full documentation
- 🐛 Report issues: https://github.com/Achyut103040/MHA-Algorithm/issues
- 💬 Questions: mha.toolbox@gmail.com

---

**Happy Optimizing! 🎉**
