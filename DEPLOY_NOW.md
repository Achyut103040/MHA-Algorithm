# 🚀 MHA Toolbox - Deployment Commands

## Ready to Deploy! ✅

Your package is built and ready. Your PyPI token has been configured.

---

## Deployment Steps

### Step 1: Upload to PyPI (Production)

Run this command in PowerShell:

```powershell
python -m twine upload dist/*
```

**What will happen:**
- Twine will read your token from `C:\Users\Achyut Maheshka\.pypirc`
- It will upload both files:
  - `mha_toolbox-2.0.0.tar.gz`
  - `mha_toolbox-2.0.0-py3-none-any.whl`
- You'll see a progress bar for each file
- When successful, you'll see: "View at: https://pypi.org/project/mha-toolbox/"

### Step 2: Verify on PyPI

After upload, visit:
```
https://pypi.org/project/mha-toolbox/
```

Check that:
- ✅ Package name shows correctly
- ✅ Version 2.0.0 is displayed
- ✅ README displays properly
- ✅ Installation command is shown

### Step 3: Test Installation

Open a **NEW** PowerShell window and run:

```powershell
# Create test environment
python -m venv test_install
.\test_install\Scripts\activate

# Install your package from PyPI
pip install mha-toolbox

# Test it works
python -c "from mha_toolbox import __version__; print('Version:', __version__)"
python -c "from mha_toolbox.algorithms import PSO, GWO, WOA; print('✓ Import successful!')"

# Test a quick optimization
python -c "from mha_toolbox.algorithms import PSO; import numpy as np; bounds = np.array([[-10, 10]] * 5); pso = PSO(lambda x: np.sum(x**2), bounds, n_particles=10, max_iter=10); best_pos, best_fit = pso.optimize(); print('✓ Optimization works! Fitness:', best_fit)"

# Clean up
deactivate
cd ..
Remove-Item -Path test_install -Recurse -Force
```

---

## Alternative: If Upload Fails

If you get any errors, you can upload manually with the token:

```powershell
python -m twine upload dist/* -u __token__ -p pypi-AgEIcHlwaS5vcmcCJDk1YTBlYmM2LWQ3YjEtNDE1My04NDg3LTZlMmQyMTU4MzY5NwACKlszLCIwODkwNmVkMi1kN2RiLTQzOTctOWI5Yi1iOWNkZmZjNjg1ODEiXQAABiC90S1xyi3xUdv02N1GWqCy08AiikEaeRylL3g3Z0Yn5A
```

---

## After Successful Upload

### Update GitHub

```powershell
# Commit all changes
git add .
git commit -m "Release v2.0.0 - Published to PyPI"

# Create version tag
git tag -a v2.0.0 -m "Version 2.0.0 - First PyPI Release"

# Push to GitHub
git push origin main
git push origin v2.0.0
```

### Create GitHub Release

1. Go to: https://github.com/Achyut103040/MHA-Algorithm/releases
2. Click "Draft a new release"
3. Choose tag: `v2.0.0`
4. Title: `MHA Toolbox v2.0.0`
5. Description:

```markdown
# 🎉 MHA Toolbox v2.0.0 - First PyPI Release!

## Installation
```bash
pip install mha-toolbox
```

## What's New
- 95+ metaheuristic optimization algorithms
- 9 hybrid algorithms
- Streamlit web interface
- Comprehensive documentation
- Easy-to-use API

## Quick Start
```python
from mha_toolbox.algorithms import PSO
import numpy as np

def sphere(x):
    return np.sum(x**2)

bounds = np.array([[-10, 10]] * 5)
pso = PSO(objective_func=sphere, bounds=bounds, n_particles=30, max_iter=100)
best_pos, best_fit = pso.optimize()
print(f"Best fitness: {best_fit}")
```

## Links
- PyPI: https://pypi.org/project/mha-toolbox/
- Documentation: See [USER_GUIDE.md](USER_GUIDE.md)
- Examples: See [examples/quick_start.py](examples/quick_start.py)

Perfect for optimization, feature selection, hyperparameter tuning, and research!
```

6. Click "Publish release"

---

## Share Your Library

### Twitter/X
```
🚀 MHA Toolbox v2.0.0 is now on PyPI!

📦 pip install mha-toolbox

✨ 104 optimization algorithms
🔬 PSO, GWO, WOA, GA, DE + 90 more
📊 Built-in web interface

#Python #MachineLearning #Optimization

https://pypi.org/project/mha-toolbox/
```

### LinkedIn
```
Excited to announce MHA Toolbox v2.0.0 on PyPI! 🎉

A comprehensive library with 104 metaheuristic optimization algorithms.

Perfect for:
✅ Function optimization
✅ Feature selection  
✅ Hyperparameter tuning
✅ Research & education

pip install mha-toolbox

https://pypi.org/project/mha-toolbox/
```

### Reddit (r/Python, r/MachineLearning)
```
[P] MHA Toolbox - 104 Metaheuristic Algorithms on PyPI

Published a comprehensive optimization library with 104 algorithms including PSO, GWO, WOA, GA, DE, and 90+ more.

Features:
- Easy API
- Web interface
- Full documentation
- 9 hybrid algorithms

pip install mha-toolbox

PyPI: https://pypi.org/project/mha-toolbox/
GitHub: https://github.com/Achyut103040/MHA-Algorithm
```

---

## Troubleshooting

### Error: "HTTPError: 403 Forbidden"
- Token might be invalid or expired
- Make sure you're using production PyPI token (not TestPyPI)
- Try the manual upload command with `-u __token__ -p <your-token>`

### Error: "File already exists"
- The version 2.0.0 is already on PyPI
- You cannot overwrite it
- Increment version to 2.0.1 in `mha_toolbox/__init__.py`
- Rebuild: `python -m build`
- Upload again

### Error: "Invalid distribution"
- Clean and rebuild:
  ```powershell
  Remove-Item -Path dist -Recurse -Force
  Remove-Item -Path build -Recurse -Force
  python -m build
  ```

---

## Quick Command Checklist

```powershell
# 1. Upload to PyPI
python -m twine upload dist/*

# 2. Test installation (in new terminal)
pip install mha-toolbox

# 3. Test imports
python -c "from mha_toolbox.algorithms import PSO; print('Success!')"

# 4. Update GitHub
git add .
git commit -m "Release v2.0.0"
git tag -a v2.0.0 -m "Version 2.0.0"
git push origin main --tags

# 5. Visit your package
# https://pypi.org/project/mha-toolbox/
```

---

## 🎉 Congratulations!

Once uploaded, anyone in the world can install your library with:

```bash
pip install mha-toolbox
```

Your contribution to the Python and optimization community is live! 🌍

---

**Next command to run:**

```powershell
python -m twine upload dist/*
```

Good luck! 🚀
