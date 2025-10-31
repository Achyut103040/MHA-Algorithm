# Publishing Guide - MHA Toolbox to PyPI

## Overview
This guide explains how to publish the MHA Toolbox library to PyPI (Python Package Index) so others can install it using `pip install mha-toolbox`.

---

## Prerequisites

### 1. Create PyPI Account
- **Production PyPI**: https://pypi.org/account/register/
- **Test PyPI** (for testing): https://test.pypi.org/account/register/

### 2. Install Required Tools
```cmd
pip install build twine wheel setuptools
```

### 3. Generate API Token
1. Log in to PyPI
2. Go to Account Settings → API tokens
3. Click "Add API token"
4. Name: `mha-toolbox-upload`
5. Scope: "Entire account" or specific to this project
6. **Copy the token** (you'll only see it once!)

---

## Step-by-Step Publishing Process

### Step 1: Verify Package Structure

Ensure your package has all required files:

```
MHA-Algorithm/
├── mha_toolbox/
│   ├── __init__.py          # Contains __version__ = "2.0.0"
│   ├── algorithms/          # All algorithm files
│   └── ...
├── setup.py                 # Package configuration
├── pyproject.toml          # Build system configuration
├── README.md               # Package description
├── LICENSE                 # MIT License
└── requirements.txt        # Dependencies
```

### Step 2: Update Version Number

Edit `mha_toolbox/__init__.py`:
```python
__version__ = "2.0.0"  # Update this for new releases
```

Version numbering guide:
- **Major** (X.0.0): Breaking changes
- **Minor** (2.X.0): New features, backwards compatible
- **Patch** (2.0.X): Bug fixes

### Step 3: Clean Previous Builds

```cmd
# Remove old build artifacts
rmdir /s /q dist
rmdir /s /q build
rmdir /s /q mha_toolbox.egg-info
```

### Step 4: Build the Package

```cmd
python -m build
```

This creates two files in `dist/`:
- `mha_toolbox-2.0.0.tar.gz` (source distribution)
- `mha_toolbox-2.0.0-py3-none-any.whl` (wheel distribution)

**Verify the build:**
```cmd
dir dist
```

### Step 5: Test on TestPyPI (Recommended)

#### Upload to TestPyPI:
```cmd
python -m twine upload --repository testpypi dist/*
```

When prompted:
- **Username**: `__token__`
- **Password**: Your TestPyPI API token (starts with `pypi-...`)

#### Test Installation:
```cmd
# Create a test environment
python -m venv test_env
test_env\Scripts\activate

# Install from TestPyPI
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ mha-toolbox

# Test it works
python -c "from mha_toolbox import __version__; print(__version__)"
python -c "from mha_toolbox.algorithms import PSO, GWO; print('Success!')"

# Deactivate and remove test environment
deactivate
rmdir /s /q test_env
```

### Step 6: Upload to Production PyPI

If TestPyPI worked correctly:

```cmd
python -m twine upload dist/*
```

When prompted:
- **Username**: `__token__`
- **Password**: Your PyPI API token

**Alternative:** Create `~/.pypirc` to avoid typing credentials:

```ini
[pypi]
username = __token__
password = pypi-your-actual-token-here

[testpypi]
username = __token__
password = pypi-your-test-token-here
```

Then just run:
```cmd
python -m twine upload dist/*
```

---

## After Publishing

### Verify on PyPI
1. Visit: https://pypi.org/project/mha-toolbox/
2. Check the package page shows correctly
3. Verify README displays properly

### Test Installation
```cmd
# In a new environment
pip install mha-toolbox

# Test it works
python -c "from mha_toolbox.algorithms import PSO; print('Installation successful!')"
```

### Update GitHub Repository

```cmd
git add .
git commit -m "Release version 2.0.0 - Published to PyPI"
git tag -a v2.0.0 -m "Version 2.0.0"
git push origin main
git push origin v2.0.0
```

---

## Making Others Aware

### 1. Update GitHub Repository

#### Add Installation Badge to README.md:
```markdown
[![PyPI version](https://badge.fury.io/py/mha-toolbox.svg)](https://badge.fury.io/py/mha-toolbox)
[![Downloads](https://pepy.tech/badge/mha-toolbox)](https://pepy.tech/project/mha-toolbox)
```

#### Update README.md with Installation Section:
```markdown
## Installation

```bash
pip install mha-toolbox
```

## Quick Start

```python
from mha_toolbox.algorithms import PSO
import numpy as np

# Define objective function
def sphere(x):
    return np.sum(x**2)

# Run optimization
bounds = np.array([[-10, 10]] * 5)
pso = PSO(sphere, bounds, n_particles=30, max_iter=100)
best_pos, best_fit = pso.optimize()
print(f"Best fitness: {best_fit}")
```
```

### 2. Create GitHub Release

1. Go to your repository: https://github.com/Achyut103040/MHA-Algorithm
2. Click "Releases" → "Create a new release"
3. Tag: `v2.0.0`
4. Title: `MHA Toolbox v2.0.0`
5. Description:
```markdown
# MHA Toolbox v2.0.0

## 🎉 First PyPI Release!

Complete metaheuristic optimization library with 104 algorithms (95+ individual + 9 hybrids).

### Installation
```bash
pip install mha-toolbox
```

### Features
- ✅ 95+ optimization algorithms
- ✅ 9 hybrid algorithms  
- ✅ Web interface with Streamlit
- ✅ Comprehensive documentation
- ✅ Easy-to-use API

### Quick Example
```python
from mha_toolbox.algorithms import PSO
import numpy as np

def sphere(x):
    return np.sum(x**2)

bounds = np.array([[-10, 10]] * 5)
pso = PSO(sphere, bounds, n_particles=30, max_iter=100)
best_pos, best_fit = pso.optimize()
```

### What's Included
- All 95+ individual algorithms (PSO, GWO, WOA, GA, etc.)
- 9 hybrid algorithms (GWO-PSO, WOA-GA, ABC-DE, etc.)
- Streamlit web interface for easy visualization
- Comprehensive documentation and examples

See [USER_GUIDE.md](USER_GUIDE.md) for detailed usage instructions.
```

### 3. Share on Social Media / Forums

**Twitter/X:**
```
🚀 Just published MHA Toolbox v2.0.0 on PyPI!

📦 pip install mha-toolbox

✨ 104 metaheuristic optimization algorithms
🔬 Perfect for optimization, feature selection, ML
📊 Built-in visualization & web interface

#Python #MachineLearning #Optimization

https://pypi.org/project/mha-toolbox/
```

**LinkedIn:**
```
Excited to announce the release of MHA Toolbox v2.0.0 on PyPI! 

This comprehensive library provides 104 metaheuristic optimization algorithms including PSO, GWO, WOA, GA, DE, and 9 hybrid combinations.

Perfect for:
- Function optimization
- Feature selection
- Hyperparameter tuning
- Engineering design
- Scientific research

Easy installation: pip install mha-toolbox

Check it out: https://pypi.org/project/mha-toolbox/
GitHub: https://github.com/Achyut103040/MHA-Algorithm
```

**Reddit** (r/Python, r/MachineLearning, r/learnpython):
```
Title: [P] MHA Toolbox - 104 Metaheuristic Optimization Algorithms

I've published a comprehensive Python library with 104 metaheuristic optimization algorithms including PSO, GWO, WOA, GA, DE, and many more.

Features:
- 95+ individual algorithms
- 9 hybrid algorithms
- Easy-to-use API
- Built-in visualization
- Streamlit web interface
- Comprehensive documentation

Installation: pip install mha-toolbox

PyPI: https://pypi.org/project/mha-toolbox/
GitHub: https://github.com/Achyut103040/MHA-Algorithm

Would love to hear your feedback!
```

### 4. Academic Sharing

If used in research, create a **CITATION.cff** file:

```yaml
cff-version: 1.2.0
message: "If you use this software, please cite it as below."
authors:
  - family-names: "Your Last Name"
    given-names: "Your First Name"
title: "MHA Toolbox: A Comprehensive Metaheuristic Algorithms Library"
version: 2.0.0
date-released: 2025-01-XX
url: "https://github.com/Achyut103040/MHA-Algorithm"
```

### 5. Create Documentation Website (Optional)

Use **Read the Docs** or **GitHub Pages**:

```bash
# Using Sphinx
pip install sphinx sphinx-rtd-theme
sphinx-quickstart docs
# Configure and build documentation
cd docs
make html
```

---

## Updating the Package (Future Releases)

### For Bug Fixes (2.0.0 → 2.0.1):
1. Fix bugs
2. Update `__version__ = "2.0.1"` in `__init__.py`
3. Clean, rebuild, upload:
```cmd
rmdir /s /q dist build mha_toolbox.egg-info
python -m build
python -m twine upload dist/*
```

### For New Features (2.0.0 → 2.1.0):
1. Add features
2. Update `__version__ = "2.1.0"`
3. Update README.md and USER_GUIDE.md
4. Clean, rebuild, upload
5. Create new GitHub release

### For Breaking Changes (2.0.0 → 3.0.0):
1. Make changes
2. Update `__version__ = "3.0.0"`
3. Update all documentation
4. Clearly document breaking changes
5. Clean, rebuild, upload
6. Announce breaking changes in release notes

---

## Usage Statistics

After publishing, track your package's usage:

### PyPI Stats:
- https://pypi.org/project/mha-toolbox/#history
- https://pepy.tech/project/mha-toolbox

### GitHub Stats:
- Stars
- Forks  
- Issues
- Contributors

### Add Analytics Badge to README:
```markdown
![PyPI Downloads](https://img.shields.io/pypi/dm/mha-toolbox)
![GitHub stars](https://img.shields.io/github/stars/Achyut103040/MHA-Algorithm)
![GitHub forks](https://img.shields.io/github/forks/Achyut103040/MHA-Algorithm)
```

---

## Troubleshooting

### Error: "Package already exists"
- You cannot overwrite a version on PyPI
- Increment version number and rebuild
- Delete from TestPyPI if needed (production PyPI doesn't allow deletion)

### Error: "Invalid username or password"
- Make sure username is `__token__` (not your PyPI username)
- Check token starts with `pypi-`
- Verify token has upload permissions

### Error: "README rendering failed"
- Check README.md is valid Markdown
- Test locally: `python -m readme_renderer README.md`

### Error: "Missing required metadata"
- Check setup.py has all required fields
- Verify pyproject.toml is correct

---

## Complete Publishing Checklist

- [ ] All tests pass
- [ ] All imports work correctly
- [ ] Version number updated
- [ ] README.md is complete
- [ ] LICENSE file exists
- [ ] requirements.txt is up to date
- [ ] Clean old builds
- [ ] Build package (`python -m build`)
- [ ] Test on TestPyPI
- [ ] Upload to PyPI
- [ ] Test installation from PyPI
- [ ] Create GitHub release
- [ ] Update repository README with installation instructions
- [ ] Share on social media
- [ ] Update documentation website (if applicable)

---

## Quick Reference Commands

```cmd
# Complete publishing workflow
rmdir /s /q dist build mha_toolbox.egg-info
python -m build
python -m twine upload --repository testpypi dist/*
python -m twine upload dist/*

# Test installation
pip install mha-toolbox
python -c "from mha_toolbox import __version__; print(__version__)"

# Update GitHub
git add .
git commit -m "Release v2.0.0"
git tag -a v2.0.0 -m "Version 2.0.0"
git push origin main --tags
```

---

**Ready to share your library with the world! 🌍**
