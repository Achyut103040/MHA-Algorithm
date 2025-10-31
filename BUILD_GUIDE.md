# 📦 Build & Distribution Guide
## MHA Toolbox Pro v2.0.0

This guide shows you how to build and distribute the MHA Toolbox as a Python library.

---

## ✅ Prerequisites

All prerequisites are now installed:
- ✅ build
- ✅ twine  
- ✅ wheel
- ✅ setuptools

---

## 🚀 Method 1: Build Distribution Packages

### Step 1: Clean Previous Builds

```powershell
# Remove old builds
if (Test-Path dist) { Remove-Item -Recurse -Force dist }
if (Test-Path build) { Remove-Item -Recurse -Force build }
if (Test-Path mha_toolbox.egg-info) { Remove-Item -Recurse -Force mha_toolbox.egg-info }
```

### Step 2: Build the Package

```powershell
python -m build
```

This creates:
- `dist/mha_toolbox-2.0.0.tar.gz` (source distribution)
- `dist/mha_toolbox-2.0.0-py3-none-any.whl` (wheel distribution)

### Step 3: Test Installation Locally

```powershell
pip install dist/mha_toolbox-2.0.0-py3-none-any.whl
```

Or install in editable mode for development:

```powershell
pip install -e .
```

---

## 🌐 Method 2: Upload to PyPI

### Option A: Test on TestPyPI First (Recommended)

```powershell
# Upload to TestPyPI
python -m twine upload --repository testpypi dist/*

# Install from TestPyPI to test
pip install --index-url https://test.pypi.org/simple/ mha-toolbox
```

### Option B: Upload to PyPI (Production)

```powershell
# Upload to PyPI
python -m twine upload dist/*

# Anyone can now install with:
# pip install mha-toolbox
```

**Note**: You'll need PyPI credentials. Create an account at:
- TestPyPI: https://test.pypi.org/account/register/
- PyPI: https://pypi.org/account/register/

---

## 💻 Method 3: Local Installation

### Install from Source Directory

```powershell
# Navigate to project root
cd D:\MHA-Algorithm

# Install in editable mode (for development)
pip install -e .

# Or regular installation
pip install .
```

---

## 🔧 Method 4: Direct Usage (No Installation)

You can use the library directly without installation:

```powershell
# Just run the Streamlit interface
streamlit run mha_toolbox_pro_ultimate.py
```

Or use it in Python:

```python
# Add parent directory to path
import sys
sys.path.insert(0, 'D:/MHA-Algorithm')

from mha_toolbox.algorithms import PSO, GWO
```

---

## 📝 Quick Commands

### Build Package
```powershell
python -m build
```

### Test Locally
```powershell
pip install -e .
```

### Upload to TestPyPI
```powershell
python -m twine upload --repository testpypi dist/*
```

### Upload to PyPI
```powershell
python -m twine upload dist/*
```

### Run Web Interface
```powershell
.\launch.bat
# Or directly:
streamlit run mha_toolbox_pro_ultimate.py
```

---

## 🧪 Testing Your Build

After installation, test it:

```python
# Test import
from mha_toolbox import __version__
print(f"MHA Toolbox version: {__version__}")

# Test algorithm
from mha_toolbox.algorithms import PSO
import numpy as np

def sphere(x):
    return np.sum(x**2)

optimizer = PSO(30, 100)
best_pos, best_fit, _, _, _ = optimizer.optimize(sphere, (-100, 100), 10)
print(f"Best fitness: {best_fit}")
```

---

## 📂 What Gets Packaged

The build includes:
- ✅ All 95 algorithm files
- ✅ All 9 hybrid algorithms
- ✅ Base optimizer class
- ✅ Utility modules
- ✅ Documentation (README.md)
- ✅ License (LICENSE)
- ✅ Dependencies (requirements.txt)

---

## 🐛 Troubleshooting

### Issue: "Module not found"
**Solution**: Install in editable mode:
```powershell
pip install -e .
```

### Issue: "Permission denied" on PyPI upload
**Solution**: Create a PyPI token and use it:
```powershell
python -m twine upload --username __token__ --password YOUR_TOKEN dist/*
```

### Issue: Build fails
**Solution**: Ensure all dependencies are installed:
```powershell
pip install --upgrade build twine wheel setuptools
```

### Issue: "Package already exists" on PyPI
**Solution**: Increment version in `mha_toolbox/__init__.py`:
```python
__version__ = "2.0.1"  # Change this
```

---

## 🎯 Distribution Options

Choose the best option for your needs:

| Method | Best For | Command |
|--------|----------|---------|
| **Local Dev** | Development | `pip install -e .` |
| **Local Install** | Personal use | `pip install .` |
| **TestPyPI** | Testing | `twine upload --repository testpypi dist/*` |
| **PyPI** | Public release | `twine upload dist/*` |
| **Direct Use** | Quick demo | `streamlit run mha_toolbox_pro_ultimate.py` |

---

## 🎉 Success Checklist

- ✅ Build tools installed
- ✅ Package builds without errors
- ✅ Tests pass locally
- ✅ Documentation complete
- ✅ Version number correct
- ✅ License included
- ✅ README comprehensive

---

## 📧 Need Help?

- GitHub Issues: https://github.com/Achyut103040/MHA-Algorithm/issues
- Email: mha.toolbox@gmail.com

---

**Your library is ready for distribution!** 🚀
