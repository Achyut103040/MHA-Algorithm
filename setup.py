"""
Setup script for MHA Toolbox - Professional Metaheuristic Algorithm Library
"""

from setuptools import setup, find_packages
import os

# Read long description from README
def read_long_description():
    """Read long description from README file."""
    try:
        with open("README.md", "r", encoding="utf-8") as fh:
            return fh.read()
    except FileNotFoundError:
        return "MHA Toolbox - Professional Metaheuristic Algorithm Library"

# Read version from __init__.py
def get_version():
    """Extract version from __init__.py"""
    with open("mha_toolbox/__init__.py", "r") as f:
        for line in f:
            if line.startswith("__version__"):
                return line.split("=")[1].strip().strip('"').strip("'")
    return "1.0.0"

# Define requirements
install_requires = [
    "numpy>=1.19.0",
    "pandas>=1.2.0",
    "matplotlib>=3.3.0",
    "seaborn>=0.11.0",
    "scikit-learn>=0.24.0",
    "scipy>=1.6.0",
    "joblib>=1.0.0",
]

# Optional dependencies for advanced features
extras_require = {
    "web": [
        "streamlit>=1.25.0",
        "plotly>=5.0.0",
        "dash>=2.0.0",
    ],
    "jupyter": [
        "jupyter>=1.0.0",
        "ipython>=7.0.0",
        "notebook>=6.0.0",
    ],
    "advanced": [
        "tensorflow>=2.6.0",
        "torch>=1.9.0",
        "optuna>=2.10.0",
        "hyperopt>=0.2.5",
    ],
    "dev": [
        "pytest>=6.0.0",
        "pytest-cov>=2.10.0",
        "black>=21.0.0",
        "flake8>=3.8.0",
        "sphinx>=4.0.0",
        "sphinx-rtd-theme>=1.0.0",
    ],
    "full": [
        "streamlit>=1.25.0",
        "plotly>=5.0.0",
        "dash>=2.0.0",
        "jupyter>=1.0.0",
        "ipython>=7.0.0",
        "notebook>=6.0.0",
        "tensorflow>=2.6.0",
        "torch>=1.9.0",
        "optuna>=2.10.0",
        "hyperopt>=0.2.5",
    ]
}

# All optional dependencies
extras_require["all"] = list(set(sum(extras_require.values(), [])))

setup(
    name="mha-toolbox",
    version=get_version(),
    author="MHA Development Team",
    author_email="mha.toolbox@gmail.com",
    description="Professional Metaheuristic Algorithm Library with 36+ algorithms, hybrid combinations, and comprehensive analysis tools",
    long_description=read_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/Achyut103040/MHA-Algorithm",
    project_urls={
        "Bug Tracker": "https://github.com/Achyut103040/MHA-Algorithm/issues",
        "Documentation": "https://github.com/Achyut103040/MHA-Algorithm/wiki",
        "Source Code": "https://github.com/Achyut103040/MHA-Algorithm",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=install_requires,
    extras_require=extras_require,
    include_package_data=True,
    package_data={
        "mha_toolbox": [
            "data/*.csv",
            "data/*.json",
            "templates/*.html",
            "static/*.css",
            "static/*.js",
        ],
    },
    entry_points={
        "console_scripts": [
            "mha-toolbox=mha_toolbox.cli:main",
            "mha-web=mha_toolbox.launcher:launch_web_interface",
            "mha-demo=mha_toolbox.launcher:run_demo_system",
        ],
    },
    keywords=[
        "metaheuristic", "optimization", "evolutionary-algorithm", 
        "swarm-intelligence", "feature-selection", "machine-learning",
        "artificial-intelligence", "bio-inspired", "physics-based",
        "hybrid-algorithms", "pso", "gwo", "sca", "woa", "ga", "de"
    ],
    zip_safe=False,
)