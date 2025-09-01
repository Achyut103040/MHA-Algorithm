"""
MHA Toolbox Setup Script
"""

import os
import sys
import shutil
from pathlib import Path

def create_directory_structure():
    """Create required directories"""
    directories = [
        'toolbox_algorithms',
        'original_codes', 
        'objective_functions',
        'test_cases',
        'utils',
        'examples',
        'documentation'
    ]
    
    print("📁 Creating directory structure...")
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created/verified: {directory}/")
    print("✅ Directory structure ready!")

def backup_original_codes():
    """Backup original codes"""
    print("\n💾 Backing up original codes...")
    
    if os.path.exists('Code'):
        if os.path.exists('original_codes/Code'):
            print("⚠️  Original codes already backed up")
        else:
            try:
                shutil.copytree('Code', 'original_codes/Code')
                print("✓ Original codes backed up to original_codes/Code/")
            except Exception as e:
                print(f"❌ Failed to backup codes: {e}")
    else:
        print("⚠️  Code/ directory not found - skipping backup")

def verify_installation():
    """Verify required components"""
    print("\n🔍 Verifying installation...")
    
    required_files = [
        'mha_toolbox.py',
        'utils/toolbox_utils.py',
        'objective_functions/benchmark_functions.py',
        'toolbox_algorithms/SCA.py',
        'README.md'
    ]
    
    missing_files = []
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✓ Found: {file_path}")
        else:
            print(f"❌ Missing: {file_path}")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\n⚠️  {len(missing_files)} files are missing!")
        return False
    else:
        print("\n✅ All required files present!")
        return True

def test_imports():
    """Test all modules import correctly"""
    print("\n🧪 Testing imports...")
    
    try:
        import numpy as np
        print("✓ numpy imported")
        
        sys.path.insert(0, os.getcwd())
        
        from utils.toolbox_utils import handle_bounds
        print("✓ toolbox_utils imported")
        
        from objective_functions.benchmark_functions import BENCHMARK_FUNCTIONS
        print("✓ benchmark_functions imported")
        
        from toolbox_algorithms.SCA import SCA
        print("✓ SCA algorithm imported")
        
        from mha_toolbox import MHAToolbox
        print("✓ MHAToolbox imported")
        
        print("\n✅ All imports successful!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def run_quick_test():
    """Run quick functionality test"""
    print("\n🚀 Running quick test...")
    
    try:
        from mha_toolbox import MHAToolbox
        import numpy as np
        
        toolbox = MHAToolbox()
        
        def test_func(x):
            return np.sum(x**2)
        
        result = toolbox.optimize(
            algorithm_name='SCA',
            objective_function=test_func,
            pop_size=10,
            max_iter=50,
            dim=5,
            lb=-10,
            ub=10
        )
        
        print(f"✓ Test completed! Best score: {result['best_score']:.6f}")
        print("✅ Toolbox working correctly!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def show_next_steps():
    """Show user what to do next"""
    print("\n" + "="*50)
    print("🎉 SETUP COMPLETE!")
    print("="*50)
    
    print("\n Quick Start:")
    print("```python")
    print("from mha_toolbox import MHAToolbox")
    print("toolbox = MHAToolbox()")
    print("result = toolbox.optimize('SCA', 'sphere', pop_size=30, max_iter=200, dim=10)")
    print("```")
    
    print("\n📚 Available:")
    print("- Algorithm: SCA")
    print("- Functions: sphere, rastrigin, ackley, rosenbrock, griewank")
    print("- Documentation: README.md")

def main():
    """Main setup function"""
    print("🚀 MHA Toolbox Setup Script")
    print("="*50)
    
    create_directory_structure()
    backup_original_codes()
    
    if not verify_installation():
        print("\n❌ Setup incomplete - missing files")
        return False
    
    if not test_imports():
        print("\n❌ Setup incomplete - import errors")
        return False
    
    if not run_quick_test():
        print("\n❌ Setup incomplete - test failed")
        return False
    
    show_next_steps()
    return True

if __name__ == "__main__":
    success = main()
    
    if success:
        print("\n🎊 Setup completed successfully!")
        sys.exit(0)
    else:
        print("\n💥 Setup failed.")
        sys.exit(1)
