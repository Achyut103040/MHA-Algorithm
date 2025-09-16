#!/usr/bin/env python3
"""
Script to update all MHA algorithms with automatic parameter calculation
"""

import os
import re

# Template for the automatic parameter calculation code
PARAM_CALC_TEMPLATE = '''    def _optimize(self, objective_function, X=None, y=None, **kwargs):
        """Algorithm optimization implementation with automatic parameter calculation"""
        # Automatically determine problem type and set bounds/dimensions
        if X is not None:
            # Feature selection problem
            self.dimensions = X.shape[1]
            self.lower_bound = np.zeros(self.dimensions)
            self.upper_bound = np.ones(self.dimensions)
        else:
            # Function optimization problem
            if not hasattr(self, 'dimensions') or self.dimensions is None:
                self.dimensions = kwargs.get('dimensions', 10)
            
            # Set bounds if not already set
            if not hasattr(self, 'lower_bound') or self.lower_bound is None:
                lb = kwargs.get('lower_bound', kwargs.get('lb', -10.0))
                self.lower_bound = np.full(self.dimensions, lb) if np.isscalar(lb) else np.array(lb)
            
            if not hasattr(self, 'upper_bound') or self.upper_bound is None:
                ub = kwargs.get('upper_bound', kwargs.get('ub', 10.0))
                self.upper_bound = np.full(self.dimensions, ub) if np.isscalar(ub) else np.array(ub)
        
        # Initialize population'''

def update_algorithm_file(filepath):
    """Update a single algorithm file with automatic parameter calculation"""
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        
        # Pattern to match the _optimize method definition
        pattern = r'def _optimize\(self, objective_function, \*\*kwargs\):(.*?)# Initialize population'
        
        # Replace with new pattern
        replacement = PARAM_CALC_TEMPLATE.replace('    def _optimize', 'def _optimize') + '\n        '
        
        new_content = re.sub(pattern, replacement, content, flags=re.DOTALL)
        
        # Also update the method signature if it's different
        old_signature = 'def _optimize(self, objective_function, **kwargs):'
        new_signature = 'def _optimize(self, objective_function, X=None, y=None, **kwargs):'
        new_content = new_content.replace(old_signature, new_signature)
        
        if new_content != content:
            with open(filepath, 'w') as f:
                f.write(new_content)
            print(f"✅ Updated: {os.path.basename(filepath)}")
        else:
            print(f"⚠️  No changes needed: {os.path.basename(filepath)}")
            
    except Exception as e:
        print(f"❌ Error updating {filepath}: {e}")

def main():
    """Update all algorithm files"""
    algorithms_dir = 'd:/MHA-Algorithm/mha_toolbox/algorithms'
    
    # Get all Python files except __init__.py
    algorithm_files = [f for f in os.listdir(algorithms_dir) 
                      if f.endswith('.py') and f != '__init__.py']
    
    print("🔄 Updating all algorithm files with automatic parameter calculation...")
    print("=" * 70)
    
    for filename in algorithm_files:
        filepath = os.path.join(algorithms_dir, filename)
        update_algorithm_file(filepath)
    
    print("=" * 70)
    print("✅ Algorithm update process completed!")

if __name__ == "__main__":
    main()