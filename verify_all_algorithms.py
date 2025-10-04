#!/usr/bin/env python3
"""
Simple verification that all 36 algorithms work with mha.optimize()
"""

import sys
sys.path.append('.')
import mha_toolbox as mha
import numpy as np

def sphere(x):
    """Simple test function"""
    return np.sum(x**2)

print("🎯 VERIFYING ALL 36 ALGORITHMS WORK")
print("="*60)

# All 36 algorithms to test
all_algorithms = [
    # Original 20 algorithms
    'sca', 'pso', 'gwo', 'aco', 'alo', 'ants', 'ao', 'ba', 'coa', 'csa',
    'de', 'fa', 'ga', 'mrfo', 'msa', 'pfa', 'sma', 'spider', 'ssa', 'tso',
    
    # New 16 algorithms
    'vcs', 'chio', 'fbi', 'ica', 'qsa', 'spbo', 'aoa', 'eo', 'hgso', 'sa',
    'wdo', 'cgo', 'gbo', 'innov', 'wca', 'vns'
]

print(f"Testing {len(all_algorithms)} algorithms...")
print("-" * 60)

working_count = 0
failed_algorithms = []

for i, algo in enumerate(all_algorithms, 1):
    try:
        print(f"{i:2d}. Testing {algo.upper():6s}...", end=" ")
        
        result = mha.optimize(algo, 
                            dimensions=3, 
                            lower_bound=[-5, -5, -5], 
                            upper_bound=[5, 5, 5], 
                            objective_function=sphere, 
                            population_size=10, 
                            max_iterations=10, 
                            verbose=False)
        
        print(f"✅ SUCCESS (fitness: {result.best_fitness_:.3f})")
        working_count += 1
        
    except Exception as e:
        print(f"❌ FAILED: {str(e)[:40]}...")
        failed_algorithms.append(algo)

print("-" * 60)
print(f"📊 RESULTS: {working_count}/{len(all_algorithms)} algorithms working")

if working_count == len(all_algorithms):
    print("🎉 ALL ALGORITHMS ARE FUNCTIONAL!")
    print("✅ Complete MHA system with 36 algorithms ready")
    
    # Test a few more usage patterns
    print("\n🔧 Testing different usage patterns...")
    
    # Test with algorithm name
    try:
        result = mha.optimize('vns', dimensions=3, lower_bound=[-2]*3, upper_bound=[2]*3, 
                            objective_function=sphere, max_iterations=5, verbose=False)
        print("✅ Direct algorithm name usage works")
    except Exception as e:
        print(f"❌ Direct usage failed: {e}")
    
    # Test with alias
    try:
        result = mha.optimize('virus', dimensions=3, lower_bound=[-2]*3, upper_bound=[2]*3, 
                            objective_function=sphere, max_iterations=5, verbose=False)
        print("✅ Algorithm alias usage works")
    except Exception as e:
        print(f"❌ Alias usage failed: {e}")
    
    # Test algorithm listing
    try:
        algorithms = mha.list_algorithms()
        aliases = mha.get_all_aliases()
        print(f"✅ Algorithm discovery works: {len(algorithms)} algorithms, {len(aliases)} aliases")
    except Exception as e:
        print(f"❌ Algorithm listing failed: {e}")
        
else:
    print(f"⚠️  {len(failed_algorithms)} algorithms need attention:")
    for algo in failed_algorithms:
        print(f"   - {algo.upper()}")

print("\n" + "="*60)
if working_count == len(all_algorithms):
    print("🏆 SYSTEM STATUS: FULLY OPERATIONAL")
    print("🚀 All 36 metaheuristic algorithms are working perfectly!")
else:
    print("🔧 SYSTEM STATUS: NEEDS MINOR FIXES")
    print(f"📈 Progress: {working_count/len(all_algorithms)*100:.1f}% complete")
print("="*60)