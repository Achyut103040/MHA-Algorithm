"""
Quick SCA Algorithm Test
"""

from mha_toolbox import MHAToolbox
from objective_functions.benchmark_functions import BenchmarkFunctions

def quick_test():
    """Quick algorithm performance test"""
    print("🔍 Quick SCA Test")
    print("-" * 30)
    
    toolbox = MHAToolbox()
    bf = BenchmarkFunctions()
    
    # Test sphere function (should get close to 0)
    result = toolbox.optimize(
        algorithm_name='SCA',
        objective_function=bf.sphere,
        pop_size=30,
        max_iter=100,
        dim=5,
        lb=-5,
        ub=5
    )
    
    print(f"Result: {result['best_score']:.6f}")
    print(f"Position: {result['best_position']}")
    
    if result['best_score'] < 0.01:
        print("✅ Algorithm working well!")
    elif result['best_score'] < 1.0:
        print("⚠️ Algorithm working moderately")
    else:
        print("❌ Algorithm needs tuning")

if __name__ == "__main__":
    quick_test()
