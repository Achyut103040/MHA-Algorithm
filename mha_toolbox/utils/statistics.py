"""
Statistical Analysis Module for MHA Toolbox
==========================================

This module provides comprehensive statistical analysis and comparison tools
for metaheuristic optimization results.
"""

import numpy as np
from scipy import stats
import pandas as pd

class StatisticalAnalyzer:
    """Statistical analysis tools for optimization results."""
    
    def __init__(self, results):
        """
        Initialize analyzer with optimization results.
        
        Parameters
        ----------
        results : list of OptimizationModel
            List of optimization results to analyze
        """
        self.results = results if isinstance(results, list) else [results]
    
    def performance_ranking(self):
        """
        Rank algorithms by performance metrics.
        
        Returns
        -------
        pandas.DataFrame
            Ranked performance table
        """
        data = []
        for result in self.results:
            algo_name = result.algorithm_name if hasattr(result, 'algorithm_name') else 'Unknown'
            data.append({
                'Algorithm': algo_name,
                'Best_Fitness': result.best_fitness,
                'Execution_Time': result.execution_time,
                'Convergence_Rate': np.mean(np.abs(np.diff(result.convergence_curve))),
                'Final_Improvement': (result.convergence_curve[0] - result.convergence_curve[-1]) / result.convergence_curve[0] if result.convergence_curve[0] != 0 else 0,
                'Iterations': len(result.convergence_curve)
            })
        
        df = pd.DataFrame(data)
        
        # Add rankings
        df['Fitness_Rank'] = df['Best_Fitness'].rank()
        df['Speed_Rank'] = df['Execution_Time'].rank()
        df['Improvement_Rank'] = df['Final_Improvement'].rank(ascending=False)
        df['Overall_Rank'] = (df['Fitness_Rank'] + df['Speed_Rank'] + df['Improvement_Rank']) / 3
        
        return df.sort_values('Overall_Rank')
    
    def statistical_tests(self):
        """
        Perform statistical significance tests between algorithms.
        
        Returns
        -------
        dict
            Statistical test results
        """
        if len(self.results) < 2:
            return {"message": "Need at least 2 results for statistical comparison"}
        
        tests = {}
        
        for i in range(len(self.results)):
            for j in range(i + 1, len(self.results)):
                algo1 = self.results[i].algorithm_name if hasattr(self.results[i], 'algorithm_name') else f'Algo_{i}'
                algo2 = self.results[j].algorithm_name if hasattr(self.results[j], 'algorithm_name') else f'Algo_{j}'
                
                conv1 = self.results[i].convergence_curve
                conv2 = self.results[j].convergence_curve
                
                # Wilcoxon rank-sum test
                try:
                    statistic, p_value = stats.ranksums(conv1, conv2)
                    tests[f'{algo1}_vs_{algo2}'] = {
                        'test': 'Wilcoxon rank-sum',
                        'statistic': statistic,
                        'p_value': p_value,
                        'significant': p_value < 0.05,
                        'interpretation': 'Significantly different' if p_value < 0.05 else 'No significant difference'
                    }
                except Exception as e:
                    tests[f'{algo1}_vs_{algo2}'] = {
                        'error': str(e)
                    }
        
        return tests
    
    def convergence_analysis(self):
        """
        Analyze convergence patterns across algorithms.
        
        Returns
        -------
        dict
            Convergence analysis results
        """
        analysis = {}
        
        for result in self.results:
            algo_name = result.algorithm_name if hasattr(result, 'algorithm_name') else 'Unknown'
            convergence = np.array(result.convergence_curve)
            
            # Convergence speed analysis
            if len(convergence) > 10:
                # Find when algorithm reaches 90% of final improvement
                total_improvement = convergence[0] - convergence[-1]
                target_fitness = convergence[0] - 0.9 * total_improvement
                
                convergence_90_iter = None
                for i, fitness in enumerate(convergence):
                    if fitness <= target_fitness:
                        convergence_90_iter = i
                        break
                
                # Stagnation analysis
                improvements = np.abs(np.diff(convergence))
                stagnation_threshold = 0.001 * np.mean(improvements) if np.mean(improvements) > 0 else 1e-6
                stagnant_iterations = np.sum(improvements < stagnation_threshold)
                
                # Premature convergence detection
                last_quarter = len(convergence) // 4
                premature_convergence = np.std(convergence[-last_quarter:]) < 0.01 * np.std(convergence)
                
                analysis[algo_name] = {
                    'convergence_90_percent': convergence_90_iter,
                    'stagnation_ratio': stagnant_iterations / len(improvements) if len(improvements) > 0 else 0,
                    'premature_convergence': premature_convergence,
                    'convergence_smoothness': 1.0 / (1.0 + np.std(improvements)),
                    'exploration_phases': self._identify_exploration_phases(convergence),
                    'exploitation_intensity': self._calculate_exploitation_intensity(convergence)
                }
        
        return analysis
    
    def efficiency_metrics(self):
        """
        Calculate efficiency metrics for each algorithm.
        
        Returns
        -------
        pandas.DataFrame
            Efficiency metrics table
        """
        data = []
        
        for result in self.results:
            algo_name = result.algorithm_name if hasattr(result, 'algorithm_name') else 'Unknown'
            convergence = np.array(result.convergence_curve)
            
            # Calculate various efficiency metrics
            total_improvement = convergence[0] - convergence[-1] if convergence[0] != convergence[-1] else 1e-10
            
            metrics = {
                'Algorithm': algo_name,
                'Fitness_per_Second': abs(total_improvement) / result.execution_time if result.execution_time > 0 else 0,
                'Improvement_Rate': total_improvement / len(convergence) if len(convergence) > 0 else 0,
                'Efficiency_Score': abs(total_improvement) / (result.execution_time * len(convergence)) if result.execution_time > 0 and len(convergence) > 0 else 0,
                'Robustness_Score': 1.0 / (1.0 + np.std(convergence) / abs(np.mean(convergence))) if np.mean(convergence) != 0 else 0,
                'Speed_Score': 1.0 / result.execution_time if result.execution_time > 0 else 0
            }
            
            data.append(metrics)
        
        return pd.DataFrame(data)
    
    def generate_report(self):
        """
        Generate a comprehensive statistical analysis report.
        
        Returns
        -------
        str
            Formatted analysis report
        """
        report = []
        report.append("="*80)
        report.append("COMPREHENSIVE STATISTICAL ANALYSIS REPORT")
        report.append("="*80)
        
        # Performance ranking
        ranking = self.performance_ranking()
        report.append("\n📊 PERFORMANCE RANKING:")
        report.append("-" * 40)
        report.append(ranking.to_string(index=False))
        
        # Statistical tests
        tests = self.statistical_tests()
        report.append("\n🔬 STATISTICAL SIGNIFICANCE TESTS:")
        report.append("-" * 40)
        for comparison, test_result in tests.items():
            if 'error' not in test_result:
                report.append(f"{comparison}:")
                report.append(f"  p-value: {test_result['p_value']:.6f}")
                report.append(f"  Result: {test_result['interpretation']}")
            else:
                report.append(f"{comparison}: Error - {test_result['error']}")
        
        # Convergence analysis
        conv_analysis = self.convergence_analysis()
        report.append("\n📈 CONVERGENCE ANALYSIS:")
        report.append("-" * 40)
        for algo, analysis in conv_analysis.items():
            report.append(f"{algo}:")
            for metric, value in analysis.items():
                if isinstance(value, (int, float)):
                    report.append(f"  {metric}: {value:.4f}")
                else:
                    report.append(f"  {metric}: {value}")
        
        # Efficiency metrics
        efficiency = self.efficiency_metrics()
        report.append("\n⚡ EFFICIENCY METRICS:")
        report.append("-" * 40)
        report.append(efficiency.to_string(index=False))
        
        report.append("\n" + "="*80)
        
        return "\n".join(report)
    
    def _identify_exploration_phases(self, convergence):
        """Identify exploration vs exploitation phases."""
        if len(convergence) < 10:
            return "Insufficient data"
        
        # Calculate moving standard deviation as exploration indicator
        window_size = max(5, len(convergence) // 10)
        exploration_indicator = []
        
        for i in range(window_size, len(convergence)):
            window = convergence[i-window_size:i]
            exploration_indicator.append(np.std(window))
        
        # Phases where std is above median are considered exploration
        if exploration_indicator:
            median_std = np.median(exploration_indicator)
            exploration_phases = np.sum(np.array(exploration_indicator) > median_std)
            return exploration_phases / len(exploration_indicator)
        
        return 0
    
    def _calculate_exploitation_intensity(self, convergence):
        """Calculate average exploitation intensity."""
        if len(convergence) < 5:
            return 0
        
        intensities = []
        for i in range(5, len(convergence)):
            recent_improvement = abs(convergence[i] - convergence[i-5])
            intensity = 1.0 / (1.0 + recent_improvement + 1e-10)
            intensities.append(intensity)
        
        return np.mean(intensities) if intensities else 0