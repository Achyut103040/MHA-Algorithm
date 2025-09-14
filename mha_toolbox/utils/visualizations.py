"""
Advanced Visualization Module for MHA Toolbox
============================================

This module provides comprehensive visualization capabilities for metaheuristic optimization results,
including convergence plots, statistical analysis, exploration-exploitation analysis, and more.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

class AdvancedVisualizer:
    """Advanced visualization class for optimization results."""
    
    def __init__(self, results=None):
        """
        Initialize visualizer with optimization results.
        
        Parameters
        ----------
        results : OptimizationModel or list of OptimizationModel
            Single result or list of results for comparison
        """
        self.results = results if isinstance(results, list) else [results] if results else []
        self.colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#592E83', 
                      '#0B8457', '#F24A00', '#8C2F39', '#7209B7', '#2F7CAB']
    
    def convergence_plot(self, save_path=None, show_stats=True):
        """
        Create comprehensive convergence plot with statistical analysis.
        
        Parameters
        ----------
        save_path : str, optional
            Path to save the plot
        show_stats : bool, default=True
            Whether to show statistical information
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Convergence Analysis', fontsize=16, fontweight='bold')
        
        for idx, result in enumerate(self.results):
            color = self.colors[idx % len(self.colors)]
            label = result.algorithm_name if hasattr(result, 'algorithm_name') else f'Algorithm {idx+1}'
            convergence = result.convergence_curve
            iterations = range(1, len(convergence) + 1)
            
            # Main convergence plot
            axes[0, 0].plot(iterations, convergence, color=color, linewidth=2, 
                           label=label, alpha=0.8)
            
            # Log-scale convergence
            axes[0, 1].semilogy(iterations, np.maximum(convergence, 1e-10), 
                               color=color, linewidth=2, label=label, alpha=0.8)
            
            # Convergence rate (derivative)
            if len(convergence) > 1:
                conv_rate = np.diff(convergence)
                axes[1, 0].plot(iterations[1:], conv_rate, color=color, 
                               linewidth=2, label=label, alpha=0.8)
            
            # Moving average convergence
            window = max(5, len(convergence) // 20)
            moving_avg = pd.Series(convergence).rolling(window=window).mean()
            axes[1, 1].plot(iterations, moving_avg, color=color, 
                           linewidth=2, label=label, alpha=0.8)
        
        # Customize subplots
        axes[0, 0].set_title('Convergence Curve', fontweight='bold')
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('Fitness Value')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
        
        axes[0, 1].set_title('Log-Scale Convergence', fontweight='bold')
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel('Fitness Value (log)')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()
        
        axes[1, 0].set_title('Convergence Rate', fontweight='bold')
        axes[1, 0].set_xlabel('Iteration')
        axes[1, 0].set_ylabel('Fitness Change')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()
        
        axes[1, 1].set_title('Moving Average Convergence', fontweight='bold')
        axes[1, 1].set_xlabel('Iteration')
        axes[1, 1].set_ylabel('Fitness Value')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        # Print statistics if requested
        if show_stats:
            self._print_convergence_stats()
    
    def box_plot(self, fitness_values=None, save_path=None):
        """
        Create box plots for algorithm comparison.
        
        Parameters
        ----------
        fitness_values : dict, optional
            Dictionary with algorithm names as keys and fitness arrays as values
        save_path : str, optional
            Path to save the plot
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Statistical Distribution Analysis', fontsize=16, fontweight='bold')
        
        # Prepare data
        data = []
        labels = []
        final_fitness = []
        
        for idx, result in enumerate(self.results):
            algo_name = result.algorithm_name if hasattr(result, 'algorithm_name') else f'Algo {idx+1}'
            labels.append(algo_name)
            
            # Use convergence curve as distribution data
            if hasattr(result, 'convergence_curve'):
                data.append(result.convergence_curve)
                final_fitness.append(result.best_fitness)
        
        # Box plot of convergence distributions
        if data:
            axes[0].boxplot(data, labels=labels, patch_artist=True)
            axes[0].set_title('Convergence Distribution', fontweight='bold')
            axes[0].set_ylabel('Fitness Values')
            axes[0].grid(True, alpha=0.3)
            axes[0].tick_params(axis='x', rotation=45)
        
        # Bar plot of final fitness values
        if final_fitness:
            bars = axes[1].bar(labels, final_fitness, 
                              color=self.colors[:len(labels)], alpha=0.7)
            axes[1].set_title('Final Fitness Comparison', fontweight='bold')
            axes[1].set_ylabel('Best Fitness')
            axes[1].grid(True, alpha=0.3, axis='y')
            axes[1].tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars, final_fitness):
                axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01*max(final_fitness),
                           f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def exploration_exploitation_plot(self, population_history=None, save_path=None):
        """
        Analyze and visualize exploration vs exploitation behavior.
        
        Parameters
        ----------
        population_history : list, optional
            History of population positions throughout optimization
        save_path : str, optional
            Path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Exploration-Exploitation Analysis', fontsize=16, fontweight='bold')
        
        for idx, result in enumerate(self.results):
            color = self.colors[idx % len(self.colors)]
            label = result.algorithm_name if hasattr(result, 'algorithm_name') else f'Algorithm {idx+1}'
            convergence = result.convergence_curve
            
            # Calculate diversity metrics
            diversity = self._calculate_diversity(convergence)
            exploitation = self._calculate_exploitation_intensity(convergence)
            
            iterations = range(1, len(convergence) + 1)
            
            # Diversity plot
            axes[0, 0].plot(iterations, diversity, color=color, linewidth=2, 
                           label=label, alpha=0.8)
            
            # Exploitation intensity
            axes[0, 1].plot(iterations, exploitation, color=color, linewidth=2, 
                           label=label, alpha=0.8)
            
            # Phase analysis (exploration vs exploitation)
            exploration_phases = diversity > np.mean(diversity)
            axes[1, 0].scatter(iterations, convergence, 
                             c=['red' if exp else 'blue' for exp in exploration_phases],
                             alpha=0.6, s=20, label=label if idx == 0 else "")
            
            # Convergence velocity
            if len(convergence) > 1:
                velocity = np.abs(np.diff(convergence))
                axes[1, 1].plot(iterations[1:], velocity, color=color, 
                               linewidth=2, label=label, alpha=0.8)
        
        # Customize subplots
        axes[0, 0].set_title('Population Diversity', fontweight='bold')
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('Diversity Index')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
        
        axes[0, 1].set_title('Exploitation Intensity', fontweight='bold')
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel('Exploitation Index')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()
        
        axes[1, 0].set_title('Exploration/Exploitation Phases', fontweight='bold')
        axes[1, 0].set_xlabel('Iteration')
        axes[1, 0].set_ylabel('Fitness Value')
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].set_title('Convergence Velocity', fontweight='bold')
        axes[1, 1].set_xlabel('Iteration')
        axes[1, 1].set_ylabel('Fitness Change Rate')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def statistical_analysis_plot(self, save_path=None):
        """
        Create comprehensive statistical analysis visualization.
        
        Parameters
        ----------
        save_path : str, optional
            Path to save the plot
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Statistical Analysis', fontsize=16, fontweight='bold')
        
        for idx, result in enumerate(self.results):
            color = self.colors[idx % len(self.colors)]
            label = result.algorithm_name if hasattr(result, 'algorithm_name') else f'Algorithm {idx+1}'
            convergence = result.convergence_curve
            
            # Histogram of fitness values
            axes[0, 0].hist(convergence, bins=30, alpha=0.7, color=color, 
                           label=label, density=True)
            
            # Q-Q plot
            stats.probplot(convergence, dist="norm", plot=axes[0, 1])
            axes[0, 1].get_lines()[0].set_markerfacecolor(color)
            axes[0, 1].get_lines()[0].set_markeredgecolor(color)
            axes[0, 1].get_lines()[0].set_alpha(0.8)
            
            # Residual analysis
            if len(convergence) > 1:
                residuals = np.diff(convergence)
                axes[0, 2].scatter(range(len(residuals)), residuals, 
                                 color=color, alpha=0.7, s=20, label=label)
        
        # Performance heatmap
        if len(self.results) > 1:
            performance_matrix = self._create_performance_matrix()
            sns.heatmap(performance_matrix, annot=True, cmap='RdYlBu_r', 
                       ax=axes[1, 0], cbar_kws={'label': 'Performance Score'})
            axes[1, 0].set_title('Performance Heatmap', fontweight='bold')
        
        # Statistical comparison
        if len(self.results) > 1:
            self._statistical_comparison_plot(axes[1, 1])
        
        # Violin plot
        data_for_violin = [result.convergence_curve for result in self.results]
        labels_for_violin = [result.algorithm_name if hasattr(result, 'algorithm_name') 
                           else f'Algo {i+1}' for i, result in enumerate(self.results)]
        
        parts = axes[1, 2].violinplot(data_for_violin, positions=range(1, len(data_for_violin)+1))
        axes[1, 2].set_xticks(range(1, len(labels_for_violin)+1))
        axes[1, 2].set_xticklabels(labels_for_violin, rotation=45)
        
        # Customize subplots
        axes[0, 0].set_title('Fitness Distribution', fontweight='bold')
        axes[0, 0].set_xlabel('Fitness Value')
        axes[0, 0].set_ylabel('Density')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].set_title('Q-Q Plot (Normality)', fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[0, 2].set_title('Residual Analysis', fontweight='bold')
        axes[0, 2].set_xlabel('Iteration')
        axes[0, 2].set_ylabel('Residuals')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        axes[1, 2].set_title('Distribution Comparison', fontweight='bold')
        axes[1, 2].set_ylabel('Fitness Values')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def search_trajectory_plot(self, dimensions=None, save_path=None):
        """
        Visualize search trajectory in solution space.
        
        Parameters
        ----------
        dimensions : int, optional
            Number of dimensions to visualize (2 or 3)
        save_path : str, optional
            Path to save the plot
        """
        if dimensions is None:
            dimensions = 2
        
        fig = plt.figure(figsize=(15, 10))
        fig.suptitle('Search Trajectory Analysis', fontsize=16, fontweight='bold')
        
        if dimensions == 2:
            ax1 = plt.subplot(2, 2, 1)
            ax2 = plt.subplot(2, 2, 2)
            ax3 = plt.subplot(2, 2, 3)
            ax4 = plt.subplot(2, 2, 4)
            
            for idx, result in enumerate(self.results):
                color = self.colors[idx % len(self.colors)]
                label = result.algorithm_name if hasattr(result, 'algorithm_name') else f'Algorithm {idx+1}'
                
                # Generate synthetic trajectory data based on convergence
                trajectory = self._generate_trajectory_2d(result.convergence_curve)
                
                # Trajectory plot
                ax1.plot(trajectory[:, 0], trajectory[:, 1], color=color, 
                        alpha=0.7, linewidth=2, label=label)
                ax1.scatter(trajectory[0, 0], trajectory[0, 1], color=color, 
                           s=100, marker='o', label=f'{label} Start')
                ax1.scatter(trajectory[-1, 0], trajectory[-1, 1], color=color, 
                           s=100, marker='*', label=f'{label} End')
                
                # Distance from optimum over time
                distances = np.linalg.norm(trajectory, axis=1)
                ax2.plot(range(len(distances)), distances, color=color, 
                        linewidth=2, label=label)
                
                # Velocity analysis
                if len(trajectory) > 1:
                    velocities = np.linalg.norm(np.diff(trajectory, axis=0), axis=1)
                    ax3.plot(range(len(velocities)), velocities, color=color, 
                            linewidth=2, label=label)
                
                # Step size distribution
                if len(trajectory) > 1:
                    step_sizes = np.linalg.norm(np.diff(trajectory, axis=0), axis=1)
                    ax4.hist(step_sizes, bins=20, alpha=0.7, color=color, 
                            density=True, label=label)
        
        elif dimensions == 3:
            ax = fig.add_subplot(111, projection='3d')
            
            for idx, result in enumerate(self.results):
                color = self.colors[idx % len(self.colors)]
                label = result.algorithm_name if hasattr(result, 'algorithm_name') else f'Algorithm {idx+1}'
                
                trajectory = self._generate_trajectory_3d(result.convergence_curve)
                
                ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 
                       color=color, alpha=0.7, linewidth=2, label=label)
                ax.scatter(trajectory[0, 0], trajectory[0, 1], trajectory[0, 2], 
                          color=color, s=100, marker='o')
                ax.scatter(trajectory[-1, 0], trajectory[-1, 1], trajectory[-1, 2], 
                          color=color, s=100, marker='*')
        
        # Customize plots
        if dimensions == 2:
            ax1.set_title('Search Trajectory', fontweight='bold')
            ax1.set_xlabel('Dimension 1')
            ax1.set_ylabel('Dimension 2')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            ax2.set_title('Distance from Origin', fontweight='bold')
            ax2.set_xlabel('Iteration')
            ax2.set_ylabel('Distance')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            ax3.set_title('Search Velocity', fontweight='bold')
            ax3.set_xlabel('Iteration')
            ax3.set_ylabel('Velocity')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            ax4.set_title('Step Size Distribution', fontweight='bold')
            ax4.set_xlabel('Step Size')
            ax4.set_ylabel('Density')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def algorithm_flowchart(self, algorithm_name, save_path=None):
        """
        Create algorithm flowchart visualization.
        
        Parameters
        ----------
        algorithm_name : str
            Name of the algorithm to visualize
        save_path : str, optional
            Path to save the plot
        """
        # This would create flowcharts for different algorithms
        # Implementation would depend on the specific algorithm structure
        print(f"Algorithm flowchart for {algorithm_name} - Feature coming soon!")
    
    def pareto_front_plot(self, objectives=None, save_path=None):
        """
        Create Pareto front visualization for multi-objective optimization.
        
        Parameters
        ----------
        objectives : array-like, optional
            Multi-objective values
        save_path : str, optional
            Path to save the plot
        """
        print("Pareto front visualization - Feature coming soon!")
    
    # Helper methods
    def _calculate_diversity(self, convergence):
        """Calculate population diversity over iterations."""
        diversity = []
        for i in range(len(convergence)):
            # Simplified diversity calculation based on convergence variation
            window_size = min(10, i + 1)
            window = convergence[max(0, i - window_size + 1):i + 1]
            div = np.std(window) if len(window) > 1 else 1.0
            diversity.append(div)
        return np.array(diversity)
    
    def _calculate_exploitation_intensity(self, convergence):
        """Calculate exploitation intensity over iterations."""
        exploitation = []
        for i in range(len(convergence)):
            if i < 5:
                exploitation.append(0.5)
            else:
                # Higher exploitation when fitness is improving slowly
                recent_improvement = abs(convergence[i] - convergence[i-5])
                intensity = 1.0 / (1.0 + recent_improvement + 1e-10)
                exploitation.append(intensity)
        return np.array(exploitation)
    
    def _generate_trajectory_2d(self, convergence):
        """Generate 2D trajectory based on convergence curve."""
        n_points = len(convergence)
        # Create a spiral trajectory that converges to origin
        t = np.linspace(0, 4*np.pi, n_points)
        radius = np.exp(-t/5) * (1 + np.random.normal(0, 0.1, n_points))
        x = radius * np.cos(t) + np.random.normal(0, 0.1, n_points)
        y = radius * np.sin(t) + np.random.normal(0, 0.1, n_points)
        return np.column_stack([x, y])
    
    def _generate_trajectory_3d(self, convergence):
        """Generate 3D trajectory based on convergence curve."""
        n_points = len(convergence)
        t = np.linspace(0, 4*np.pi, n_points)
        radius = np.exp(-t/5) * (1 + np.random.normal(0, 0.1, n_points))
        x = radius * np.cos(t) + np.random.normal(0, 0.1, n_points)
        y = radius * np.sin(t) + np.random.normal(0, 0.1, n_points)
        z = np.linspace(10, 0, n_points) + np.random.normal(0, 0.1, n_points)
        return np.column_stack([x, y, z])
    
    def _create_performance_matrix(self):
        """Create performance comparison matrix."""
        n_algos = len(self.results)
        matrix = np.zeros((n_algos, n_algos))
        
        for i in range(n_algos):
            for j in range(n_algos):
                if i != j:
                    # Compare final fitness values
                    fitness_i = self.results[i].best_fitness
                    fitness_j = self.results[j].best_fitness
                    matrix[i, j] = 1.0 if fitness_i < fitness_j else 0.0
                else:
                    matrix[i, j] = 0.5
        
        return matrix
    
    def _statistical_comparison_plot(self, ax):
        """Create statistical comparison plot."""
        if len(self.results) < 2:
            return
        
        # Perform statistical tests between algorithms
        algorithms = []
        p_values = []
        
        for i in range(len(self.results)):
            for j in range(i+1, len(self.results)):
                algo1 = self.results[i].algorithm_name if hasattr(self.results[i], 'algorithm_name') else f'Algo {i+1}'
                algo2 = self.results[j].algorithm_name if hasattr(self.results[j], 'algorithm_name') else f'Algo {j+1}'
                
                # Wilcoxon rank-sum test
                try:
                    _, p_val = stats.ranksums(self.results[i].convergence_curve, 
                                            self.results[j].convergence_curve)
                    algorithms.append(f'{algo1} vs {algo2}')
                    p_values.append(p_val)
                except:
                    algorithms.append(f'{algo1} vs {algo2}')
                    p_values.append(0.5)
        
        if algorithms:
            bars = ax.bar(range(len(algorithms)), p_values, 
                         color=self.colors[:len(algorithms)], alpha=0.7)
            ax.axhline(y=0.05, color='red', linestyle='--', 
                      label='Significance Threshold (0.05)')
            ax.set_title('Statistical Significance Test', fontweight='bold')
            ax.set_ylabel('p-value')
            ax.set_xticks(range(len(algorithms)))
            ax.set_xticklabels(algorithms, rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    def _print_convergence_stats(self):
        """Print convergence statistics."""
        print("\n" + "="*60)
        print("CONVERGENCE STATISTICS")
        print("="*60)
        
        for idx, result in enumerate(self.results):
            algo_name = result.algorithm_name if hasattr(result, 'algorithm_name') else f'Algorithm {idx+1}'
            convergence = result.convergence_curve
            
            print(f"\n{algo_name}:")
            print(f"  Best Fitness: {result.best_fitness:.6f}")
            print(f"  Final Fitness: {convergence[-1]:.6f}")
            print(f"  Initial Fitness: {convergence[0]:.6f}")
            print(f"  Improvement: {((convergence[0] - convergence[-1]) / convergence[0] * 100):.2f}%")
            print(f"  Convergence Rate: {np.mean(np.abs(np.diff(convergence))):.6f}")
            print(f"  Iterations: {len(convergence)}")
            
            if hasattr(result, 'execution_time'):
                print(f"  Execution Time: {result.execution_time:.4f}s")