"""
Hybrid Metaheuristic Algorithms Package
======================================

This package contains hybrid algorithms that combine multiple optimization strategies
for enhanced performance and robustness.
"""

from .pso_ga_hybrid import PSO_GA_Hybrid
from .woa_sma_hybrid import WOA_SMA_Hybrid
from .ga_sa_hybrid import GeneticSimulatedAnnealingHybrid
from .de_pso_hybrid import DifferentialEvolutionPSOHybrid
from .abc_de_hybrid import ABC_DE_Hybrid
from .gwo_pso_hybrid import GWO_PSO_Hybrid
from .woa_ga_hybrid import WOA_GA_Hybrid
from .sma_de_hybrid import SMA_DE_Hybrid
from .fa_ga_hybrid import FA_GA_Hybrid

__all__ = [
    'PSO_GA_Hybrid',
    'WOA_SMA_Hybrid',
    'GeneticSimulatedAnnealingHybrid',
    'DifferentialEvolutionPSOHybrid',
    'ABC_DE_Hybrid',
    'GWO_PSO_Hybrid',
    'WOA_GA_Hybrid',
    'SMA_DE_Hybrid',
    'FA_GA_Hybrid',
    'GA_SA_Hybrid',
    'DE_PSO_Hybrid',
]

# Aliases for convenience
GA_SA_Hybrid = GeneticSimulatedAnnealingHybrid
DE_PSO_Hybrid = DifferentialEvolutionPSOHybrid

# Algorithm mapping for easy access
HYBRID_ALGORITHMS = {
    'PSO_GA_Hybrid': PSO_GA_Hybrid,
    'WOA_SMA_Hybrid': WOA_SMA_Hybrid,
    'GeneticSimulatedAnnealingHybrid': GeneticSimulatedAnnealingHybrid,
    'DifferentialEvolutionPSOHybrid': DifferentialEvolutionPSOHybrid,
    'ABC_DE_Hybrid': ABC_DE_Hybrid,
    'GWO_PSO_Hybrid': GWO_PSO_Hybrid,
    'WOA_GA_Hybrid': WOA_GA_Hybrid,
    'SMA_DE_Hybrid': SMA_DE_Hybrid,
    'FA_GA_Hybrid': FA_GA_Hybrid,
}