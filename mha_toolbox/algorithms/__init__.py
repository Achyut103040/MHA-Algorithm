"""
Metaheuristic algorithms package
"""

from .sca import SineCosinAlgorithm
from .pso import ParticleSwarmOptimization
from .ga import GeneticAlgorithm
from .gwo import GreyWolfOptimizer
from .woa import WhaleOptimizationAlgorithm
from .fa import FireflyAlgorithm
from .ba import BatAlgorithm
from .aco import AntColonyOptimization
from .de import DifferentialEvolution
from .alo import AntLionOptimizer
from .csa import CapuchinSearchAlgorithm
from .coa import CoyoteOptimizationAlgorithm
from .mrfo import MantaRayForagingOptimization
from .msa import MothSearchAlgorithm
from .pfa import PathfinderAlgorithm
from .ssa import SalpSwarmAlgorithm
from .spider import SocialSpiderAlgorithm
from .tso import TunaSwarmOptimization
from .sma import SlimeMouldAlgorithm
from .ants import ApproximatedNondeterministicTreeSearch
from .ao import AquilaOptimizer
# New algorithms batch 1
from .vcs import VirusColonySearch
from .chio import CoronavirusHerdImmunityOptimization
from .fbi import ForensicBasedInvestigationOptimization
from .ica import ImperialistCompetitiveAlgorithm
from .qsa import QueuingSearchAlgorithm
from .spbo import StudentPsychologyBasedOptimization
from .aoa import ArchimedesOptimizationAlgorithm
from .eo import EquilibriumOptimizer
# New algorithms batch 2
from .hgso import HenryGasSolubilityOptimization
from .sa import SimulatedAnnealing
from .wdo import WindDrivenOptimization
from .cgo import ChaosGameOptimization
from .gbo import GradientBasedOptimizer
from .innov import WeightedMeanOfVectors
from .wca import WaterCycleAlgorithm
from .vns import VariableNeighborhoodSearch

__all__ = [
    'SineCosinAlgorithm',
    'ParticleSwarmOptimization', 
    'GeneticAlgorithm',
    'GreyWolfOptimizer',
    'WhaleOptimizationAlgorithm',
    'FireflyAlgorithm',
    'BatAlgorithm',
    'AntColonyOptimization',
    'DifferentialEvolution',
    'AntLionOptimizer',
    'CapuchinSearchAlgorithm',
    'CoyoteOptimizationAlgorithm',
    'MantaRayForagingOptimization',
    'MothSearchAlgorithm',
    'PathfinderAlgorithm',
    'SalpSwarmAlgorithm',
    'SocialSpiderAlgorithm',
    'TunaSwarmOptimization',
    'SlimeMouldAlgorithm',
    'ApproximatedNondeterministicTreeSearch',
    'AquilaOptimizer',
    # New algorithms batch 1
    'VirusColonySearch',
    'CoronavirusHerdImmunityOptimization',
    'ForensicBasedInvestigationOptimization',
    'ImperialistCompetitiveAlgorithm',
    'QueuingSearchAlgorithm',
    'StudentPsychologyBasedOptimization',
    'ArchimedesOptimizationAlgorithm',
    'EquilibriumOptimizer',
    # New algorithms batch 2
    'HenryGasSolubilityOptimization',
    'SimulatedAnnealing',
    'WindDrivenOptimization',
    'ChaosGameOptimization',
    'GradientBasedOptimizer',
    'WeightedMeanOfVectors',
    'WaterCycleAlgorithm',
    'VariableNeighborhoodSearch'
]