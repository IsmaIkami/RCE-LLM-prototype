"""
RCE-LLM Optimization Module

Implements three optimization strategies for Ω* = arg max μ(Ω | C) (Eq. 14):
    1. Exact optimization via Integer Linear Programming (small graphs)
    2. Beam search approximation (medium/large graphs)
    3. Differentiable relaxation (for end-to-end training)

Author: Ismail Sialyen
Based on: DOI 10.5281/zenodo.17360372
"""

from rce_llm.optimization.optimizer import ActualizationOptimizer
from rce_llm.optimization.beam_search import BeamSearchOptimizer
from rce_llm.optimization.ilp import ILPOptimizer

__all__ = [
    "ActualizationOptimizer",
    "BeamSearchOptimizer",
    "ILPOptimizer",
]
