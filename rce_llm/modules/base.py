"""
RCE-LLM Base Coherence Module

Abstract base class for all coherence modules μ_k.

Author: Ismail Sialyen
Based on: DOI 10.5281/zenodo.17360372
"""

from abc import ABC, abstractmethod
from typing import Tuple, List, Dict, Any, Optional
import time

from rce_llm.core.types import Graph, Context


class CoherenceModule(ABC):
    """
    Abstract base class for coherence modules μ_k(Ω | C).

    Each module evaluates a specific dimension of coherence (Eq. 8):
        μ(Ω | C) = Σ_{k=1}^K w_k(C)·μ_k(Ω | C)

    The five standard modules are:
        - μ_units: Dimensional analysis (Eq. 9)
        - μ_time: Temporal ordering (Eq. 10)
        - μ_arith: Arithmetic validity (Eq. 11)
        - μ_coref: Coreference resolution (Eq. 12)
        - μ_entail: Factual entailment (Eq. 13)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize coherence module.

        Args:
            config: Module-specific configuration
        """
        self.config = config or {}
        self.name = self.get_name()

    @abstractmethod
    def get_name(self) -> str:
        """
        Get module name (e.g., 'units', 'temporal', 'arithmetic').

        Returns:
            Module name
        """
        pass

    @abstractmethod
    def get_description(self) -> str:
        """
        Get human-readable module description.

        Returns:
            Description of what this module evaluates
        """
        pass

    @abstractmethod
    def evaluate(
        self,
        graph: Graph,
        context: Context
    ) -> Tuple[float, List[str], str]:
        """
        Evaluate coherence μ_k(Ω | C) for a graph.

        This is the core method implementing the module's coherence evaluation.

        Args:
            graph: Graph Ω to evaluate
            context: Context C

        Returns:
            Tuple of (score, violations, explanation):
                - score: Coherence score in [0, 1] where 1 = fully coherent
                - violations: List of detected violations/inconsistencies
                - explanation: Human-readable explanation of the score
        """
        pass

    def evaluate_with_timing(
        self,
        graph: Graph,
        context: Context
    ) -> Tuple[float, List[str], str, float]:
        """
        Evaluate coherence with timing information.

        Args:
            graph: Graph to evaluate
            context: Context

        Returns:
            Tuple of (score, violations, explanation, time_ms)
        """
        start_time = time.time()
        score, violations, explanation = self.evaluate(graph, context)
        elapsed_ms = (time.time() - start_time) * 1000

        return score, violations, explanation, elapsed_ms

    def get_default_weight(self, context: Context) -> float:
        """
        Get default weight for this module given context.

        Can be overridden by subclasses to provide context-adaptive weights.

        Args:
            context: Context

        Returns:
            Default weight in [0, 1]
        """
        # Base implementation: uniform weights
        return 0.2  # 1/5 for 5 modules

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
