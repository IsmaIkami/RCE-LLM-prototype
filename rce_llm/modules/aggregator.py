"""
RCE-LLM Coherence Aggregator

Implements the modular coherence functional (Eq. 8):
    μ(Ω | C) = Σ_{k=1}^K w_k(C)·μ_k(Ω | C)

Author: Ismail Sialyen
Based on: DOI 10.5281/zenodo.17360372
"""

from typing import Dict, List, Tuple, Optional, Any
import time

from rce_llm.core.types import Graph, Context, CoherenceScore
from rce_llm.modules.base import CoherenceModule
from rce_llm.modules.units import UnitsCoherenceModule
from rce_llm.modules.temporal import TemporalCoherenceModule
from rce_llm.modules.arithmetic import ArithmeticCoherenceModule
from rce_llm.modules.coreference import CoreferenceCoherenceModule
from rce_llm.modules.entailment import EntailmentCoherenceModule


class CoherenceAggregator:
    """
    Coherence Aggregator implementing Eq. 8:
        μ(Ω | C) = Σ_{k=1}^K w_k(C)·μ_k(Ω | C)

    Combines scores from all five coherence modules with context-adaptive weights.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize aggregator with all five modules."""
        self.config = config or {}

        # Initialize all five modules
        self.modules: List[CoherenceModule] = [
            UnitsCoherenceModule(self.config.get("units", {})),
            TemporalCoherenceModule(self.config.get("temporal", {})),
            ArithmeticCoherenceModule(self.config.get("arithmetic", {})),
            CoreferenceCoherenceModule(self.config.get("coreference", {})),
            EntailmentCoherenceModule(self.config.get("entailment", {})),
        ]

    def evaluate(self, graph: Graph, context: Context) -> CoherenceScore:
        """
        Evaluate coherence μ(Ω | C) for graph.

        Implements Eq. 8 from paper.

        Args:
            graph: Graph Ω to evaluate
            context: Context C

        Returns:
            CoherenceScore with modular breakdown
        """
        start_time = time.time()

        module_scores = {}
        module_weights = {}
        all_violations = []

        # Evaluate each module
        for module in self.modules:
            score, violations, explanation = module.evaluate(graph, context)
            weight = module.get_default_weight(context)

            module_scores[module.get_name()] = score
            module_weights[module.get_name()] = weight

            all_violations.extend(violations)

        # Normalize weights to sum to 1.0
        weight_sum = sum(module_weights.values())
        if weight_sum > 0:
            module_weights = {k: v / weight_sum for k, v in module_weights.items()}
        else:
            # Uniform weights
            module_weights = {k: 1.0 / len(self.modules) for k in module_weights}

        # Compute overall score (Eq. 8)
        overall_score = sum(
            module_weights[name] * module_scores[name]
            for name in module_scores
        )

        computation_time_ms = (time.time() - start_time) * 1000

        coherence_score = CoherenceScore(
            overall=overall_score,
            module_scores=module_scores,
            module_weights=module_weights,
            violations=all_violations,
            confidence=1.0 if len(all_violations) == 0 else 0.8,
            computation_time_ms=computation_time_ms,
        )

        return coherence_score
