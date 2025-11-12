"""
RCE-LLM Main Actualization Optimizer

Implements Ω* = arg max_{Ω⊆G} μ(Ω | C) subject to Φ(Ω) (Eq. 14).

Automatically selects best strategy based on graph size.

Author: Ismail Sialyen
Based on: DOI 10.5281/zenodo.17360372
"""

from typing import Tuple, Dict, Any, Optional

from rce_llm.core.types import Graph, Context, CoherenceScore
from rce_llm.optimization.beam_search import BeamSearchOptimizer
from rce_llm.optimization.ilp import ILPOptimizer


class ActualizationOptimizer:
    """
    Main Actualization Optimizer implementing Eq. 14.

    Automatically selects optimization strategy based on graph size:
    - Small (|R| ≤ 50): ILP for exact solution
    - Medium (50 < |R| ≤ 500): Beam search
    - Large (|R| > 500): Beam search with larger beam
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize optimizer.

        Args:
            config: Configuration dict
                - strategy: Force specific strategy ("ilp", "beam", "auto")
                - beam_size: Beam size for beam search (default: 10)
        """
        self.config = config or {}

        self.strategy = self.config.get("strategy", "auto")
        self.beam_size = self.config.get("beam_size", 10)

        # Initialize optimizers
        self.ilp_optimizer = ILPOptimizer()
        self.beam_optimizer = BeamSearchOptimizer(beam_size=self.beam_size)

    def optimize(
        self,
        graph: Graph,
        context: Context,
        coherence_evaluator: Any,
    ) -> Tuple[Graph, CoherenceScore, Dict[str, Any]]:
        """
        Find optimal subgraph Ω*.

        Args:
            graph: Candidate graph G
            context: Context C
            coherence_evaluator: Coherence aggregator μ

        Returns:
            (optimal_subgraph, coherence_score, optimization_info)
        """
        num_relations = len(graph.relations)

        # Select strategy
        if self.strategy == "ilp":
            optimizer = self.ilp_optimizer
        elif self.strategy == "beam":
            optimizer = self.beam_optimizer
        else:  # auto
            if num_relations <= 50:
                optimizer = self.ilp_optimizer
            else:
                optimizer = self.beam_optimizer

        # Optimize
        optimal_subgraph, coherence_score, opt_info = optimizer.optimize(
            graph, context, coherence_evaluator
        )

        opt_info["graph_size"] = graph.size
        opt_info["subgraph_size"] = optimal_subgraph.size
        opt_info["strategy_used"] = opt_info.get("method", "unknown")

        return optimal_subgraph, coherence_score, opt_info
