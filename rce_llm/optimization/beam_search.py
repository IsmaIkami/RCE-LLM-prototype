"""
RCE-LLM Beam Search Optimizer

Implements beam search approximation for actualization optimization.
Complexity: O(B·|R|·log|R|) where B is beam size.

Author: Ismail Sialyen
Based on: DOI 10.5281/zenodo.17360372
"""

from typing import List, Tuple, Set, Dict, Any, Optional
import heapq
import time

from rce_llm.core.types import Graph, Context, CoherenceScore


class BeamSearchOptimizer:
    """
    Beam Search Optimizer for actualization (Eq. 14).

    Efficiently finds high-coherence subgraphs using beam search.
    Suitable for graphs with 50-5000 relations.
    """

    def __init__(self, beam_size: int = 10, max_iterations: Optional[int] = None):
        """
        Initialize beam search optimizer.

        Args:
            beam_size: Number of candidates to keep (B in complexity analysis)
            max_iterations: Maximum iterations (default: |R|)
        """
        self.beam_size = beam_size
        self.max_iterations = max_iterations

    def optimize(
        self,
        graph: Graph,
        context: Context,
        coherence_evaluator: Any,
    ) -> Tuple[Graph, CoherenceScore, Dict[str, Any]]:
        """
        Find optimal subgraph using beam search.

        Args:
            graph: Candidate graph G
            context: Context C
            coherence_evaluator: Coherence aggregator

        Returns:
            (optimal_subgraph, coherence_score, optimization_info)
        """
        start_time = time.time()

        # Initialize beam with empty subgraph
        beam: List[Tuple[float, Set[str]]] = [(0.0, set())]  # (score, entity_ids)

        max_iter = self.max_iterations or len(graph.relations)

        for iteration in range(max_iter):
            candidates = []

            # Expand each beam element
            for score, entity_ids in beam:
                # Try adding each remaining entity
                remaining_entities = set(graph.entities.keys()) - entity_ids

                for entity_id in remaining_entities:
                    new_entity_ids = entity_ids | {entity_id}

                    # Create subgraph
                    subgraph = graph.get_subgraph(new_entity_ids)

                    # Evaluate coherence
                    coherence = coherence_evaluator.evaluate(subgraph, context)

                    candidates.append((coherence.overall, new_entity_ids))

                    if len(candidates) >= self.beam_size * 10:
                        # Prune early to save memory
                        candidates = heapq.nlargest(self.beam_size, candidates)

            if not candidates:
                break

            # Keep top B candidates
            beam = heapq.nlargest(self.beam_size, candidates)

        # Get best subgraph
        best_score, best_entity_ids = beam[0] if beam else (0.0, set())

        optimal_subgraph = graph.get_subgraph(best_entity_ids)
        final_coherence = coherence_evaluator.evaluate(optimal_subgraph, context)

        optimization_info = {
            "method": "beam_search",
            "beam_size": self.beam_size,
            "iterations": min(iteration + 1, max_iter),
            "entities_selected": len(best_entity_ids),
            "relations_selected": len(optimal_subgraph.relations),
            "time_ms": (time.time() - start_time) * 1000,
        }

        return optimal_subgraph, final_coherence, optimization_info
