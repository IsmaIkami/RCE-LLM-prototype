"""
RCE-LLM Coreference Coherence Module: μ_coref(Ω | C)

Implements coreference resolution and entity stability checking (Eq. 12).

Author: Ismail Sialyen
Based on: DOI 10.5281/zenodo.17360372
"""

from typing import Tuple, List, Dict, Any, Optional

from rce_llm.modules.base import CoherenceModule
from rce_llm.core.types import Graph, Context


class CoreferenceCoherenceModule(CoherenceModule):
    """Coreference Coherence Module: μ_coref(Ω | C) from Eq. 12."""

    def get_name(self) -> str:
        return "coreference"

    def get_description(self) -> str:
        return "Coreference resolution and entity stability checking"

    def evaluate(self, graph: Graph, context: Context) -> Tuple[float, List[str], str]:
        """Evaluate coreference coherence μ_coref(Ω | C)."""
        violations = []
        total_checks = 0
        passed_checks = 0

        # Find coreference relations
        coref_relations = [r for r in graph.relations
                          if r.predicate in ["refers_to", "same_as", "coreference"]]

        if not coref_relations:
            return 1.0, [], "No coreference relations found"

        # Check consistency of coreferences
        for relation in coref_relations:
            total_checks += 1
            is_valid, violation = self._check_coreference(relation, graph)
            if is_valid:
                passed_checks += 1
            else:
                violations.append(violation)

        score = passed_checks / total_checks if total_checks > 0 else 1.0
        explanation = f"Passed {passed_checks}/{total_checks} coreference checks"

        return score, violations, explanation

    def _check_coreference(self, relation, graph: Graph) -> Tuple[bool, str]:
        """Check coreference consistency."""
        subj = graph.get_entity(relation.subject)
        obj = graph.get_entity(relation.object)

        if not subj or not obj:
            return True, ""

        # Check type compatibility
        if subj.semantic_type != obj.semantic_type:
            if not self._are_types_compatible(subj.semantic_type, obj.semantic_type):
                return False, f"Type mismatch in coreference: {subj.text} ({subj.semantic_type}) vs {obj.text} ({obj.semantic_type})"

        return True, ""

    def _are_types_compatible(self, type1: str, type2: str) -> bool:
        """Check if types can corefer."""
        # Define compatible type pairs
        compatible = {
            ("Person", "Entity"),
            ("Organization", "Entity"),
            ("Location", "Entity"),
        }

        return (type1, type2) in compatible or (type2, type1) in compatible

    def get_default_weight(self, context: Context) -> float:
        """Context-adaptive weight."""
        if context.domain in ["narrative", "dialogue"]:
            return 0.30
        else:
            return 0.15
