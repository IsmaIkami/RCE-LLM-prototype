"""
RCE-LLM Entailment Coherence Module: μ_entail(Ω | C)

Implements factual entailment and evidence grounding checking (Eq. 13).

Author: Ismail Sialyen
Based on: DOI 10.5281/zenodo.17360372
"""

from typing import Tuple, List, Dict, Any, Optional

from rce_llm.modules.base import CoherenceModule
from rce_llm.core.types import Graph, Context


class EntailmentCoherenceModule(CoherenceModule):
    """Entailment Coherence Module: μ_entail(Ω | C) from Eq. 13."""

    def get_name(self) -> str:
        return "entailment"

    def get_description(self) -> str:
        return "Factual entailment and evidence grounding checking"

    def evaluate(self, graph: Graph, context: Context) -> Tuple[float, List[str], str]:
        """Evaluate entailment coherence μ_entail(Ω | C)."""
        violations = []
        total_checks = 0
        passed_checks = 0

        # Check relations have evidence
        for relation in graph.relations:
            total_checks += 1

            if not relation.inferred:  # Only check extracted relations
                if relation.evidence and len(relation.evidence) > 0:
                    passed_checks += 1
                else:
                    violations.append(f"Relation lacks evidence: {relation.predicate}({relation.subject}, {relation.object})")

        # Check against retrieved evidence if available
        if context.retrieved_evidence:
            for entity in graph.entities.values():
                total_checks += 1
                is_grounded = self._is_entity_grounded(entity, context.retrieved_evidence)
                if is_grounded:
                    passed_checks += 1
                else:
                    violations.append(f"Entity not grounded in evidence: {entity.text}")

        score = passed_checks / total_checks if total_checks > 0 else 1.0
        explanation = f"Passed {passed_checks}/{total_checks} entailment checks"

        return score, violations, explanation

    def _is_entity_grounded(self, entity, evidence_docs: List[Dict[str, Any]]) -> bool:
        """Check if entity appears in evidence."""
        entity_text_lower = entity.text.lower()

        for doc in evidence_docs:
            content = doc.get("content", "").lower()
            if entity_text_lower in content:
                return True

        return False

    def get_default_weight(self, context: Context) -> float:
        """Context-adaptive weight."""
        if context.intent.value == "validate":
            return 0.40
        elif context.domain in ["medical", "legal", "scientific"]:
            return 0.35
        elif len(context.retrieved_evidence) > 0:
            return 0.30
        else:
            return 0.20
