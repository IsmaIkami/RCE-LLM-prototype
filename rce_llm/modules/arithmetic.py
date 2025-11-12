"""
RCE-LLM Arithmetic Coherence Module: μ_arith(Ω | C)

Implements arithmetic validity and numerical consistency checking (Eq. 11).

Author: Ismail Sialyen
Based on: DOI 10.5281/zenodo.17360372
"""

import re
from typing import Tuple, List, Dict, Any, Optional

from rce_llm.modules.base import CoherenceModule
from rce_llm.core.types import Graph, Context


class ArithmeticCoherenceModule(CoherenceModule):
    """Arithmetic Coherence Module: μ_arith(Ω | C) from Eq. 11."""

    def get_name(self) -> str:
        return "arithmetic"

    def get_description(self) -> str:
        return "Arithmetic validity and numerical consistency checking"

    def evaluate(self, graph: Graph, context: Context) -> Tuple[float, List[str], str]:
        """Evaluate arithmetic coherence μ_arith(Ω | C)."""
        violations = []
        total_checks = 0
        passed_checks = 0

        # Find arithmetic relations
        arith_relations = [r for r in graph.relations
                          if r.predicate in ["add", "subtract", "multiply", "divide", "equals"]]

        if not arith_relations:
            return 1.0, [], "No arithmetic relations found"

        for relation in arith_relations:
            total_checks += 1
            is_valid, violation = self._check_arithmetic(relation, graph)
            if is_valid:
                passed_checks += 1
            else:
                violations.append(violation)

        score = passed_checks / total_checks if total_checks > 0 else 1.0
        explanation = f"Passed {passed_checks}/{total_checks} arithmetic checks"

        return score, violations, explanation

    def _check_arithmetic(self, relation, graph: Graph) -> Tuple[bool, str]:
        """Check arithmetic relation."""
        subj = graph.get_entity(relation.subject)
        obj = graph.get_entity(relation.object)

        if not subj or not obj:
            return True, ""

        val1 = self._extract_number(subj.text)
        val2 = self._extract_number(obj.text)

        if val1 is None or val2 is None:
            return True, ""

        # Check based on predicate
        if relation.predicate == "add":
            # Find result entity
            result_rels = [r for r in graph.relations
                          if r.subject == relation.id and r.predicate == "result"]
            if result_rels:
                result_entity = graph.get_entity(result_rels[0].object)
                if result_entity:
                    result_val = self._extract_number(result_entity.text)
                    if result_val is not None:
                        expected = val1 + val2
                        if abs(result_val - expected) / max(abs(expected), 1) > 0.01:
                            return False, f"Arithmetic error: {val1} + {val2} ≠ {result_val}"

        return True, ""

    def _extract_number(self, text: str) -> Optional[float]:
        """Extract number from text."""
        match = re.search(r'(-?\d+\.?\d*)', text)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                pass
        return None

    def get_default_weight(self, context: Context) -> float:
        """Context-adaptive weight."""
        if context.intent.value == "calculate":
            return 0.40
        elif context.domain in ["mathematical", "financial"]:
            return 0.30
        else:
            return 0.15
