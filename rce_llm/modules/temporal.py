"""
RCE-LLM Temporal Coherence Module: μ_time(Ω | C)

Implements temporal ordering and chronological constraint checking (Eq. 10).

Author: Ismail Sialyen
Based on: DOI 10.5281/zenodo.17360372
"""

import re
from typing import Tuple, List, Dict, Any, Optional
from datetime import datetime, timedelta

from rce_llm.modules.base import CoherenceModule
from rce_llm.core.types import Graph, Context


class TemporalCoherenceModule(CoherenceModule):
    """
    Temporal Coherence Module: μ_time(Ω | C) from Eq. 10.

    Evaluates temporal consistency and chronological ordering.
    Critical for F2 benchmark tasks (Temporal Reasoning).
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

    def get_name(self) -> str:
        return "temporal"

    def get_description(self) -> str:
        return "Temporal ordering and chronological constraint checking"

    def evaluate(self, graph: Graph, context: Context) -> Tuple[float, List[str], str]:
        """Evaluate temporal coherence μ_time(Ω | C)."""
        violations = []
        total_checks = 0
        passed_checks = 0

        # Extract temporal relations
        temporal_relations = [r for r in graph.relations
                            if r.predicate in ["before", "after", "during", "since", "until", "temporal"]]

        if not temporal_relations:
            return 1.0, [], "No temporal relations found"

        # Check for temporal cycles
        total_checks += 1
        if not self._has_temporal_cycles(temporal_relations, graph):
            passed_checks += 1
        else:
            violations.append("Temporal cycle detected (e.g., A before B, B before C, C before A)")

        # Check temporal consistency
        for relation in temporal_relations:
            total_checks += 1
            is_valid, violation = self._check_temporal_relation(relation, graph)
            if is_valid:
                passed_checks += 1
            else:
                violations.append(violation)

        score = passed_checks / total_checks if total_checks > 0 else 1.0
        explanation = f"Passed {passed_checks}/{total_checks} temporal consistency checks"

        return score, violations, explanation

    def _has_temporal_cycles(self, relations: List, graph: Graph) -> bool:
        """Check for cycles in temporal ordering using DFS."""
        # Build adjacency list
        adj = {}
        for rel in relations:
            if rel.predicate in ["before"]:
                if rel.subject not in adj:
                    adj[rel.subject] = []
                adj[rel.subject].append(rel.object)

        # DFS to detect cycle
        visited = set()
        rec_stack = set()

        def has_cycle_util(node):
            visited.add(node)
            rec_stack.add(node)

            if node in adj:
                for neighbor in adj[node]:
                    if neighbor not in visited:
                        if has_cycle_util(neighbor):
                            return True
                    elif neighbor in rec_stack:
                        return True

            rec_stack.remove(node)
            return False

        for node in adj:
            if node not in visited:
                if has_cycle_util(node):
                    return True

        return False

    def _check_temporal_relation(self, relation, graph: Graph) -> Tuple[bool, str]:
        """Check if temporal relation is consistent."""
        subject = graph.get_entity(relation.subject)
        obj = graph.get_entity(relation.object)

        if not subject or not obj:
            return True, ""

        # Try to extract dates/times
        time1 = self._extract_time(subject.text)
        time2 = self._extract_time(obj.text)

        if time1 is not None and time2 is not None:
            if relation.predicate == "before":
                if time1 >= time2:
                    return False, f"Temporal violation: {subject.text} not before {obj.text}"
            elif relation.predicate == "after":
                if time1 <= time2:
                    return False, f"Temporal violation: {subject.text} not after {obj.text}"

        return True, ""

    def _extract_time(self, text: str) -> Optional[float]:
        """Extract time in seconds from text."""
        text_lower = text.lower()

        # Try to parse time expressions
        patterns = [
            (r'(\d+\.?\d*)\s*hour?s?', 3600),
            (r'(\d+\.?\d*)\s*h\b', 3600),
            (r'(\d+\.?\d*)\s*minute?s?', 60),
            (r'(\d+\.?\d*)\s*min\b', 60),
            (r'(\d+\.?\d*)\s*second?s?', 1),
            (r'(\d+\.?\d*)\s*sec\b', 1),
            (r'(\d+\.?\d*)\s*s\b', 1),
        ]

        for pattern, multiplier in patterns:
            match = re.search(pattern, text_lower)
            if match:
                try:
                    value = float(match.group(1))
                    return value * multiplier
                except ValueError:
                    continue

        return None

    def get_default_weight(self, context: Context) -> float:
        """Context-adaptive weight."""
        if any("temporal" in c.lower() for c in context.constraints):
            return 0.35
        elif context.domain in ["historical", "scheduling"]:
            return 0.30
        else:
            return 0.15
