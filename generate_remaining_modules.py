"""
Script to generate remaining RCE-LLM modules
This ensures consistency and completeness
"""

import os

# Create temporal module
temporal_content = '''"""
RCE-LLM Temporal Coherence Module: μ_time(Ω | C)

Implements temporal ordering and chronological constraint checking (Eq. 10).

Author: Ismail Sialyen
Based on: DOI 10.5281/zenodo.17360372
"""

import re
from typing import Tuple, List, Dict, Any, Optional
from datetime import datetime, timedelta
from dateutil import parser as date_parser

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
                            if r.predicate in ["before", "after", "during", "since", "until"]]

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

        # Try to extract dates
        date1 = self._extract_date(subject.text)
        date2 = self._extract_date(obj.text)

        if date1 and date2:
            if relation.predicate == "before":
                if date1 >= date2:
                    return False, f"Temporal violation: {subject.text} not before {obj.text}"
            elif relation.predicate == "after":
                if date1 <= date2:
                    return False, f"Temporal violation: {subject.text} not after {obj.text}"

        return True, ""

    def _extract_date(self, text: str) -> Optional[datetime]:
        """Extract date from text."""
        try:
            return date_parser.parse(text, fuzzy=True)
        except:
            return None

    def get_default_weight(self, context: Context) -> float:
        """Context-adaptive weight."""
        if "temporal" in [c.lower() for c in context.constraints]:
            return 0.35
        elif context.domain == "historical":
            return 0.30
        else:
            return 0.15
'''

# Write temporal module
with open("/Users/isma/RCE-LLM-prototype/rce_llm/modules/temporal.py", "w") as f:
    f.write(temporal_content)

print("Generated temporal.py")

