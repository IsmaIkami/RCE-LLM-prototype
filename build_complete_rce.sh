#!/bin/bash
# Complete RCE-LLM Prototype Builder
# Author: Ismail Sialyen
# This script generates all remaining production modules

set -e

echo "Building Complete RCE-LLM Prototype..."
echo "======================================"

# Create arithmetic coherence module
cat > rce_llm/modules/arithmetic.py << 'ARITH_END'
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
ARITH_END

echo "✓ Created arithmetic.py"

# Create coreference coherence module
cat > rce_llm/modules/coreference.py << 'COREF_END'
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
COREF_END

echo "✓ Created coreference.py"

# Create entailment coherence module
cat > rce_llm/modules/entailment.py << 'ENTAIL_END'
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
ENTAIL_END

echo "✓ Created entailment.py"

# Create coherence aggregator
cat > rce_llm/modules/aggregator.py << 'AGG_END'
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
AGG_END

echo "✓ Created aggregator.py"

echo ""
echo "Module creation complete!"
echo "========================="
echo "Created:"
echo "  - arithmetic.py"
echo "  - coreference.py"
echo "  - entailment.py"
echo "  - aggregator.py"
echo ""
echo "All five coherence modules μ_k now implemented!"
