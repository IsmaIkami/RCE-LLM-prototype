"""
RCE-LLM Units Coherence Module: μ_units(Ω | C)

Implements dimensional analysis and unit consistency checking (Eq. 9).

Ensures that:
- Unit conversions are correct
- Dimensional compatibility in operations
- Magnitude consistency

Author: Ismail Sialyen
Based on: DOI 10.5281/zenodo.17360372
"""

import re
from typing import Tuple, List, Dict, Any, Optional

from rce_llm.modules.base import CoherenceModule
from rce_llm.core.types import Graph, Context, Entity


class UnitsCoherenceModule(CoherenceModule):
    """
    Units Coherence Module: μ_units(Ω | C) from Eq. 9.

    Evaluates dimensional consistency and unit compatibility.
    Critical for F1 benchmark tasks (Units Consistency).
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

        # Unit conversion factors (to base SI units)
        self.conversion_factors = {
            # Length
            "m": 1.0,
            "km": 1000.0,
            "cm": 0.01,
            "mm": 0.001,
            "mile": 1609.34,
            "miles": 1609.34,
            "foot": 0.3048,
            "feet": 0.3048,
            "ft": 0.3048,
            "inch": 0.0254,
            "inches": 0.0254,
            "in": 0.0254,
            # Time
            "s": 1.0,
            "sec": 1.0,
            "second": 1.0,
            "seconds": 1.0,
            "min": 60.0,
            "minute": 60.0,
            "minutes": 60.0,
            "h": 3600.0,
            "hour": 3600.0,
            "hours": 3600.0,
            "hr": 3600.0,
            "day": 86400.0,
            "days": 86400.0,
            # Speed
            "m/s": 1.0,
            "km/h": 0.277778,
            "km/hr": 0.277778,
            "mph": 0.44704,
            "miles/hour": 0.44704,
            # Mass
            "kg": 1.0,
            "g": 0.001,
            "gram": 0.001,
            "grams": 0.001,
            "mg": 0.000001,
            "lb": 0.453592,
            "lbs": 0.453592,
            "pound": 0.453592,
            "pounds": 0.453592,
            "oz": 0.0283495,
        }

        # Unit dimensions (for dimensional analysis)
        self.unit_dimensions = {
            # Length [L]
            "m": {"L": 1},
            "km": {"L": 1},
            "cm": {"L": 1},
            "mile": {"L": 1},
            "foot": {"L": 1},
            "feet": {"L": 1},
            # Time [T]
            "s": {"T": 1},
            "min": {"T": 1},
            "h": {"T": 1},
            "hour": {"T": 1},
            "day": {"T": 1},
            # Speed [L/T]
            "m/s": {"L": 1, "T": -1},
            "km/h": {"L": 1, "T": -1},
            "mph": {"L": 1, "T": -1},
            # Mass [M]
            "kg": {"M": 1},
            "g": {"M": 1},
            "lb": {"M": 1},
        }

    def get_name(self) -> str:
        return "units"

    def get_description(self) -> str:
        return "Dimensional analysis and unit consistency checking"

    def evaluate(
        self,
        graph: Graph,
        context: Context
    ) -> Tuple[float, List[str], str]:
        """
        Evaluate unit consistency μ_units(Ω | C).

        Checks:
        1. Unit conversions are mathematically correct
        2. Dimensional compatibility in relations
        3. Magnitude reasonableness

        Args:
            graph: Graph to evaluate
            context: Context

        Returns:
            (score, violations, explanation)
        """
        violations = []
        total_checks = 0
        passed_checks = 0

        # Extract quantity entities with units
        quantity_entities = self._extract_quantity_entities(graph)

        if not quantity_entities:
            # No units to check - perfect score
            return 1.0, [], "No unit-bearing quantities found"

        # Check unit conversions in relations
        for relation in graph.relations:
            if relation.predicate in ["has_unit", "converts_to", "equals"]:
                total_checks += 1

                subject = graph.get_entity(relation.subject)
                obj = graph.get_entity(relation.object)

                if subject and obj:
                    is_valid, violation = self._check_unit_compatibility(
                        subject, obj, relation.predicate
                    )

                    if is_valid:
                        passed_checks += 1
                    else:
                        violations.append(violation)

        # Check for dimensional consistency
        for entity in quantity_entities:
            total_checks += 1

            is_consistent, violation = self._check_dimensional_consistency(entity)

            if is_consistent:
                passed_checks += 1
            else:
                violations.append(violation)

        # Calculate score
        if total_checks == 0:
            score = 1.0
            explanation = "No unit checks performed"
        else:
            score = passed_checks / total_checks
            explanation = f"Passed {passed_checks}/{total_checks} unit consistency checks"

        return score, violations, explanation

    def _extract_quantity_entities(self, graph: Graph) -> List[Entity]:
        """Extract entities with units or quantities."""
        quantity_entities = []

        for entity in graph.entities.values():
            if entity.semantic_type in ["Quantity", "Money", "Number", "Measurement"]:
                quantity_entities.append(entity)
            elif "unit" in entity.attributes or "units" in entity.attributes:
                quantity_entities.append(entity)
            elif self._has_unit_pattern(entity.text):
                quantity_entities.append(entity)

        return quantity_entities

    def _has_unit_pattern(self, text: str) -> bool:
        """Check if text contains unit pattern (e.g., '60 km')."""
        # Pattern: number followed by unit
        pattern = r'\d+\.?\d*\s*[a-zA-Z]+/?[a-zA-Z]*'
        return bool(re.search(pattern, text))

    def _check_unit_compatibility(
        self,
        entity1: Entity,
        entity2: Entity,
        relation_type: str
    ) -> Tuple[bool, str]:
        """
        Check if units are compatible for the relation.

        Args:
            entity1: First entity
            entity2: Second entity
            relation_type: Type of relation

        Returns:
            (is_valid, violation_message)
        """
        unit1 = self._extract_unit(entity1)
        unit2 = self._extract_unit(entity2)

        if not unit1 or not unit2:
            return True, ""  # No units to check

        # Get dimensions
        dim1 = self.unit_dimensions.get(unit1)
        dim2 = self.unit_dimensions.get(unit2)

        if not dim1 or not dim2:
            # Unknown units - assume compatible
            return True, ""

        # Check dimensional compatibility
        if dim1 != dim2:
            return False, f"Dimensional mismatch: {entity1.text} ({unit1}) vs {entity2.text} ({unit2})"

        # If relation is "equals" or "converts_to", check conversion factor
        if relation_type in ["equals", "converts_to"]:
            val1 = self._extract_numeric_value(entity1)
            val2 = self._extract_numeric_value(entity2)

            if val1 is not None and val2 is not None:
                # Convert to base units
                base1 = val1 * self.conversion_factors.get(unit1, 1.0)
                base2 = val2 * self.conversion_factors.get(unit2, 1.0)

                # Check if values are approximately equal (within 5% tolerance)
                if abs(base1 - base2) / max(abs(base1), abs(base2), 1e-10) > 0.05:
                    return False, f"Unit conversion error: {entity1.text} != {entity2.text}"

        return True, ""

    def _check_dimensional_consistency(self, entity: Entity) -> Tuple[bool, str]:
        """Check dimensional consistency within an entity."""
        # Extract value and unit
        value = self._extract_numeric_value(entity)
        unit = self._extract_unit(entity)

        if value is None or unit is None:
            return True, ""

        # Check for unreasonable magnitudes
        base_value = value * self.conversion_factors.get(unit, 1.0)

        # Define reasonable ranges for different dimensions
        if unit in ["m", "km", "mile"]:  # Length
            if base_value < 0:
                return False, f"Negative length: {entity.text}"
            if base_value > 1e9:  # > 1 million km
                return False, f"Unreasonably large length: {entity.text}"

        elif unit in ["s", "min", "h", "hour"]:  # Time
            if base_value < 0:
                return False, f"Negative time: {entity.text}"
            if base_value > 3.15e9:  # > 100 years
                return False, f"Unreasonably large time: {entity.text}"

        elif unit in ["kg", "g", "lb"]:  # Mass
            if base_value < 0:
                return False, f"Negative mass: {entity.text}"
            if base_value > 1e6:  # > 1000 tons
                return False, f"Unreasonably large mass: {entity.text}"

        return True, ""

    def _extract_unit(self, entity: Entity) -> Optional[str]:
        """Extract unit from entity."""
        # Check attributes first
        if "unit" in entity.attributes:
            return entity.attributes["unit"]
        if "units" in entity.attributes:
            return entity.attributes["units"]

        # Try to parse from text
        text = entity.text.lower()
        for unit in self.conversion_factors.keys():
            if unit in text:
                return unit

        return None

    def _extract_numeric_value(self, entity: Entity) -> Optional[float]:
        """Extract numeric value from entity."""
        # Check attributes
        if "numeric_value" in entity.attributes:
            return entity.attributes["numeric_value"]

        # Try to parse from text
        text = entity.text
        match = re.search(r'(\d+\.?\d*)', text)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                pass

        return None

    def get_default_weight(self, context: Context) -> float:
        """Get context-adaptive weight for units module."""
        # Higher weight for calculation and scientific domains
        if context.intent.value == "calculate":
            return 0.35
        elif context.domain in ["scientific", "mathematical", "technical"]:
            return 0.30
        else:
            return 0.15
