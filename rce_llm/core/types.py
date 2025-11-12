"""
RCE-LLM Core Type Definitions

Mathematical Foundation from Paper:
    G = (V, R, τ, σ)                    [Eq. 6]
    C = (intent, constraints, evidence, domain)    [Eq. 7]
    μ(Ω | C) = Σ_{k=1}^K w_k(C)·μ_k(Ω | C)       [Eq. 8]
    Ω* = arg max_{Ω⊆G} μ(Ω | C)  s.t. Φ(Ω)      [Eq. 14]

Author: Ismail Sialyen
Based on: DOI 10.5281/zenodo.17360372
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional, Any
from enum import Enum
from datetime import datetime
import uuid


# ═════════════════════════════════════════════════════════════════════
# GRAPH COMPONENTS (Eq. 6: G = (V, R, τ, σ))
# ═════════════════════════════════════════════════════════════════════


@dataclass
class Entity:
    """
    Entity in relational graph (element of V in Eq. 6).

    Represents typed entities extracted from input with confidence scores
    and semantic type annotations.

    Attributes:
        id: Unique identifier
        text: Original text span
        canonical_form: Normalized representation
        semantic_type: Type τ(v) from Eq. 6 (e.g., "Quantity", "Date", "Person")
        attributes: Additional attributes (units, modifiers, etc.)
        confidence: Extraction confidence σ(v) from Eq. 6, in [0, 1]
        source_span: Character positions in input text (start, end)
        knowledge_links: External KB references
    """

    id: str
    text: str
    canonical_form: str
    semantic_type: str
    attributes: Dict[str, Any]
    confidence: float
    source_span: Tuple[int, int]
    knowledge_links: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be in [0,1], got {self.confidence}")
        if self.source_span[0] < 0 or self.source_span[1] < self.source_span[0]:
            raise ValueError(f"Invalid source span: {self.source_span}")


@dataclass
class Relation:
    """
    Typed relation between entities (element of R in Eq. 6).

    Represents semantic relations with confidence, modality, and temporality.
    Part of the relational graph R ⊆ V × L × V where L is the label set.

    Attributes:
        id: Unique identifier
        subject: Subject entity ID
        predicate: Relation type from L (e.g., "has_unit", "before", "causes")
        object: Object entity ID
        confidence: Relation confidence σ(r) from Eq. 6, in [0, 1]
        modality: Modal qualification ("factual", "must", "may", "cannot")
        temporality: Temporal qualification
        evidence: Supporting evidence (text spans, KB entries)
        inferred: True if inferred, False if directly extracted
    """

    id: str
    subject: str  # Entity ID
    predicate: str
    object: str  # Entity ID
    confidence: float
    modality: str = "factual"
    temporality: Optional[str] = None
    evidence: List[str] = field(default_factory=list)
    inferred: bool = False

    def __post_init__(self):
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be in [0,1], got {self.confidence}")


@dataclass
class Graph:
    """
    Complete relational graph G = (V, R, τ, σ) from Eq. 6.

    Represents all possible interpretations of the input text.
    - V: Entities (vertices)
    - R: Relations (edges)
    - τ: Type function (Entity.semantic_type)
    - σ: Confidence function (Entity.confidence, Relation.confidence)

    Attributes:
        id: Unique identifier
        entities: Dictionary of entity_id → Entity (represents V)
        relations: List of relations (represents R)
        metadata: Additional graph metadata
    """

    id: str
    entities: Dict[str, Entity]
    relations: List[Relation]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get entity by ID."""
        return self.entities.get(entity_id)

    def get_relations_for_entity(self, entity_id: str) -> List[Relation]:
        """Get all relations involving an entity."""
        return [r for r in self.relations
                if r.subject == entity_id or r.object == entity_id]

    def get_subgraph(self, entity_ids: Set[str]) -> 'Graph':
        """
        Extract subgraph Ω ⊆ G containing specified entities.

        Args:
            entity_ids: Set of entity IDs to include

        Returns:
            Subgraph containing only specified entities and their relations
        """
        subgraph_entities = {
            eid: entity for eid, entity in self.entities.items()
            if eid in entity_ids
        }

        subgraph_relations = [
            r for r in self.relations
            if r.subject in entity_ids and r.object in entity_ids
        ]

        return Graph(
            id=f"{self.id}_subgraph_{uuid.uuid4().hex[:8]}",
            entities=subgraph_entities,
            relations=subgraph_relations,
            metadata={**self.metadata, "parent_graph": self.id}
        )

    @property
    def size(self) -> Tuple[int, int]:
        """Return (|V|, |R|) - number of entities and relations."""
        return len(self.entities), len(self.relations)

    @property
    def density(self) -> float:
        """
        Compute relational density ρ = |R| / |V|².

        From Eq. 31 in paper: measures sparsity advantage of RCE.
        """
        n = len(self.entities)
        if n == 0:
            return 0.0
        return len(self.relations) / (n * n)


# ═════════════════════════════════════════════════════════════════════
# CONTEXT (Eq. 7: C = (intent, constraints, evidence, domain))
# ═════════════════════════════════════════════════════════════════════


class ContextIntent(Enum):
    """
    User intent classification for context-adaptive processing.

    Different intents require different module weights w_k(C) in Eq. 8.
    """

    QUERY = "query"              # Information retrieval
    VALIDATE = "validate"        # Consistency checking
    EXPLAIN = "explain"          # Explanation generation
    CALCULATE = "calculate"      # Numerical computation
    DIAGNOSE = "diagnose"        # Problem identification
    COMPARE = "compare"          # Comparison task
    REASON = "reason"            # Multi-step reasoning
    UNKNOWN = "unknown"          # Intent unclear


@dataclass
class Context:
    """
    Rich context C for context-adaptive coherence evaluation (Eq. 7).

    From paper Eq. 7: C = (intent, constraints, evidence, domain)
    Context determines module weights w_k(C) in coherence functional (Eq. 8).

    Attributes:
        intent: User intent classification
        domain: Domain of discourse ("medical", "legal", "financial", etc.)
        constraints: Must/cannot conditions from query
        evidence_requirements: What counts as valid evidence
        confidence_threshold: Minimum acceptable confidence
        retrieved_evidence: Retrieved documents/passages (for RAG integration)
        user_background: Optional user expertise level
    """

    intent: ContextIntent
    domain: str
    constraints: List[str] = field(default_factory=list)
    evidence_requirements: List[str] = field(default_factory=list)
    confidence_threshold: float = 0.6
    retrieved_evidence: List[Dict[str, Any]] = field(default_factory=list)
    user_background: Optional[str] = None

    def __post_init__(self):
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError(
                f"confidence_threshold must be in [0,1], got {self.confidence_threshold}"
            )


# ═════════════════════════════════════════════════════════════════════
# COHERENCE (Eq. 8: μ(Ω | C) = Σ w_k·μ_k(Ω | C))
# ═════════════════════════════════════════════════════════════════════


@dataclass
class CoherenceScore:
    """
    Coherence score with complete modular breakdown (Eq. 8).

    From paper Eq. 8: μ(Ω | C) = Σ_{k=1}^K w_k(C)·μ_k(Ω | C)

    The five modules μ_k are (Eqs. 9-13):
        - μ_units: Dimensional analysis and unit consistency
        - μ_time: Temporal ordering and chronological constraints
        - μ_arith: Arithmetic validity and numerical consistency
        - μ_coref: Coreference resolution and entity stability
        - μ_entail: Factual entailment and evidence grounding

    Attributes:
        overall: Final aggregated coherence score μ(Ω | C) in [0, 1]
        module_scores: Individual module scores {module_name: μ_k(Ω | C)}
        module_weights: Context-dependent weights {module_name: w_k(C)}
        violations: Detected inconsistencies
        confidence: Confidence in the scoring itself
        computation_time_ms: Time taken for coherence evaluation
    """

    overall: float
    module_scores: Dict[str, float]
    module_weights: Dict[str, float]
    violations: List[str] = field(default_factory=list)
    confidence: float = 1.0
    computation_time_ms: float = 0.0

    def __post_init__(self):
        if not 0.0 <= self.overall <= 1.0:
            raise ValueError(f"Overall score must be in [0,1], got {self.overall}")

        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be in [0,1], got {self.confidence}")

        # Validate module weights sum to 1.0 (allow small floating point error)
        if self.module_weights:
            weight_sum = sum(self.module_weights.values())
            if not (0.99 <= weight_sum <= 1.01):
                raise ValueError(
                    f"Module weights must sum to 1.0, got {weight_sum}: {self.module_weights}"
                )

    @property
    def is_coherent(self) -> bool:
        """Return True if no violations and score > 0.5."""
        return len(self.violations) == 0 and self.overall > 0.5


# ═════════════════════════════════════════════════════════════════════
# ANSWER (Eq. 15: (y, c) = R(Ω*, C))
# ═════════════════════════════════════════════════════════════════════


@dataclass
class Answer:
    """
    Final answer with full provenance and traceability (Eq. 15).

    From paper Eq. 15: (y, c) = R(Ω*, C)
    where y is the natural language answer and c is confidence.

    Provides complete output from RCE pipeline including:
    - Natural language answer
    - Supporting evidence subgraph Ω*
    - Coherence scores
    - Confidence and caveats
    - Evidence mapping for explainability

    Attributes:
        text: Generated natural language answer
        subgraph: Supporting evidence subgraph Ω*
        coherence: Overall coherence score
        confidence: Confidence in answer [0, 1]
        caveats: Limitations/uncertainties to communicate
        evidence_map: Mapping from claims to supporting evidence
        reasoning_trace: Complete reasoning trace for explainability
        computation_stats: Computational efficiency metrics
    """

    text: str
    subgraph: Graph
    coherence: CoherenceScore
    confidence: float
    caveats: List[str] = field(default_factory=list)
    evidence_map: Dict[str, List[str]] = field(default_factory=dict)
    reasoning_trace: List[Dict[str, Any]] = field(default_factory=list)
    computation_stats: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be in [0,1], got {self.confidence}")

    def explain(self, claim: str) -> str:
        """
        Generate human-readable explanation for a specific claim.

        Provides transparency and explainability by tracing the claim
        through the reasoning trace and evidence map.

        Args:
            claim: Claim to explain

        Returns:
            Human-readable explanation with evidence
        """
        if claim not in self.evidence_map:
            return f"No direct evidence found for claim: '{claim}'"

        evidence_items = self.evidence_map[claim]

        explanation_parts = [
            f"Explanation for claim: '{claim}'",
            "",
            f"Confidence: {self.confidence:.2%}",
            f"Coherence: {self.coherence.overall:.2%}",
            "",
            "Supporting Evidence:",
        ]

        for i, evidence in enumerate(evidence_items, 1):
            explanation_parts.append(f"  {i}. {evidence}")

        if self.caveats:
            explanation_parts.extend(["", "Caveats:"])
            for caveat in self.caveats:
                explanation_parts.append(f"  - {caveat}")

        return "\n".join(explanation_parts)

    def to_dict(self) -> Dict[str, Any]:
        """Convert answer to dictionary for serialization."""
        return {
            "text": self.text,
            "confidence": self.confidence,
            "coherence": {
                "overall": self.coherence.overall,
                "module_scores": self.coherence.module_scores,
                "module_weights": self.coherence.module_weights,
                "violations": self.coherence.violations,
            },
            "caveats": self.caveats,
            "evidence_count": len(self.evidence_map),
            "subgraph_size": self.subgraph.size,
            "subgraph_density": self.subgraph.density,
            "computation_stats": self.computation_stats,
        }
