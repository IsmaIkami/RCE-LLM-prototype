"""
RCE-LLM Graphizer: G: X → G

Implements candidate graph construction from input text (Eq. 6).
Maps input text X to complete relational graph G = (V, R, τ, σ).

This is a production implementation using:
- SpaCy for NER, dependency parsing, and POS tagging
- Pattern matching for relation extraction
- Heuristic rules for typed relation construction

Author: Ismail Sialyen
Based on: DOI 10.5281/zenodo.17360372
"""

import uuid
import re
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict
import time

try:
    import spacy
    from spacy.tokens import Doc, Token as SpacyToken
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    print("Warning: spaCy not available. Install with: pip install spacy && python -m spacy download en_core_web_sm")

from rce_llm.core.types import Entity, Relation, Graph


class Graphizer:
    """
    Graphizer G: X → G - Maps input text to candidate relational graph.

    Implements Eq. 6 from paper: G = (V, R, τ, σ)
    where:
        - V: entities extracted from text
        - R: relations between entities
        - τ: semantic type function
        - σ: confidence function

    This implementation uses spaCy for:
        1. Named Entity Recognition (NER) → entities V
        2. Dependency parsing → relations R
        3. POS tagging → type function τ
        4. Confidence estimation → σ
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Graphizer.

        Args:
            config: Configuration dictionary
                - spacy_model: SpaCy model name (default: "en_core_web_sm")
                - max_entities: Maximum entities to extract (default: 500)
                - max_relations: Maximum relations to extract (default: 2000)
                - confidence_threshold: Minimum confidence for extraction (default: 0.3)
        """
        self.config = config or {}

        self.spacy_model_name = self.config.get("spacy_model", "en_core_web_sm")
        self.max_entities = self.config.get("max_entities", 500)
        self.max_relations = self.config.get("max_relations", 2000)
        self.confidence_threshold = self.config.get("confidence_threshold", 0.3)

        # Initialize spaCy
        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load(self.spacy_model_name)
            except OSError:
                print(f"SpaCy model '{self.spacy_model_name}' not found. Downloading...")
                import subprocess
                subprocess.run(["python", "-m", "spacy", "download", self.spacy_model_name])
                self.nlp = spacy.load(self.spacy_model_name)
        else:
            self.nlp = None
            print("Warning: Graphizer initialized without spaCy - functionality limited")

        # Relation patterns (dependency-based)
        self.relation_patterns = self._initialize_relation_patterns()

    def _initialize_relation_patterns(self) -> Dict[str, List[str]]:
        """
        Initialize relation extraction patterns based on dependency labels.

        Returns mapping from relation types to dependency patterns.
        """
        return {
            "has_attribute": ["amod", "attr", "npadvmod"],
            "has_quantity": ["nummod", "quantmod"],
            "located_in": ["prep_in", "prep_at", "prep_on"],
            "part_of": ["prep_of", "poss"],
            "agent_of": ["nsubj", "agent"],
            "patient_of": ["dobj", "nsubjpass"],
            "temporal": ["prep_during", "prep_before", "prep_after", "prep_since"],
            "causal": ["prep_because", "prep_due"],
            "comparison": ["acomp", "prep_than"],
        }

    def graphize(self, text: str) -> Graph:
        """
        Main graphization function: X → G.

        Constructs complete candidate graph from input text.

        Args:
            text: Input text X

        Returns:
            Graph G = (V, R, τ, σ) representing all possible interpretations

        Raises:
            ValueError: If text is empty or invalid
            RuntimeError: If spaCy is not available
        """
        if not text or not text.strip():
            raise ValueError("Input text cannot be empty")

        if not self.nlp:
            raise RuntimeError("SpaCy not available - cannot perform graphization")

        start_time = time.time()

        # Process text with spaCy
        doc = self.nlp(text)

        # Extract entities (V)
        entities = self._extract_entities(doc, text)

        # Extract relations (R)
        relations = self._extract_relations(doc, entities, text)

        # Build graph
        graph = self._build_graph(entities, relations, text)

        graph.metadata["graphization_time_ms"] = (time.time() - start_time) * 1000

        return graph

    def _extract_entities(self, doc: Doc, original_text: str) -> List[Entity]:
        """
        Extract entities V from spaCy Doc.

        Uses:
        - Named entities from NER
        - Noun chunks
        - Numerical expressions
        - Dates and times

        Args:
            doc: SpaCy processed document
            original_text: Original input text

        Returns:
            List of extracted entities with types and confidence
        """
        entities = []
        entity_texts_seen = set()  # Avoid duplicates

        # Extract named entities
        for ent in doc.ents:
            if ent.text in entity_texts_seen:
                continue

            entity_id = f"ent_{uuid.uuid4().hex[:8]}"

            # Map spaCy entity types to semantic types
            semantic_type = self._map_spacy_label(ent.label_)

            # Confidence estimation (spaCy doesn't provide this, so we use heuristics)
            confidence = self._estimate_entity_confidence(ent)

            if confidence < self.confidence_threshold:
                continue

            entity = Entity(
                id=entity_id,
                text=ent.text,
                canonical_form=ent.text.lower().strip(),
                semantic_type=semantic_type,
                attributes={
                    "spacy_label": ent.label_,
                    "start_char": ent.start_char,
                    "end_char": ent.end_char,
                },
                confidence=confidence,
                source_span=(ent.start_char, ent.end_char),
            )

            entities.append(entity)
            entity_texts_seen.add(ent.text)

            if len(entities) >= self.max_entities:
                break

        # Extract noun chunks (additional entities)
        for chunk in doc.noun_chunks:
            if chunk.text in entity_texts_seen or len(entities) >= self.max_entities:
                continue

            # Skip if too short or too long
            if len(chunk.text) < 2 or len(chunk.text) > 100:
                continue

            entity_id = f"ent_{uuid.uuid4().hex[:8]}"

            entity = Entity(
                id=entity_id,
                text=chunk.text,
                canonical_form=chunk.text.lower().strip(),
                semantic_type="Concept",
                attributes={
                    "source": "noun_chunk",
                    "root": chunk.root.text,
                    "start_char": chunk.start_char,
                    "end_char": chunk.end_char,
                },
                confidence=0.7,  # Lower confidence for noun chunks
                source_span=(chunk.start_char, chunk.end_char),
            )

            entities.append(entity)
            entity_texts_seen.add(chunk.text)

        # Extract numerical expressions
        for token in doc:
            if token.like_num and token.text not in entity_texts_seen:
                entity_id = f"ent_{uuid.uuid4().hex[:8]}"

                entity = Entity(
                    id=entity_id,
                    text=token.text,
                    canonical_form=token.text,
                    semantic_type="Quantity",
                    attributes={
                        "numeric_value": self._parse_number(token.text),
                        "pos": token.pos_,
                    },
                    confidence=0.95,
                    source_span=(token.idx, token.idx + len(token.text)),
                )

                entities.append(entity)
                entity_texts_seen.add(token.text)

                if len(entities) >= self.max_entities:
                    break

        return entities

    def _extract_relations(
        self,
        doc: Doc,
        entities: List[Entity],
        original_text: str
    ) -> List[Relation]:
        """
        Extract relations R from spaCy dependency parse.

        Uses dependency patterns to identify typed relations between entities.

        Args:
            doc: SpaCy processed document
            entities: Extracted entities
            original_text: Original input text

        Returns:
            List of extracted relations
        """
        relations = []

        # Build entity lookup by text and position
        entity_by_span = {}
        for entity in entities:
            start, end = entity.source_span
            entity_by_span[(start, end)] = entity

        # Extract dependency-based relations
        for token in doc:
            # Find entity containing this token
            subject_entity = self._find_entity_by_token(token, entity_by_span)

            if not subject_entity:
                continue

            # Check dependencies
            for child in token.children:
                object_entity = self._find_entity_by_token(child, entity_by_span)

                if not object_entity or object_entity.id == subject_entity.id:
                    continue

                # Determine relation type from dependency label
                dep_label = child.dep_
                relation_type = self._map_dependency_to_relation(dep_label)

                if not relation_type:
                    continue

                # Create relation
                relation_id = f"rel_{uuid.uuid4().hex[:8]}"

                relation = Relation(
                    id=relation_id,
                    subject=subject_entity.id,
                    predicate=relation_type,
                    object=object_entity.id,
                    confidence=0.75,  # Dependency-based relations have moderate confidence
                    modality="factual",
                    evidence=[f"{token.text} --{dep_label}--> {child.text}"],
                    inferred=False,
                )

                relations.append(relation)

                if len(relations) >= self.max_relations:
                    break

            if len(relations) >= self.max_relations:
                break

        # Extract co-occurrence relations (entities near each other)
        relations.extend(self._extract_cooccurrence_relations(entities, original_text))

        return relations[:self.max_relations]

    def _extract_cooccurrence_relations(
        self,
        entities: List[Entity],
        text: str
    ) -> List[Relation]:
        """
        Extract relations based on entity co-occurrence in text.

        Entities appearing within a window are likely related.

        Args:
            entities: Extracted entities
            text: Original text

        Returns:
            List of co-occurrence relations
        """
        relations = []
        window_size = 50  # characters

        # Sort entities by position
        sorted_entities = sorted(entities, key=lambda e: e.source_span[0])

        for i, entity1 in enumerate(sorted_entities):
            for entity2 in sorted_entities[i + 1:]:
                # Check if within window
                distance = entity2.source_span[0] - entity1.source_span[1]

                if distance > window_size:
                    break  # Too far, stop

                if distance < 0:
                    continue  # Overlapping

                # Create co-occurrence relation
                relation_id = f"rel_{uuid.uuid4().hex[:8]}"

                relation = Relation(
                    id=relation_id,
                    subject=entity1.id,
                    predicate="related_to",
                    object=entity2.id,
                    confidence=0.5,  # Lower confidence for co-occurrence
                    modality="factual",
                    evidence=[f"Co-occurrence within {distance} chars"],
                    inferred=True,
                )

                relations.append(relation)

                if len(relations) >= 100:  # Limit co-occurrence relations
                    return relations

        return relations

    def _build_graph(
        self,
        entities: List[Entity],
        relations: List[Relation],
        original_text: str
    ) -> Graph:
        """
        Construct Graph object from entities and relations.

        Args:
            entities: Extracted entities
            relations: Extracted relations
            original_text: Original input text

        Returns:
            Complete graph G = (V, R, τ, σ)
        """
        entity_dict = {e.id: e for e in entities}

        # Count entity types
        entity_type_counts = defaultdict(int)
        for entity in entities:
            entity_type_counts[entity.semantic_type] += 1

        # Count relation types
        relation_type_counts = defaultdict(int)
        for relation in relations:
            relation_type_counts[relation.predicate] += 1

        metadata = {
            "source_text_length": len(original_text),
            "source_text_preview": original_text[:200],
            "entity_count": len(entities),
            "relation_count": len(relations),
            "entity_types": dict(entity_type_counts),
            "relation_types": dict(relation_type_counts),
            "density": len(relations) / (len(entities) ** 2) if entities else 0.0,
        }

        graph_id = f"graph_{uuid.uuid4().hex[:8]}"

        return Graph(
            id=graph_id,
            entities=entity_dict,
            relations=relations,
            metadata=metadata,
        )

    # Helper methods

    def _map_spacy_label(self, label: str) -> str:
        """Map spaCy entity label to semantic type."""
        mapping = {
            "PERSON": "Person",
            "ORG": "Organization",
            "GPE": "Location",
            "LOC": "Location",
            "DATE": "Date",
            "TIME": "Time",
            "MONEY": "Money",
            "QUANTITY": "Quantity",
            "CARDINAL": "Number",
            "ORDINAL": "Ordinal",
            "PERCENT": "Percentage",
            "PRODUCT": "Product",
            "EVENT": "Event",
            "WORK_OF_ART": "WorkOfArt",
            "LAW": "Law",
            "LANGUAGE": "Language",
        }
        return mapping.get(label, "Entity")

    def _estimate_entity_confidence(self, ent) -> float:
        """
        Estimate confidence for entity extraction.

        Uses heuristics:
        - Length of entity
        - Capitalization
        - Entity type
        """
        confidence = 0.8  # Base confidence

        # Adjust based on entity type (some types are more reliable)
        reliable_types = {"PERSON", "ORG", "GPE", "DATE", "MONEY", "QUANTITY"}
        if ent.label_ in reliable_types:
            confidence += 0.1

        # Adjust based on length
        if len(ent.text) < 2:
            confidence -= 0.3
        elif len(ent.text) > 50:
            confidence -= 0.2

        # Adjust based on capitalization (proper nouns more reliable)
        if ent.text[0].isupper():
            confidence += 0.05

        return min(1.0, max(0.0, confidence))

    def _parse_number(self, text: str) -> Optional[float]:
        """Parse number from text."""
        try:
            # Remove commas and parse
            cleaned = text.replace(",", "")
            return float(cleaned)
        except ValueError:
            return None

    def _find_entity_by_token(
        self,
        token: SpacyToken,
        entity_by_span: Dict[Tuple[int, int], Entity]
    ) -> Optional[Entity]:
        """Find entity containing this token."""
        token_start = token.idx
        token_end = token.idx + len(token.text)

        for (start, end), entity in entity_by_span.items():
            if start <= token_start < end or start < token_end <= end:
                return entity

        return None

    def _map_dependency_to_relation(self, dep_label: str) -> Optional[str]:
        """Map dependency label to relation type."""
        for relation_type, patterns in self.relation_patterns.items():
            if any(pattern in dep_label for pattern in patterns):
                return relation_type

        # Default mappings
        mapping = {
            "nsubj": "agent_of",
            "dobj": "patient_of",
            "pobj": "related_to",
            "amod": "has_attribute",
            "nummod": "has_quantity",
            "compound": "part_of",
        }

        return mapping.get(dep_label)
