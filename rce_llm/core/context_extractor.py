"""
RCE-LLM Context Extractor: E: X → C

Implements context extraction from input text and graph (Eq. 7).
Maps input X to context C = (intent, constraints, evidence, domain).

The context C determines module weights w_k(C) in the coherence functional (Eq. 8).

Author: Ismail Sialyen
Based on: DOI 10.5281/zenodo.17360372
"""

import re
from typing import Dict, List, Optional, Any
from collections import Counter

from rce_llm.core.types import Context, ContextIntent, Graph


class ContextExtractor:
    """
    Context Extractor E: X → C

    Implements Eq. 7 from paper: C = (intent, domain, constraints, evidence)

    Extracts rich contextual information that determines:
    1. Module weights w_k(C) in coherence functional (Eq. 8)
    2. Optimization constraints Φ(Ω) in actualization (Eq. 14)
    3. Rendering strategy R(Ω*, C) (Eq. 15)

    Uses:
    - Intent classification from query patterns
    - Domain identification from entity types and keywords
    - Constraint extraction from modal and temporal expressions
    - Evidence requirements from domain and intent
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Context Extractor.

        Args:
            config: Configuration dictionary
                - default_domain: Default domain if undetected (default: "general")
                - confidence_threshold: Default confidence threshold (default: 0.6)
        """
        self.config = config or {}

        self.default_domain = self.config.get("default_domain", "general")
        self.default_confidence_threshold = self.config.get("confidence_threshold", 0.6)

        # Intent patterns
        self.intent_patterns = self._initialize_intent_patterns()

        # Domain keywords
        self.domain_keywords = self._initialize_domain_keywords()

        # Constraint patterns
        self.constraint_patterns = self._initialize_constraint_patterns()

    def _initialize_intent_patterns(self) -> Dict[ContextIntent, List[str]]:
        """
        Initialize intent classification patterns.

        Returns:
            Mapping from intents to regex patterns
        """
        return {
            ContextIntent.QUERY: [
                r"^what\s+(is|are|was|were)",
                r"^who\s+(is|are|was|were)",
                r"^where\s+(is|are|was|were)",
                r"^when\s+(did|does|do|will)",
                r"^how\s+many",
                r"tell me about",
                r"explain",
                r"describe",
            ],
            ContextIntent.CALCULATE: [
                r"\bhow\s+much\b",
                r"\bhow\s+many\b",
                r"\bcalculate\b",
                r"\bcompute\b",
                r"\bwhat\s+is.*\+",
                r"\d+\s*[\+\-\*/]\s*\d+",
                r"\bsum\b",
                r"\btotal\b",
                r"\baverage\b",
            ],
            ContextIntent.VALIDATE: [
                r"\bis\s+it\s+true",
                r"\bverify\b",
                r"\bcheck\b",
                r"\bcorrect\b",
                r"\bconsistent\b",
                r"\bvalid\b",
            ],
            ContextIntent.COMPARE: [
                r"\bcompare\b",
                r"\bdifference\s+between\b",
                r"\bversus\b",
                r"\bvs\.?\b",
                r"\bbetter\b",
                r"\bworse\b",
            ],
            ContextIntent.REASON: [
                r"\bwhy\b",
                r"\bhow\s+does\b",
                r"\breason\b",
                r"\bcause\b",
                r"\bexplain\s+why\b",
            ],
            ContextIntent.EXPLAIN: [
                r"\bexplain\b",
                r"\bdescribe\b",
                r"\bwhat\s+does.*mean\b",
                r"\btell\s+me\s+about\b",
            ],
            ContextIntent.DIAGNOSE: [
                r"\bdiagnose\b",
                r"\bidentify\s+the\s+problem\b",
                r"\bwhat\s+is\s+wrong\b",
                r"\btroubleshoot\b",
            ],
        }

    def _initialize_domain_keywords(self) -> Dict[str, List[str]]:
        """
        Initialize domain classification keywords.

        Returns:
            Mapping from domains to keyword lists
        """
        return {
            "medical": [
                "patient", "drug", "dosage", "disease", "symptom", "treatment",
                "diagnosis", "physician", "hospital", "medicine", "therapy",
                "prescription", "clinical", "medical", "healthcare",
            ],
            "legal": [
                "law", "court", "judge", "attorney", "lawsuit", "contract",
                "regulation", "statute", "legal", "defendant", "plaintiff",
                "jurisdiction", "ruling", "legislation",
            ],
            "financial": [
                "money", "investment", "stock", "bond", "portfolio", "revenue",
                "profit", "loss", "bank", "financial", "currency", "market",
                "trading", "asset", "liability", "credit", "debt",
            ],
            "scientific": [
                "experiment", "hypothesis", "theory", "research", "study",
                "data", "analysis", "result", "conclusion", "method",
                "scientific", "laboratory", "measurement", "observation",
            ],
            "technical": [
                "system", "software", "hardware", "algorithm", "code",
                "program", "computer", "network", "database", "server",
                "technical", "technology", "implementation",
            ],
            "mathematical": [
                "equation", "theorem", "proof", "calculation", "formula",
                "function", "variable", "mathematics", "algebra", "geometry",
                "calculus", "number", "integer", "fraction",
            ],
        }

    def _initialize_constraint_patterns(self) -> Dict[str, List[str]]:
        """
        Initialize constraint extraction patterns.

        Returns:
            Mapping from constraint types to patterns
        """
        return {
            "must": [
                r"\bmust\b",
                r"\brequired\b",
                r"\bmandatory\b",
                r"\bnecessary\b",
                r"\bhas\s+to\b",
            ],
            "cannot": [
                r"\bcannot\b",
                r"\bcan't\b",
                r"\bmust\s+not\b",
                r"\bmustn't\b",
                r"\bprohibited\b",
                r"\bforbidden\b",
            ],
            "should": [
                r"\bshould\b",
                r"\brecommended\b",
                r"\bsuggested\b",
                r"\bpreferred\b",
            ],
            "temporal": [
                r"\bbefore\b",
                r"\bafter\b",
                r"\bduring\b",
                r"\bwhile\b",
                r"\bsince\b",
                r"\buntil\b",
            ],
        }

    def extract(self, text: str, graph: Graph) -> Context:
        """
        Main extraction function: (X, G) → C.

        Extracts context from input text and candidate graph.

        Args:
            text: Input text X
            graph: Candidate graph G

        Returns:
            Context C = (intent, domain, constraints, evidence)

        Raises:
            ValueError: If text is empty
        """
        if not text or not text.strip():
            raise ValueError("Input text cannot be empty")

        # Extract intent
        intent = self._extract_intent(text)

        # Extract domain
        domain = self._extract_domain(text, graph)

        # Extract constraints
        constraints = self._extract_constraints(text)

        # Determine evidence requirements
        evidence_requirements = self._determine_evidence_requirements(domain, intent)

        # Determine confidence threshold
        confidence_threshold = self._determine_confidence_threshold(domain, intent)

        context = Context(
            intent=intent,
            domain=domain,
            constraints=constraints,
            evidence_requirements=evidence_requirements,
            confidence_threshold=confidence_threshold,
        )

        return context

    def _extract_intent(self, text: str) -> ContextIntent:
        """
        Extract user intent from text.

        Uses pattern matching to classify intent.

        Args:
            text: Input text

        Returns:
            Classified intent
        """
        text_lower = text.lower()

        # Score each intent
        intent_scores = {}

        for intent, patterns in self.intent_patterns.items():
            score = 0
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    score += 1

            if score > 0:
                intent_scores[intent] = score

        # Return highest scoring intent
        if intent_scores:
            return max(intent_scores, key=intent_scores.get)

        return ContextIntent.UNKNOWN

    def _extract_domain(self, text: str, graph: Graph) -> str:
        """
        Extract domain from text and graph.

        Uses:
        1. Keyword matching in text
        2. Entity type distribution in graph

        Args:
            text: Input text
            graph: Candidate graph

        Returns:
            Detected domain
        """
        text_lower = text.lower()

        # Score domains by keyword presence
        domain_scores = {}

        for domain, keywords in self.domain_keywords.items():
            score = 0
            for keyword in keywords:
                if keyword in text_lower:
                    score += 1

            if score > 0:
                domain_scores[domain] = score

        # Also consider entity types in graph
        entity_types = graph.metadata.get("entity_types", {})

        # Medical domain indicators
        if "Drug" in entity_types or "Disease" in entity_types:
            domain_scores["medical"] = domain_scores.get("medical", 0) + 2

        # Financial domain indicators
        if "Money" in entity_types or "Percentage" in entity_types:
            domain_scores["financial"] = domain_scores.get("financial", 0) + 2

        # Date/Time suggests temporal reasoning
        if "Date" in entity_types or "Time" in entity_types:
            domain_scores["general"] = domain_scores.get("general", 0) + 1

        # Quantity/Number suggests mathematical
        if "Quantity" in entity_types or "Number" in entity_types:
            domain_scores["mathematical"] = domain_scores.get("mathematical", 0) + 1

        # Return highest scoring domain
        if domain_scores:
            return max(domain_scores, key=domain_scores.get)

        return self.default_domain

    def _extract_constraints(self, text: str) -> List[str]:
        """
        Extract constraints from text.

        Identifies modal and temporal constraints:
        - Must/cannot conditions
        - Temporal ordering
        - Should/preferred conditions

        Args:
            text: Input text

        Returns:
            List of extracted constraints
        """
        constraints = []
        text_lower = text.lower()

        for constraint_type, patterns in self.constraint_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text_lower)
                for match in matches:
                    # Extract context around match
                    start = max(0, match.start() - 30)
                    end = min(len(text), match.end() + 30)
                    context = text[start:end].strip()

                    constraint = f"[{constraint_type}] {context}"
                    constraints.append(constraint)

        return constraints

    def _determine_evidence_requirements(
        self,
        domain: str,
        intent: ContextIntent
    ) -> List[str]:
        """
        Determine what counts as valid evidence for this context.

        Different domains and intents require different evidence standards.

        Args:
            domain: Detected domain
            intent: User intent

        Returns:
            List of evidence requirements
        """
        requirements = []

        # Domain-specific requirements
        if domain == "medical":
            requirements.extend([
                "Peer-reviewed clinical studies",
                "FDA approvals",
                "Medical guidelines",
            ])
        elif domain == "legal":
            requirements.extend([
                "Legal precedents",
                "Statutes and regulations",
                "Court rulings",
            ])
        elif domain == "financial":
            requirements.extend([
                "Financial reports",
                "Market data",
                "Regulatory filings",
            ])
        elif domain == "scientific":
            requirements.extend([
                "Peer-reviewed publications",
                "Experimental data",
                "Replicated results",
            ])

        # Intent-specific requirements
        if intent == ContextIntent.VALIDATE:
            requirements.append("Multiple independent sources")
        elif intent == ContextIntent.CALCULATE:
            requirements.append("Verifiable numerical data")
        elif intent == ContextIntent.REASON:
            requirements.append("Causal chain of evidence")

        return requirements if requirements else ["General reliable sources"]

    def _determine_confidence_threshold(
        self,
        domain: str,
        intent: ContextIntent
    ) -> float:
        """
        Determine confidence threshold based on domain and intent.

        High-stakes domains require higher confidence.

        Args:
            domain: Detected domain
            intent: User intent

        Returns:
            Confidence threshold in [0, 1]
        """
        # Base threshold
        threshold = self.default_confidence_threshold

        # Adjust based on domain
        if domain in ["medical", "legal"]:
            threshold += 0.2  # High-stakes domains
        elif domain in ["financial"]:
            threshold += 0.15
        elif domain in ["scientific"]:
            threshold += 0.1

        # Adjust based on intent
        if intent == ContextIntent.VALIDATE:
            threshold += 0.1  # Validation requires high confidence
        elif intent == ContextIntent.DIAGNOSE:
            threshold += 0.15  # Diagnosis requires high confidence

        return min(0.95, threshold)  # Cap at 0.95

    def extract_with_evidence(
        self,
        text: str,
        graph: Graph,
        retrieved_documents: List[Dict[str, Any]]
    ) -> Context:
        """
        Extract context with retrieved evidence (for RAG integration).

        Args:
            text: Input text
            graph: Candidate graph
            retrieved_documents: Retrieved evidence documents

        Returns:
            Context with evidence
        """
        context = self.extract(text, graph)

        # Add retrieved evidence
        context.retrieved_evidence = retrieved_documents

        # Update evidence requirements based on retrieved docs
        if retrieved_documents:
            doc_sources = [doc.get("source", "unknown") for doc in retrieved_documents]
            context.evidence_requirements.append(
                f"Evidence from: {', '.join(set(doc_sources))}"
            )

        return context
