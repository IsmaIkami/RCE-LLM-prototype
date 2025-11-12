"""
RCE-LLM Main Engine

Complete implementation of the RCE-LLM pipeline from publication.

Pipeline stages (Algorithm 1):
    1. Graph Construction: G = G(x)
    2. Context Extraction: C = E(x, G)
    3. Coherence Evaluation: μ(Ω | C)
    4. Actualization: Ω* = arg max μ(Ω | C)
    5. Rendering: (y, c) = R(Ω*, C)

Author: Ismail Sialyen
Based on: DOI 10.5281/zenodo.17360372
"""

import time
from typing import Optional, Dict, Any

from rce_llm.core.types import Answer, Graph, Context
from rce_llm.core.graphizer import Graphizer
from rce_llm.core.context_extractor import ContextExtractor
from rce_llm.core.renderer import AnswerRenderer
from rce_llm.modules.aggregator import CoherenceAggregator
from rce_llm.optimization.optimizer import ActualizationOptimizer


class RCEEngine:
    """
    Main RCE-LLM Engine implementing Algorithm 1 from paper.

    Replaces token-level likelihood maximization:
        Transformer: max Σ log P(x_t | x_{<t})

    With coherence optimization:
        RCE: max μ(Ω | C) subject to Φ(Ω)

    Usage:
        engine = RCEEngine()
        answer = engine.process("What is 60 km/h for 30 minutes in meters?")
        print(answer.text)
        print(f"Confidence: {answer.confidence:.2%}")
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize RCE Engine with all components.

        Args:
            config: Configuration dictionary with component configs:
                - graphizer: Graphizer config
                - context_extractor: Context extractor config
                - coherence: Coherence modules config
                - optimization: Optimization config
                - renderer: Renderer config
        """
        self.config = config or {}

        print("Initializing RCE-LLM Engine...")
        print("=" * 50)

        # Stage 1: Graphizer G: X → G (Eq. 6)
        print("Loading Graphizer...")
        self.graphizer = Graphizer(
            self.config.get("graphizer", {})
        )

        # Stage 2: Context Extractor E: X → C (Eq. 7)
        print("Loading Context Extractor...")
        self.context_extractor = ContextExtractor(
            self.config.get("context_extractor", {})
        )

        # Stage 3: Coherence Aggregator μ(Ω | C) (Eq. 8)
        print("Loading Coherence Modules...")
        self.coherence_aggregator = CoherenceAggregator(
            self.config.get("coherence", {})
        )

        # Stage 4: Actualization Optimizer Ω* = arg max μ(Ω | C) (Eq. 14)
        print("Loading Actualization Optimizer...")
        self.optimizer = ActualizationOptimizer(
            self.config.get("optimization", {})
        )

        # Stage 5: Answer Renderer R: 2^G × C → Y × [0,1] (Eq. 15)
        print("Loading Answer Renderer...")
        self.renderer = AnswerRenderer(
            self.config.get("renderer", {})
        )

        print("=" * 50)
        print("RCE-LLM Engine initialized successfully!")
        print()

    def process(
        self,
        text: str,
        retrieved_evidence: Optional[list] = None
    ) -> Answer:
        """
        Process query through complete RCE pipeline (Algorithm 1).

        Args:
            text: Input query text X
            retrieved_evidence: Optional retrieved documents for RAG integration

        Returns:
            Complete Answer with provenance and traceability

        Example:
            >>> engine = RCEEngine()
            >>> answer = engine.process("A car travels 60 km/h for 30 minutes. How far in meters?")
            >>> print(answer.text)
            >>> print(f"Coherence: {answer.coherence.overall:.2%}")
            >>> print(f"Confidence: {answer.confidence:.2%}")
        """
        if not text or not text.strip():
            raise ValueError("Input text cannot be empty")

        total_start_time = time.time()

        print(f"Processing query: {text[:100]}...")
        print()

        # Stage 1: Graph Construction G = G(x)
        print("[1/5] Graph Construction...")
        graph = self.graphizer.graphize(text)
        print(f"      → Extracted {len(graph.entities)} entities, {len(graph.relations)} relations")
        print(f"      → Density ρ = {graph.density:.4f}")
        print()

        # Stage 2: Context Extraction C = E(x, G)
        print("[2/5] Context Extraction...")
        if retrieved_evidence:
            context = self.context_extractor.extract_with_evidence(
                text, graph, retrieved_evidence
            )
        else:
            context = self.context_extractor.extract(text, graph)
        print(f"      → Intent: {context.intent.value}")
        print(f"      → Domain: {context.domain}")
        print(f"      → Constraints: {len(context.constraints)}")
        print()

        # Stage 3: Coherence Evaluation (integrated in optimization)
        print("[3/5] Coherence Evaluation...")
        initial_coherence = self.coherence_aggregator.evaluate(graph, context)
        print(f"      → Initial coherence: {initial_coherence.overall:.3f}")
        print(f"      → Module scores: {initial_coherence.module_scores}")
        print()

        # Stage 4: Actualization Optimization Ω* = arg max μ(Ω | C)
        print("[4/5] Actualization Optimization...")
        optimal_subgraph, coherence, opt_info = self.optimizer.optimize(
            graph, context, self.coherence_aggregator
        )
        print(f"      → Strategy: {opt_info.get('strategy_used', 'unknown')}")
        print(f"      → Selected {opt_info.get('entities_selected', 0)} entities")
        print(f"      → Final coherence: {coherence.overall:.3f}")
        print()

        # Stage 5: Answer Rendering (y, c) = R(Ω*, C)
        print("[5/5] Answer Rendering...")
        answer = self.renderer.render(
            optimal_subgraph, context, coherence, opt_info
        )
        print(f"      → Confidence: {answer.confidence:.3f}")
        print(f"      → Caveats: {len(answer.caveats)}")
        print()

        total_time_ms = (time.time() - total_start_time) * 1000
        answer.computation_stats["total_time_ms"] = total_time_ms

        print(f"Processing complete in {total_time_ms:.1f}ms")
        print("=" * 50)
        print()

        return answer

    def process_batch(self, texts: list) -> list:
        """
        Process multiple queries.

        Args:
            texts: List of query texts

        Returns:
            List of Answer objects
        """
        return [self.process(text) for text in texts]

    def get_stats(self) -> Dict[str, Any]:
        """
        Get engine statistics and configuration.

        Returns:
            Dictionary with engine info
        """
        return {
            "version": "1.0.0",
            "author": "Ismail Sialyen",
            "paper_doi": "10.5281/zenodo.17360372",
            "components": {
                "graphizer": "SpaCy-based NLP extraction",
                "context_extractor": "Rule-based intent and domain classification",
                "coherence_modules": [
                    "units", "temporal", "arithmetic", "coreference", "entailment"
                ],
                "optimizer": "Adaptive (ILP for small, beam search for large)",
                "renderer": "Template-based with evidence mapping",
            },
            "complexity": {
                "graphizer": "O(n) where n is text length",
                "coherence": "O(|R|·K·d) where |R|=relations, K=5 modules, d=feature_dim",
                "optimization_beam": "O(B·|R|·log|R|) where B=beam_size",
                "optimization_ilp": "O(2^|R|) exact for small graphs",
                "total": "O(|R|·K·d) dominant for sparse graphs",
            },
        }

    def explain_answer(self, answer: Answer) -> str:
        """
        Generate human-readable explanation of answer.

        Args:
            answer: Answer to explain

        Returns:
            Explanation text
        """
        explanation = [
            "Answer Explanation",
            "=" * 50,
            "",
            f"Query: {answer.subgraph.metadata.get('source_text_preview', 'N/A')}",
            "",
            "Answer:",
            f"  {answer.text}",
            "",
            f"Confidence: {answer.confidence:.2%}",
            f"Coherence: {answer.coherence.overall:.2%}",
            "",
            "Coherence Breakdown:",
        ]

        for module, score in answer.coherence.module_scores.items():
            weight = answer.coherence.module_weights.get(module, 0)
            explanation.append(f"  • {module}: {score:.2%} (weight: {weight:.2%})")

        if answer.caveats:
            explanation.extend(["", "Caveats:"])
            for caveat in answer.caveats:
                explanation.append(f"  • {caveat}")

        if answer.coherence.violations:
            explanation.extend(["", "Violations:"])
            for violation in answer.coherence.violations[:5]:  # Show first 5
                explanation.append(f"  • {violation}")

        explanation.extend([
            "",
            "Computation Statistics:",
            f"  • Total time: {answer.computation_stats.get('total_time_ms', 0):.1f}ms",
            f"  • Graphization: {answer.computation_stats.get('graphization_time_ms', 0):.1f}ms",
            f"  • Coherence: {answer.computation_stats.get('coherence_time_ms', 0):.1f}ms",
            f"  • Optimization: {answer.computation_stats.get('optimization_time_ms', 0):.1f}ms",
            "",
            "Graph Statistics:",
            f"  • Entities: {len(answer.subgraph.entities)}",
            f"  • Relations: {len(answer.subgraph.relations)}",
            f"  • Density: {answer.subgraph.density:.4f}",
        ])

        return "\n".join(explanation)
