"""
RCE-LLM: Relational Coherence Engine for Consistent and Energy-Efficient Language Modeling

Production implementation based on the research paper:
"RCE-LLM: A Relational Coherence Engine for Consistent and Energy-Efficient Language Modeling"
DOI: 10.5281/zenodo.17360372

Author: Ismail Sialyen
Email: is.sialyen@gmail.com
Date: October 15, 2025

This implementation provides:
1. Graphizer G: X → G for candidate graph construction (Eq. 6)
2. Context Extractor E: X → C for contextual information (Eq. 7)
3. Modular Coherence Functional μ(Ω | C) with 5 modules (Eq. 8)
4. Actualization Optimization Ω* = arg max μ(Ω | C) (Eq. 14)
5. Answer Renderer R: 2^G × C → Y × [0,1] (Eq. 15)

Theoretical Foundation:
    Replaces next-token prediction:    max Σ log P(x_t | x_{<t})
    With coherence optimization:       max μ(Ω | C) subject to Φ(Ω)
"""

__version__ = "1.0.0"
__author__ = "Ismail Sialyen"
__email__ = "is.sialyen@gmail.com"
__paper_doi__ = "10.5281/zenodo.17360372"
__license__ = "MIT"

from rce_llm.core.types import (
    Entity,
    Relation,
    Graph,
    Context,
    ContextIntent,
    CoherenceScore,
    Answer,
)

from rce_llm.core.engine import RCEEngine

__all__ = [
    "__version__",
    "__author__",
    "__email__",
    "__paper_doi__",
    # Core Types
    "Entity",
    "Relation",
    "Graph",
    "Context",
    "ContextIntent",
    "CoherenceScore",
    "Answer",
    # Engine
    "RCEEngine",
]
