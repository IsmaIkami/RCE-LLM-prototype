"""
RCE-LLM Coherence Modules

Implements the five coherence modules μ_k from Eq. 8-13:
    μ_units: Dimensional analysis and unit consistency (Eq. 9)
    μ_time: Temporal ordering and chronological constraints (Eq. 10)
    μ_arith: Arithmetic validity and numerical consistency (Eq. 11)
    μ_coref: Coreference resolution and entity stability (Eq. 12)
    μ_entail: Factual entailment and evidence grounding (Eq. 13)

Author: Ismail Sialyen
Based on: DOI 10.5281/zenodo.17360372
"""

from rce_llm.modules.base import CoherenceModule
from rce_llm.modules.units import UnitsCoherenceModule
from rce_llm.modules.temporal import TemporalCoherenceModule
from rce_llm.modules.arithmetic import ArithmeticCoherenceModule
from rce_llm.modules.coreference import CoreferenceCoherenceModule
from rce_llm.modules.entailment import EntailmentCoherenceModule
from rce_llm.modules.aggregator import CoherenceAggregator

__all__ = [
    "CoherenceModule",
    "UnitsCoherenceModule",
    "TemporalCoherenceModule",
    "ArithmeticCoherenceModule",
    "CoreferenceCoherenceModule",
    "EntailmentCoherenceModule",
    "CoherenceAggregator",
]
