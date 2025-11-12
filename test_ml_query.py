#!/usr/bin/env python3
"""
Test script to trace ML query execution
"""

import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from rce_llm import RCEEngine

print("=" * 80)
print("TESTING: Machine Learning Query")
print("=" * 80)

engine = RCEEngine()

query = "what is machine learning?"
print(f"\nQuery: {query}\n")

answer = engine.process(query)

print("\n" + "=" * 80)
print("FINAL ANSWER:")
print("=" * 80)
print(answer.text)
print()
print(f"Confidence: {answer.confidence:.2%}")
print(f"Coherence: {answer.coherence.overall:.2%}")
print()
