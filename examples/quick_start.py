"""
RCE-LLM Quick Start Example

Demonstrates basic usage of the RCE Engine.

Author: Ismail Sialyen
Based on: DOI 10.5281/zenodo.17360372
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rce_llm import RCEEngine


def main():
    """Run quick start examples."""
    print("="  * 70)
    print("RCE-LLM Quick Start Examples")
    print("=" * 70)
    print()

    # Initialize engine
    print("Initializing RCE Engine...")
    engine = RCEEngine()
    print()

    # Example 1: Unit conversion (F1 benchmark type)
    print("\n" + "=" * 70)
    print("Example 1: Unit Conversion (Tests μ_units module)")
    print("=" * 70)

    query1 = "A car travels 60 km/h for 30 minutes. How far in meters?"
    print(f"\nQuery: {query1}")
    print("\n" + "-" * 70)

    answer1 = engine.process(query1)

    print("\nRESULTS:")
    print(f"Answer: {answer1.text}")
    print(f"Confidence: {answer1.confidence:.2%}")
    print(f"Coherence: {answer1.coherence.overall:.2%}")
    print(f"\nModule Scores:")
    for module, score in answer1.coherence.module_scores.items():
        weight = answer1.coherence.module_weights[module]
        print(f"  • {module:12s}: {score:.2%} (weight: {weight:.2%})")

    if answer1.caveats:
        print(f"\nCaveats:")
        for caveat in answer1.caveats:
            print(f"  • {caveat}")

    # Example 2: Simple factual query
    print("\n" + "=" * 70)
    print("Example 2: Factual Query (Tests μ_entail module)")
    print("=" * 70)

    query2 = "Who is the president of the United States?"
    print(f"\nQuery: {query2}")
    print("\n" + "-" * 70)

    answer2 = engine.process(query2)

    print("\nRESULTS:")
    print(f"Answer: {answer2.text}")
    print(f"Confidence: {answer2.confidence:.2%}")
    print(f"Coherence: {answer2.coherence.overall:.2%}")

    # Example 3: Arithmetic reasoning (F3 benchmark type)
    print("\n" + "=" * 70)
    print("Example 3: Arithmetic Reasoning (Tests μ_arith module)")
    print("=" * 70)

    query3 = "If Alice has 5 apples and Bob has 3 more apples than Alice, how many apples does Bob have?"
    print(f"\nQuery: {query3}")
    print("\n" + "-" * 70)

    answer3 = engine.process(query3)

    print("\nRESULTS:")
    print(f"Answer: {answer3.text}")
    print(f"Confidence: {answer3.confidence:.2%}")
    print(f"Coherence: {answer3.coherence.overall:.2%}")

    # Show detailed explanation for last example
    print("\n" + "=" * 70)
    print("Detailed Explanation for Example 3")
    print("=" * 70)
    print()
    explanation = engine.explain_answer(answer3)
    print(explanation)

    # Show engine statistics
    print("\n" + "=" * 70)
    print("Engine Statistics")
    print("=" * 70)
    print()
    stats = engine.get_stats()
    print(f"Version: {stats['version']}")
    print(f"Author: {stats['author']}")
    print(f"Paper DOI: {stats['paper_doi']}")
    print()
    print("Components:")
    for component, desc in stats['components'].items():
        if isinstance(desc, list):
            print(f"  • {component}: {', '.join(desc)}")
        else:
            print(f"  • {component}: {desc}")
    print()
    print("Complexity Analysis:")
    for operation, complexity in stats['complexity'].items():
        print(f"  • {operation}: {complexity}")

    print("\n" + "=" * 70)
    print("Quick Start Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
