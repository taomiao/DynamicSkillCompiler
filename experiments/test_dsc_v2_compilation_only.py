#!/usr/bin/env python3
"""
DSC 2.0 Compilation Test - No Environment Required
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tests compilation performance across all three domains
without requiring full environment setup.
"""

import sys
from pathlib import Path

# Add path
sys.path.insert(0, str(Path(__file__).parent.parent / "skillnet-ai" / "src"))

from skillnet_ai.compiler_v2 import CompilerV2


def test_compilation_performance():
    """Test compilation across domains"""

    # Initialize compiler
    signatures_path = Path(__file__).parent.parent / "skillnet-ai" / "data" / "skill_signatures.json"

    if not signatures_path.exists():
        print(f"❌ Signatures file not found: {signatures_path}")
        return

    print("\n" + "="*70)
    print("DSC 2.0 COMPILATION PERFORMANCE TEST")
    print("="*70 + "\n")

    # Test tasks for each domain
    test_cases = {
        "alfworld": [
            "put a hot cup in the cabinet",
            "find a pen and put it on the desk",
            "take the knife from the drawer and clean it",
            "heat a mug then place it in the fridge",
            "cool a tomato and put it on the countertop",
            "examine the laptop in bright light",
            "put two pillows on the couch",
        ],
        "scienceworld": [
            "boil water in a pot",
            "measure the temperature of the substance",
            "mix chemical A with chemical B",
            "grow a plant in the greenhouse",
            "identify which animal is living",
            "measure the melting point of ice",
            "heat a substance using the bunsen burner",
        ],
        "webshop": [
            "find a red shirt under $30",
            "search for wireless headphones",
            "buy the cheapest running shoes",
            "find a laptop with 16GB RAM",
            "search for a blue backpack",
        ],
    }

    all_results = {}

    for domain, tasks in test_cases.items():
        print(f"\n{'─'*70}")
        print(f"TESTING: {domain.upper()}")
        print(f"{'─'*70}\n")

        compiler = CompilerV2(domain=domain, signatures_path=str(signatures_path))

        results = []

        for idx, task in enumerate(tasks, 1):
            print(f"[{idx}/{len(tasks)}] {task}")

            try:
                optimized_ops, report = compiler.compile(task)

                result = {
                    "task": task,
                    "success": True,
                    "entities": report.parsed_entities,
                    "intents": report.parsed_intents,
                    "atomic_ops_before": report.atomic_ops_before_opt,
                    "atomic_ops_after": report.atomic_ops_after_opt,
                    "reduction_pct": report.optimization_stats["reduction_pct"],
                    "matched_signatures": report.matched_signatures,
                    "token_cost": report.estimated_token_cost,
                }

                results.append(result)

                print(f"  ✅ {result['atomic_ops_before']} → {result['atomic_ops_after']} ops "
                      f"({result['reduction_pct']:.1f}% reduction)")
                print(f"     Matched: {result['matched_signatures']}/{result['atomic_ops_after']}, "
                      f"Tokens: {result['token_cost']:.1f}")

            except Exception as e:
                print(f"  ❌ {e}")
                results.append({
                    "task": task,
                    "success": False,
                    "error": str(e)
                })

        all_results[domain] = results

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}\n")

    for domain, results in all_results.items():
        successful = [r for r in results if r.get("success", False)]

        if successful:
            avg_reduction = sum(r["reduction_pct"] for r in successful) / len(successful)
            avg_token_cost = sum(r["token_cost"] for r in successful) / len(successful)
            avg_ops_before = sum(r["atomic_ops_before"] for r in successful) / len(successful)
            avg_ops_after = sum(r["atomic_ops_after"] for r in successful) / len(successful)
            match_rate = sum(r["matched_signatures"] for r in successful) / sum(r["atomic_ops_after"] for r in successful if r["atomic_ops_after"] > 0) if any(r["atomic_ops_after"] > 0 for r in successful) else 0

            print(f"{domain.upper()}:")
            print(f"  Success: {len(successful)}/{len(results)}")
            print(f"  Avg operations: {avg_ops_before:.1f} → {avg_ops_after:.1f}")
            print(f"  Avg reduction: {avg_reduction:.1f}%")
            print(f"  Avg token cost: {avg_token_cost:.1f}")
            print(f"  Signature match rate: {match_rate:.1%}")
            print()


if __name__ == "__main__":
    test_compilation_performance()
