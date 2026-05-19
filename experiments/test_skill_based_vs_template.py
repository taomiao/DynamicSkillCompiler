#!/usr/bin/env python3
"""
Test DSC 2.0 with Skill-Based Operation Extraction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tests the new skill-based approach vs template-based approach
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "skillnet-ai" / "src"))

from skillnet_ai.compiler_v2 import CompilerV2


def test_skill_based_extraction():
    """Test skill-based extraction on all three domains"""

    # Test tasks for each domain
    test_cases = {
        "alfworld": [
            "put a hot cup in the cabinet",
            "find a pen and put it on the desk",
            "cool a tomato and put it on the countertop",
        ],
        "scienceworld": [
            "boil water in a pot",
            "measure the temperature of the substance",
            "find the thermometer",
        ],
        "webshop": [
            "find a red shirt under $30",
            "buy the cheapest running shoes",
        ],
    }

    signatures_path = Path(__file__).parent.parent / "skillnet-ai" / "data" / "skill_signatures.json"

    print("\n" + "="*80)
    print("DSC 2.0: SKILL-BASED vs TEMPLATE-BASED COMPARISON")
    print("="*80 + "\n")

    for domain, tasks in test_cases.items():
        print(f"\n{'─'*80}")
        print(f"DOMAIN: {domain.upper()}")
        print(f"{'─'*80}\n")

        skills_dir = Path(__file__).parent / "src" / "skills" / domain

        # Test both approaches
        for approach, use_skills in [("SKILL-BASED", True), ("TEMPLATE-BASED", False)]:
            print(f"\n  {'■'*40}")
            print(f"  APPROACH: {approach}")
            print(f"  {'■'*40}\n")

            compiler = CompilerV2(
                domain=domain,
                signatures_path=str(signatures_path),
                skills_dir=str(skills_dir) if use_skills else None,
                use_skills=use_skills
            )

            # Show library stats
            stats = compiler.stats()
            if "local_skill_library" in stats:
                lib_stats = stats["local_skill_library"]
                print(f"  Local Skills: {lib_stats['total_skills']} loaded")
                print(f"  Intent Types: {lib_stats['intent_types']}")
                print(f"  Intents: {lib_stats['intents']}")
            print()

            results = []

            for task in tasks:
                print(f"  Task: \"{task}\"")

                try:
                    optimized_ops, report = compiler.compile(task)

                    result = {
                        "task": task,
                        "success": True,
                        "approach": approach,
                        "atomic_ops": report.atomic_ops_after_opt,
                        "matched_signatures": report.matched_signatures,
                        "token_cost": report.estimated_token_cost,
                    }

                    print(f"    ✓ {report.atomic_ops_after_opt} ops, "
                          f"{report.matched_signatures} matched, "
                          f"{report.estimated_token_cost:.1f} tokens\n")

                except Exception as e:
                    print(f"    ✗ Error: {e}\n")
                    result = {
                        "task": task,
                        "success": False,
                        "approach": approach,
                        "error": str(e)
                    }

                results.append(result)

    print(f"\n{'='*80}")
    print("TEST COMPLETE")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    test_skill_based_extraction()
