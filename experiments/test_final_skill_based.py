#!/usr/bin/env python3
"""
Final Test: Skill-Based DSC 2.0
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "skillnet-ai" / "src"))

from skillnet_ai.compiler_v2 import CompilerV2


def main():
    test_cases = {
        "alfworld": [
            "put a hot cup in the cabinet",
            "find a pen and put it on the desk",
            "cool a tomato and put it on the countertop",
        ],
        "scienceworld": [
            "find the thermometer",
            "boil water in a pot",
            "measure the temperature",
        ],
        "webshop": [
            "find a red shirt",
            "search for shoes",
        ],
    }

    signatures_path = Path(__file__).parent.parent / "skillnet-ai" / "data" / "skill_signatures.json"

    print("\n" + "="*80)
    print("FINAL TEST: Skill-Based DSC 2.0")
    print("="*80 + "\n")

    for domain, tasks in test_cases.items():
        print(f"\n{domain.upper()}:")
        print(f"{'─'*80}")

        skills_dir = Path(__file__).parent / "src" / "skills" / domain

        compiler = CompilerV2(
            domain=domain,
            signatures_path=str(signatures_path),
            skills_dir=str(skills_dir),
            use_skills=True
        )

        for task in tasks:
            try:
                optimized_ops, report = compiler.compile(task)
                print(f"\n  ✓ \"{task}\"")
                print(f"     {report.atomic_ops_after_opt} ops, "
                      f"{report.matched_signatures} matched, "
                      f"{report.estimated_token_cost:.1f} tokens")
            except Exception as e:
                print(f"\n  ✗ \"{task}\"")
                print(f"     Error: {e}")

    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()
