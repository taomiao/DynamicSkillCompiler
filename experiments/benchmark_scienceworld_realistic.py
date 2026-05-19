#!/usr/bin/env python3
"""
ScienceWorld Benchmark - Realistic Task Descriptions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tests DSC 2.0 with more realistic task descriptions instead of just task IDs.
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent / "skillnet-ai" / "src"))

from skillnet_ai.compiler_v2 import CompilerV2


# Realistic task descriptions for ScienceWorld tasks
REALISTIC_TASKS = {
    "boil": "boil water in the pot using the stove",
    "find-animal": "find the ant in the environment and observe it",
    "find-plant": "locate the plant in the greenhouse and examine it",
    "freeze": "freeze the water by placing it in the freezer",
    "grow-fruit": "grow a fruit by planting seeds and watering them",
    "grow-plant": "grow a plant in the greenhouse using soil and water",
    "use-thermometer": "use the thermometer to measure the temperature of the substance",
    "test-conductivity": "test which objects conduct electricity using the circuit",
    "chemistry-mix": "mix two chemicals in a beaker",
    "melt": "melt the ice by heating it",
}


def main():
    print("\n" + "="*80)
    print("SCIENCEWORLD REALISTIC TASK BENCHMARK")
    print("="*80 + "\n")

    # Initialize compiler
    current_dir = Path(__file__).parent
    signatures_path = current_dir.parent / "skillnet-ai" / "data" / "skill_signatures.json"
    skills_dir = current_dir / "src" / "skills" / "scienceworld"

    compiler = CompilerV2(
        domain="scienceworld",
        signatures_path=str(signatures_path),
        skills_dir=str(skills_dir),
        use_skills=True
    )

    print(f"✓ Compiler initialized with {len(compiler.local_skill_library.skills)} skills\n")

    results = []

    for task_id, task_description in REALISTIC_TASKS.items():
        print(f"\n{'─'*80}")
        print(f"Task: {task_id}")
        print(f"Description: {task_description}")
        print(f"{'─'*80}")

        try:
            # Compile task
            optimized_ops, report = compiler.compile(task_description)

            result = {
                "task_id": task_id,
                "description": task_description,
                "success": len(optimized_ops) > 0,
                "ops_extracted": len(optimized_ops),
                "ops_before": report.atomic_ops_before_opt,
                "reduction_pct": report.optimization_stats.get("reduction_pct", 0),
                "token_cost": report.estimated_token_cost,
                "parsed_entities": report.parsed_entities,
                "parsed_intents": report.parsed_intents,
            }

            results.append(result)

            if len(optimized_ops) > 0:
                print(f"✅ Success: {report.atomic_ops_before_opt} → {len(optimized_ops)} ops "
                      f"({result['reduction_pct']:.1f}% reduction)")
                print(f"   Token cost: {result['token_cost']:.1f}")
                print(f"   Operations:")
                for i, op in enumerate(optimized_ops[:5], 1):
                    print(f"     {i}. {op.op_type.name}({op.target})")
                if len(optimized_ops) > 5:
                    print(f"     ... and {len(optimized_ops) - 5} more")
            else:
                print(f"⚠️  No operations extracted")
                print(f"   Parsed: {result['parsed_entities']} entities, {result['parsed_intents']} intents")

        except Exception as e:
            print(f"❌ Error: {e}")
            results.append({
                "task_id": task_id,
                "description": task_description,
                "success": False,
                "error": str(e)
            })

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}\n")

    successful = [r for r in results if r.get("success", False)]

    print(f"Total tasks: {len(results)}")
    print(f"Successful: {len(successful)}/{len(results)} ({len(successful)/len(results)*100:.1f}%)")

    if successful:
        avg_ops_before = sum(r['ops_before'] for r in successful) / len(successful)
        avg_ops_after = sum(r['ops_extracted'] for r in successful) / len(successful)
        avg_reduction = sum(r['reduction_pct'] for r in successful) / len(successful)
        avg_token = sum(r['token_cost'] for r in successful) / len(successful)

        print(f"Avg operations: {avg_ops_before:.1f} → {avg_ops_after:.1f}")
        print(f"Avg reduction: {avg_reduction:.1f}%")
        print(f"Avg token cost: {avg_token:.1f}")

    # Save results
    output_path = Path("results/scienceworld_realistic_benchmark.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        "timestamp": datetime.now().isoformat(),
        "mode": "realistic-descriptions",
        "results": results,
        "summary": {
            "total": len(results),
            "successful": len(successful),
            "success_rate": len(successful) / len(results) if results else 0,
        }
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
