#!/usr/bin/env python3
"""
DSC 2.0 Benchmark - Compilation Performance Test
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tests DSC 2.0 compilation on sample tasks from all three domains.
Measures: compilation success, operation count, token cost, optimization rate.
"""

import sys
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent / "skillnet-ai" / "src"))

from skillnet_ai.compiler_v2 import CompilerV2


def load_sample_tasks():
    """Load sample tasks from each domain"""
    return {
        "alfworld": [
            "put a clean apple in the fridge",
            "heat a mug and put it in the coffee maker",
            "examine the painting under the lamp",
            "put a cool potato on the counter",
            "find two books and put them on the shelf",
            "clean a knife and put it in the drawer",
            "cool a tomato and put it on the dining table",
            "heat a cup and place it on the desk",
            "find a pen and put it in the cabinet",
            "put a hot plate in the sink",
        ],
        "scienceworld": [
            "find the thermometer in the workshop",
            "measure the temperature of the substance",
            "boil water in the pot",
            "mix the two chemicals in a beaker",
            "grow a plant in the greenhouse",
            "use the bunsen burner to heat the flask",
            "identify which object conducts electricity",
            "measure the melting point of ice",
            "find a metal object",
            "activate the circuit",
        ],
        "webshop": [
            "find a red shirt under $30",
            "search for running shoes",
            "find a laptop with 16GB RAM",
            "buy the cheapest backpack",
            "find wireless headphones",
            "search for a blue dress",
            "find shoes size 10",
            "search for books about python",
            "find a phone case",
            "search for a coffee maker",
        ],
    }


def benchmark_domain(domain, tasks, skills_dir, signatures_path):
    """Benchmark DSC 2.0 on a specific domain"""

    print(f"\n{'='*80}")
    print(f"BENCHMARKING: {domain.upper()}")
    print(f"{'='*80}\n")

    # Initialize compiler with skill-based extraction
    compiler = CompilerV2(
        domain=domain,
        signatures_path=str(signatures_path),
        skills_dir=str(skills_dir),
        use_skills=True
    )

    print(f"Compiler initialized:")
    stats = compiler.stats()
    if 'local_skill_library' in stats:
        lib_stats = stats['local_skill_library']
        print(f"  Local skills: {lib_stats['total_skills']}")
        print(f"  Intent types: {lib_stats['intent_types']}")
    print()

    results = []

    for idx, task in enumerate(tasks, 1):
        print(f"[{idx}/{len(tasks)}] {task}")

        try:
            # Compile task
            optimized_ops, report = compiler.compile(task)

            result = {
                "task_id": idx,
                "task": task,
                "success": True,
                "parsed_entities": report.parsed_entities,
                "parsed_intents": report.parsed_intents,
                "atomic_ops_before": report.atomic_ops_before_opt,
                "atomic_ops_after": report.atomic_ops_after_opt,
                "reduction_pct": report.optimization_stats.get("reduction_pct", 0),
                "matched_signatures": report.matched_signatures,
                "token_cost": report.estimated_token_cost,
            }

            results.append(result)

            # Summary
            if result['atomic_ops_after'] > 0:
                print(f"  ✓ {result['atomic_ops_before']} → {result['atomic_ops_after']} ops "
                      f"({result['reduction_pct']:.1f}% reduction), "
                      f"{result['matched_signatures']} matched, "
                      f"{result['token_cost']:.1f} tokens\n")
            else:
                print(f"  ⚠️  No operations extracted (0 ops)\n")

        except Exception as e:
            print(f"  ✗ Error: {e}\n")
            results.append({
                "task_id": idx,
                "task": task,
                "success": False,
                "error": str(e)
            })

    return results


def compute_summary(domain, results):
    """Compute summary statistics"""
    successful = [r for r in results if r.get('success', False)]
    with_ops = [r for r in successful if r.get('atomic_ops_after', 0) > 0]

    if not successful:
        return {
            "domain": domain,
            "total_tasks": len(results),
            "success_rate": 0.0,
        }

    return {
        "domain": domain,
        "total_tasks": len(results),
        "successful_compilations": len(successful),
        "success_rate": len(successful) / len(results),
        "tasks_with_ops": len(with_ops),
        "ops_extraction_rate": len(with_ops) / len(results) if results else 0,
        "avg_ops_before": sum(r.get('atomic_ops_before', 0) for r in with_ops) / len(with_ops) if with_ops else 0,
        "avg_ops_after": sum(r.get('atomic_ops_after', 0) for r in with_ops) / len(with_ops) if with_ops else 0,
        "avg_reduction": sum(r.get('reduction_pct', 0) for r in with_ops) / len(with_ops) if with_ops else 0,
        "avg_token_cost": sum(r.get('token_cost', 0) for r in successful) / len(successful),
        "avg_matched_rate": sum(r.get('matched_signatures', 0) / max(r.get('atomic_ops_after', 1), 1) for r in with_ops) / len(with_ops) if with_ops else 0,
    }


def main():
    print("\n" + "="*80)
    print("DSC 2.0 SKILL-BASED BENCHMARK")
    print("="*80)

    current_dir = Path(__file__).parent
    signatures_path = current_dir.parent / "skillnet-ai" / "data" / "skill_signatures.json"

    if not signatures_path.exists():
        print(f"\n⚠️  Signatures file not found: {signatures_path}")
        print("Please run: python skillnet-ai/tools/extract_signatures.py")
        return

    # Load sample tasks
    sample_tasks = load_sample_tasks()

    # Run benchmarks
    all_results = {}
    all_summaries = {}

    for domain in ["alfworld", "scienceworld", "webshop"]:
        skills_dir = current_dir / "src" / "skills" / domain

        if not skills_dir.exists():
            print(f"\n⚠️  Skills directory not found: {skills_dir}")
            continue

        results = benchmark_domain(
            domain,
            sample_tasks[domain],
            skills_dir,
            signatures_path
        )

        all_results[domain] = results
        all_summaries[domain] = compute_summary(domain, results)

    # Print summary
    print(f"\n{'='*80}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*80}\n")

    for domain, summary in all_summaries.items():
        print(f"{domain.upper()}:")
        print(f"  Total tasks: {summary['total_tasks']}")
        print(f"  Compilation success: {summary.get('successful_compilations', 0)}/{summary['total_tasks']} ({summary.get('success_rate', 0):.1%})")
        print(f"  Tasks with ops: {summary.get('tasks_with_ops', 0)}/{summary['total_tasks']} ({summary.get('ops_extraction_rate', 0):.1%})")
        if summary.get('tasks_with_ops', 0) > 0:
            print(f"  Avg operations: {summary.get('avg_ops_before', 0):.1f} → {summary.get('avg_ops_after', 0):.1f}")
            print(f"  Avg reduction: {summary.get('avg_reduction', 0):.1f}%")
            print(f"  Avg token cost: {summary.get('avg_token_cost', 0):.1f}")
            print(f"  Avg match rate: {summary.get('avg_matched_rate', 0):.1%}")
        print()

    # Save results
    output_path = Path("results/dsc_v2_skill_benchmark.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        "timestamp": datetime.now().isoformat(),
        "mode": "skill-based",
        "results": all_results,
        "summaries": all_summaries,
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"Results saved to: {output_path}\n")


if __name__ == "__main__":
    main()
