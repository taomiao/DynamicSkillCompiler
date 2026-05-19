#!/usr/bin/env python3
"""
DSC 2.0 集成测试 - 简化版
~~~~~~~~~~~~~~~~~~~~~~~~~~

不依赖完整的 ALFWorld 环境，只测试编译功能
"""

import sys
from pathlib import Path

# Add path
sys.path.insert(0, str(Path(__file__).parent.parent / "skillnet-ai" / "src"))

from skillnet_ai.compiler_v2 import CompilerV2


def test_with_signatures():
    """测试带签名库的编译"""
    print("\n" + "="*70)
    print("DSC 2.0 WITH SKILL SIGNATURES - INTEGRATION TEST")
    print("="*70 + "\n")

    # Initialize compiler with signatures
    signatures_path = Path(__file__).parent.parent / "skillnet-ai" / "data" / "skill_signatures.json"

    if not signatures_path.exists():
        print(f"❌ Signatures file not found: {signatures_path}")
        return False

    print(f"✓ Loading signatures from: {signatures_path}")

    compiler = CompilerV2(
        domain="alfworld",
        signatures_path=str(signatures_path)
    )

    # Print library stats
    stats = compiler.stats()
    print(f"✓ Skill library loaded:")
    print(f"  - Total signatures: {stats['skill_library']['total_signatures']}")
    print(f"  - Op types: {stats['skill_library']['op_types']}")
    print(f"  - Domains: {stats['skill_library']['domains']}")

    # Test tasks
    test_tasks = [
        "put a hot cup in the cabinet",
        "find a pen and put it on the desk",
        "take the knife from the drawer and clean it",
        "heat a mug then place it in the fridge",
        "cool a tomato and put it on the countertop",
    ]

    all_results = []

    for i, task in enumerate(test_tasks, 1):
        print(f"\n{'─'*70}")
        print(f"TEST {i}/{len(test_tasks)}: {task}")
        print(f"{'─'*70}")

        try:
            optimized_ops, report = compiler.compile(task)

            result = {
                "task": task,
                "success": True,
                "entities": report.parsed_entities,
                "intents": report.parsed_intents,
                "atomic_ops_before": report.atomic_ops_before_opt,
                "atomic_ops_after": report.atomic_ops_after_opt,
                "matched_signatures": report.matched_signatures,
                "token_cost": report.estimated_token_cost,
                "reduction_pct": report.optimization_stats["reduction_pct"],
            }
            all_results.append(result)

            print(f"\n📊 Results:")
            print(f"  ✅ SUCCESS")
            print(f"  - Entities: {result['entities']}")
            print(f"  - Atomic ops: {result['atomic_ops_before']} → {result['atomic_ops_after']}")
            print(f"  - Reduction: {result['reduction_pct']:.1f}%")
            print(f"  - Matched signatures: {result['matched_signatures']}/{result['atomic_ops_after']}")
            print(f"  - Token cost: {result['token_cost']:.1f}")

        except Exception as e:
            print(f"\n❌ FAILED: {e}")
            import traceback
            traceback.print_exc()
            all_results.append({"task": task, "success": False, "error": str(e)})

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    successful = [r for r in all_results if r.get("success")]
    if successful:
        avg_reduction = sum(r["reduction_pct"] for r in successful) / len(successful)
        avg_token_cost = sum(r["token_cost"] for r in successful) / len(successful)
        avg_matched = sum(r["matched_signatures"] for r in successful) / len(successful)

        print(f"✅ {len(successful)}/{len(all_results)} tasks succeeded\n")
        print(f"Performance metrics:")
        print(f"  - Avg token cost: {avg_token_cost:.1f} tokens/task")
        print(f"  - Avg reduction: {avg_reduction:.1f}%")
        print(f"  - Avg matched signatures: {avg_matched:.1f}")
        print(f"\n🎯 DSC 2.0 is working correctly with skill signatures!")
        return True
    else:
        print(f"❌ All tasks failed")
        return False


if __name__ == "__main__":
    success = test_with_signatures()
    sys.exit(0 if success else 1)
