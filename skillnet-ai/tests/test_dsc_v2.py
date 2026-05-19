#!/usr/bin/env python3
"""
DSC 2.0 Test Script
~~~~~~~~~~~~~~~~~~~

Tests the complete CompilerV2 pipeline without environment.
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from skillnet_ai.compiler_v2 import CompilerV2


def test_compilation():
    """Test basic compilation"""
    print("\n" + "="*70)
    print("DSC 2.0 - COMPILATION TEST")
    print("="*70 + "\n")

    # Initialize compiler (without signatures for now)
    compiler = CompilerV2(domain="alfworld")

    # Test tasks
    test_tasks = [
        "put a hot cup in the cabinet",
        "find a pen and put it on the desk",
        "take the knife from the drawer",
        "heat a mug then place it in the fridge",
    ]

    for i, task in enumerate(test_tasks, 1):
        print(f"\n{'─'*70}")
        print(f"TEST {i}/{len(test_tasks)}")
        print(f"{'─'*70}")

        try:
            optimized_ops, report = compiler.compile(task)

            print(f"\n📊 Results:")
            print(f"   Entities extracted: {report.parsed_entities}")
            print(f"   Intents identified: {report.parsed_intents}")
            print(f"   Atomic operations: {report.atomic_ops_before_opt}")
            print(f"   After optimization: {report.atomic_ops_after_opt}")
            print(f"   Reduction: {report.optimization_stats['reduction_pct']:.1f}%")
            print(f"   Estimated token cost: {report.estimated_token_cost:.1f}")

            print(f"\n✅ PASS")

        except Exception as e:
            print(f"\n❌ FAIL: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*70}")
    print("ALL TESTS COMPLETE")
    print(f"{'='*70}\n")


def test_components():
    """Test individual components"""
    print("\n" + "="*70)
    print("DSC 2.0 - COMPONENT TESTS")
    print("="*70 + "\n")

    # Test QueryParser
    print("1. Testing QueryParser...")
    from skillnet_ai.compiler_v2.frontend import QueryParser

    parser = QueryParser(domain="alfworld")
    ast = parser.parse("put a hot cup in the cabinet")

    print(f"   Entities: {[e.name for e in ast.entities]}")
    print(f"   Intents: {[i.intent_type for i in ast.intents]}")
    print(f"   ✅ QueryParser OK\n")

    # Test IntentDecomposer
    print("2. Testing IntentDecomposer...")
    from skillnet_ai.compiler_v2.frontend import IntentDecomposer

    decomposer = IntentDecomposer(domain="alfworld")
    atomic_ops = decomposer.decompose(ast)

    print(f"   Atomic ops generated: {len(atomic_ops)}")
    print(f"   Op types: {[op.op_type.name for op in atomic_ops[:5]]}")
    print(f"   ✅ IntentDecomposer OK\n")

    # Test IROptimizer
    print("3. Testing IROptimizer...")
    from skillnet_ai.compiler_v2.midend import IROptimizer

    optimizer = IROptimizer()
    optimized = optimizer.optimize(atomic_ops)
    stats = optimizer.stats(atomic_ops, optimized)

    print(f"   Before: {stats['original_ops']} ops")
    print(f"   After: {stats['optimized_ops']} ops")
    print(f"   Reduction: {stats['reduction_pct']:.1f}%")
    print(f"   ✅ IROptimizer OK\n")

    # Test StateTracker
    print("4. Testing StateTracker...")
    from skillnet_ai.compiler_v2.backend import StateTracker

    tracker = StateTracker(domain="alfworld")
    tracker.update(
        "go to loc 1",
        "You arrive at loc 1. On the countertop 1, you see a cup 1."
    )
    tracker.update(
        "take cup 1 from countertop 1",
        "You pick up the cup 1 from the countertop 1."
    )

    print(f"   Current location: {tracker.state.current_location}")
    print(f"   Inventory: {tracker.state.inventory}")
    print(f"   Object locations: {tracker.state.object_locations}")
    print(f"   ✅ StateTracker OK\n")

    print(f"{'='*70}")
    print("ALL COMPONENT TESTS PASSED")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    test_components()
    test_compilation()
